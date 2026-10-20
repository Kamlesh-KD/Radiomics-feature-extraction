import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from radiomics import featureextractor, imageoperations

from timer import Timer

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PIPELINE_DIR.parent
PARAMS_PATH = PROJECT_DIR / "config_files" / "pyradiomics_params.yaml"
for folder in (PIPELINE_DIR, PROJECT_DIR / "scripts", PROJECT_DIR / "pre_processing", PROJECT_DIR):
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

from scripts import (config, load_image, image_path, mask_path, list_case, task_name, largest_slice_index,  # noqa: E402
                     load_reference_features, compare_features, summarize_comparison)
from filter import FILTERS, apply_filter, filter_args  # noqa: E402

ID_COLUMNS = ["case", "side", "channel", "task"]

def load_params(path=PARAMS_PATH):
    """pyradiomics_params.yaml with `setting` (setup.yaml PARAMS.setting + overrides), `options` and dirs."""
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    settings = dict(config["PARAMS"]["setting"])
    settings.update(params.get("setting_overrides") or {})
    # setup.yaml writes Python's None (e.g. weightingNorm: None), which YAML reads as the string 'None'.
    # PyRadiomics would treat 'None' as an unknown norm and merge the 13 GLCM/GLRLM angles instead of
    # averaging them, so convert it back to a real None.
    params["setting"] = {k: (None if v == "None" else v) for k, v in settings.items()}
    params["options"] = params.get("options") or dict(config["PIPELINE_OPTIONS"])
    params["feature_dir"] = PROJECT_DIR / params["output"]["feature_dir"]
    params["figure_dir"] = PROJECT_DIR / params["output"]["figure_dir"]
    return params


def enabled_filters(params):
    return [name for name, entry in params["filters"].items() if entry.get("enabled", True)]


def filter_table(params):
    """Overview: output file, PyRadiomics image type and arguments of every filter."""
    rows = []
    for name in params["filters"]:
        module = FILTERS[name]
        rows.append({"filter": name, "csv": f"features_{name}.csv", "pyradiomics_image_type": module.IMAGE_TYPE,
                     "custom": module.CUSTOM, "enabled": params["filters"][name].get("enabled", True),
                     "args": filter_args(name, params["filters"][name].get("args"))})
    return pd.DataFrame(rows)


def all_tasks():
    """(case, side, channel) for every case, breast side and MRI channel."""
    return [(case, side, channel) for case in list_case() for side in config["SIDES"]
            for channel in range(config["N_CHANNELS"])]


# ----------------------------------------------------------------------------------------- pre-processing
def load_task_inputs(case, side, channel, options):
    """Raw image and mask (SimpleITK) of one task after optional bias correction / registration to channel 0."""
    image = load_image(image_path(case, channel))
    if options.get("bias_correction", False):
        from n4 import load_bias_corrected_channel
        image = load_bias_corrected_channel(case, channel)
    if options.get("registration", False) and channel != 0:
        from registration import load_registered_channel
        image = load_registered_channel(case, channel, image)
    mask = load_image(mask_path(case, side))
    return image, mask


def make_extractor(params):
    """RadiomicsFeatureExtractor with the pipeline settings and feature classes (plus shape)."""
    extractor = featureextractor.RadiomicsFeatureExtractor(**params["setting"])
    extractor.disableAllFeatures()
    for feature_class in params["featureClass"]:
        extractor.enableFeatureClassByName(feature_class)
    if params.get("shape_in_raw_image", True):
        extractor.enableFeatureClassByName("shape")
    return extractor


def preprocess(extractor, image, mask):
    """PyRadiomics pre-processing: (image, mask, bounding box) exactly as used inside execute()."""
    image, mask = extractor.loadImage(image, mask, None, **extractor.settings)
    bounding_box, corrected_mask = imageoperations.checkMask(image, mask, **extractor.settings)
    if corrected_mask is not None:
        mask = corrected_mask
    return image, mask, bounding_box


def preprocess_task(task, params, extractor=None, options=None):
    """Dict with the pre-processed SimpleITK image/mask, their arrays [z, y, x] and spacing."""
    extractor = extractor or make_extractor(params)
    options = params["options"] if options is None else options
    image, mask, bounding_box = preprocess(extractor, *load_task_inputs(*task, options))
    label = extractor.settings.get("label", 1)
    return {"task": task, "image": image, "mask": mask, "bounding_box": bounding_box,
            "image_array": sitk.GetArrayFromImage(image).astype(np.float64),
            "mask_array": sitk.GetArrayFromImage(mask) == label,
            "spacing_zyx": np.array(image.GetSpacing())[::-1]}


# ---------------------------------------------------------------------------------------------- extraction
def extract_filter(extractor, image, mask, bounding_box, name, args=None, include_shape=False):
    """Features of every image produced by one filter ('<image type>_<class>_<feature>': float)."""
    settings = extractor.settings
    features = OrderedDict()
    if include_shape:
        features.update(extractor.computeShape(image, mask, bounding_box, **settings))
    kwargs = dict(settings)
    kwargs.update(filter_args(name, args))
    for filtered, type_name, type_kwargs in FILTERS[name].generate(image, mask, **kwargs):
        cropped_image, cropped_mask = imageoperations.cropToTumorMask(filtered, mask, bounding_box)
        features.update(extractor.computeFeatures(cropped_image, cropped_mask, type_name, **type_kwargs))
    return OrderedDict((key, float(value)) for key, value in features.items())


def includes_shape(name, params):
    return name == "raw_image" and params.get("shape_in_raw_image", True)


def filter_signature(name, params, options):
    """Everything that changes a filter's features; a cached result is reused only if this is unchanged."""
    return json.dumps({"setting": params["setting"], "options": options, "featureClass": params["featureClass"],
                       "shape": includes_shape(name, params),
                       "args": filter_args(name, params["filters"][name].get("args"))}, sort_keys=True, default=str)


def extract_task(task, params, extractor=None, filters=None, options=None):
    """Pre-process one task once and extract every requested filter -> ({filter: features}, {filter: seconds})."""
    extractor = extractor or make_extractor(params)
    pre = preprocess_task(task, params, extractor, options)
    results, seconds = OrderedDict(), OrderedDict()
    for name in filters or enabled_filters(params):
        start = time.perf_counter()
        results[name] = extract_filter(extractor, pre["image"], pre["mask"], pre["bounding_box"], name,
                                       params["filters"][name].get("args"), includes_shape(name, params))
        seconds[name] = time.perf_counter() - start
    return results, seconds


def task_cache_path(params, task):
    return Path(params["feature_dir"]) / "tasks" / f"{task_name(*task)}.json"


def run_extraction(tasks, params, filters=None, resume=True):
    """Extract all tasks (resuming from the per-task cache) and write one CSV per filter. Returns the tables."""
    filters = filters or enabled_filters(params)
    extractor = make_extractor(params)
    failures = []
    for i, task in enumerate(tasks, start=1):
        path = task_cache_path(params, task)
        cache = json.loads(path.read_text()) if (resume and path.exists()) else {}
        cache.setdefault("features", {})
        cache.setdefault("seconds", {})
        cache.setdefault("signature", {})
        todo = [name for name in filters
                if cache["signature"].get(name) != filter_signature(name, params, params["options"])]
        label = f"[{i}/{len(tasks)}] {task_name(*task)}"
        if not todo:
            print(f"{label}: cached")
            continue
        try:
            results, seconds = extract_task(task, params, extractor, todo)
        except Exception as error:                      # keep going; report at the end
            failures.append((task_name(*task), repr(error)))
            print(f"{label}: FAILED {error!r}")
            continue
        for name in todo:
            cache["features"][name] = results[name]
            cache["seconds"][name] = seconds[name]
            cache["signature"][name] = filter_signature(name, params, params["options"])
        cache["task"] = list(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
        timing = ", ".join(f"{name} {s:.0f}s" for name, s in seconds.items())
        print(f"{label}: {sum(len(r) for r in results.values())} features in {sum(seconds.values()):.0f} s ({timing})")
    if failures:
        print("\nFailed tasks:", *failures, sep="\n  ")
    return write_feature_tables(tasks, params, filters)


def write_feature_tables(tasks, params, filters=None):
    """Collect the per-task cache into features_<filter>.csv (one row per task). Returns {filter: DataFrame}."""
    filters = filters or enabled_filters(params)
    feature_dir = Path(params["feature_dir"])
    feature_dir.mkdir(parents=True, exist_ok=True)
    caches = {}
    for task in tasks:
        path = task_cache_path(params, task)
        if path.exists():
            caches[task] = json.loads(path.read_text())
    tables = OrderedDict()
    for name in filters:
        rows = []
        for task, cache in caches.items():
            if name in cache["features"]:
                case, side, channel = task
                rows.append({"case": case, "side": side, "channel": channel, "task": task_name(*task),
                             **cache["features"][name]})
        table = pd.DataFrame(rows)
        table.to_csv(feature_dir / f"features_{name}.csv", index=False)
        tables[name] = table
    return tables


def summarize_tables(tables):
    """Rows, image types and features per class of every table."""
    rows = []
    for name, table in tables.items():
        features = [c for c in table.columns if c not in ID_COLUMNS]
        image_types = sorted({c.split("_")[0] for c in features})
        classes = pd.Series([c.split("_")[1] for c in features]).value_counts()
        rows.append({"file": f"features_{name}.csv", "rows": len(table), "features": len(features),
                     "image_types": len(image_types), **classes.to_dict(),
                     "missing_values": int(table[features].isna().sum().sum()) if features else 0})
    return pd.DataFrame(rows).fillna(0)


def compare_with_reference(task, params, filters=None):
    """Our per-filter features for one task (with REFERENCE_OPTIONS) vs the PyRadiomics 3.1.0 reference JSON."""
    filters = [f for f in (filters or enabled_filters(params)) if not FILTERS[f].CUSTOM]
    results, _ = extract_task(task, params, filters=filters, options=config["REFERENCE_OPTIONS"])
    reference = load_reference_features(*task)
    tables = []
    for name, features in results.items():
        table = compare_features(features, reference)
        table.insert(0, "filter", name)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


# ------------------------------------------------------------------------------------------------- display
def filter_arrays(name, pre, params):
    """OrderedDict(image type -> float array [z, y, x]) of one filter applied to a pre-processed task."""
    images = apply_filter(name, pre["image"], pre["mask"], params["setting"], params["filters"][name].get("args"))
    return OrderedDict((k, sitk.GetArrayFromImage(v).astype(np.float64)) for k, v in images.items())


def roi_statistics(arrays, pre):
    """ROI statistics of every image, plus its Pearson correlation with the raw image inside the ROI."""
    mask, original = pre["mask_array"], pre["image_array"][pre["mask_array"]]
    rows = []
    for name, array in arrays.items():
        values = array[mask]
        correlation = np.corrcoef(values, original)[0, 1] if values.std() > 0 else np.nan
        rows.append({"image": name, "min": values.min(), "p01": np.percentile(values, 1), "mean": values.mean(),
                     "p99": np.percentile(values, 99), "max": values.max(), "std": values.std(),
                     "corr_with_raw": correlation})
    return pd.DataFrame(rows)


def roi_crop_slices(mask_array, axis_index, margin=5):
    """In-plane (row, column) slices of the ROI bounding box (+margin) on an axial slice."""
    rows, cols = np.nonzero(mask_array[axis_index])
    return (slice(max(rows.min() - margin, 0), rows.max() + margin + 1),
            slice(max(cols.min() - margin, 0), cols.max() + margin + 1))


def wrap_title(name, width=24):
    """Split a long image type name over two lines at the hyphen closest to its middle."""
    if len(name) <= width or "-" not in name:
        return name
    hyphens = [i for i, ch in enumerate(name) if ch == "-"]
    cut = min(hyphens, key=lambda i: abs(i - len(name) / 2))
    return name[:cut + 1] + "\n" + name[cut + 1:]


def show_filter(name, arrays, pre, include_raw=True, z=None, save_dir=None):
    """Raw vs filtered images of one filter: axial slice with ROI contour, ROI-only zoom, ROI histogram."""
    import matplotlib.pyplot as plt
    from visualize import show_image_with_mask

    mask, spacing = pre["mask_array"], pre["spacing_zyx"]
    z = largest_slice_index(mask, 0) if z is None else z
    images = OrderedDict([("raw image", pre["image_array"])]) if include_raw else OrderedDict()
    images.update(arrays)
    rows_crop, cols_crop = roi_crop_slices(mask, z)

    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(2.7 * n + 0.5, 8.4), squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1, 0.6]})
    for column, (image_name, array) in enumerate(images.items()):
        values = array[mask]
        window = tuple(np.percentile(values, [1, 99]))
        if window[0] == window[1]:
            window = (window[0] - 1, window[1] + 1)
        show_image_with_mask(axes[0, column], array, mask, spacing, 0, z, title=wrap_title(image_name), window=window)
        roi_only = np.where(mask, array, np.nan)[:, rows_crop, cols_crop]
        show_image_with_mask(axes[1, column], roi_only, None, spacing, 0, z, window=window, cmap="viridis")
        ax = axes[2, column]
        ax.hist(values, bins=80, range=window, color="#4a6fa5" if column or not include_raw else "#777777")
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("axial slice\n(orange = ROI)")
    axes[1, 0].set_ylabel("inside ROI")
    axes[2, 0].set_ylabel("ROI histogram")
    fig.suptitle(f"{task_name(*pre['task'])}: {name} (axial slice z = {z})", y=1.0)
    fig.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f"{task_name(*pre['task'])}_{name}.png", dpi=90, bbox_inches="tight")
    return fig
