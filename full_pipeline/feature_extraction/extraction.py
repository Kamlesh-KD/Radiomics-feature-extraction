import json
from pathlib import Path

import numpy as np
import pandas as pd
from radiomics import featureextractor, imageoperations

from scripts import load_image, image_path, mask_path, task_name


def make_extractor(settings, feature_classes):
    """PyRadiomics extractor with the given settings and only the given feature classes enabled."""
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for feature_class in feature_classes:
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def load_image_and_mask(case, side, channel, registration, bias_correction):
    """Raw MRI channel and breast mask (SimpleITK images), with optional bias correction / registration."""
    image = load_image(image_path(case, channel))
    if bias_correction:
        from n4 import load_bias_corrected_channel
        image = load_bias_corrected_channel(case, channel)
    if registration and channel != 0:
        from registration import load_registered_channel
        image = load_registered_channel(case, channel, image)
    mask = load_image(mask_path(case, side))
    return image, mask


def preprocess(extractor, image, mask):
    """PyRadiomics pre-processing (normalize, resample, check mask), exactly as inside extractor.execute()."""
    image, mask = extractor.loadImage(image, mask, None, **extractor.settings)
    bounding_box, corrected_mask = imageoperations.checkMask(image, mask, **extractor.settings)
    if corrected_mask is not None:
        mask = corrected_mask
    return image, mask, bounding_box


def extract_features(extractor, image, mask, bounding_box, filter_module, filter_args, include_shape=False):
    """Features of every image made by one filter, as {'<image type>_<class>_<feature>': value}."""
    features = {}
    if include_shape:
        features.update(extractor.computeShape(image, mask, bounding_box, **extractor.settings))
    kwargs = {**extractor.settings, **filter_args}
    for filtered_image, image_type, type_kwargs in filter_module.generate(image, mask, **kwargs):
        cropped_image, cropped_mask = imageoperations.cropToTumorMask(filtered_image, mask, bounding_box)
        features.update(extractor.computeFeatures(cropped_image, cropped_mask, image_type, **type_kwargs))
    return {name: float(np.real(value)) for name, value in features.items()}


def run_all_tasks(extractor, tasks, filters, registration, bias_correction, cache_dir, overwrite=False):
    """Extract every filter for every (case, side, channel) task.

    The features of each task are saved to <cache_dir>/<task>.json. Filters already in that file are skipped,
    so the run can be stopped and restarted. Use overwrite=True after changing settings or filter arguments.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, (case, side, channel) in enumerate(tasks, start=1):
        name = task_name(case, side, channel)
        cache_path = cache_dir / f"{name}.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() and not overwrite else {}
        todo = [f for f in filters if f not in cache]
        if not todo:
            print(f"[{i}/{len(tasks)}] {name}: already done")
            continue
        try:
            image, mask = load_image_and_mask(case, side, channel, registration, bias_correction)
            image, mask, bounding_box = preprocess(extractor, image, mask)
            for filter_name in todo:
                module, args = filters[filter_name]
                cache[filter_name] = extract_features(extractor, image, mask, bounding_box, module, args,
                                                      include_shape=(filter_name == "raw_image"))
        except Exception as error:                      # keep going with the next task
            print(f"[{i}/{len(tasks)}] {name}: FAILED {error!r}")
            continue
        cache_path.write_text(json.dumps(cache))
        print(f"[{i}/{len(tasks)}] {name}: done ({', '.join(todo)})")


def save_feature_tables(tasks, filter_names, cache_dir, feature_dir):
    """Combine the per-task JSON files into one CSV per filter: features_<filter>.csv. Returns the tables."""
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for filter_name in filter_names:
        rows = []
        for case, side, channel in tasks:
            cache_path = Path(cache_dir) / f"{task_name(case, side, channel)}.json"
            if not cache_path.exists():
                continue
            cache = json.loads(cache_path.read_text())
            if filter_name in cache:
                rows.append({"case": case, "side": side, "channel": channel,
                             "task": task_name(case, side, channel), **cache[filter_name]})
        table = pd.DataFrame(rows)
        table.to_csv(feature_dir / f"features_{filter_name}.csv", index=False)
        tables[filter_name] = table
    return tables
