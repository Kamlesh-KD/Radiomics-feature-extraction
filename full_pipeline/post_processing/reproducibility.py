"""Perturbed feature extractions for the reproducibility (ICC) check.

Every task (case, side, channel) is extracted again under small perturbations, applied on top of the pipeline
pre-processing (PIPELINE_OPTIONS of setup.yaml, i.e. registration on):
    volume_minus10   ROI shrunk by 10 % of its volume
    volume_plus10    ROI grown by 10 %
    n4               image divided by its N4 bias field
    noise            Gaussian noise added with the image's own estimated noise level

Features are extracted exactly like the feature CSVs (feature_extraction/extraction.py with PARAMS.setting of
setup.yaml and the filters + arguments of pyradiomics_params.yaml, Gabor and Laws included), so the CSV can be used
as the baseline. Results: <REPRO_DIR>/<perturbation>/<task>.json -> {"features": {feature name: value}, ...}.
ICC_check.ipynb reads them.
"""
import json
import math
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml
from scipy import ndimage

from scripts import array_from_image, image_from_array, list_case, task_name
from extraction import make_extractor, load_image_and_mask, preprocess, extract_features
from filter import FILTERS, filter_args

CONFIG_DIR = Path("/Users/kamleshranabhat/Desktop/radiomics/full_pipeline/config_files")
with open(CONFIG_DIR / "setup.yaml", "r") as f:
    config = yaml.safe_load(f)
with open(CONFIG_DIR / "pyradiomics_params.yaml", "r") as f:
    params = yaml.safe_load(f)

PERTURBATIONS = ["volume_minus10", "volume_plus10", "n4", "noise"]
REPRO_DIR = Path(config["REPRO_DIR"])
# setup.yaml writes Python's None (weightingNorm: None), which YAML reads as the text 'None'
SETTINGS = {k: (None if v == "None" else v) for k, v in config["PARAMS"]["setting"].items()}
FEATURE_CLASSES = config["PARAMS"]["featureClass"]
FILTER_ARGS = {name: filter_args(name, entry.get("args")) for name, entry in params["filters"].items()
               if entry.get("enabled", True)}


# ------------------------------------------------------------------------------------------- perturbations
def adapt_volume(mask_image, fraction, label=1):
    """Grow (fraction > 0) or shrink (fraction < 0) a binary mask by |fraction| of its voxel count."""
    mask = array_from_image(mask_image) == label
    distance = array_from_image(sitk.SignedMaurerDistanceMap(sitk.Cast(mask_image == label, sitk.sitkUInt8),
                                                             insideIsPositive=False, squaredDistance=False,
                                                             useImageSpacing=True))
    n_change = int(round(abs(fraction) * mask.sum()))
    flat_mask, flat_distance = mask.ravel().copy(), distance.ravel()
    if fraction < 0:
        inside = np.nonzero(flat_mask)[0]
        order = np.argsort(-flat_distance[inside], kind="stable")   # closest to the surface first (-0.5 > -3)
        flat_mask[inside[order[:n_change]]] = False
    else:
        # candidates: outside voxels within 15 mm of the ROI (far fewer than all background voxels)
        outside = np.nonzero(~flat_mask & (flat_distance <= 15.0))[0]
        if len(outside) < n_change:
            outside = np.nonzero(~flat_mask)[0]
        order = np.argsort(flat_distance[outside], kind="stable")   # closest outside voxels first
        flat_mask[outside[order[:n_change]]] = True
    new_mask = flat_mask.reshape(mask.shape).astype(np.uint32) * label
    return image_from_array(new_mask, mask_image)


IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)


def estimate_noise_sigma(image_array_zyx):
    """Median over axial slices of Immerkaer's noise standard deviation estimate."""
    sigmas = []
    for plane in np.asarray(image_array_zyx, dtype=float):
        H, W = plane.shape
        response = ndimage.convolve(plane, IMMERKAER_KERNEL)[1:-1, 1:-1]
        sigmas.append(math.sqrt(math.pi / 2) * np.abs(response).sum() / (6.0 * (W - 2) * (H - 2)))
    return float(np.median(sigmas))


def add_gaussian_noise(image, sigma, seed):
    """Image (float32) plus N(0, sigma^2) noise from a seeded generator."""
    array = array_from_image(image).astype(np.float32)
    noise = np.random.default_rng(seed).normal(0.0, sigma, array.shape).astype(np.float32)
    return image_from_array(array + noise, image)


def perturbed_task_inputs(case, side, channel, perturbation):
    """Image and mask of a task with one perturbation applied (on top of PIPELINE_OPTIONS)."""
    options = dict(config["PIPELINE_OPTIONS"])
    if perturbation == "n4":
        options["bias_correction"] = True
    image, mask = load_image_and_mask(case, side, channel, options["registration"], options["bias_correction"])
    if perturbation == "volume_minus10":
        mask = adapt_volume(mask, -0.10)
    elif perturbation == "volume_plus10":
        mask = adapt_volume(mask, +0.10)
    elif perturbation == "noise":
        sigma = estimate_noise_sigma(array_from_image(image))
        seed = zlib.crc32(f"{case}_ch{channel}".encode())       # same noise for both breasts of one image
        image = add_gaussian_noise(image, sigma, seed)
    elif perturbation not in ("baseline", "n4"):
        raise ValueError(perturbation)
    return image, mask


# extraction
def all_tasks():
    """Every (case, side, channel) combination: 5 x 2 x 4 = 40 for this dataset."""
    return [(case, side, channel) for case in list_case() for side in config["SIDES"]
            for channel in range(config["N_CHANNELS"])]


def run_one_perturbed(case, side, channel, perturbation, folder, overwrite=False):
    """Extract one perturbed task (all filters) and save it as <folder>/<task>.json, unless it already exists."""
    path = Path(folder) / f"{task_name(case, side, channel)}.json"
    status = {"task": task_name(case, side, channel), "perturbation": perturbation}
    if path.exists() and not overwrite:
        return {**status, "status": "cached"}
    start = time.perf_counter()
    try:
        extractor = make_extractor(SETTINGS, FEATURE_CLASSES)
        image, mask = perturbed_task_inputs(case, side, channel, perturbation)
        image, mask, bounding_box = preprocess(extractor, image, mask)
        features = {}
        for name, args in FILTER_ARGS.items():
            features.update(extract_features(extractor, image, mask, bounding_box, FILTERS[name], args,
                                             include_shape=(name == "raw_image")))
    except Exception as error:                                       # keep going, report at the end
        return {**status, "status": "error", "error": f"{type(error).__name__}: {error}"}
    payload = {"PatientID": case, "Side": side, "Channel": channel,
               "options": {**config["PIPELINE_OPTIONS"], "perturbation": perturbation},
               "seconds": round(time.perf_counter() - start, 1), "features": features}
    path.write_text(json.dumps(payload))
    return {**status, "status": "ok", "seconds": payload["seconds"]}


def limit_library_threads(n_threads):
    """Worker initializer: let each process use only a few SimpleITK threads (avoids oversubscription)."""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n_threads)


def run_perturbations(tasks=None, perturbations=PERTURBATIONS, workers=3, threads_per_worker=2, overwrite=False):
    """Run every (task, perturbation) in `workers` processes and print progress. Saved results are reused."""
    tasks = all_tasks() if tasks is None else tasks
    jobs = []
    for perturbation in perturbations:
        folder = REPRO_DIR / perturbation
        folder.mkdir(parents=True, exist_ok=True)
        jobs += [(case, side, channel, perturbation, folder, overwrite) for case, side, channel in tasks]

    results = []
    start = time.perf_counter()

    def report(result):
        results.append(result)
        print("%3d/%d %-15s %-22s %-6s %s" % (len(results), len(jobs), result["perturbation"], result["task"],
                                              result["status"], result.get("seconds", result.get("error", ""))),
              flush=True)

    if workers <= 1:
        for job in jobs:
            report(run_one_perturbed(*job))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=limit_library_threads,
                                 initargs=(threads_per_worker,)) as pool:
            for future in as_completed([pool.submit(run_one_perturbed, *job) for job in jobs]):
                report(future.result())

    errors = [r for r in results if r["status"] == "error"]
    print("wall time %.0f s | %d errors" % (time.perf_counter() - start, len(errors)))
    for e in errors:
        print("   ", e)
    return results
