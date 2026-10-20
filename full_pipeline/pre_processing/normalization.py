import SimpleITK as sitk
import os
import numpy as np
from pathlib import Path

from scripts import *
from n4 import *
from registration import load_registered_channel

import yaml
config_path = Path("/Users/kamleshranabhat/Desktop/radiomics/full_pipeline/config_files/setup.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config

def load_task_inputs(case, side, channel, options=None):
    """Image and mask (SimpleITK) of one (case, side, channel) task after the optional pre-processing.

    options: {"registration": bool, "bias_correction": bool}; defaults to PIPELINE_OPTIONS.
    """
    options = PIPELINE_OPTIONS if options is None else options
    image = load_image(image_path(case, channel))
    if options.get("bias_correction", False):
        image = load_bias_corrected_channel(case, channel)
    if options.get("registration", False) and channel != 0:
        image = load_registered_channel(case, channel, image)
    mask = sitk.Cast(load_image(mask_path(case, side)), sitk.sitkUInt32)
    return image, mask


def check_same_grid(image, mask, tolerance=1e-4):
    """Raise ValueError unless image and mask have the same size, spacing, origin and direction."""
    if image.GetSize() != mask.GetSize():
        raise ValueError(f"size differs: image {image.GetSize()} vs mask {mask.GetSize()}")
    for name in ("GetSpacing", "GetOrigin", "GetDirection"):
        a, b = np.array(getattr(image, name)()), np.array(getattr(mask, name)())
        if not np.allclose(a, b, atol=tolerance):
            raise ValueError(f"{name[3:].lower()} differs: {a} vs {b}")
    return True

def zscore_statistics(array):
    """Mean and sample standard deviation (n-1), as ITK's StatisticsImageFilter computes them."""
    values = np.asarray(array, dtype=np.float64)
    return values.mean(), values.std(ddof=1)


def normalize_image(image, scale=100, outliers=3):
    """z-score the whole image, clip at +-outliers SD, multiply by scale. Returns a float64 image."""
    array = array_from_image(image).astype(np.float64)
    mean, std = zscore_statistics(array)
    z = (array - mean) / std
    if outliers is not None:
        z = np.clip(z, -outliers, outliers)
    return image_from_array(z * scale, image)

def bounding_box_index_size(mask_array_zyx, label=1):
    """(Lx, Ly, Lz, Sx, Sy, Sz): lower corner and size of the ROI in (x, y, z) voxel indices."""
    z, y, x = np.nonzero(mask_array_zyx == label)
    lower = np.array([x.min(), y.min(), z.min()])
    upper = np.array([x.max(), y.max(), z.max()])
    return np.concatenate([lower, upper - lower + 1])


def bounding_box_lower_upper(mask_array_zyx, label=1):
    """(Lx, Ux, Ly, Uy, Lz, Uz): first and last ROI index in every dimension."""
    bb = bounding_box_index_size(mask_array_zyx, label)
    lower, size = bb[:3], bb[3:]
    out = np.empty(6, dtype=int)
    out[0::2] = lower
    out[1::2] = lower + size - 1
    return out


def check_roi(mask_array_zyx, label=1, minimum_dimensions=3, minimum_size=100):
    """Raise ValueError if the ROI is too small or not enough dimensions; return the voxel count."""
    n_voxels = int((mask_array_zyx == label).sum())
    if n_voxels == 0:
        raise ValueError(f"label {label} not present in mask")
    size = bounding_box_index_size(mask_array_zyx, label)[3:]
    n_dims = int((size > 1).sum())
    if n_dims == 0:
        raise ValueError("mask contains a single voxel")
    if n_dims < minimum_dimensions:
        raise ValueError(f"ROI has {n_dims} dimension(s) with more than one voxel, {minimum_dimensions} required")
    if minimum_size is not None and n_voxels <= minimum_size:
        raise ValueError(f"ROI has {n_voxels} voxels, more than {minimum_size} required")
    return n_voxels

def compute_resampling_grid(mask_image, new_spacing=(1, 1, 1), pad_distance=10, label=1):
    """Size, origin, spacing and direction of the cropped isotropic grid around the ROI.

    Returns a dict; all vectors in SimpleITK (x, y, z) order.
    """
    old_spacing = np.array(mask_image.GetSpacing(), dtype=float)
    new_spacing = np.array(new_spacing, dtype=float)
    new_spacing = np.where(new_spacing == 0, old_spacing, new_spacing)        # 0 = keep original

    bb = bounding_box_index_size(array_from_image(mask_image), label)
    lower_index, size = bb[:3].astype(float), bb[3:].astype(float)
    new_spacing = np.where(size != 1, new_spacing, old_spacing)               # 1-voxel thick: keep

    ratio = old_spacing / new_spacing
    lower = np.floor((lower_index - 0.5) * ratio - pad_distance)
    upper = np.ceil((lower_index + size - 0.5) * ratio + pad_distance)
    max_upper = np.ceil(np.array(mask_image.GetSize()) * ratio) - 1
    lower = np.where(lower < 0, 0, lower)
    upper = np.where(upper > max_upper, max_upper, upper)
    new_size = np.array(upper - lower + 1, dtype=int)

    origin_index = 0.5 * (new_spacing - old_spacing) / old_spacing + lower / ratio
    origin = mask_image.TransformContinuousIndexToPhysicalPoint(origin_index.tolist())
    return {"size": new_size.tolist(), "spacing": new_spacing.tolist(), "origin": list(origin),
            "direction": list(mask_image.GetDirection()), "bounding_box": bb.tolist(),
            "lower": lower.tolist(), "upper": upper.tolist(), "spacing_ratio": ratio.tolist()}


def resample_onto_grid(image, grid, interpolator, pixel_type):
    """Resample a SimpleITK image onto the grid dictionary of compute_resampling_grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(grid["spacing"])
    resampler.SetOutputDirection(grid["direction"])
    resampler.SetOutputOrigin(grid["origin"])
    resampler.SetSize([int(v) for v in grid["size"]])
    resampler.SetOutputPixelType(pixel_type)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def interpolator_from_name(name):
    """'sitkBSpline' -> sitk.sitkBSpline"""
    return getattr(sitk, name) if isinstance(name, str) else name


def crop_image_to_bounding_box(image, mask, bb_lower_upper, pad_distance=0):
    """Crop image and mask (SimpleITK) to the ROI bounding box plus a margin."""
    size = np.array(mask.GetSize())
    lower_crop = np.maximum(bb_lower_upper[0::2] - pad_distance, 0)
    upper_crop = np.maximum(size - bb_lower_upper[1::2] - pad_distance - 1, 0)
    cropper = sitk.CropImageFilter()
    cropper.SetLowerBoundaryCropSize([int(v) for v in lower_crop])
    cropper.SetUpperBoundaryCropSize([int(v) for v in upper_crop])
    return cropper.Execute(image), cropper.Execute(mask)


def resample_image_and_mask(image, mask, new_spacing=(1, 1, 1), pad_distance=10, label=1,
                            interpolator="sitkBSpline"):
    """Image (B-spline, keeps its pixel type) and mask (nearest neighbour) on the isotropic ROI grid."""
    if np.allclose(image.GetSpacing(), new_spacing):
        # Already at the requested spacing: no interpolation, just crop to the ROI + margin
        bb = bounding_box_lower_upper(array_from_image(mask), label)
        return crop_image_to_bounding_box(image, mask, bb, pad_distance)
    grid = compute_resampling_grid(mask, new_spacing, pad_distance, label)
    resampled_image = resample_onto_grid(image, grid, interpolator_from_name(interpolator), image.GetPixelID())
    resampled_mask = resample_onto_grid(mask, grid, sitk.sitkNearestNeighbor, mask.GetPixelID())
    return resampled_image, resampled_mask

def preprocess(image, mask, settings=None):
    """Normalization -> ROI checks -> resampling, following the PyRadiomics order of operations."""
    s = config["PARAMS"]["setting"] if settings is None else settings
    label = s["label"]
    check_same_grid(image, mask)
    original_voxels = check_roi(array_from_image(mask), label, s["minimumROIDimensions"], s["minimumROISize"])
    diagnostics = {"original_size": list(image.GetSize()), "original_spacing": list(image.GetSpacing()),
                   "original_voxels": original_voxels}

    if s.get("normalize", False):
        image = normalize_image(image, s["normalizeScale"], s["removeOutliers"])
    if s.get("resampledPixelSpacing") is not None:
        image, mask = resample_image_and_mask(image, mask, s["resampledPixelSpacing"], s["padDistance"],
                                              label, s["interpolator"])

    image_array = array_from_image(image)
    mask_array = array_from_image(mask) == label
    check_roi(mask_array.astype(np.uint8), 1, s["minimumROIDimensions"], s["minimumROISize"])
    roi_values = image_array[mask_array]
    diagnostics.update({
        "interpolated_size": list(image.GetSize()),
        "interpolated_spacing": list(image.GetSpacing()),
        "interpolated_bounding_box": bounding_box_index_size(mask_array.astype(np.uint8)).tolist(),
        "interpolated_voxels": int(mask_array.sum()),
        "image_mean": float(image_array.mean()), "image_min": float(image_array.min()),
        "image_max": float(image_array.max()),
        "roi_mean": float(roi_values.mean()), "roi_min": float(roi_values.min()),
        "roi_max": float(roi_values.max()),
    })
    return {"image": image, "mask": mask, "image_array": image_array, "mask_array": mask_array,
            "spacing_zyx": spacing_zyx(image), "diagnostics": diagnostics}


def load_reference_diagnostics(case, side, channel):
    """Diagnostics PyRadiomics stored for one task."""
    return json.loads(reference_feature_json_path(case, side, channel).read_text())["diagnostics"]