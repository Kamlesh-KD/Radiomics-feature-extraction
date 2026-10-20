import numpy as np
import math

import SimpleITK as sitk
import os
from pathlib import Path

from scripts import *

import yaml
config_path = Path("/Users/kamleshranabhat/Desktop/radiomics/full_pipeline/config_files/setup.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config

def fixed_bin_count_edges(values, bin_count):
    """N+1 equally spaced edges between min and max (like numpy.histogram), last edge + 1."""
    values = np.asarray(values)
    low, high = values.min(), values.max()
    if low == high:                                   # flat region: one bin around the single value
        low, high = low - 0.5, high + 0.5
    dtype = np.result_type(low, high, values)
    if np.issubdtype(dtype, np.integer):
        dtype = np.result_type(dtype, float)
    edges = np.linspace(low, high, bin_count + 1, endpoint=True, dtype=dtype)
    edges[-1] += 1                                    # include the maximum in the top bin
    return edges


def fixed_bin_width_edges(values, bin_width):
    """Edges at multiples of `bin_width`, from the largest multiple <= min to beyond max."""
    values = np.asarray(values)
    low = values.min() - (values.min() % bin_width)
    edges = np.arange(low, values.max() + 2 * bin_width, bin_width)
    if len(edges) == 1:                               # flat region
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    return edges


def digitize(values, edges):
    """Bin index of every value: i such that edges[i-1] <= value < edges[i] (1-based, like numpy.digitize)."""
    return np.searchsorted(edges, values, side="right")

def crop_to_roi(array, mask_array, pad=0):
    """Crop an array and its boolean mask to the ROI bounding box (+ pad voxels)."""
    z, y, x = np.nonzero(mask_array)
    box = tuple(slice(max(int(c.min()) - pad, 0), min(int(c.max()) + pad + 1, n))
                for c, n in zip((z, y, x), array.shape))
    return array[box], mask_array[box]


def discretize_roi(array, mask_array, bin_count=None, bin_width=None):
    """Grey levels 1..N inside the ROI, 0 outside. Returns (integer array, bin edges).

    Exactly one of bin_count / bin_width must be given (PARAMS uses binCount = 32).
    """
    values = array[mask_array]
    if bin_count is not None:
        edges = fixed_bin_count_edges(values, bin_count)
    else:
        edges = fixed_bin_width_edges(values, bin_width)
    discrete = np.zeros(array.shape, dtype=np.int64)
    discrete[mask_array] = digitize(values, edges)
    return discrete, edges


def discretize_with_settings(array, mask_array, settings=None):
    """discretize_roi with the binCount / binWidth of PARAMS['setting']."""
    s = config["PARAMS"]["setting"] if settings is None else settings
    if s.get("binCount") is not None:
        return discretize_roi(array, mask_array, bin_count=s["binCount"])
    return discretize_roi(array, mask_array, bin_width=s.get("binWidth", 25))


def grey_levels_present(discrete, mask_array):
    """Sorted grey levels that actually occur in the ROI (some bins can be empty)."""
    return np.unique(discrete[mask_array])