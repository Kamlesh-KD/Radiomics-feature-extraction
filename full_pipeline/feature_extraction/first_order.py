import numpy as np
import yaml
from pathlib import Path
from collections import OrderedDict
import yaml
config_path = Path("/Users/kamleshranabhat/Desktop/radiomics/full_pipeline/config_files/setup.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config


def central_moment(X, order):
    """mu_k = mean((X - mean(X))^k)."""
    return np.mean((X - X.mean()) ** order)

def grey_level_probabilities(discrete_values):
    """p(i): fraction of voxels in each grey level that occurs (empty bins are not included)."""
    _, counts = np.unique(discrete_values, return_counts=True)
    return counts / counts.sum()

def fo_energy(X, shift):
    return np.sum((X + shift) ** 2)

def fo_total_energy(X, shift, voxel_volume):
    return fo_energy(X, shift) * voxel_volume

def fo_entropy(p):
    return -np.sum(p * np.log2(p + np.spacing(1)))

def fo_minimum(X):
    return np.min(X)

def fo_10_percentile(X):
    return np.percentile(X, 10)

def fo_90_percentile(X):
    return np.percentile(X, 90)

def fo_maximum(X):
    return np.max(X)

def fo_mean(X):
    return np.mean(X)

def fo_median(X):
    return np.median(X)

def fo_interquartile_range(X):
    return np.percentile(X, 75) - np.percentile(X, 25)

def fo_range(X):
    return np.max(X) - np.min(X)

def fo_mean_absolute_deviation(X):
    return np.mean(np.abs(X - X.mean()))

def fo_robust_mean_absolute_deviation(X):
    """MAD of the voxels whose value lies in the closed interval [P10, P90]."""
    p10, p90 = np.percentile(X, 10), np.percentile(X, 90)
    inner = X[(X >= p10) & (X <= p90)]
    return np.mean(np.abs(inner - inner.mean()))

def fo_root_mean_squared(X, shift):
    return np.sqrt(np.sum((X + shift) ** 2) / X.size) if X.size else 0.0

def fo_skewness(X):
    m2 = central_moment(X, 2)
    return 0.0 if m2 == 0 else central_moment(X, 3) / m2 ** 1.5

def fo_kurtosis(X):
    m2 = central_moment(X, 2)
    return 0.0 if m2 == 0 else central_moment(X, 4) / m2 ** 2

def fo_variance(X):
    return central_moment(X, 2)

def fo_uniformity(p):
    return np.sum(p ** 2)

def first_order_features(image, mask_array, discrete, spacing_zyx, shift=None):
    """The 18 first-order features of the voxels inside the ROI."""
    shift = config["PARAMS"]["setting"]["voxelArrayShift"] if shift is None else shift
    X = image[mask_array].astype(np.float64)
    p = grey_level_probabilities(discrete[mask_array])
    voxel_volume = float(np.prod(spacing_zyx))
    return OrderedDict([
        ("Energy", fo_energy(X, shift)),
        ("TotalEnergy", fo_total_energy(X, shift, voxel_volume)),
        ("Entropy", fo_entropy(p)),
        ("Minimum", fo_minimum(X)),
        ("10Percentile", fo_10_percentile(X)),
        ("90Percentile", fo_90_percentile(X)),
        ("Maximum", fo_maximum(X)),
        ("Mean", fo_mean(X)),
        ("Median", fo_median(X)),
        ("InterquartileRange", fo_interquartile_range(X)),
        ("Range", fo_range(X)),
        ("MeanAbsoluteDeviation", fo_mean_absolute_deviation(X)),
        ("RobustMeanAbsoluteDeviation", fo_robust_mean_absolute_deviation(X)),
        ("RootMeanSquared", fo_root_mean_squared(X, shift)),
        ("Skewness", fo_skewness(X)),
        ("Kurtosis", fo_kurtosis(X)),
        ("Variance", fo_variance(X)),
        ("Uniformity", fo_uniformity(p)),
    ])