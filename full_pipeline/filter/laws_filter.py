"""Laws texture energy filter:
Laws kernels are separable 3D kernels built from 1D kernels :
    L5 = [1, 4, 6, 4, 1] / sqrt(70)    level        L3 = [1, 2, 1] / sqrt(6)
    E5 = [-1, -2, 0, 2, 1] / sqrt(10)  edge         E3 = [-1, 0, 1] / sqrt(2)
    S5 = [-1, 0, 2, 0, -1] / sqrt(6)   spot         S3 = [-1, 2, -1] / sqrt(6)
    W5 = [-1, 2, 0, -2, 1] / sqrt(10)  wave
    R5 = [1, -4, 6, -4, 1] / sqrt(70)  ripple
A kernel name lists the 1D kernel for x, y and z, e.g. 'L5E5E5' = L5 along x, E5 along y and z.
Steps per kernel:
  1. response = I convolved with the separable kernel (mirror padding);
  2. rotation invariance (optional): repeat step 1 for the 24 right-angle rotations of the kernel and pool
     the responses voxel-wise (max or average);
  3. energy (optional): local mean of |response| in a (2 d + 1)^3 box.
One image per kernel, named 'laws-<kernel>' or 'laws-<kernel>-energy'.
"""
import itertools
import re

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

NAME = "laws_filter"
IMAGE_TYPE = "Laws"
CUSTOM = True
DEFAULT_ARGS = {"lawsKernels": ["L5E5E5", "E5E5E5", "L5S5S5", "S5S5S5", "E5W5R5", "R5R5R5"],
                "lawsRotationInvariant": True, "lawsPooling": "max", "lawsEnergy": True, "lawsEnergyDistance": 7}

LAWS_1D = {
    "L3": np.array([1, 2, 1]) / np.sqrt(6),
    "E3": np.array([-1, 0, 1]) / np.sqrt(2),
    "S3": np.array([-1, 2, -1]) / np.sqrt(6),
    "L5": np.array([1, 4, 6, 4, 1]) / np.sqrt(70),
    "E5": np.array([-1, -2, 0, 2, 1]) / np.sqrt(10),
    "S5": np.array([-1, 0, 2, 0, -1]) / np.sqrt(6),
    "W5": np.array([-1, 2, 0, -2, 1]) / np.sqrt(10),
    "R5": np.array([1, -4, 6, -4, 1]) / np.sqrt(70),
}

def parse_kernel_name(name):
    """'L5E5E5' -> ['L5', 'E5', 'E5'] (kernels for x, y, z)."""
    parts = re.findall(r"[LESWR][35]", name.upper())
    if len(parts) != 3 or "".join(parts) != name.upper() or any(p not in LAWS_1D for p in parts):
        raise ValueError(f"invalid 3D Laws kernel name {name!r}")
    return parts


def right_angle_rotations():
    """The 24 proper rotations of a cube as (permutation, signs): direction i -> signs[i] * direction perm[i]."""
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for i in range(3):
                matrix[perm[i], i] = signs[i]
            if np.linalg.det(matrix) > 0:
                rotations.append((perm, signs))
    return rotations


def separable_response(array, kernels_by_axis):
    """Convolve a [z, y, x] array with one 1D kernel per numpy axis (mirror padding)."""
    result = array
    for axis, kernel in enumerate(kernels_by_axis):
        result = ndimage.convolve1d(result, kernel, axis=axis, mode="reflect")
    return result


def rotated_kernels(kernel_xyz, rotation_invariant=True):
    """Distinct per-axis kernel triples ([z, y, x] order) of all rotations, with how often each occurs."""
    rotations = right_angle_rotations() if rotation_invariant else [((0, 1, 2), (1, 1, 1))]
    unique = {}
    for perm, signs in rotations:
        by_direction = [None] * 3                          # direction 0 = x, 1 = y, 2 = z
        for i in range(3):
            k = LAWS_1D[kernel_xyz[i]]
            by_direction[perm[i]] = k[::-1] if signs[i] < 0 else k
        by_axis = tuple(by_direction[::-1])                # numpy axis 0 = z, 1 = y, 2 = x
        key = tuple(tuple(np.round(k, 12)) for k in by_axis)
        count, _ = unique.get(key, (0, by_axis))
        unique[key] = (count + 1, by_axis)
    return list(unique.values())


def laws_response(array, kernel_name, rotation_invariant=True, pooling="max", energy=True, energy_distance=7):
    """Laws response (or energy) map of a [z, y, x] array for one 3D kernel."""
    array = np.asarray(array, dtype=np.float64)
    pooled, n = None, 0
    for count, kernels in rotated_kernels(parse_kernel_name(kernel_name), rotation_invariant):
        response = separable_response(array, kernels)
        if pooling == "max":
            pooled = response if pooled is None else np.maximum(pooled, response)
        elif pooling == "average":
            pooled = count * response if pooled is None else pooled + count * response
            n += count
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
    if pooling == "average":
        pooled /= n
    if energy:
        pooled = ndimage.uniform_filter(np.abs(pooled), size=2 * int(energy_distance) + 1, mode="reflect")
    return pooled


def laws_image_name(kernel_name, energy=True):
    return "laws-%s%s" % (kernel_name.upper(), "-energy" if energy else "")


def generate(image, mask, **kwargs):
    """PyRadiomics generator: yields one 'laws-<kernel>[-energy]' image per kernel."""
    args = dict(DEFAULT_ARGS)
    args.update({k: v for k, v in kwargs.items() if k in DEFAULT_ARGS})
    array = sitk.GetArrayFromImage(image)
    for kernel_name in args["lawsKernels"]:
        response = laws_response(array, kernel_name, bool(args["lawsRotationInvariant"]), args["lawsPooling"],
                                 bool(args["lawsEnergy"]), int(args["lawsEnergyDistance"]))
        filtered = sitk.GetImageFromArray(response)
        filtered.CopyInformation(image)
        yield filtered, laws_image_name(kernel_name, bool(args["lawsEnergy"])), kwargs
