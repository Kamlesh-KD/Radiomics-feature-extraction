import re
import numpy as np
import json
from pathlib import Path
import pandas as pd

import gzip
import struct
import math
import SimpleITK as sitk

import yaml
config_path = Path("/Users/kamleshranabhat/Desktop/radiomics/full_pipeline/config_files/setup.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config


def load_image(path, pixel_type=None):
    image = sitk.ReadImage(str(path))
    return sitk.Cast(image, pixel_type) if pixel_type is not None else image

def spacing_zyx(image):
    x, y, z = image.GetSpacing()
    return np.array([z, y, x])
    
def read_nifti(path):
    header = read_nifti_header(path)

    dtype = config["NIFTI_DTYPES"][header["datatype"]]

    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rb") as f:
        buffer = f.read()

    shape_xyz = header["dim"][:3]

    data = np.frombuffer(
        buffer,
        dtype=dtype,
        count=int(np.prod(shape_xyz)),
        offset=int(header["vox_offset"])
    )

    array = data.reshape(shape_xyz[::-1])

    slope = header["scl_slope"]
    inter = header["scl_inter"]

    if slope not in (0.0, 1.0) or inter != 0.0:
        array = array * slope + inter

    return header, array

def read_nifti_header(path):
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rb") as f:
        raw = f.read(352)

    dim = struct.unpack("<8h", raw[40:56])
    n_dims = dim[0]

    header = {
        "dim": dim[1:n_dims + 1],
        "datatype": struct.unpack("<h", raw[70:72])[0],
        "pixdim": struct.unpack("<8f", raw[76:108])[1:n_dims + 1],
        "vox_offset": struct.unpack("<f", raw[108:112])[0],
        "scl_slope": struct.unpack("<f", raw[112:116])[0],
        "scl_inter": struct.unpack("<f", raw[116:120])[0],
        "qform_code": struct.unpack("<h", raw[252:254])[0],
        "sform_code": struct.unpack("<h", raw[254:256])[0],
        "quatern_bcd": struct.unpack("<3f", raw[256:268]),
        "qoffset": struct.unpack("<3f", raw[268:280]),
        "srow_x": struct.unpack("<4f", raw[280:296]),
        "srow_y": struct.unpack("<4f", raw[296:312]),
        "srow_z": struct.unpack("<4f", raw[312:328]),
        "qfac": struct.unpack("<f", raw[76:80])[0],
    }

    return header

def nifti_affine_ras(header):
    """4x4 voxel-index -> RAS(mm) matrix, from sform if valid, otherwise from the qform quaternion."""
    if header["sform_code"] > 0:
        affine = np.array([header["srow_x"], header["srow_y"], header["srow_z"], [0, 0, 0, 1]], float)
        return affine, "sform"
    b, c, d = header["quatern_bcd"]
    a = math.sqrt(max(0.0, 1.0 - (b * b + c * c + d * d)))
    rotation = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
    ])
    qfac = -1.0 if header["qfac"] < 0 else 1.0
    scale = np.array(header["pixdim"][:3]) * np.array([1.0, 1.0, qfac])
    affine = np.eye(4)
    affine[:3, :3] = rotation * scale[None, :]
    affine[:3, 3] = header["qoffset"]
    return affine, "qform"

def nifti_geometry_lps(header):
    affine, source = nifti_affine_ras(header)

    ras_to_lps = np.diag([-1.0, -1.0, 1.0])

    matrix = ras_to_lps @ affine[:3, :3]
    origin = ras_to_lps @ affine[:3, 3]

    spacing = np.linalg.norm(matrix, axis=0)
    direction = matrix / spacing

    return {
        "origin": origin,
        "direction": direction,
        "spacing": spacing,
        "source": source
    }

def list_case():
    dir_path = Path(config["IMAGES_DIR"])

    return sorted({
        re.split(r"[_\.]", im.name)[0]
        for im in dir_path.glob("*.nii.gz")
    })

def label_path(case):
    return Path(config["LABELS_DIR"]) / f"{case}.nii.gz"

def image_path(case, n_channel):
    return Path(config["IMAGES_DIR"]) / f"{case}_000{n_channel}.nii.gz"

def mask_path(case, side):
    return Path(config["MASK_DIR"]) / f"{case}_{side}.nii.gz"

def largest_slice_index(mask_zyx, axis):
    """Index along `axis` of the slice that contains the most mask voxels."""
    other_axes = tuple(a for a in range(3) if a != axis)
    return int(np.argmax(mask_zyx.sum(axis=other_axes)))

def array_from_image(image):
    return sitk.GetArrayFromImage(image)

def image_from_array(array, reference_image):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference_image)
    return image

def radiological_view(array_zyx, axis, index):
    plane = np.take(array_zyx, index, axis=axis)
    return plane[:, ::-1]

def load_reference_features(case, side, channel):
    """Feature dictionary of the PyRadiomics 3.1.0 run for one task (without diagnostics)."""
    with open(reference_feature_json_path(case, side, channel)) as f:
        return json.load(f)["features"]

def reference_feature_json_path(case, side, channel):
    """PyRadiomics 3.1.0 result for one task."""
    return Path(config["REFERENCE_FEATURE_DIR"]) / f"{task_name(case, side, channel)}.json"

def task_name(case, side, channel):
    """Unique text name of one extraction task, e.g. 'BreastMRI002_L_ch0'."""
    return f"{case}_{side}_ch{channel}"

def summarize_comparison(table, tolerance=1e-6):
    """One-line verdict for a compare_features table."""
    n = len(table)
    n_ok = int((table["rel_error"] <= tolerance).sum())
    worst = table["rel_error"].max()
    print(f"{n_ok}/{n} features within relative error {tolerance:g} of PyRadiomics "
          f"(largest relative error {worst:.2e})")

def compare_features(ours, reference, prefix=""):
    """Table of our value vs the reference value for every key present in both.

    `prefix` is added in front of our keys (e.g. 'original_firstorder_') to match reference names.
    The relative error is |ours - ref| / max(|ref|, 1e-12).
    """
    rows = []
    for name, value in ours.items():
        key = prefix + name
        if key not in reference:
            rows.append({"feature": key, "ours": float(value), "reference": np.nan, "rel_error": np.nan})
            continue
        ref = float(reference[key])
        rel = abs(float(value) - ref) / max(abs(ref), 1e-12)
        rows.append({"feature": key, "ours": float(value), "reference": ref, "rel_error": rel})
    return pd.DataFrame(rows)