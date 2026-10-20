import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scripts import *

import yaml
config_path = Path("/Users/kamleshranabhat/Desktop/radiomics/full_pipeline/config_files/setup.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config

def binarize_label(label_array):
    """Foreground = every voxel with a label value > 0."""
    return label_array > 0


def connectivity_structure(connectivity):
    """3x3x3 boolean neighbourhood for 6-, 18- or 26-connectivity."""
    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    return ndimage.generate_binary_structure(3, rank)


def relabel_by_size(component_array, n_components):
    """Renumber components so that 1 = largest, 2 = second largest, ...

    Returns the new label array and the component sizes in the new order.
    Ties keep their original (scan) order because the sort is stable.
    """
    sizes = np.bincount(component_array.ravel(), minlength=n_components + 1)[1:]
    order = np.argsort(-sizes, kind="stable")            # old label - 1, largest first
    lookup = np.zeros(n_components + 1, dtype=np.int32)  # old label -> new label
    lookup[order + 1] = np.arange(1, n_components + 1)
    return lookup[component_array], sizes[order]


def connected_components(mask, connectivity=6):
    """Label the connected components of a boolean mask, largest first.

    Returns (labels, sizes): labels is an int array (0 = background), sizes[k-1] = voxels of component k.
    """
    labels, n = ndimage.label(mask, structure=connectivity_structure(connectivity))
    return relabel_by_size(labels, n)

def index_to_physical(index_xyz, image):
    """Continuous voxel index (x, y, z) -> physical point (mm, LPS): p = origin + D * diag(spacing) * index."""
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    return origin + direction @ (spacing * np.asarray(index_xyz, dtype=float))


def component_centroids(labels, n_components):
    """Mean (z, y, x) voxel index of every component (row k-1 = component k)."""
    coordinates = np.nonzero(labels)                          # z, y, x index of every foreground voxel
    component_of_voxel = labels[coordinates]
    counts = np.bincount(component_of_voxel, minlength=n_components + 1)[1:].astype(float)
    centroids = np.zeros((n_components, 3))
    for axis in range(3):
        sums = np.bincount(component_of_voxel, weights=coordinates[axis], minlength=n_components + 1)[1:]
        centroids[:, axis] = sums / counts
    return centroids


def component_table(labels, sizes, image):
    """One row per component: size, centroid index and physical centroid (LPS, mm)."""
    centroids_zyx = component_centroids(labels, len(sizes))
    rows = []
    for k, (size, c_zyx) in enumerate(zip(sizes, centroids_zyx), start=1):
        physical = index_to_physical(c_zyx[::-1], image)
        rows.append({"component": k, "n_voxels": int(size),
                     "centroid_x_mm": physical[0], "centroid_y_mm": physical[1], "centroid_z_mm": physical[2]})
    return pd.DataFrame(rows)

def find_midline_x(components):
    """Physical x (mm) halfway between the centroids of the two largest components."""
    if len(components) < 2:
        raise RuntimeError("expected at least two components (left and right breast)")
    largest_two = components.sort_values("n_voxels", ascending=False).iloc[:2]
    return float(largest_two["centroid_x_mm"].mean())


def side_of(centroid_x_mm, midline_x_mm):
    """LPS: +x points to the patient's LEFT, so x >= midline is the left breast."""
    return "L" if centroid_x_mm >= midline_x_mm else "R"


def assign_sides(components, midline_x_mm, min_voxels):
    """Add columns 'kept' (size >= min_voxels) and 'side' (L/R) to the component table."""
    table = components.copy()
    table["kept"] = table["n_voxels"] >= min_voxels
    table["side"] = [side_of(x, midline_x_mm) for x in table["centroid_x_mm"]]
    return table

def prepare_rois(case, min_voxels=int(config["MIN_COMPONENT_VOXELS"]), connectivity=6, write=True):
    """Split one bilateral label into despeckled left/right masks. Returns (records, component table)."""
    label_image = load_image(label_path(case))
    label = array_from_image(label_image)
    foreground = binarize_label(label)

    labels, sizes = connected_components(foreground, connectivity)
    components = component_table(labels, sizes, label_image)
    midline = find_midline_x(components)
    components = assign_sides(components, midline, min_voxels)

    voxel_mm3 = float(np.prod(label_image.GetSpacing()))
    dropped = components[~components["kept"]]
    records = []
    for side in config["SIDES"]:
        chosen = components[components["kept"] & (components["side"] == side)]["component"].to_numpy()
        side_mask = np.isin(labels, chosen).astype(np.uint8)
        if write:
            sitk.WriteImage(image_from_array(side_mask, label_image), str(mask_path(case, side)), True)
        n_voxels = int(side_mask.sum())
        records.append({
            "PatientID": case, "Side": side,
            "n_components_kept": int(len(chosen)),
            "n_voxels": n_voxels,
            "volume_mm3": round(n_voxels * voxel_mm3, 1),
            "spacing": [round(s, 4) for s in label_image.GetSpacing()],
            "midline_x_mm": round(midline, 2),
            "mask_file": f"masks/{case}_{side}.nii.gz",
            "n_components_dropped_all_sides": int(len(dropped)),
            "voxels_dropped_all_sides": int(dropped["n_voxels"].sum()),
        })
    return records, components

