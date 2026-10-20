import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scripts import *

import yaml
config_path = Path("/Users/kamleshranabhat/Desktop/radiomics/full_pipeline/config_files/setup.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config


def percentile_window(values, low=1.0, high=99.5):
    """Display range (vmin, vmax) from percentiles, robust to a few very bright voxels."""
    values = np.asarray(values, dtype=float)
    return tuple(np.percentile(values, [low, high]))

def display_geometry(spacing_zyx, axis):
    """imshow aspect ratio (row spacing / column spacing) and origin for axial or coronal slices."""
    row_spacing = spacing_zyx[1] if axis == 0 else spacing_zyx[0]
    aspect = row_spacing / spacing_zyx[2]            # x is always the column axis here
    origin = "lower" if axis == 0 else "upper"       # anterior up (axial), superior up (coronal)
    return aspect, origin

def show_image_with_mask(
    ax,
    image_zyx,
    mask_zyx=None,
    spacing_zyx=(1, 1, 1),
    axis=0,
    index=None,
    color="#e08214",
    title=None,
    contour=True,
    alpha=0.35,
    window=None,
    cmap="gray",
    radiological=True
):
    """
    Show a 3D image slice with a mask correctly overlaid.

    image_zyx : numpy array [z, y, x]
    mask_zyx  : numpy array [z, y, x]
    spacing_zyx : spacing in [z, y, x]

    axis:
        0 = axial
        1 = coronal
        2 = sagittal
    """

    image_zyx = np.asarray(image_zyx)

    if mask_zyx is not None:
        mask_zyx = np.asarray(mask_zyx)

        # Important check
        if image_zyx.shape != mask_zyx.shape:
            raise ValueError(
                f"Image and mask shapes do not match:\n"
                f"image = {image_zyx.shape}\n"
                f"mask  = {mask_zyx.shape}"
            )

    # ----------------------------------------------------
    # Select slice
    # ----------------------------------------------------
    if index is None:
        if mask_zyx is not None:
            other_axes = tuple(i for i in range(3) if i != axis)
            index = np.argmax(mask_zyx.sum(axis=other_axes))
        else:
            index = image_zyx.shape[axis] // 2

    # ----------------------------------------------------
    # Extract SAME slice from image and mask
    # ----------------------------------------------------
    if axis == 0:                 # axial -> [y, x]

        image_slice = image_zyx[index, :, :]

        if mask_zyx is not None:
            mask_slice = mask_zyx[index, :, :]

        row_spacing = spacing_zyx[1]
        col_spacing = spacing_zyx[2]

    elif axis == 1:               # coronal -> [z, x]

        image_slice = image_zyx[:, index, :]

        if mask_zyx is not None:
            mask_slice = mask_zyx[:, index, :]

        row_spacing = spacing_zyx[0]
        col_spacing = spacing_zyx[2]

    elif axis == 2:               # sagittal -> [z, y]

        image_slice = image_zyx[:, :, index]

        if mask_zyx is not None:
            mask_slice = mask_zyx[:, :, index]

        row_spacing = spacing_zyx[0]
        col_spacing = spacing_zyx[1]

    else:
        raise ValueError("axis must be 0, 1, or 2")

    # ----------------------------------------------------
    # Radiological orientation
    # Flip IMAGE AND MASK together
    # ----------------------------------------------------
    if radiological:
        image_slice = image_slice[:, ::-1]

        if mask_zyx is not None:
            mask_slice = mask_slice[:, ::-1]

    # ----------------------------------------------------
    # Physical coordinates
    # ----------------------------------------------------
    height = image_slice.shape[0] * row_spacing
    width = image_slice.shape[1] * col_spacing

    extent = [0, width, 0, height]

    # ----------------------------------------------------
    # Intensity window
    # ----------------------------------------------------
    if window is None:
        vmin, vmax = np.percentile(
            image_slice[np.isfinite(image_slice)],
            [1, 99.5]
        )
    else:
        vmin, vmax = window

    # ----------------------------------------------------
    # Image
    # ----------------------------------------------------
    ax.imshow(
        image_slice,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=extent,
        interpolation="nearest"
    )

    # ----------------------------------------------------
    # Mask overlay
    # ----------------------------------------------------
    if mask_zyx is not None and np.any(mask_slice):

        mask_slice = mask_slice > 0

        if contour:

            ax.contour(
                mask_slice.astype(float),
                levels=[0.5],
                colors=[color],
                linewidths=1.2,
                origin="lower",
                extent=extent
            )

        else:

            rgba = np.zeros(
                mask_slice.shape + (4,),
                dtype=float
            )

            rgb = matplotlib.colors.to_rgb(color)

            rgba[mask_slice, 0] = rgb[0]
            rgba[mask_slice, 1] = rgb[1]
            rgba[mask_slice, 2] = rgb[2]
            rgba[mask_slice, 3] = alpha

            ax.imshow(
                rgba,
                origin="lower",
                extent=extent,
                interpolation="nearest"
            )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title, fontsize=9)

    return index

def overlay_mask(ax, mask, spacing, axis=0, index=None, color="red", contour=False, radiological=True):
    if index is None:
        index = mask.shape[axis] // 2

    if axis == 0:      # axial
        mask_slice = mask[index, :, :]
        extent = [
            0, mask.shape[2] * spacing[2],
            0, mask.shape[1] * spacing[1]
        ]

    elif axis == 1:    # coronal
        mask_slice = mask[:, index, :]
        extent = [
            0, mask.shape[2] * spacing[2],
            0, mask.shape[0] * spacing[0]
        ]

    elif axis == 2:    # sagittal
        mask_slice = mask[:, :, index]
        extent = [
            0, mask.shape[1] * spacing[1],
            0, mask.shape[0] * spacing[0]
        ]

    # flip left-right like show_image_with_mask(radiological=True), so the mask lies on the image
    if radiological:
        mask_slice = mask_slice[:, ::-1]

    if contour:
        ax.contour(
            mask_slice,
            levels=[0.5],
            colors=[color],
            extent=extent,
            linewidths=1
        )
    else:
        masked = np.ma.masked_where(mask_slice == 0, mask_slice)

        ax.imshow(
            masked,
            cmap=plt.matplotlib.colors.ListedColormap([color]),
            alpha=0.4,
            origin="lower",
            extent=extent
        )