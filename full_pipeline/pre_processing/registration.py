import SimpleITK as sitk
import time
import numpy as np
from scripts import *
from roi_preparation import *

import yaml
config_path = Path("/Users/kamleshranabhat/Desktop/radiomics/full_pipeline/config_files/setup.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config


def registration_mask(case, dilation_voxels=5):
    """Region where the metric is evaluated: both breasts (label > 0), dilated by a few voxels."""
    label = load_image(label_path(case)) > 0
    return sitk.BinaryDilate(sitk.Cast(label, sitk.sitkUInt8), [dilation_voxels] * 3)


def as_euler_transform(transform):
    """Unwrap the result of ImageRegistrationMethod.Execute into a plain Euler3DTransform."""
    if transform.GetName() == "CompositeTransform":
        transform = sitk.CompositeTransform(transform).GetNthTransform(0)
    return sitk.Euler3DTransform(transform)


def register_rigid(fixed, moving, fixed_mask=None, shrink_factors=(4, 2, 1), smoothing_sigmas=(2, 1, 0),
                   sampling_percentage=0.05, iterations=200, seed=42, histogram_bins=50):
    """Rigid (Euler 3D) mutual-information registration of `moving` onto `fixed`.

    Returns (transform, info). The transform maps points of the fixed image to points of the moving image,
    i.e. moving(T(x)) ~ fixed(x): that is the direction ITK needs to resample `moving` onto the fixed grid.
    """
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=histogram_bins)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(sampling_percentage, seed)
    if fixed_mask is not None:
        method.SetMetricFixedMask(fixed_mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-3,
                                                    numberOfIterations=iterations,
                                                    gradientMagnitudeTolerance=1e-6)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(list(shrink_factors))
    method.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
    method.SetInitialTransform(initial, inPlace=False)

    # metric before registration (identity = geometry-only alignment)
    identity = sitk.Euler3DTransform(initial)
    identity.SetParameters((0, 0, 0, 0, 0, 0))
    method.SetInitialTransform(identity, inPlace=False)
    metric_before = method.MetricEvaluate(fixed, moving)
    method.SetInitialTransform(initial, inPlace=False)

    start = time.perf_counter()
    composite = method.Execute(fixed, moving)
    seconds = time.perf_counter() - start

    transform = as_euler_transform(composite)
    info = {"metric_before": metric_before, "metric_after": method.GetMetricValue(),
            "iterations_last_level": method.GetOptimizerIteration(),
            "stop_condition": method.GetOptimizerStopConditionDescription(), "seconds": seconds}
    return transform, info


def euler_parameters(transform):
    """Rotation angles (degrees) and translation (mm) of an Euler3DTransform."""
    angles = np.degrees(transform.GetParameters()[:3])
    translation = np.array(transform.GetParameters()[3:])
    return {"rot_x_deg": angles[0], "rot_y_deg": angles[1], "rot_z_deg": angles[2],
            "tx_mm": translation[0], "ty_mm": translation[1], "tz_mm": translation[2]}


def roi_points(mask_image, every=25):
    """Physical coordinates (LPS, mm) of every `every`-th ROI voxel."""
    array = array_from_image(mask_image) > 0
    zyx = np.array(np.nonzero(array)).T[::every]
    return np.array([mask_image.TransformIndexToPhysicalPoint([int(v) for v in p[::-1]]) for p in zyx])


def displacement_mm(transform, points, reference_transform=None):
    """Distance (mm) each point moves under `transform` (or its difference to `reference_transform`)."""
    moved = np.array([transform.TransformPoint(tuple(p)) for p in points])
    target = points if reference_transform is None else \
        np.array([reference_transform.TransformPoint(tuple(p)) for p in points])
    return np.linalg.norm(moved - target, axis=1)


# def apply_transform(moving, fixed, transform, interpolator=sitk.sitkBSpline, default_value=0.0,
#                     pixel_type=sitk.sitkFloat32):
#     """Resample `moving` onto the grid of `fixed` using the registration transform.

#     The output is float32 by default so interpolated values are not rounded to the integer input type.
#     Voxels that map outside the moving image get `default_value`.
#     """
#     return sitk.Resample(moving, fixed, transform, interpolator, default_value, pixel_type)

def apply_transform(moving, fixed, transform):
    return sitk.Resample(
        moving,                 # image being moved
        fixed,                  # output grid = fixed image
        transform,              # DO NOT use GetInverse()
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID()
    )

def rescale_01(image, low=1.0, high=99.5):
    """Linear rescale to [0, 1] using percentiles (for display only)."""
    array = array_from_image(image).astype(np.float32)
    lo, hi = np.percentile(array, [low, high])
    return image_from_array(np.clip((array - lo) / (hi - lo), 0, 1).astype(np.float32), image)


def checkerboard(image_a, image_b, tiles=(8, 8, 1)):
    """SimpleITK checkerboard of two images on the same grid (both rescaled to 0-1 first)."""
    return sitk.CheckerBoard(rescale_01(image_a), rescale_01(image_b), list(tiles))

def registration_transform_path(case, channel):
    """File written by this notebook for channel k -> channel 0."""
    return Path(config["REGISTRATION_DIR"]) / f"{case}_ch{channel}_to_ch0.tfm"


def load_registration_transform(case, channel):
    """Rigid transform of notebook 03 (identity for channel 0)."""
    if channel == 0:
        return sitk.Euler3DTransform()
    path = registration_transform_path(case, channel)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run notebook 03 first")
    return as_euler_transform(sitk.ReadTransform(str(path)))


def load_registered_channel(case, channel, image=None):
    """Channel `channel` of `case` on the channel-0 grid (B-spline, float32).

    `image` can be passed if the channel was already loaded (and e.g. bias-corrected).
    """
    image = image if image is not None else load_image(image_path(case, channel))
    if channel == 0:
        return image
    reference = load_image(image_path(case, 0))
    return apply_transform(sitk.Cast(image, sitk.sitkFloat32), reference,
                           load_registration_transform(case, channel))