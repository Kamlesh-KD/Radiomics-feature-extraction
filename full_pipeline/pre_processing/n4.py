import SimpleITK as sitk
from scripts import *

def body_mask(image, bins=200):
    """Foreground (body) mask: Otsu threshold separates tissue from background air."""
    return sitk.OtsuThreshold(sitk.Cast(image, sitk.sitkFloat32), 0, 1, bins)


def estimate_log_bias_field(image, mask=None, shrink_factor=4, fitting_levels=3, iterations=50,
                            convergence_threshold=1e-3):
    """Run N4 and return the log bias field on the full-resolution grid of `image` (float32).

    corrected = image / exp(log_bias)
    """
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = body_mask(image) if mask is None else mask
    small_image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    small_mask = sitk.Shrink(mask, [shrink_factor] * image.GetDimension())

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([iterations] * fitting_levels)
    n4.SetConvergenceThreshold(convergence_threshold)
    n4.Execute(small_image, small_mask)
    # The fitted B-spline can be evaluated on any grid: here, the full-resolution image.
    return n4.GetLogBiasFieldAsImage(image)


def apply_log_bias_field(image, log_bias):
    """Divide the image by the bias field exp(log_bias). Output float32."""
    image = sitk.Cast(image, sitk.sitkFloat32)
    return sitk.Divide(image, sitk.Exp(log_bias))


def n4_bias_field_correction(image, mask=None, **n4_settings):
    """Convenience wrapper: returns (corrected image, bias field b = exp(log_bias))."""
    log_bias = estimate_log_bias_field(image, mask, **n4_settings)
    return apply_log_bias_field(image, log_bias), sitk.Exp(log_bias)


def load_bias_corrected_channel(case, channel):
    """Channel image divided by its own N4 bias field (native grid, float32)."""
    image = load_image(image_path(case, channel))
    corrected, _ = n4_bias_field_correction(image)
    return corrected