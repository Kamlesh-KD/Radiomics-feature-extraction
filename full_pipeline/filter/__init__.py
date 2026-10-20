"""Registry of the filter modules in this folder (used by feature_extraction/pipeline.py).

Every filter module defines NAME, IMAGE_TYPE, CUSTOM, DEFAULT_ARGS and generate(image, mask, **kwargs), a PyRadiomics
style generator yielding (filtered image, image type name, kwargs).
"""
from collections import OrderedDict

from . import (raw_image, binary_patterns, gradient_filter, log_filter, wavelet_filter, square_filter,
               square_root_filter, logarithmic_filter, exponential_filter, gabor_filter, laws_filter)

FILTERS = OrderedDict((module.NAME, module) for module in (
    raw_image, binary_patterns, gradient_filter, log_filter, wavelet_filter, square_filter, square_root_filter,
    logarithmic_filter, exponential_filter, gabor_filter, laws_filter))


def filter_args(name, args=None):
    """Arguments of one filter: its DEFAULT_ARGS, updated with `args`."""
    return {**FILTERS[name].DEFAULT_ARGS, **(args or {})}


def apply_filter(name, image, mask, settings, args=None):
    """OrderedDict(image type name -> filtered SimpleITK image) of one filter."""
    kwargs = {**settings, **filter_args(name, args)}
    return OrderedDict((type_name, filtered) for filtered, type_name, _ in FILTERS[name].generate(image, mask, **kwargs))


def filtered_images(pre, settings=None, filters=None):
    """Yield (image type name, float array [z, y, x]) of every filter (default: all in FILTERS).

    `pre` is the dictionary returned by normalization.preprocess() (it holds the SimpleITK 'image' and 'mask').
    """
    import SimpleITK as sitk
    for name in (filters or FILTERS):
        for type_name, filtered in apply_filter(name, pre["image"], pre["mask"], settings or {}).items():
            yield type_name, sitk.GetArrayFromImage(filtered).astype(float)
