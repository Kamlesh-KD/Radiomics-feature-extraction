"""Logarithm:
    f(x) = sign(x) * log(|x| + 1) * M / log(M + 1),  M = max|x| over the whole image
Strongly compresses bright values; rescaled so that the output keeps the input range.
"""
from radiomics import imageoperations

NAME = "logarithmic_filter"
IMAGE_TYPE = "Logarithm"
CUSTOM = False
DEFAULT_ARGS = {}

def generate(image, mask, **kwargs):
    return imageoperations.getLogarithmImage(image, mask, **kwargs)
