"""
Exponential filter
    f(x) = exp(c x),  c = log(M) / M,  M = max|x| over the whole image   (so f(M) = M)
Strongly expands bright values; all outputs are positive.
"""
from radiomics import imageoperations

NAME = "exponential_filter"
IMAGE_TYPE = "Exponential"
CUSTOM = False
DEFAULT_ARGS = {}

def generate(image, mask, **kwargs):
    return imageoperations.getExponentialImage(image, mask, **kwargs)
