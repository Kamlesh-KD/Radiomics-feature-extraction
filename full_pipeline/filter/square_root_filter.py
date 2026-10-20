"""Square root filter:
    f(x) = sign(x) * sqrt(M |x|),  M = max|x| over the whole image
Compresses bright values and expands dark ones; the sign of negative values is kept.
"""
from radiomics import imageoperations

NAME = "square_root_filter"
IMAGE_TYPE = "SquareRoot"
CUSTOM = False
DEFAULT_ARGS = {}

def generate(image, mask, **kwargs):
    return imageoperations.getSquareRootImage(image, mask, **kwargs)
