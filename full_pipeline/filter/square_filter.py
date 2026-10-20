"""Square filter:
    f(x) = (c x)^2,  c = 1 / sqrt(max|x|)
max|x| is taken over the whole (resampled, pre-cropped) image, so the output stays in the input range.
Emphasizes bright values; negative values become positive.
"""
from radiomics import imageoperations

NAME = "square_filter"
IMAGE_TYPE = "Square"
CUSTOM = False
DEFAULT_ARGS = {}

def generate(image, mask, **kwargs):
    return imageoperations.getSquareImage(image, mask, **kwargs)
