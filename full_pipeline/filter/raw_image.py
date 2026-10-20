"""No filter: the normalized, resampled image is used as it is. Features are named 'original_<class>_<feature>'.
Shape features (which do not depend on intensities) are stored with this image type only.
"""
from radiomics import imageoperations

NAME = "raw_image"
IMAGE_TYPE = "Original"
CUSTOM = False
DEFAULT_ARGS = {}

def generate(image, mask, **kwargs):
    return imageoperations.getOriginalImage(image, mask, **kwargs)
