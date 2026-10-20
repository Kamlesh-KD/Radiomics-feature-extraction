"""Gradient magnitude:
    |grad I| = sqrt((dI/dx)^2 + (dI/dy)^2 + (dI/dz)^2)
With `gradientUseSpacing` = True: the derivatives are in intensity units per mm, otherwise per voxel.
Highlights edges; flat regions become ~0.
"""
from radiomics import imageoperations

NAME = "gradient_filter"
IMAGE_TYPE = "Gradient"
CUSTOM = False
DEFAULT_ARGS = {"gradientUseSpacing": True}

def generate(image, mask, **kwargs):
    return imageoperations.getGradientImage(image, mask, **kwargs)
