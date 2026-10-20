"""Laplacian of Gaussian:
    LoG_sigma[I] = sigma^2 * (d2/dx2 + d2/dy2 + d2/dz2) (G_sigma * I)
LaplacianRecursiveGaussianImageFilter with scale normalization; sigma in mm.
Bright blobs of radius ~ sigma*sqrt(3) give strongly negative values,
dark blobs positive values. One image per sigma, named 'log-sigma-<s>-mm-3D'.
Sigmas too large for the image (< ceil(sigma/spacing)+1 voxels) are skipped.
"""
from radiomics import imageoperations

NAME = "log_filter"
IMAGE_TYPE = "LoG"
CUSTOM = False
DEFAULT_ARGS = {"sigma": [1.0, 2.0, 3.0, 4.0, 5.0]}

def generate(image, mask, **kwargs):
    return imageoperations.getLoGImage(image, mask, **kwargs)
