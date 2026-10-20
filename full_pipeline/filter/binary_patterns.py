"""In local binary patterns, for every roi voxel image is sampled on icosphere around it, and each sample is binarized.
Binary pattern on the sphere is expanded in spheraical harmonics; energy of degree n gives rotation-invariant map 'lbp-3D-m<n+1>'
and kurtosis of raw samples gives 'lbp-3D-k'. Values are computed inside roi."""
#Reference: Banerjee et al., "A 3D rotation invariant local binary pattern", 2013.

from radiomics import imageoperations

NAME = "binary_patterns"
IMAGE_TYPE = "LBP3D"
CUSTOM = False
DEFAULT_ARGS = {"lbp3DLevels": 2, "lbp3DIcosphereRadius": 1, "lbp3DIcosphereSubdivision": 1}

def generate(image, mask, **kwargs):
    return imageoperations.getLBP3DImage(image, mask, **kwargs)
