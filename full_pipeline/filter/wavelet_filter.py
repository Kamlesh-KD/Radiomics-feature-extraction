"""Wavelet transform:
Undecimated (PyWavelets swtn) decomposition with the chosen wavelet (default coif1). At each level the low-pass
(L) or high-pass (H) filter is applied along x, y and z, giving 8 sub-bands named 'wavelet-<xyz letters>'
(e.g. 'wavelet-LLH' = low-pass along x and y, high-pass along z). Sub-bands have the same size as the input,
so the ROI mask applies unchanged. Levels > 1 are named 'wavelet2-...', etc.; only the last level's LLL is kept.
"""
from radiomics import imageoperations

NAME = "wavelet_filter"
IMAGE_TYPE = "Wavelet"
CUSTOM = False
DEFAULT_ARGS = {"wavelet": "coif1", "level": 1, "start_level": 0}

def generate(image, mask, **kwargs):
    return imageoperations.getWaveletImage(image, mask, **kwargs)
