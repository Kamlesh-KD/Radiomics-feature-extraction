"""
Gabor filter
Gabor kernel; a complex sinusoid (wavelength lambda) under an elongated Gaussian envelope, defined in a
2D plane:
    h(u, v) = gamma / (2 pi sigma^2) * exp(-(u'^2 + gamma^2 v'^2) / (2 sigma^2)) * exp(i 2 pi u' / lambda)
    u' =  u cos(theta) + v sin(theta)
    v' = -u sin(theta) + v cos(theta)
sigma and lambda are in mm (the kernel is sampled at the image spacing and multiplied by the pixel area, so
the envelope sums to ~1).
The response map is the modulus |I * h|, computed slice by slice in each selected plane. 
Rotation invariance: the modulus is averaged over `gaborOrientations` angles theta = k pi / n
(theta and theta + pi give the same modulus) and over the selected planes (axial / coronal / sagittal).
Borders are handled by mirror padding. One image per (sigma, lambda) pair, named e.g.
'gabor-sigma-2-0-mm-lambda-4-0-mm-axial'.

Large responses = stripes/edges with period ~lambda; good for oriented texture.
"""
import itertools
import numpy as np
import SimpleITK as sitk
from scipy.signal import fftconvolve

NAME = "gabor_filter"
IMAGE_TYPE = "Gabor"
CUSTOM = True
DEFAULT_ARGS = {"gaborSigma": [2.0, 4.0], "gaborLambda": [4.0, 8.0], "gaborGamma": 0.5,
                "gaborOrientations": 8, "gaborPlanes": ["axial"], "gaborTruncate": 3.0}

# numpy axes (of a [z, y, x] array) spanned by each plane, as (row axis, column axis)
PLANE_AXES = {"axial": (1, 2), "coronal": (0, 2), "sagittal": (0, 1)}

def gabor_kernel(sigma_mm, wavelength_mm, gamma, theta, spacing_rc, truncate=3.0):
    """Complex 2D Gabor kernel on a (row, column) grid with the given spacing (mm)."""
    reach_mm = truncate * sigma_mm / min(gamma, 1.0)         # envelope is sigma/gamma long along v'
    radius = [int(np.ceil(reach_mm / s)) for s in spacing_rc]
    v = np.arange(-radius[0], radius[0] + 1) * spacing_rc[0]
    u = np.arange(-radius[1], radius[1] + 1) * spacing_rc[1]
    v, u = np.meshgrid(v, u, indexing="ij")
    u_rot = u * np.cos(theta) + v * np.sin(theta)
    v_rot = -u * np.sin(theta) + v * np.cos(theta)
    envelope = gamma / (2 * np.pi * sigma_mm ** 2) * np.exp(-(u_rot ** 2 + (gamma * v_rot) ** 2) / (2 * sigma_mm ** 2))
    return envelope * np.exp(2j * np.pi * u_rot / wavelength_mm) * spacing_rc[0] * spacing_rc[1]


def gabor_response(array, spacing_zyx, sigma_mm, wavelength_mm, gamma=0.5, n_orientations=8,
                   planes=("axial",), truncate=3.0):
    """Orientation- (and plane-) averaged Gabor modulus of a [z, y, x] array."""
    array = np.asarray(array, dtype=np.float64)
    total = np.zeros(array.shape)
    for plane in planes:
        row_axis, col_axis = PLANE_AXES[plane]
        spacing_rc = (spacing_zyx[row_axis], spacing_zyx[col_axis])
        kernels = [gabor_kernel(sigma_mm, wavelength_mm, gamma, k * np.pi / n_orientations, spacing_rc, truncate)
                   for k in range(n_orientations)]
        r_row, r_col = kernels[0].shape[0] // 2, kernels[0].shape[1] // 2
        pad = [(0, 0)] * 3
        pad[row_axis], pad[col_axis] = (r_row, r_row), (r_col, r_col)
        padded = np.pad(array, pad, mode="symmetric")
        for kernel in kernels:
            shape = [1, 1, 1]
            shape[row_axis], shape[col_axis] = kernel.shape
            response = fftconvolve(padded, kernel.reshape(shape), mode="valid", axes=(row_axis, col_axis))
            total += np.abs(response)
    return total / (len(planes) * n_orientations)


def gabor_image_name(sigma_mm, wavelength_mm, planes):
    """PyRadiomics-style name (no underscores), e.g. 'gabor-sigma-2-0-mm-lambda-4-0-mm-axial'."""
    def number(v):
        return str(float(v)).replace(".", "-")
    return "gabor-sigma-%s-mm-lambda-%s-mm-%s" % (number(sigma_mm), number(wavelength_mm), "-".join(planes))


def generate(image, mask, **kwargs):
    """PyRadiomics generator: yields one 'gabor-...' image per (sigma, lambda) pair."""
    args = dict(DEFAULT_ARGS)
    args.update({k: v for k, v in kwargs.items() if k in DEFAULT_ARGS})
    array = sitk.GetArrayFromImage(image)
    spacing_zyx = np.array(image.GetSpacing())[::-1]
    planes = list(args["gaborPlanes"])
    for sigma, wavelength in itertools.product(args["gaborSigma"], args["gaborLambda"]):
        response = gabor_response(array, spacing_zyx, float(sigma), float(wavelength), float(args["gaborGamma"]),
                                  int(args["gaborOrientations"]), planes, float(args["gaborTruncate"]))
        filtered = sitk.GetImageFromArray(response)
        filtered.CopyInformation(image)
        yield filtered, gabor_image_name(sigma, wavelength, planes), kwargs
