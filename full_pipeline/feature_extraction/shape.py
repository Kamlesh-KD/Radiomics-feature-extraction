import numpy as np
import yaml
from pathlib import Path
from collections import OrderedDict
from scipy.spatial import ConvexHull
import math

import yaml
config_path = Path("/Users/kamleshranabhat/Desktop/radiomics/full_pipeline/config_files/setup.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config

MC_CORNERS = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                       [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]])

MC_EDGE_MIDPOINTS = np.array([[0, 0, 0.5], [0, 0.5, 1], [0, 1, 0.5], [0, 0.5, 0],
                              [1, 0, 0.5], [1, 0.5, 1], [1, 1, 0.5], [1, 0.5, 0],
                              [0.5, 0, 0], [0.5, 0, 1], [0.5, 1, 1], [0.5, 1, 0]])

MC_TRIANGLES = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  8,  3,  1,  9,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1],
    [11,  2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  2,  0,  0,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  2,  1,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1],
    [ 3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10,  1,  0,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  0,  3,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  8, 10, 11, 10,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  4,  3,  4,  7,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  4,  1,  4,  7,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  2, 10,  9,  0,  2,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1],
    [ 2, 10,  9,  2,  9,  7,  2,  7,  3,  4,  7,  9, -1, -1, -1, -1],
    [ 4,  7,  8, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0,  4,  7,  8, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  7, 11,  9,  4, 11, 11,  2,  9,  1,  9,  2, -1, -1, -1, -1],
    [ 3, 10,  1, 11, 10,  3,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1],
    [11, 10,  1,  1,  4, 11,  1,  0,  4,  4,  7, 11, -1, -1, -1, -1],
    [ 4,  7,  8,  9,  0,  3,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1],
    [ 4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  5,  4,  0,  1,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3,  1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 2, 10,  5,  5,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1],
    [ 2, 10,  5,  3,  2,  5,  3,  5,  4,  8,  3,  4, -1, -1, -1, -1],
    [ 9,  5,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  2,  0,  0,  8, 11,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  5,  4,  0,  1,  5, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  1,  5,  2,  5,  8, 11,  2,  8,  5,  4,  8, -1, -1, -1, -1],
    [ 3, 10,  1, 11, 10,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 5,  4,  9, 10,  1,  0,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1],
    [ 5,  4,  0,  5,  0, 11, 10,  5, 11, 11,  0,  3, -1, -1, -1, -1],
    [ 5,  4,  8, 10,  5,  8, 11, 10,  8, -1, -1, -1, -1, -1, -1, -1],
    [ 7,  8,  9,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  0,  9,  5,  3,  9,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 7,  8,  0,  0,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  5,  7,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 7,  8,  9,  5,  7,  9,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10,  3,  0,  9,  3,  9,  5,  3,  5,  7, -1, -1, -1, -1],
    [ 0,  2,  8,  8,  2,  5,  8,  5,  7,  2, 10,  5, -1, -1, -1, -1],
    [ 2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 5,  7,  9,  7,  8,  9, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 5,  7,  9,  9,  7,  2,  2,  0,  9, 11,  2,  7, -1, -1, -1, -1],
    [11,  2,  3,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1],
    [ 2,  1, 11,  1,  7, 11,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 7,  8,  9,  5,  7,  9,  3, 10,  1, 11, 10,  3, -1, -1, -1, -1],
    [ 5,  7,  0,  5,  0,  9,  7, 11,  0, 10,  1,  0, 11, 10,  0, -1],
    [11, 10,  0,  0,  3, 11, 10,  5,  0,  0,  7,  8,  5,  7,  0, -1],
    [11, 10,  5,  5,  7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 6,  5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  8,  3,  1,  9,  8,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  6,  5,  1,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  6,  5,  1,  2,  6,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1],
    [ 5,  9,  8,  5,  8,  2,  5,  2,  6,  8,  3,  2, -1, -1, -1, -1],
    [11,  2,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  2,  0,  0,  8, 11,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0, 11,  2,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1],
    [11,  2,  1,  1,  9, 11,  9,  8, 11,  6,  5, 10, -1, -1, -1, -1],
    [ 6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1],
    [ 6,  3, 11,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1],
    [ 6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  7,  8,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  4,  3,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0,  4,  7,  8,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  4,  1,  4,  7,  1,  7,  3,  6,  5, 10, -1, -1, -1, -1],
    [ 4,  7,  8,  1,  6,  5,  1,  2,  6, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  4,  3,  4,  7,  3,  1,  6,  5,  1,  2,  6, -1, -1, -1, -1],
    [ 4,  7,  8,  9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1],
    [ 7,  3,  9,  4,  7,  9,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1],
    [11,  2,  3,  4,  7,  8,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1],
    [11,  4,  7, 11,  2,  4,  2,  0,  4,  6,  5, 10, -1, -1, -1, -1],
    [ 1,  9,  0, 11,  2,  3,  4,  7,  8,  6,  5, 10, -1, -1, -1, -1],
    [ 4,  7, 11,  9,  4, 11, 11,  2,  9,  1,  9,  2,  6,  5, 10, -1],
    [ 4,  7,  8,  6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1],
    [ 5,  1, 11,  5, 11,  6,  1,  0, 11,  4,  7, 11,  0,  4, 11, -1],
    [ 4,  7,  8,  6,  3, 11,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1],
    [ 6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1],
    [10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10,  4,  9,  6,  4, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  1, 10, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  8,  3,  1,  6,  8,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1],
    [ 1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  4,  9,  1,  2,  4,  2,  6,  4,  0,  8,  3, -1, -1, -1, -1],
    [ 0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1],
    [11,  2,  3, 10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1],
    [11,  2,  0,  0,  8, 11, 10,  4,  9,  6,  4, 10, -1, -1, -1, -1],
    [11,  2,  3,  0,  1, 10, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1],
    [ 6,  4,  1,  6,  1, 10,  1,  4,  8, 11,  2,  1,  1,  8, 11, -1],
    [ 9,  6,  4,  3,  6,  9,  1,  3,  9, 11,  6,  3, -1, -1, -1, -1],
    [ 1,  8, 11,  0,  8,  1, 11,  6,  1,  1,  4,  9,  6,  4,  1, -1],
    [ 6,  3, 11,  0,  3,  6,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  6,  4,  6,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 6,  7, 10,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 6,  7, 10,  0, 10,  7,  0,  9, 10,  0,  7,  3, -1, -1, -1, -1],
    [ 6,  7, 10,  1, 10,  7,  1,  7,  8,  0,  1,  8, -1, -1, -1, -1],
    [ 6,  7, 10,  1, 10,  7,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2,  6,  1,  6,  8,  1,  8,  9,  6,  7,  8, -1, -1, -1, -1],
    [ 2,  6,  9,  1,  2,  9,  6,  7,  9,  3,  0,  9,  7,  3,  9, -1],
    [ 0,  7,  8,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  7,  3,  2,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  2,  3,  6,  7, 10,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1],
    [ 2,  0,  7, 11,  2,  7,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1],
    [ 6,  7, 10,  1, 10,  7,  1,  7,  8,  0,  1,  8, 11,  2,  3, -1],
    [11,  2,  1,  1,  7, 11, 10,  6,  1,  1,  6,  7, -1, -1, -1, -1],
    [ 8,  9,  6,  6,  7,  8,  1,  6,  9, 11,  6,  3,  1,  3,  6, -1],
    [ 0,  9,  1,  6,  7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  7,  8,  7,  0,  6,  0,  3, 11, 11,  6,  0, -1, -1, -1, -1],
    [ 6,  7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
], dtype=np.int64)

def cube_indices(padded_mask):
    """8-bit marching-cubes index of every 2x2x2 cube of a (padded) binary mask."""
    m = padded_mask.astype(np.int32)
    Z, Y, X = m.shape
    index = np.zeros((Z - 1, Y - 1, X - 1), dtype=np.int32)
    for bit, (dz, dy, dx) in enumerate(MC_CORNERS):
        index |= m[dz:dz + Z - 1, dy:dy + Y - 1, dx:dx + X - 1] << bit
    return index


def mesh_triangles(mask_array, spacing_zyx):
    """All mesh triangles as three (N, 3) arrays of physical (z, y, x) vertices, plus a +-1 sign per triangle.

    Cubes with index >= 128 are looked up as their complement (255 - index) with sign -1.
    """
    padded = np.pad(mask_array.astype(bool), 1)
    index = cube_indices(padded)
    sign = np.where(index >= 128, -1.0, 1.0)
    index = np.where(index >= 128, 255 - index, index)
    cz, cy, cx = np.nonzero(index)                         # cubes 0 (all out) / 255 (all in) have no faces
    cube_index = index[cz, cy, cx]
    cube_sign = sign[cz, cy, cx]
    cube_origin = np.stack([cz, cy, cx], axis=1).astype(float)
    spacing = np.asarray(spacing_zyx, dtype=float)

    a_list, b_list, c_list, s_list = [], [], [], []
    for t in range(5):                                     # at most 5 triangles per cube
        edges = MC_TRIANGLES[cube_index, 3 * t:3 * t + 3]
        has_triangle = edges[:, 0] >= 0
        if not has_triangle.any():
            continue
        origin = cube_origin[has_triangle]
        e = edges[has_triangle]
        a_list.append((origin + MC_EDGE_MIDPOINTS[e[:, 0]]) * spacing)
        b_list.append((origin + MC_EDGE_MIDPOINTS[e[:, 1]]) * spacing)
        c_list.append((origin + MC_EDGE_MIDPOINTS[e[:, 2]]) * spacing)
        s_list.append(cube_sign[has_triangle])
    return np.concatenate(a_list), np.concatenate(b_list), np.concatenate(c_list), np.concatenate(s_list)


def mesh_volume(a, b, c, sign):
    """Sum of signed tetrahedron volumes a . (b x c) / 6."""
    return float(np.sum(sign * np.einsum("ij,ij->i", np.cross(a, b), c)) / 6.0)


def mesh_surface_area(a, b, c):
    """Sum of triangle areas |(a - c) x (b - c)| / 2."""
    return float(np.sum(0.5 * np.linalg.norm(np.cross(a - c, b - c), axis=1)))


def mesh_vertices(mask_array, spacing_zyx):
    """All mesh vertices: midpoints between face-adjacent voxels where the mask changes (physical z,y,x)."""
    padded = np.pad(mask_array.astype(bool), 1)
    spacing = np.asarray(spacing_zyx, dtype=float)
    vertices = []
    for axis in range(3):
        lower = [slice(None)] * 3; upper = [slice(None)] * 3
        lower[axis] = slice(0, -1); upper[axis] = slice(1, None)
        changes = np.array(np.nonzero(padded[tuple(lower)] != padded[tuple(upper)]), dtype=float).T
        changes[:, axis] += 0.5
        vertices.append(changes * spacing)
    return np.concatenate(vertices)

def max_pairwise_distance(points, chunk=2000):
    """Largest Euclidean distance between any two of the points (exact)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    if len(points) > points.shape[1] + 1:
        try:
            points = points[ConvexHull(points).vertices]
        except Exception:                                  # flat/degenerate point set: keep all points
            pass
    best = 0.0
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        d2 = ((block[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        best = max(best, float(d2.max()))
    return math.sqrt(best)


def maximum_diameters(vertices):
    """(slice, column, row, 3D) maximum diameters of a vertex cloud in (z, y, x) mm coordinates."""
    planar = []
    for axis in range(3):                                  # 0: same z (slice), 1: same y (column), 2: same x (row)
        other = [k for k in range(3) if k != axis]
        best = 0.0
        for value in np.unique(vertices[:, axis]):
            in_plane = vertices[vertices[:, axis] == value][:, other]
            best = max(best, max_pairwise_distance(in_plane))
        planar.append(best)
    return planar[0], planar[1], planar[2], max_pairwise_distance(vertices)

#principal eigenvalues
def principal_eigenvalues(mask_array, spacing_zyx):
    """Eigenvalues (ascending) of the covariance of the physical voxel-centre coordinates.

    Covariance with divisor N (population), as PyRadiomics. Tiny negative values from rounding are set to 0.
    """
    coordinates = np.array(np.nonzero(mask_array), dtype=float).T * np.asarray(spacing_zyx, dtype=float)
    coordinates -= coordinates.mean(axis=0)
    coordinates /= math.sqrt(len(coordinates))
    eigenvalues = np.linalg.eigvals(coordinates.T @ coordinates).real
    eigenvalues[(eigenvalues < 0) & (eigenvalues > -1e-10)] = 0
    return np.sort(eigenvalues)

#feature functions
def sh_mesh_volume(V):                      return V
def sh_voxel_volume(n_voxels, spacing_zyx): return n_voxels * float(np.prod(spacing_zyx))
def sh_surface_area(A):                     return A
def sh_surface_volume_ratio(A, V):          return A / V
def sh_sphericity(A, V):                    return (36 * math.pi * V ** 2) ** (1.0 / 3.0) / A
def sh_major_axis_length(ev):               return 4 * math.sqrt(ev[2]) if ev[2] >= 0 else np.nan
def sh_minor_axis_length(ev):               return 4 * math.sqrt(ev[1]) if ev[1] >= 0 else np.nan
def sh_least_axis_length(ev):               return 4 * math.sqrt(ev[0]) if ev[0] >= 0 else np.nan
def sh_elongation(ev):                      return math.sqrt(ev[1] / ev[2]) if ev[2] > 0 and ev[1] >= 0 else np.nan
def sh_flatness(ev):                        return math.sqrt(ev[0] / ev[2]) if ev[2] > 0 and ev[0] >= 0 else np.nan


def shape_features(mask_array, spacing_zyx):
    """The 14 3D shape features of a binary ROI (any crop of the mask; padding is added internally)."""
    a, b, c, sign = mesh_triangles(mask_array, spacing_zyx)
    V = mesh_volume(a, b, c, sign)
    A = mesh_surface_area(a, b, c)
    d_slice, d_column, d_row, d_3d = maximum_diameters(mesh_vertices(mask_array, spacing_zyx))
    ev = principal_eigenvalues(mask_array, spacing_zyx)
    return OrderedDict([
        ("Elongation", sh_elongation(ev)),
        ("Flatness", sh_flatness(ev)),
        ("LeastAxisLength", sh_least_axis_length(ev)),
        ("MajorAxisLength", sh_major_axis_length(ev)),
        ("Maximum2DDiameterColumn", d_column),
        ("Maximum2DDiameterRow", d_row),
        ("Maximum2DDiameterSlice", d_slice),
        ("Maximum3DDiameter", d_3d),
        ("MeshVolume", sh_mesh_volume(V)),
        ("MinorAxisLength", sh_minor_axis_length(ev)),
        ("Sphericity", sh_sphericity(A, V)),
        ("SurfaceArea", sh_surface_area(A)),
        ("SurfaceVolumeRatio", sh_surface_volume_ratio(A, V)),
        ("VoxelVolume", sh_voxel_volume(int(mask_array.sum()), spacing_zyx)),
    ])