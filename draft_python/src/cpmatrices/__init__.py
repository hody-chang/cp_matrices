"""Draft Python core for Closest Point Method sparse matrices."""

from .bands import make_invbandmap, rm_bandwidth
from .grid import TensorGrid, uniform_axis
from .interpolation import interp2_matrix, interp3_matrix, interpn_matrix
from .laplacian import laplacian_2d_matrix, laplacian_3d_matrix, laplacian_nd_matrix
from .surfaces import circle_closest_point, sphere_closest_point

__all__ = [
    "TensorGrid",
    "circle_closest_point",
    "interp2_matrix",
    "interp3_matrix",
    "interpn_matrix",
    "laplacian_2d_matrix",
    "laplacian_3d_matrix",
    "laplacian_nd_matrix",
    "make_invbandmap",
    "rm_bandwidth",
    "sphere_closest_point",
    "uniform_axis",
]
