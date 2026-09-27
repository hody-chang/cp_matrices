"""diffcpm -- a differentiable closest point method layer in JAX.

See README.md in this directory for what it is for and what is verified.
"""

from .grid import BandedGrid, band_from_cp, band_from_dist, make_grid1d, rm_bandwidth
from .interp import InterpPattern, apply_sparse, apply_sparse_T, lagrange_weights_1d
from .inverse import Observer, check_gradient, lbfgs, misfit, tikhonov
from .operators import CPMOperator, build_operator, laplacian_dropped, laplacian_pattern
from .sdf import (adam, cp_level_set, level_set_residual_norm, siren_apply,
                  siren_init)
from .solve import linear_solve, solve_operator

__all__ = [
    "BandedGrid", "band_from_cp", "band_from_dist", "make_grid1d", "rm_bandwidth",
    "InterpPattern", "apply_sparse", "apply_sparse_T", "lagrange_weights_1d",
    "Observer", "check_gradient", "lbfgs", "misfit", "tikhonov",
    "CPMOperator", "build_operator", "laplacian_dropped", "laplacian_pattern",
    "adam", "cp_level_set", "level_set_residual_norm", "siren_apply", "siren_init",
    "linear_solve", "solve_operator",
]
