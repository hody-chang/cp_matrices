"""Interpolation matrix builders for the Closest Point Method."""

from __future__ import annotations

from itertools import product
import warnings

import numpy as np
from scipy import sparse
from scipy.special import comb

from .bands import make_invbandmap
from .indexing import meshgrid_linear_index, ndgrid_linear_index


def lagrange_weights_1d(base_points, points, dx, stencil_size: int) -> np.ndarray:
    """Barycentric Lagrange interpolation weights on an equispaced grid."""

    base_points = np.asarray(base_points, dtype=float).ravel()
    points = np.asarray(points, dtype=float).ravel()
    dx = np.asarray(dx, dtype=float)

    if base_points.shape != points.shape:
        base_points, points = np.broadcast_arrays(base_points, points)
        base_points = base_points.ravel()
        points = points.ravel()

    if stencil_size < 1:
        raise ValueError("stencil_size must be positive")

    j = np.arange(stencil_size, dtype=float)
    bary = np.array(
        [(-1.0) ** k * comb(stencil_size - 1, k, exact=False) for k in range(stencil_size)],
        dtype=float,
    )

    grid_points = base_points[:, None] + j[None, :] * dx
    denom = points[:, None] - grid_points
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = bary[None, :] / denom

    exact = denom == 0
    exact_rows = np.any(exact, axis=1)
    if np.any(exact_rows):
        weights[exact_rows, :] = exact[exact_rows, :].astype(float)

    weights /= np.sum(weights, axis=1)[:, None]
    return weights


def find_grid_interp_base_points(points, degree: int, lower_left, dx):
    """Return base stencil indices and coordinates for interpolation points.

    The returned indices are 0-based. This follows the asymmetric MATLAB
    stencil convention used by `findGridInterpBasePt_vec.m`.
    """

    coords = _as_coordinate_list(points)
    dim = len(coords)
    lower_left = np.asarray(lower_left, dtype=float).ravel()
    dx = np.asarray(dx, dtype=float).ravel()
    if lower_left.size != dim or dx.size != dim:
        raise ValueError("lower_left and dx must match the point dimension")

    base_indices = []
    base_coords = []
    for axis, values in enumerate(coords):
        values = np.asarray(values, dtype=float).ravel()
        if degree % 2 == 0:
            base = np.floor((values - lower_left[axis]) / dx[axis]).astype(np.int64) - degree // 2
        else:
            base = np.floor((values - lower_left[axis]) / dx[axis]).astype(np.int64) - (degree - 1) // 2
        base_indices.append(base)
        base_coords.append(lower_left[axis] + base * dx[axis])

    return base_indices, base_coords


def interp2_matrix(x, y, xi, yi, degree: int = 3, band=None, ordering: str = "meshgrid"):
    """Build a 2D sparse interpolation matrix.

    `band`, if supplied, must contain 0-based natural linear grid indices.
    """

    return interpn_matrix([x, y], [xi, yi], degree=degree, band=band, ordering=ordering)


def interp3_matrix(x, y, z, xi, yi, zi, degree: int = 3, band=None, ordering: str = "meshgrid"):
    """Build a 3D sparse interpolation matrix.

    `band`, if supplied, must contain 0-based natural linear grid indices.
    """

    return interpn_matrix([x, y, z], [xi, yi, zi], degree=degree, band=band, ordering=ordering)


def interpn_matrix(xs, xi, degree: int = 3, band=None, ordering: str = "ndgrid"):
    """Build an n-D sparse interpolation matrix.

    Parameters
    ----------
    xs:
        Sequence of 1D coordinate arrays defining an equispaced grid.
    xi:
        Sequence of coordinate arrays for interpolation points.
    degree:
        Polynomial interpolation degree. The stencil size is `degree + 1`.
    band:
        Optional 0-based natural linear indices. If given, output columns are
        restricted to the band.
    ordering:
        `"ndgrid"` or `"meshgrid"`. The latter is supported only for 2D/3D.
    """

    xs = [np.asarray(axis, dtype=float).ravel() for axis in xs]
    xi = _as_coordinate_list(xi)
    dim = len(xs)
    if len(xi) != dim:
        raise ValueError("xs and xi dimensions do not match")
    if any(axis.size < 2 for axis in xs):
        raise ValueError("each grid axis must contain at least two points")

    n_points = xi[0].size
    if any(coord.size != n_points for coord in xi):
        raise ValueError("all interpolation coordinate arrays must have equal length")

    shape = tuple(axis.size for axis in xs)
    dx = np.array([axis[1] - axis[0] for axis in xs], dtype=float)
    lower_left = np.array([axis[0] for axis in xs], dtype=float)
    grid_size = int(np.prod(shape))
    stencil_size_1d = degree + 1
    stencil_size = stencil_size_1d**dim

    base_indices, base_coords = find_grid_interp_base_points(xi, degree, lower_left, dx)
    weights_by_axis = [
        lagrange_weights_1d(base_coords[axis], xi[axis], dx[axis], stencil_size_1d)
        for axis in range(dim)
    ]

    rows = np.tile(np.arange(n_points, dtype=np.int64), stencil_size)
    cols = np.empty(n_points * stencil_size, dtype=np.int64)
    vals = np.empty(n_points * stencil_size, dtype=float)

    linear_index = _linear_indexer(ordering, dim)
    cursor = 0
    for offsets in product(range(stencil_size_1d), repeat=dim):
        stencil_indices = [base_indices[axis] + offsets[axis] for axis in range(dim)]
        cols[cursor : cursor + n_points] = linear_index(stencil_indices, shape)

        weight = weights_by_axis[0][:, offsets[0]]
        for axis in range(1, dim):
            weight = weight * weights_by_axis[axis][:, offsets[axis]]
        vals[cursor : cursor + n_points] = weight
        cursor += n_points

    if band is None:
        return sparse.coo_matrix((vals, (rows, cols)), shape=(n_points, grid_size)).tocsr()

    band = np.asarray(band, dtype=np.int64).ravel()
    invband = make_invbandmap(grid_size, band)
    mapped_cols = invband[cols]
    keep = mapped_cols >= 0
    if not np.all(keep):
        warnings.warn(
            "non-zero interpolation coefficients outside the band were discarded",
            RuntimeWarning,
            stacklevel=2,
        )

    return sparse.coo_matrix(
        (vals[keep], (rows[keep], mapped_cols[keep])),
        shape=(n_points, band.size),
    ).tocsr()


def _as_coordinate_list(points):
    if isinstance(points, (list, tuple)):
        return [np.asarray(coord, dtype=float).ravel() for coord in points]

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError("points must be a coordinate list or an (n_points, dim) array")
    return [arr[:, axis].ravel() for axis in range(arr.shape[1])]


def _linear_indexer(ordering: str, dim: int):
    ordering = ordering.lower()
    if ordering == "ndgrid":
        return ndgrid_linear_index
    if ordering == "meshgrid":
        if dim not in (2, 3):
            raise ValueError("meshgrid ordering is only supported for 2D and 3D")
        return meshgrid_linear_index
    raise ValueError("ordering must be 'ndgrid' or 'meshgrid'")
