"""Finite-difference Laplacian matrix builders."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy import sparse

from .bands import make_invbandmap
from .indexing import (
    meshgrid_linear_index,
    ndgrid_linear_index,
    unravel_meshgrid_index,
    unravel_ndgrid_index,
)


def laplacian_2d_matrix(
    x,
    y,
    order: int = 2,
    band1=None,
    band2=None,
    ordering: str = "meshgrid",
):
    """Build a 2D sparse finite-difference Laplacian matrix."""

    return laplacian_nd_matrix([x, y], order=order, band1=band1, band2=band2, ordering=ordering)


def laplacian_3d_matrix(
    x,
    y,
    z,
    order: int = 2,
    band1=None,
    band2=None,
    ordering: str = "meshgrid",
):
    """Build a 3D sparse finite-difference Laplacian matrix."""

    return laplacian_nd_matrix([x, y, z], order=order, band1=band1, band2=band2, ordering=ordering)


def laplacian_nd_matrix(xs, order: int = 2, band1=None, band2=None, ordering: str = "ndgrid"):
    """Build an n-D sparse finite-difference Laplacian matrix.

    Bands use 0-based natural linear indices. If `band2` is supplied, output
    columns are restricted to that band and stencil entries outside it are
    discarded.
    """

    xs = [np.asarray(axis, dtype=float).ravel() for axis in xs]
    if any(axis.size < 2 for axis in xs):
        raise ValueError("each grid axis must contain at least two points")

    shape = tuple(axis.size for axis in xs)
    grid_size = int(np.prod(shape))
    dim = len(shape)

    if band1 is None:
        band1 = np.arange(grid_size, dtype=np.int64)
    else:
        band1 = np.asarray(band1, dtype=np.int64).ravel()

    if band2 is None:
        band2 = band1
    else:
        band2 = np.asarray(band2, dtype=np.int64).ravel()

    offsets, weights = _laplacian_stencil(xs, order)
    return _diff_matrix(shape, band1, band2, offsets, weights, ordering, dim)


def second_derivative_matrices_nd(xs, order: int = 2, band1=None, band2=None, ordering: str = "ndgrid"):
    """Build centered second-derivative matrices for each coordinate axis."""

    xs = [np.asarray(axis, dtype=float).ravel() for axis in xs]
    shape = tuple(axis.size for axis in xs)
    grid_size = int(np.prod(shape))
    dim = len(shape)

    if band1 is None:
        band1 = np.arange(grid_size, dtype=np.int64)
    else:
        band1 = np.asarray(band1, dtype=np.int64).ravel()
    if band2 is None:
        band2 = band1
    else:
        band2 = np.asarray(band2, dtype=np.int64).ravel()

    matrices = []
    for axis, coord in enumerate(xs):
        dx = coord[1] - coord[0]
        if order == 2:
            weights = np.array([1.0, -2.0, 1.0]) / dx**2
            axis_offsets = [-1, 0, 1]
        elif order == 4:
            weights = np.array([-1.0 / 12.0, 4.0 / 3.0, -2.5, 4.0 / 3.0, -1.0 / 12.0]) / dx**2
            axis_offsets = [-2, -1, 0, 1, 2]
        else:
            raise ValueError("only order 2 and order 4 are supported")

        offsets = np.zeros((len(axis_offsets), dim), dtype=np.int64)
        offsets[:, axis] = axis_offsets
        matrices.append(_diff_matrix(shape, band1, band2, offsets, weights, ordering, dim))

    return matrices


def _laplacian_stencil(xs, order: int):
    dim = len(xs)
    dx = np.array([axis[1] - axis[0] for axis in xs], dtype=float)

    offsets = [np.zeros(dim, dtype=np.int64)]
    weights = []

    if order == 2:
        center = -2.0 * np.sum(1.0 / dx**2)
        weights.append(center)
        for axis in range(dim):
            for step in (1, -1):
                off = np.zeros(dim, dtype=np.int64)
                off[axis] = step
                offsets.append(off)
                weights.append(1.0 / dx[axis] ** 2)
    elif order == 4:
        center = -2.5 * np.sum(1.0 / dx**2)
        weights.append(center)
        for axis in range(dim):
            for step, coeff in [(-2, -1.0 / 12.0), (-1, 4.0 / 3.0), (1, 4.0 / 3.0), (2, -1.0 / 12.0)]:
                off = np.zeros(dim, dtype=np.int64)
                off[axis] = step
                offsets.append(off)
                weights.append(coeff / dx[axis] ** 2)
    else:
        raise ValueError("only order 2 and order 4 are supported")

    return np.vstack(offsets), np.asarray(weights, dtype=float)


def _diff_matrix(shape, band1, band2, offsets, weights, ordering: str, dim: int):
    band1 = np.asarray(band1, dtype=np.int64).ravel()
    band2 = np.asarray(band2, dtype=np.int64).ravel()
    grid_size = int(np.prod(shape))

    if np.any(band1 < 0) or np.any(band1 >= grid_size):
        raise ValueError("band1 contains indices outside the grid")
    if np.any(band2 < 0) or np.any(band2 >= grid_size):
        raise ValueError("band2 contains indices outside the grid")

    unravel, linear = _index_fns(ordering, dim)
    base_multi = unravel(band1, shape)
    invband = make_invbandmap(grid_size, band2)

    rows_list = []
    cols_list = []
    vals_list = []
    row_base = np.arange(band1.size, dtype=np.int64)

    for offset, weight in zip(offsets, weights):
        shifted = [np.asarray(axis_idx, dtype=np.int64) + offset[axis] for axis, axis_idx in enumerate(base_multi)]
        in_bounds = np.ones(band1.size, dtype=bool)
        for axis, idx in enumerate(shifted):
            in_bounds &= (idx >= 0) & (idx < shape[axis])

        if not np.all(in_bounds):
            shifted = [idx[in_bounds] for idx in shifted]
            rows = row_base[in_bounds]
        else:
            rows = row_base

        natural_cols = linear(shifted, shape)
        mapped_cols = invband[natural_cols]
        keep = mapped_cols >= 0
        rows_list.append(rows[keep])
        cols_list.append(mapped_cols[keep])
        vals_list.append(np.full(np.count_nonzero(keep), weight, dtype=float))

    rows = np.concatenate(rows_list) if rows_list else np.array([], dtype=np.int64)
    cols = np.concatenate(cols_list) if cols_list else np.array([], dtype=np.int64)
    vals = np.concatenate(vals_list) if vals_list else np.array([], dtype=float)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(band1.size, band2.size)).tocsr()


def _index_fns(ordering: str, dim: int):
    ordering = ordering.lower()
    if ordering == "ndgrid":
        return unravel_ndgrid_index, ndgrid_linear_index
    if ordering == "meshgrid":
        if dim not in (2, 3):
            raise ValueError("meshgrid ordering is only supported for 2D and 3D")
        return unravel_meshgrid_index, meshgrid_linear_index
    raise ValueError("ordering must be 'ndgrid' or 'meshgrid'")
