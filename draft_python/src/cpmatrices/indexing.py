"""Linear-index helpers matching MATLAB matrix-builder conventions."""

from __future__ import annotations

import numpy as np


def meshgrid_linear_index(indices, shape) -> np.ndarray:
    """Return 0-based linear indices for MATLAB `meshgrid` ordering.

    For 2D this matches MATLAB linear indexing of arrays created by
    `[xx, yy] = meshgrid(x, y)`, where the y-index varies fastest.
    """

    shape = tuple(int(n) for n in shape)
    indices = [np.asarray(i, dtype=np.int64) for i in indices]

    if len(shape) == 2:
        ix, iy = indices
        nx, ny = shape
        _check_bounds((ix, iy), shape)
        return ix * ny + iy

    if len(shape) == 3:
        ix, iy, iz = indices
        nx, ny, nz = shape
        _check_bounds((ix, iy, iz), shape)
        return iz * (nx * ny) + ix * ny + iy

    raise ValueError("meshgrid ordering is implemented only for 2D and 3D")


def ndgrid_linear_index(indices, shape) -> np.ndarray:
    """Return 0-based linear indices for MATLAB `ndgrid`/column-major ordering."""

    shape = tuple(int(n) for n in shape)
    indices = [np.asarray(i, dtype=np.int64) for i in indices]
    _check_bounds(indices, shape)
    return np.ravel_multi_index(tuple(indices), shape, order="F")


def unravel_meshgrid_index(linear, shape):
    """Inverse of `meshgrid_linear_index` for 2D and 3D."""

    linear = np.asarray(linear, dtype=np.int64)
    shape = tuple(int(n) for n in shape)

    if len(shape) == 2:
        nx, ny = shape
        ix = linear // ny
        iy = linear % ny
        return ix, iy

    if len(shape) == 3:
        nx, ny, nz = shape
        iz = linear // (nx * ny)
        rem = linear % (nx * ny)
        ix = rem // ny
        iy = rem % ny
        return ix, iy, iz

    raise ValueError("meshgrid ordering is implemented only for 2D and 3D")


def unravel_ndgrid_index(linear, shape):
    """Inverse of `ndgrid_linear_index`."""

    return np.unravel_index(np.asarray(linear, dtype=np.int64), tuple(shape), order="F")


def _check_bounds(indices, shape) -> None:
    for axis, (idx, n) in enumerate(zip(indices, shape)):
        if np.any(idx < 0) or np.any(idx >= n):
            raise ValueError(f"index outside grid along axis {axis}")
