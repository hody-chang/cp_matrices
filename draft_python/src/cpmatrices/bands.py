"""Band and index helpers for Closest Point Method matrices."""

from __future__ import annotations

import numpy as np


def make_invbandmap(size: int, band) -> np.ndarray:
    """Return a dense natural-index to band-index map.

    Parameters
    ----------
    size:
        Total number of points in the underlying tensor-product grid.
    band:
        0-based natural linear indices included in the band.

    Returns
    -------
    numpy.ndarray
        Array of length `size`. Entries outside the band are `-1`; entries
        inside the band contain the corresponding column index in `band`.
    """

    band = np.asarray(band, dtype=np.int64).ravel()
    if np.any(band < 0) or np.any(band >= size):
        raise ValueError("band contains indices outside the grid")

    inv = np.full(int(size), -1, dtype=np.int64)
    inv[band] = np.arange(band.size, dtype=np.int64)
    return inv


def rm_bandwidth(dim: int, degree: int = 3, fd_radius: float = 1.0, safety: float = 1.0001) -> float:
    """Ruuth-Merriman closest-point computational bandwidth."""

    return float(
        safety
        * np.sqrt((dim - 1) * ((degree + 1) / 2) ** 2 + (fd_radius + (degree + 1) / 2) ** 2)
    )
