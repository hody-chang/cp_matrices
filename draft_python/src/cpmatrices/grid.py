"""Small grid helpers for draft Closest Point Method examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorGrid:
    """Tensor-product embedding grid metadata."""

    axes: tuple[np.ndarray, ...]
    ordering: str = "meshgrid"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(float(axis[1] - axis[0]) for axis in self.axes)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))


def uniform_axis(start: float, stop: float, dx: float) -> np.ndarray:
    """Create an inclusive uniform axis similar to MATLAB `start:dx:stop`."""

    count = int(np.floor((stop - start) / dx + 0.5)) + 1
    return start + dx * np.arange(count, dtype=float)
