"""Minimal closest-point surface functions."""

from __future__ import annotations

import numpy as np


def circle_closest_point(x, y, radius: float = 1.0, center=(0.0, 0.0)):
    """Closest point and signed distance to a circle."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    center = np.asarray(center, dtype=float)

    xs = x - center[0]
    ys = y - center[1]
    r = np.hypot(xs, ys)
    scale = np.divide(radius, r, out=np.zeros_like(r, dtype=float), where=r != 0)

    cpx = np.where(r == 0, radius, scale * xs)
    cpy = np.where(r == 0, 0.0, scale * ys)

    return cpx + center[0], cpy + center[1], r - radius


def sphere_closest_point(x, y, z, radius: float = 1.0, center=(0.0, 0.0, 0.0)):
    """Closest point and signed distance to a sphere."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    center = np.asarray(center, dtype=float)

    xs = x - center[0]
    ys = y - center[1]
    zs = z - center[2]
    r = np.sqrt(xs * xs + ys * ys + zs * zs)
    scale = np.divide(radius, r, out=np.zeros_like(r, dtype=float), where=r != 0)

    cpx = np.where(r == 0, radius, scale * xs)
    cpy = np.where(r == 0, 0.0, scale * ys)
    cpz = np.where(r == 0, 0.0, scale * zs)

    return cpx + center[0], cpy + center[1], cpz + center[2], r - radius
