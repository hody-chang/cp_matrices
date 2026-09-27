"""Surfaces with differentiable shape parameters.

Each surface provides

  ``cp(theta, x)``  closest points, (m, dim) -> (m, dim), differentiable in the
                    shape parameters ``theta``
  ``level_set(theta, y)`` a scalar function whose zero set is the surface, for
                    use with :mod:`diffcpm.sdf`

Closed forms are given where they exist (circle, sphere, torus) because they
make finite-difference gradient checks unambiguous; the ellipse goes through the
generic implicit solver, which is the same code path a learned SDF uses.
"""

import jax
import jax.numpy as jnp

from .sdf import cp_level_set

_EPS = 1e-30


# ---------------------------------------------------------------------------
# circle / sphere: cp is closed form, so is its derivative in R
# ---------------------------------------------------------------------------


def sphere_cp(R, xs, center=None):
    """Closest point on a sphere (or circle in 2D) of radius ``R``."""
    xs = jnp.asarray(xs)
    c = jnp.zeros(xs.shape[1]) if center is None else jnp.asarray(center)
    d = xs - c
    r = jnp.linalg.norm(d, axis=1, keepdims=True)
    return c + R * d / jnp.sqrt(r**2 + _EPS)


def sphere_level_set(R, y, center=None):
    c = jnp.zeros(y.shape[0]) if center is None else jnp.asarray(center)
    return jnp.linalg.norm(y - c) - R


# ---------------------------------------------------------------------------
# torus in R^3, axis along z: major radius R, minor radius r
# ---------------------------------------------------------------------------


def torus_cp(Rr, xs):
    R, r = Rr[0], Rr[1]
    xs = jnp.asarray(xs)
    rho = jnp.sqrt(xs[:, 0] ** 2 + xs[:, 1] ** 2 + _EPS)
    # nearest point on the centre circle
    cc = jnp.stack([R * xs[:, 0] / rho, R * xs[:, 1] / rho, jnp.zeros_like(rho)], 1)
    d = xs - cc
    dn = jnp.linalg.norm(d, axis=1, keepdims=True)
    return cc + r * d / jnp.sqrt(dn**2 + _EPS)


def torus_level_set(Rr, y):
    R, r = Rr[0], Rr[1]
    rho = jnp.sqrt(y[0] ** 2 + y[1] ** 2 + _EPS)
    return jnp.sqrt((rho - R) ** 2 + y[2] ** 2 + _EPS) - r


# ---------------------------------------------------------------------------
# ellipse / ellipsoid, via the generic implicit closest point solver
# ---------------------------------------------------------------------------


def ellipsoid_level_set(axes, y):
    """Scaled implicit function.  Not a distance function -- deliberately.

    Using ``sum (y_i/a_i)^2 - 1`` rather than a true distance exercises the same
    code path as a learned SDF, where ``|grad f| != 1``.
    """
    return jnp.sum((y / axes) ** 2) - 1.0


def ellipsoid_cp(axes, xs, **kwargs):
    return cp_level_set(ellipsoid_level_set, axes, xs, **kwargs)


# ---------------------------------------------------------------------------
# helpers for building bands (numpy, non-differentiable, called once)
# ---------------------------------------------------------------------------


def numpy_cp(cp_fun, theta):
    """Wrap a differentiable cp function for use by :mod:`diffcpm.grid`."""
    import numpy as np

    def f(pts):
        return np.asarray(cp_fun(theta, jnp.asarray(pts)))

    return f
