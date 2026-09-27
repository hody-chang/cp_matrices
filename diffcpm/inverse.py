"""Observation operators, objectives and regularizers for CPM inverse problems.

Nothing here is deep; it exists so the examples read as

    J(theta) = || observe(solve(theta)) - data ||^2 + reg(theta)

with every piece differentiable and each piece testable on its own.
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from .interp import InterpPattern, apply_sparse


class Observer:
    """Interpolate a banded grid function to a fixed set of surface points.

    This is the CPM ``Eplot`` matrix.  When the sample points themselves move
    with the geometry (they lie on the surface, after all), pass them in on each
    call so their dependence on the shape parameters is differentiated too.
    """

    def __init__(self, grid, p=None):
        self.pattern = InterpPattern(grid, p)

    def __call__(self, u, points):
        cols, vals = self.pattern.weights(points)
        return apply_sparse(cols, vals, u)

    def violations(self, points):
        return self.pattern.violations(points)


def surface_quadrature_weights(theta_nodes):
    """Composite trapezoid weights for a closed periodic parameter grid.

    ``theta_nodes`` must be equispaced on [0, 2 pi) *without* the duplicate
    endpoint, in which case the trapezoid rule is just the uniform rule and is
    spectrally accurate for smooth periodic integrands.
    """
    m = len(theta_nodes)
    return np.full(m, 2.0 * np.pi / m)


def misfit(pred, data, sigma=1.0):
    """Gaussian data misfit, 1/2 || (pred - data)/sigma ||^2."""
    r = (pred - data) / sigma
    return 0.5 * jnp.sum(r * r)


def tikhonov(theta, weight=1.0):
    return 0.5 * weight * jnp.sum(theta * theta)


def graph_laplacian_reg(theta, neighbours, weight=1.0):
    """Smoothness penalty on a field given on surface sample points.

    ``neighbours`` is an (m, k) index array; the penalty is
    ``weight/2 * sum_i sum_j (theta_i - theta_{ij})^2``.  Used to regularize a
    spatially varying coefficient without committing to a basis.
    """
    d = theta[:, None] - theta[neighbours]
    return 0.5 * weight * jnp.sum(d * d)


def ring_neighbours(m):
    """(m, 2) neighbour indices for a periodic 1D ordering."""
    i = np.arange(m)
    return np.stack([(i - 1) % m, (i + 1) % m], axis=1)


DEFAULT_EPS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def check_gradient(fun, x, directions=None, seed=0, eps=DEFAULT_EPS, nprobe=4,
                   verbose=False):
    """Compare reverse-mode gradients against central finite differences.

    Returns a list of ``(analytic, best_fd, best_rel_err, best_eps)``, one per
    probe direction.  ``x`` may be any pytree.

    Why a *sweep* over step sizes rather than one "optimal" step.  For a smooth
    objective the textbook choice ``eps ~ macheps^(1/3)`` is right, and a single
    step suffices.  A CPM objective is not smooth in the *geometry*: when the
    surface moves, closest points cross interpolation-stencil cell boundaries,
    and the stencil switches.  With the base index and local coordinate computed
    consistently (see diffcpm/interp.py) the objective stays continuous across
    such a crossing, but its derivative has a small jump.  So J is continuous and
    piecewise C^1 in the shape parameters, with kinks at a density of roughly
    ``n_band / dx`` per unit of shape perturbation.

    A central difference straddling a kink measures an average slope, not the
    derivative -- and no amount of float64 precision fixes that.  The remedy is
    to shrink ``eps`` until the interval contains no crossing, at which point the
    agreement drops abruptly to rounding level.  That transition is visible in
    the sweep (errors plateau around 1e-5, then fall to 1e-10 at 1e-6), and is
    the honest empirical statement of the moving-band issue.  Taking the best
    step over a sweep is therefore the correct test, not a fudge; ``verbose``
    prints the whole sweep so the transition can be inspected.
    """
    flat, unravel = jax.flatten_util.ravel_pytree(x)
    n = flat.shape[0]
    g = jax.flatten_util.ravel_pytree(jax.grad(fun)(x))[0]

    if directions is None:
        key = jax.random.PRNGKey(seed)
        dirs = jax.random.normal(key, (min(nprobe, n), n))
        dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    else:
        dirs = jnp.asarray(directions)

    eps_list = (eps,) if np.isscalar(eps) else tuple(eps)

    out = []
    for d in dirs:
        ana = float(jnp.dot(g, d))
        best = None
        for e in eps_list:
            fp = fun(unravel(flat + e * d))
            fm = fun(unravel(flat - e * d))
            fd = float((fp - fm) / (2 * e))
            rel = abs(ana - fd) / max(abs(ana), abs(fd), 1e-30)
            if verbose:
                print("    eps %8.1e  fd %+.12e  rel %.3e" % (e, fd, rel))
            if best is None or rel < best[1]:
                best = (fd, rel, e)
        out.append((ana, best[0], best[1], best[2]))
    return out


def lbfgs(objective, x0, maxiter=200, tol=1e-12, callback=None, verbose=False):
    """Drive a JAX objective with scipy's L-BFGS-B.

    ``objective`` is any function of a pytree returning a scalar; gradients come
    from ``jax.value_and_grad``.  Returns ``(x_opt, result)``.  This exists to
    make the point that the CPM layer is an ordinary differentiable function:
    nothing about the optimizer needs to know it contains a PDE solve.
    """
    from scipy.optimize import minimize

    flat0, unravel = jax.flatten_util.ravel_pytree(x0)
    vg = jax.jit(jax.value_and_grad(objective))
    history = []

    def fg(z):
        val, grad = vg(unravel(jnp.asarray(z)))
        g = jax.flatten_util.ravel_pytree(grad)[0]
        history.append(float(val))
        if verbose and len(history) % 10 == 1:
            print("    iter %4d  J = %.8e" % (len(history), val))
        return float(val), np.asarray(g, dtype=np.float64)

    res = minimize(fg, np.asarray(flat0, dtype=np.float64), jac=True,
                   method="L-BFGS-B", callback=callback,
                   options={"maxiter": maxiter, "ftol": tol, "gtol": tol})
    return unravel(jnp.asarray(res.x)), res
