"""Closest points on an implicit surface, differentiated by the IFT.

For a level set surface ``{f(y; phi) = 0}`` the closest point to x is
characterised by the first-order conditions

    y - x = mu * grad f(y),      f(y) = 0                              (*)

i.e. the displacement is normal to the surface and the foot lies on it.  We
solve (*) by Newton on the (d+1) unknowns ``z = (y, mu)`` and differentiate
with ``jax.lax.custom_root``, which applies the implicit function theorem at the
converged root.  So the gradient never sees the Newton iterations: it is exact
for the converged closest point, costs one small linear solve per point, and
uses no memory proportional to the iteration count.

Two things worth spelling out, because both are easy to get wrong:

*   The commonly quoted projection ``cp(x) = x - f grad f / |grad f|^2``,
    iterated, is *not* the closest point unless f is an exact signed distance
    function.  It follows the gradient flow of f from x and lands somewhere on
    the zero level set, generally not the nearest point.  For a learned SDF,
    where ``|grad f| != 1`` and the level sets are not parallel, the difference
    is a genuine O(SDF error) perturbation of the geometry.  We use it only as
    an initial guess and then enforce (*), which is the true closest point for
    any f whose zero level set is the surface -- exact or not.

*   Differentiating the Newton *iterates* (unrolling) gives a gradient of the
    approximation rather than an approximation of the gradient.  Those differ
    until the iteration is fully converged, and the discrepancy shows up as
    finite-difference checks that pass at loose tolerance and fail at tight
    tolerance.  ``custom_root`` avoids the question.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# closest point on a level set
# ---------------------------------------------------------------------------


def _residual(f, params, x, z):
    d = x.shape[0]
    y, mu = z[:d], z[d]
    g = jax.grad(f, argnums=1)(params, y)
    return jnp.concatenate([y - x - mu * g, jnp.atleast_1d(f(params, y))])


def _initial_guess(f, params, x, projection_steps):
    """A few gradient-flow projection steps, purely as a Newton starting point."""

    def step(y, _):
        val, g = jax.value_and_grad(f, argnums=1)(params, y)
        gn2 = jnp.sum(g * g) + 1e-30
        return y - val * g / gn2, None

    y, _ = jax.lax.scan(step, x, None, length=projection_steps)
    g = jax.grad(f, argnums=1)(params, y)
    mu = jnp.sum((y - x) * g) / (jnp.sum(g * g) + 1e-30)
    return jnp.concatenate([y, jnp.atleast_1d(mu)])


def _newton(res, z0, iters, damping):
    def step(z, _):
        r = res(z)
        J = jax.jacobian(res)(z)
        # Levenberg-style damping keeps the step finite if J is near-singular,
        # which happens when x sits on the medial axis (cp is not unique there).
        JtJ = J.T @ J + damping * jnp.eye(z.shape[0])
        dz = jnp.linalg.solve(JtJ, J.T @ r)
        return z - dz, None

    z, _ = jax.lax.scan(step, z0, None, length=iters)
    return z


def cp_level_set_single(f, params, x, *, projection_steps=6, newton_iters=12,
                        damping=1e-12):
    """Closest point on ``{f(params, .) = 0}`` to the single point ``x``.

    Gradients w.r.t. both ``params`` and ``x`` come from the IFT at the root.
    """
    res = lambda z: _residual(f, params, x, z)

    def solve(res_fn, z0):
        return _newton(res_fn, z0, newton_iters, damping)

    def tangent_solve(g_lin, y):
        J = jax.jacobian(g_lin)(jnp.zeros_like(y))
        return jnp.linalg.solve(J, y)

    z0 = _initial_guess(f, params, x, projection_steps)
    z = jax.lax.custom_root(res, z0, solve, tangent_solve)
    return z[: x.shape[0]]


def cp_level_set(f, params, xs, **kwargs):
    """Vectorized :func:`cp_level_set_single` over ``xs`` of shape (m, dim)."""
    fn = lambda x: cp_level_set_single(f, params, x, **kwargs)
    return jax.vmap(fn)(xs)


def level_set_residual_norm(f, params, xs, cps):
    """Diagnostics for a batch of computed closest points.

    Returns ``(max |f(cp)|, max |sin angle between (cp-x) and grad f(cp)|)``.
    Both should be at machine-precision level if Newton converged.
    """
    gs = jax.vmap(jax.grad(f, argnums=1), in_axes=(None, 0))(params, cps)
    fv = jax.vmap(f, in_axes=(None, 0))(params, cps)
    d = cps - xs
    dn = jnp.linalg.norm(d, axis=1) + 1e-30
    ghat = gs / (jnp.linalg.norm(gs, axis=1, keepdims=True) + 1e-30)
    # The sine of the angle straight from the component of d perpendicular to
    # grad f.  Going via sqrt(1 - cos^2) instead loses half the digits to
    # cancellation when the angle is small, which is exactly the regime we are
    # trying to measure, and puts a spurious floor of ~1e-8 on the diagnostic.
    perp = d - jnp.sum(d * ghat, axis=1, keepdims=True) * ghat
    sin = jnp.linalg.norm(perp, axis=1) / dn
    return float(jnp.max(jnp.abs(fv))), float(jnp.max(sin))


# ---------------------------------------------------------------------------
# a small SIREN, to stand in for DeepSDF / NeuS style reconstructions
# ---------------------------------------------------------------------------


def siren_init(key, dim, widths, w0_first=30.0, w0=30.0):
    """Sitzmann et al. initialization.  Returns ``(layers, w0_first, w0)``."""
    keys = jax.random.split(key, len(widths) + 1)
    sizes = [dim] + list(widths) + [1]
    layers = []
    for i, k in enumerate(keys):
        fan_in = sizes[i]
        if i == 0:
            bound = 1.0 / fan_in
        else:
            bound = np.sqrt(6.0 / fan_in) / w0
        W = jax.random.uniform(k, (sizes[i + 1], fan_in), minval=-bound, maxval=bound)
        b = jnp.zeros(sizes[i + 1])
        layers.append((W, b))
    return {"layers": layers, "w0_first": w0_first, "w0": w0}


def siren_apply(params, x):
    """Scalar SIREN evaluated at a single point ``x`` of shape (dim,)."""
    layers = params["layers"]
    h = jnp.sin(params["w0_first"] * (layers[0][0] @ x + layers[0][1]))
    for W, b in layers[1:-1]:
        h = jnp.sin(params["w0"] * (W @ h + b))
    W, b = layers[-1]
    return (W @ h + b)[0]


def adam(loss_and_grad, params, steps, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8,
         callback=None):
    """Minimal Adam, so the package depends only on jax/numpy."""
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    for t in range(1, steps + 1):
        loss, g = loss_and_grad(params)
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
        mh = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1**t), m)
        vh = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2**t), v)
        params = jax.tree_util.tree_map(
            lambda p_, m_, v_: p_ - lr * m_ / (jnp.sqrt(v_) + eps), params, mh, vh
        )
        if callback is not None:
            callback(t, loss, params)
    return params, loss
