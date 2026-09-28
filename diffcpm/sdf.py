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


def _initial_guess(f, params, x, projection_steps, max_step=None):
    """A few gradient-flow projection steps, purely as a Newton starting point.

    The step ``-f grad f / |grad f|^2`` has length ``|f| / |grad f|``, which is
    exactly the quantity that blows up on a learned SDF: where the gradient decays
    away from the surface, a single step can travel arbitrarily far.  Backtracking
    inside the Newton solve cannot repair this, because it only guarantees the
    residual does not *increase* from wherever it starts -- and by then the
    starting point is already useless.  A test on ``tanh(5(|y|-1))``, whose zero
    set is the unit sphere, landed 1.4e9 away from it before Newton ran at all.

    So the projection steps are clipped to ``max_step`` as well.  Clipping only
    slows the approach to the surface; it cannot move the converged root, which is
    fixed by the first-order conditions Newton then enforces.
    """

    def step(y, _):
        val, g = jax.value_and_grad(f, argnums=1)(params, y)
        gn2 = jnp.sum(g * g) + 1e-30
        d = -val * g / gn2
        if max_step is not None:
            d = d * jnp.minimum(1.0, max_step / (jnp.linalg.norm(d) + 1e-300))
        d = jnp.where(jnp.isfinite(d), d, 0.0)
        return y + d, None

    y, _ = jax.lax.scan(step, x, None, length=projection_steps)
    g = jax.grad(f, argnums=1)(params, y)
    mu = jnp.sum((y - x) * g) / (jnp.sum(g * g) + 1e-30)
    return jnp.concatenate([y, jnp.atleast_1d(mu)])


def _newton(res, z0, iters, damping, n_backtrack=6, max_step=None):
    """Damped Gauss-Newton with monotone backtracking.

    A plain damped step is not enough on a *learned* SDF.  Where the network's
    gradient is small the Jacobian is nearly singular, the step is enormous, and
    the iteration leaves the region where anything is meaningful -- on a SIREN
    fitted to the Stanford bunny this produced closest points thousands of units
    away, with the residual stuck near 1.

    So each step is accepted only if it reduces ||F||.  A geometric ladder of
    step lengths is evaluated (1, 1/2, 1/4, ...), the best is taken, and if none
    improves on the current point the iterate is left where it is.  That makes
    the residual non-increasing by construction, which is what turns divergence
    into a clean per-point failure the caller can detect and exclude.  The ladder
    is evaluated unconditionally rather than in a loop with an early exit, so it
    stays a fixed-shape computation that jit and vmap can handle.

    ``max_step`` additionally clips the step length, which bounds how far a
    single iteration can travel regardless of the Jacobian.
    """
    eye = jnp.eye(z0.shape[0])
    alphas = 0.5 ** jnp.arange(n_backtrack)

    def step(z, _):
        r = res(z)
        f0 = jnp.sum(r * r)
        J = jax.jacobian(res)(z)
        JtJ = J.T @ J
        # scale the damping to the problem so it means the same thing on any SDF
        lam = damping * (jnp.trace(JtJ) / z.shape[0] + 1e-300)
        dz = jnp.linalg.solve(JtJ + lam * eye, J.T @ r)
        if max_step is not None:
            nrm = jnp.linalg.norm(dz)
            dz = dz * jnp.minimum(1.0, max_step / (nrm + 1e-300))
        dz = jnp.where(jnp.isfinite(dz), dz, 0.0)

        cands = z - alphas[:, None] * dz
        fs = jax.vmap(lambda c: jnp.sum(res(c) ** 2))(cands)
        fs = jnp.where(jnp.isfinite(fs), fs, jnp.inf)
        i = jnp.argmin(fs)
        return jnp.where(fs[i] < f0, cands[i], z), None

    z, _ = jax.lax.scan(step, z0, None, length=iters)
    return z


def cp_level_set_single(f, params, x, *, projection_steps=6, newton_iters=12,
                        damping=1e-10, max_step=None, n_backtrack=6):
    """Closest point on ``{f(params, .) = 0}`` to the single point ``x``.

    Gradients w.r.t. both ``params`` and ``x`` come from the IFT at the root.
    On a learned SDF, set ``max_step`` to a length scale of the problem (a few
    grid cells, say) -- see :func:`_newton` for why.
    """
    res = lambda z: _residual(f, params, x, z)

    def solve(res_fn, z0):
        return _newton(res_fn, z0, newton_iters, damping,
                       n_backtrack=n_backtrack, max_step=max_step)

    def tangent_solve(g_lin, y):
        J = jax.jacobian(g_lin)(jnp.zeros_like(y))
        return jnp.linalg.solve(J, y)

    z0 = _initial_guess(f, params, x, projection_steps, max_step=max_step)
    z = jax.lax.custom_root(res, z0, solve, tangent_solve)
    return z[: x.shape[0]]


def cp_level_set(f, params, xs, **kwargs):
    """Vectorized :func:`cp_level_set_single` over ``xs`` of shape (m, dim)."""
    fn = lambda x: cp_level_set_single(f, params, x, **kwargs)
    return jax.vmap(fn)(xs)


def level_set_residuals(f, params, xs, cps):
    """Per-point closest point residuals: ``(|f(cp)|, |sin angle|)``.

    Per-point rather than reduced, because on real geometry a few points fail
    while the rest are at machine precision, and a maximum alone cannot tell
    those two situations apart.  Use these to mask the failures out and to report
    how many there were.
    """
    gs = jax.vmap(jax.grad(f, argnums=1), in_axes=(None, 0))(params, cps)
    fv = jax.vmap(f, in_axes=(None, 0))(params, cps)
    d = cps - xs
    dn = jnp.linalg.norm(d, axis=1) + 1e-30
    ghat = gs / (jnp.linalg.norm(gs, axis=1, keepdims=True) + 1e-30)
    perp = d - jnp.sum(d * ghat, axis=1, keepdims=True) * ghat
    return jnp.abs(fv), jnp.linalg.norm(perp, axis=1) / dn


def level_set_residual_norm(f, params, xs, cps):
    """Diagnostics for a batch of computed closest points.

    Returns ``(max |f(cp)|, max |sin angle between (cp-x) and grad f(cp)|)``.
    Both should be at machine-precision level if Newton converged.
    """
    # The sine of the angle comes from the component of d perpendicular to
    # grad f.  Going via sqrt(1 - cos^2) instead loses half the digits to
    # cancellation when the angle is small, which is exactly the regime we are
    # trying to measure, and puts a spurious floor of ~1e-8 on the diagnostic.
    fv, sin = level_set_residuals(f, params, xs, cps)
    return float(jnp.max(fv)), float(jnp.max(sin))


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
         callback=None, key=None):
    """Minimal Adam, so the package depends only on jax/numpy.

    The whole loop is put inside a single ``lax.scan`` and jitted, which matters:
    run step by step from Python, a few thousand steps of a small network spend
    almost all their time in dispatch rather than arithmetic.  Passing a
    ``callback`` forces the Python loop, since a callback cannot be traced --
    use it for monitoring, not for fitting.

    Pass ``key`` to train on minibatches: ``loss_and_grad`` is then called as
    ``loss_and_grad(params, subkey)`` with a fresh subkey each step, so the loss
    can draw its own batch.  Full-batch training on a large 3D sample set is
    slower than it needs to be, and on CPU it is the difference between minutes
    and hours.

    Returns ``(params, final_loss)``.
    """
    tree_map = jax.tree_util.tree_map
    m0 = tree_map(jnp.zeros_like, params)
    v0 = tree_map(jnp.zeros_like, params)
    minibatch = key is not None

    def step(carry, t):
        if minibatch:
            params, m, v, k = carry
            k, sub = jax.random.split(k)
            loss, g = loss_and_grad(params, sub)
        else:
            params, m, v = carry
            loss, g = loss_and_grad(params)
        m = tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t
        params = tree_map(
            lambda p_, m_, v_: p_ - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps),
            params, m, v,
        )
        return ((params, m, v, k) if minibatch else (params, m, v)), loss

    init = (params, m0, v0, key) if minibatch else (params, m0, v0)

    if callback is None:
        ts = jnp.arange(1, steps + 1, dtype=jnp.float64 if jax.config.read(
            "jax_enable_x64") else jnp.float32)
        carry, losses = jax.jit(lambda c: jax.lax.scan(step, c, ts))(init)
        return carry[0], float(losses[-1])

    carry = init
    jstep = jax.jit(step)
    loss = None
    for t in range(1, steps + 1):
        carry, loss = jstep(carry, float(t))
        callback(t, loss, carry[0])
    return carry[0], float(loss)
