"""Verification tests for the differentiable CPM layer.

Run directly (``python diffcpm/tests/test_diffcpm.py``) or under pytest.

The tests are organised by what could actually be wrong:

  operators   does the forward CPM solve converge at the expected rate
  adjoint     does custom_linear_solve reproduce the hand-written adjoint
  parameter   is dJ/dtheta right (finite differences)
  geometry    is dJ/d(shape) right, for closed-form and for learned surfaces
  continuous  does the discrete shape derivative converge to the continuous one
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from diffcpm.grid import band_from_cp, make_grid1d
from diffcpm.interp import InterpPattern, apply_sparse, apply_sparse_T
from diffcpm.inverse import Observer, check_gradient, misfit
from diffcpm.operators import build_operator, laplacian_pattern
from diffcpm.sdf import cp_level_set, level_set_residual_norm, siren_apply, siren_init
from diffcpm.solve import linear_solve, solve_operator
from diffcpm.surfaces import (ellipsoid_cp, ellipsoid_level_set, sphere_cp,
                              sphere_level_set, torus_cp, torus_level_set)

TOL_FD = 1e-7    # parameter gradients: plain smooth objectives
TOL_GEOM = 1e-6  # geometry gradients: best step over the sweep, see check_gradient


# ---------------------------------------------------------------------------
# fixtures, built by hand to keep the tests dependency-free
# ---------------------------------------------------------------------------


def circle_setup(dx=0.1, R=1.0, p=3, half=1.6):
    x1d = [make_grid1d(-half, half, dx)] * 2
    cpn = lambda pts: np.asarray(sphere_cp(R, jnp.asarray(pts)))
    grid = band_from_cp(x1d, cpn, p=p, stenrad=1)
    return grid


def circle_problem(grid, R, k=3, alpha=-1.0, c=1.0, pattern=None):
    """(-Delta_s + 1) u = cos(k theta), whose exact solution is known."""
    cp = sphere_cp(R, jnp.asarray(grid.xg))
    op = build_operator(grid, cp, alpha=alpha, c=c, pattern=pattern)
    th = jnp.arctan2(cp[:, 1], cp[:, 0])
    b = jnp.cos(k * th)
    return op, b, th


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------


def test_implicit_cp_matches_closed_form():
    """The Newton/IFT closest point solver agrees with the closed forms.

    Sphere and torus have closed-form closest point maps, so the generic
    level-set solver -- the code path a learned SDF uses -- can be checked
    against them directly rather than only through a downstream objective.  The
    ellipsoid is included with a deliberately non-distance level-set function
    (|grad f| != 1) to confirm the first-order conditions, not the projection
    iteration, are what is being solved.
    """
    xs = jax.random.normal(jax.random.PRNGKey(0), (60, 3)) * 0.7 \
        + jnp.array([0.0, 0.0, 1.2])
    got = cp_level_set(sphere_level_set, 1.3, xs)
    assert float(jnp.max(jnp.abs(got - sphere_cp(1.3, xs)))) < 1e-14
    fmax, sinmax = level_set_residual_norm(sphere_level_set, 1.3, xs, got)
    assert fmax < 1e-14 and sinmax < 1e-12, (fmax, sinmax)

    xt = jax.random.normal(jax.random.PRNGKey(1), (40, 3)) * 0.5 \
        + jnp.array([1.0, 0.0, 0.0])
    Rr = jnp.array([1.0, 0.35])
    got = cp_level_set(torus_level_set, Rr, xt)
    assert float(jnp.max(jnp.abs(got - torus_cp(Rr, xt)))) < 1e-13

    axes = jnp.array([1.4, 1.0, 0.7])
    got = ellipsoid_cp(axes, xs)
    fmax, sinmax = level_set_residual_norm(ellipsoid_level_set, axes, xs, got)
    assert fmax < 1e-14 and sinmax < 1e-12, (fmax, sinmax)


def test_interp_reproduces_polynomials():
    """E applied to a polynomial of degree <= p is exact at the closest points."""
    grid = circle_setup(dx=0.1, p=3)
    pat = InterpPattern(grid)
    cp = sphere_cp(1.0, jnp.asarray(grid.xg))
    assert pat.violations(cp) == 0
    xg = jnp.asarray(grid.xg)
    for powers in [(0, 0), (1, 0), (0, 1), (2, 1), (3, 0), (1, 2)]:
        f = xg[:, 0] ** powers[0] * xg[:, 1] ** powers[1]
        cols, vals = pat.weights(cp)
        got = apply_sparse(cols, vals, f)
        want = cp[:, 0] ** powers[0] * cp[:, 1] ** powers[1]
        err = float(jnp.max(jnp.abs(got - want)))
        assert err < 1e-11, (powers, err)


def test_transpose_is_the_transpose():
    grid = circle_setup(dx=0.15)
    pat = InterpPattern(grid)
    cols, vals = pat.weights(sphere_cp(1.0, jnp.asarray(grid.xg)))
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (grid.n,))
    v = jax.random.normal(jax.random.PRNGKey(1), (grid.n,))
    lhs = float(jnp.dot(v, apply_sparse(cols, vals, u)))
    rhs = float(jnp.dot(u, apply_sparse_T(cols, vals, v, grid.n)))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), 1.0), (lhs, rhs)


def test_stabilizations_are_equivalent_and_necessary():
    """diagsplit == lapsharp exactly, and the unstabilized product is unstable.

    With gamma = -diag(L) the two stabilizations are algebraically the same
    matrix:  diag(L) u + (L - diag(L)) E u  ==  L E u - gamma (I - E) u.  Checking
    it on an *anisotropic* grid is the real test, since that is where a
    hard-coded gamma = 2*dim/dx^2 would differ.  Meanwhile the raw product L E
    has an eigenvalue with positive real part on the same grid, which is why
    neither stabilization is optional.
    """
    x1d = [make_grid1d(-1.6, 1.6, 0.1), make_grid1d(-1.6, 1.6, 0.08)]
    grid = band_from_cp(x1d, lambda p: np.asarray(sphere_cp(1.0, jnp.asarray(p))),
                        p=3, stenrad=1)
    cp = sphere_cp(1.0, jnp.asarray(grid.xg))
    dense = {}
    for stab in ("diagsplit", "lapsharp", "none"):
        op = build_operator(grid, cp, alpha=1.0, c=0.0, stabilization=stab)
        dense[stab] = np.asarray(op.to_dense())
    scale = np.abs(dense["diagsplit"]).max()
    assert np.abs(dense["diagsplit"] - dense["lapsharp"]).max() < 1e-10 * scale
    for stab in ("diagsplit", "lapsharp"):
        assert np.linalg.eigvals(dense[stab]).real.max() < 1e-8 * scale, stab
    assert np.linalg.eigvals(dense["none"]).real.max() > 1.0


def test_order4_laplacian():
    """Fourth-order differences need stenrad 2, and are more accurate."""
    x1d = [make_grid1d(-1.6, 1.6, 0.1)] * 2
    cpn = lambda pts: np.asarray(sphere_cp(1.0, jnp.asarray(pts)))
    grid2 = band_from_cp(x1d, cpn, p=3, stenrad=1)
    try:
        laplacian_pattern(grid2, order=4)
        raise AssertionError("order 4 on a stenrad=1 band should be rejected")
    except ValueError:
        pass

    k = 3
    errs = {}
    for order, stenrad in ((2, 1), (4, 2)):
        grid = band_from_cp(x1d, cpn, p=3, stenrad=stenrad)
        cp = sphere_cp(1.0, jnp.asarray(grid.xg))
        op = build_operator(grid, cp, alpha=-1.0, c=1.0, order=order)
        th = jnp.arctan2(cp[:, 1], cp[:, 0])
        u = solve_operator(op, jnp.cos(k * th), method="dense")
        uex = jnp.cos(k * th) / (1.0 + k**2)
        near = np.abs(np.linalg.norm(grid.xg, axis=1) - 1.0) <= 0.1
        errs[order] = float(jnp.max(jnp.abs(u[near] - uex[near])))
    assert errs[4] < errs[2], errs


def test_band_clearance_predicts_degradation():
    """The clearance diagnostic fires before interpolation does, and it is right.

    A band built for one shape does not stay valid as the surface moves, and the
    two ways it fails are not simultaneous.  InterpPattern.violations detects the
    late failure (a stencil leaving the band); BandedGrid.clearance detects the
    early one (the band's Dirichlet outer edge coming within the Ruuth-Merriman
    radius of the surface), which is where accuracy actually starts to go.  This
    test pins down that ordering and checks the diagnostic against the error it
    is supposed to predict.
    """
    dx = 0.1
    k = 3
    grid = circle_setup(dx=dx, R=1.0, half=1.8)
    pat = InterpPattern(grid)
    J = _shape_objective(grid, pat, lambda t, xs: sphere_cp(t[0], xs), m=256, k=k)
    gJ = jax.jit(jax.grad(lambda t: J(t)))

    def dist_fun(R):
        return lambda pts: np.abs(np.linalg.norm(pts, axis=1) - R)

    def deriv_error(R):
        g = float(gJ(jnp.array([R]))[0])
        return abs(g - continuous_shape_derivative(R, k))

    def value_error(R):
        q = 1.0 + k**2 / R**2
        return abs(float(J(jnp.array([R]))) - 0.5 * np.pi * R / q**2)

    # at the design radius: full clearance
    assert grid.uncovered_near_surface(dist_fun(1.0)) == 0
    assert grid.clearance(dist_fun(1.0)) >= 1.0
    d_ok, v_ok = deriv_error(1.0), value_error(1.0)

    # drifted far enough to lose clearance, while interpolation is still fine --
    # this is the ordering the diagnostic exists to expose
    R_bad = 1.1
    assert pat.violations(sphere_cp(R_bad, jnp.asarray(grid.xg))) == 0
    assert grid.uncovered_near_surface(dist_fun(R_bad)) > 0
    assert grid.clearance(dist_fun(R_bad)) < 1.0

    # and it is the *gradient* that pays for it, far more than the value: the
    # objective degrades by a modest factor while its shape derivative loses
    # more than two orders of magnitude.
    d_bad, v_bad = deriv_error(R_bad), value_error(R_bad)
    assert d_bad / d_ok > 100.0, (d_ok, d_bad)
    assert v_bad / v_ok < 20.0, (v_ok, v_bad)
    assert (d_bad / d_ok) > 10.0 * (v_bad / v_ok), (d_ok, d_bad, v_ok, v_bad)

    # widening the band restores the clearance
    wide = band_from_cp([make_grid1d(-1.8, 1.8, dx)] * 2,
                        lambda p: np.asarray(sphere_cp(1.0, jnp.asarray(p))),
                        p=3, stenrad=1, extra_bw=3.0)
    assert wide.clearance(dist_fun(R_bad)) >= 1.0


def test_forward_solve_second_order():
    """Near-surface error of the CPM solve halves twice per grid refinement."""
    errs = []
    for dx in (0.2, 0.1, 0.05):
        grid = circle_setup(dx=dx)
        op, b, th = circle_problem(grid, 1.0, k=3)
        u = solve_operator(op, b, method="dense")
        uex = jnp.cos(3 * th) / (1.0 + 9.0)
        d = np.abs(np.linalg.norm(grid.xg, axis=1) - 1.0)
        near = d <= dx
        errs.append(float(jnp.max(jnp.abs(u[near] - uex[near]))))
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert min(rates) > 1.6, (errs, rates)


# ---------------------------------------------------------------------------
# adjoint
# ---------------------------------------------------------------------------


def test_custom_linear_solve_matches_hand_adjoint():
    """dJ/dtheta from JAX equals lam^T (db/dtheta - (dA/dtheta) u), by hand.

    This is the test that says the "layer" is a correct adjoint implementation
    rather than merely a differentiable one: it compares against the textbook
    formula assembled explicitly, with no finite differences involved.
    """
    grid = circle_setup(dx=0.12)
    pat = InterpPattern(grid)
    key = jax.random.PRNGKey(3)
    theta0 = 0.3 * jax.random.normal(key, (grid.n,))
    sel = np.arange(0, grid.n, 7)
    data = jnp.zeros(len(sel))

    def residual_pieces(theta):
        """A(theta) and b(theta) for a reaction coefficient theta."""
        cp = sphere_cp(1.0, jnp.asarray(grid.xg))
        op = build_operator(grid, cp, alpha=-1.0, c=1.0 + theta, pattern=pat)
        b = jnp.cos(3 * jnp.arctan2(cp[:, 1], cp[:, 0])) * (1.0 + 0.5 * theta)
        return op, b

    def J(theta):
        op, b = residual_pieces(theta)
        u = solve_operator(op, b, method="dense")
        return misfit(u[sel], data)

    g_jax = jax.grad(J)(theta0)

    # --- the same gradient, assembled by hand -----------------------------
    op0, b0 = residual_pieces(theta0)
    A = op0.to_dense()
    u = jnp.linalg.solve(A, b0)
    dJdu = jnp.zeros(grid.n).at[sel].add(u[sel] - data)
    lam = jnp.linalg.solve(A.T, dJdu)

    def Au_minus_b(theta):
        op, b = residual_pieces(theta)
        return op.matvec(jax.lax.stop_gradient(u)) - b

    # dJ/dtheta = -lam^T d/dtheta (A u - b)   with u held fixed
    _, vjp = jax.vjp(Au_minus_b, theta0)
    (g_hand,) = vjp(-lam)

    err = float(jnp.max(jnp.abs(g_jax - g_hand)) / jnp.max(jnp.abs(g_hand)))
    assert err < 1e-11, err


def test_gmres_and_dense_agree_including_gradients():
    grid = circle_setup(dx=0.12)
    pat = InterpPattern(grid)
    sel = np.arange(0, grid.n, 5)

    def J(theta, method):
        cp = sphere_cp(1.0, jnp.asarray(grid.xg))
        op = build_operator(grid, cp, alpha=-1.0, c=1.0 + theta, pattern=pat)
        b = jnp.cos(2 * jnp.arctan2(cp[:, 1], cp[:, 0]))
        u = solve_operator(op, b, method=method, tol=1e-13, maxiter=400, restart=120)
        return misfit(u[sel], jnp.zeros(len(sel)))

    theta0 = 0.2 * jax.random.normal(jax.random.PRNGKey(5), (grid.n,))
    jd, gd = jax.value_and_grad(lambda t: J(t, "dense"))(theta0)
    jg, gg = jax.value_and_grad(lambda t: J(t, "gmres"))(theta0)
    assert abs(jd - jg) / abs(jd) < 1e-8, (jd, jg)
    rel = float(jnp.linalg.norm(gd - gg) / jnp.linalg.norm(gd))
    assert rel < 1e-6, rel


# ---------------------------------------------------------------------------
# parameter gradients
# ---------------------------------------------------------------------------


def test_source_gradient_vs_finite_differences():
    grid = circle_setup(dx=0.12)
    pat = InterpPattern(grid)
    obs_th = jnp.linspace(0, 2 * np.pi, 24, endpoint=False)
    obs_pts = jnp.stack([jnp.cos(obs_th), jnp.sin(obs_th)], axis=1)
    observer = Observer(grid)
    assert observer.violations(obs_pts) == 0
    data = jnp.sin(3 * obs_th) * 0.05

    def J(coeffs):
        cp = sphere_cp(1.0, jnp.asarray(grid.xg))
        th = jnp.arctan2(cp[:, 1], cp[:, 0])
        modes = jnp.arange(1, coeffs.shape[0] + 1)
        b = jnp.sum(coeffs[None, :] * jnp.cos(modes[None, :] * th[:, None]), axis=1)
        op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pat)
        u = solve_operator(op, b, method="dense")
        return misfit(observer(u, obs_pts), data)

    c0 = jnp.array([0.4, -0.2, 0.1, 0.05])
    for ana, fd, rel, e in check_gradient(J, c0, seed=1):
        assert rel < TOL_FD, (ana, fd, rel, e)


def test_reaction_field_gradient_vs_finite_differences():
    grid = circle_setup(dx=0.15)
    pat = InterpPattern(grid)
    sel = np.arange(0, grid.n, 4)

    def J(theta):
        cp = sphere_cp(1.0, jnp.asarray(grid.xg))
        op = build_operator(grid, cp, alpha=-1.0, c=1.0 + theta, pattern=pat)
        b = jnp.cos(2 * jnp.arctan2(cp[:, 1], cp[:, 0]))
        u = solve_operator(op, b, method="dense")
        return misfit(u[sel], jnp.zeros(len(sel)))

    t0 = 0.3 * jax.random.normal(jax.random.PRNGKey(7), (grid.n,))
    for ana, fd, rel, e in check_gradient(J, t0, seed=2):
        assert rel < TOL_FD, (ana, fd, rel, e)


# ---------------------------------------------------------------------------
# geometry gradients
# ---------------------------------------------------------------------------


def _shape_objective(grid, pattern, cp_fun, m=64, k=3):
    """J(shape) = 1/2 * integral over Gamma of u^2 ds, with (-Delta_s + 1)u = cos(k th).

    The band is fixed; the surface moves inside it.  Both the operator and the
    quadrature points depend on the shape parameter.
    """
    th = jnp.linspace(0, 2 * np.pi, m, endpoint=False)
    w = 2 * np.pi / m

    def J(theta):
        R = theta[0]
        cp = cp_fun(theta, jnp.asarray(grid.xg))
        op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pattern)
        b = jnp.cos(k * jnp.arctan2(cp[:, 1], cp[:, 0]))
        u = solve_operator(op, b, method="dense")
        pts = R * jnp.stack([jnp.cos(th), jnp.sin(th)], axis=1)
        cols, vals = pattern.weights(pts)
        us = apply_sparse(cols, vals, u)
        return 0.5 * jnp.sum(w * us**2) * R  # ds = R dtheta

    return J


def test_circle_radius_gradient_vs_finite_differences():
    grid = circle_setup(dx=0.1, R=1.0, half=1.8)
    pat = InterpPattern(grid)
    cp_fun = lambda theta, xs: sphere_cp(theta[0], xs)
    J = _shape_objective(grid, pat, cp_fun)
    R0 = jnp.array([1.0])
    for ana, fd, rel, e in check_gradient(J, R0, directions=jnp.array([[1.0]])):
        assert rel < TOL_GEOM, (ana, fd, rel, e)


def test_ellipse_axes_gradient_vs_finite_differences():
    """Shape gradient through the implicit closest point solve (IFT path)."""
    axes0 = jnp.array([1.15, 0.9])
    cpn = lambda pts: np.asarray(ellipsoid_cp(axes0, jnp.asarray(pts)))
    x1d = [make_grid1d(-1.9, 1.9, 0.12)] * 2
    grid = band_from_cp(x1d, cpn, p=3, stenrad=1, extra_bw=1.0)
    pat = InterpPattern(grid)
    sel = np.arange(0, grid.n, 5)

    def J(axes):
        cp = ellipsoid_cp(axes, jnp.asarray(grid.xg))
        op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pat)
        b = jnp.cos(2 * jnp.arctan2(cp[:, 1], cp[:, 0]))
        u = solve_operator(op, b, method="dense")
        return misfit(u[sel], jnp.zeros(len(sel)))

    for ana, fd, rel, e in check_gradient(J, axes0, seed=3):
        assert rel < TOL_GEOM, (ana, fd, rel, e)


def test_siren_weight_gradient_vs_finite_differences():
    """dJ/dphi for a learned SDF, checked in random weight-space directions.

    The network is only lightly fitted -- what matters is that the gradient of
    the PDE-constrained objective with respect to the *network weights* is
    correct, not that the network is a good sphere.
    """
    key = jax.random.PRNGKey(11)
    params = siren_init(key, 2, (16, 16), w0_first=4.0, w0=4.0)

    # nudge the net towards a circle of radius 1 so closest points are sane
    xs = jax.random.uniform(jax.random.PRNGKey(12), (2048, 2), minval=-1.7, maxval=1.7)
    target = jnp.linalg.norm(xs, axis=1) - 1.0

    def loss(p):
        pred = jax.vmap(siren_apply, in_axes=(None, 0))(p, xs)
        return jnp.mean((pred - target) ** 2)

    from diffcpm.sdf import adam
    params, final = adam(jax.value_and_grad(loss), params, 400, lr=5e-3)
    assert final < 2e-2, final

    cpn = lambda pts: np.asarray(cp_level_set(siren_apply, params, jnp.asarray(pts)))
    x1d = [make_grid1d(-1.8, 1.8, 0.15)] * 2
    grid = band_from_cp(x1d, cpn, p=3, stenrad=1, extra_bw=0.5)
    pat = InterpPattern(grid)
    cp0 = cp_level_set(siren_apply, params, jnp.asarray(grid.xg))
    fmax, sinmax = level_set_residual_norm(siren_apply, params, jnp.asarray(grid.xg), cp0)
    assert fmax < 1e-9 and sinmax < 1e-11, (fmax, sinmax)
    assert pat.violations(cp0) == 0

    sel = np.arange(0, grid.n, 6)

    def J(p):
        cp = cp_level_set(siren_apply, p, jnp.asarray(grid.xg))
        op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pat)
        b = jnp.cos(2 * jnp.arctan2(cp[:, 1], cp[:, 0]))
        u = solve_operator(op, b, method="dense")
        return misfit(u[sel], jnp.zeros(len(sel)))

    # only the network weights carry gradients; w0 scalars are structure
    layers = params["layers"]
    rest = {"w0_first": params["w0_first"], "w0": params["w0"]}
    Jw = lambda L: J({"layers": L, **rest})
    for ana, fd, rel, e in check_gradient(Jw, layers, seed=2, nprobe=3):
        assert rel < 1e-5, (ana, fd, rel, e)


# ---------------------------------------------------------------------------
# does the discrete shape derivative converge to the continuous one?
# ---------------------------------------------------------------------------


def continuous_shape_derivative(R, k):
    """Exact dJ/dR for J(R) = 1/2 int_Gamma u^2 ds on a circle of radius R.

    On the circle, Delta_s = R^{-2} d^2/dtheta^2, so (-Delta_s + 1) u = cos(k th)
    gives u = cos(k th) / (1 + k^2/R^2) and

        J(R) = 1/2 * pi * R / (1 + k^2/R^2)^2.
    """
    q = 1.0 + k**2 / R**2
    dq = -2.0 * k**2 / R**3
    return 0.5 * np.pi * (1.0 / q**2 + R * (-2.0) * dq / q**3)


def test_discrete_objective_converges_second_order():
    """J_h -> J at second order.  The *value* is the well behaved quantity."""
    k, R = 3, 1.0
    exact = 0.5 * np.pi * R / (1.0 + k**2 / R**2) ** 2
    errs = []
    for dx in (0.2, 0.1, 0.05, 0.025):
        grid = circle_setup(dx=dx, R=R, half=1.8)
        pat = InterpPattern(grid)
        cp_fun = lambda theta, xs: sphere_cp(theta[0], xs)
        J = _shape_objective(grid, pat, cp_fun, m=256, k=k)
        errs.append(abs(float(J(jnp.array([R]))) - exact))
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert min(rates) > 1.5, (errs, rates)


def test_discrete_shape_derivative_converges():
    """dJ_h/dR -> dJ/dR, but not monotonically and not at second order.

    This is a finding, not a slack tolerance.  The CPM discretization error
    depends on where the surface sits relative to the grid; differentiating in a
    shape direction differentiates that dependence too, so the error in the
    discrete shape derivative oscillates with R/h even though the error in J_h
    itself is a clean O(h^2) (previous test).  Averaging over sub-cell grid
    offsets recovers the second-order rate --
    examples_diffcpm/ex3_shape_derivative_convergence.py runs that experiment.

    What is asserted here is what holds pointwise: the error is small and falls
    by more than an order of magnitude over a 16x refinement, i.e. roughly first
    order on average.
    """
    k, R = 3, 1.0
    exact = continuous_shape_derivative(R, k)
    errs = []
    for dx in (0.2, 0.1, 0.05, 0.025):
        grid = circle_setup(dx=dx, R=R, half=1.8)
        pat = InterpPattern(grid)
        cp_fun = lambda theta, xs: sphere_cp(theta[0], xs)
        J = _shape_objective(grid, pat, cp_fun, m=256, k=k)
        g = float(jax.grad(lambda t: J(t))(jnp.array([R]))[0])
        errs.append(abs(g - exact))
    assert max(errs) < 2e-3, errs
    avg_rate = np.log2(errs[0] / errs[-1]) / (len(errs) - 1)
    assert avg_rate > 0.9, (exact, errs, avg_rate)


# ---------------------------------------------------------------------------
# 3D sanity
# ---------------------------------------------------------------------------


def test_sphere_3d_forward():
    """Spherical harmonic on the unit sphere: -Delta_s z = 2z, so
    (-Delta_s + 1) z = 3z.  Solved matrix-free with GMRES."""
    dx = 0.2
    x1d = [make_grid1d(-1.6, 1.6, dx)] * 3
    cpn = lambda pts: np.asarray(sphere_cp(1.0, jnp.asarray(pts)))
    grid = band_from_cp(x1d, cpn, p=3, stenrad=1)
    pat = InterpPattern(grid)
    cp = sphere_cp(1.0, jnp.asarray(grid.xg))
    assert pat.violations(cp) == 0
    op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pat)
    u = solve_operator(op, 3.0 * cp[:, 2], method="gmres", tol=1e-12,
                       maxiter=600, restart=200)
    near = np.abs(np.linalg.norm(grid.xg, axis=1) - 1.0) <= dx
    err = float(jnp.max(jnp.abs(u[near] - cp[near, 2])))
    assert err < 5e-2, err


def test_sphere_3d_radius_gradient():
    """dJ/dR in 3D.  Coarse grid and a dense solve, deliberately.

    A geometry finite-difference check needs a step small enough that no closest
    point crosses a stencil cell (see check_gradient).  At such a step the
    difference quotient is only as accurate as the solve: an iterative solve
    stopped at 1e-13 puts a floor of ~1e-13/eps on the quotient, which swamps
    the signal.  So the gradient check uses dense LU on a grid small enough to
    afford it, and test_sphere_3d_forward exercises GMRES on a finer one.
    test_gmres_and_dense_agree_including_gradients ties the two together.
    """
    dx = 0.3
    x1d = [make_grid1d(-1.55, 1.55, dx)] * 3
    cpn = lambda pts: np.asarray(sphere_cp(1.0, jnp.asarray(pts)))
    grid = band_from_cp(x1d, cpn, p=3, stenrad=1)
    pat = InterpPattern(grid)
    assert pat.violations(sphere_cp(1.0, jnp.asarray(grid.xg))) == 0
    sel = np.arange(0, grid.n, 13)

    def J(Rv):
        c = sphere_cp(Rv[0], jnp.asarray(grid.xg))
        o = build_operator(grid, c, alpha=-1.0, c=1.0, pattern=pat)
        uu = solve_operator(o, 3.0 * c[:, 2], method="dense")
        return misfit(uu[sel], jnp.zeros(len(sel)))

    for ana, fd, rel, e in check_gradient(J, jnp.array([1.0]),
                                          directions=jnp.array([[1.0]])):
        assert rel < TOL_GEOM, (ana, fd, rel, e)


def test_torus_gradient():
    """dJ/d(R, r) for a torus, dense solve -- see test_sphere_3d_radius_gradient."""
    dx = 0.25
    Rr0 = jnp.array([1.0, 0.4])
    x1d = [make_grid1d(-1.9, 1.9, dx)] * 2 + [make_grid1d(-1.0, 1.0, dx)]
    cpn = lambda pts: np.asarray(torus_cp(Rr0, jnp.asarray(pts)))
    grid = band_from_cp(x1d, cpn, p=3, stenrad=1)
    pat = InterpPattern(grid)
    assert pat.violations(torus_cp(Rr0, jnp.asarray(grid.xg))) == 0
    sel = np.arange(0, grid.n, 7)

    def J(Rr):
        cp = torus_cp(Rr, jnp.asarray(grid.xg))
        op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pat)
        b = cp[:, 2] + 0.5 * cp[:, 0]
        u = solve_operator(op, b, method="dense")
        return misfit(u[sel], jnp.zeros(len(sel)))

    for ana, fd, rel, e in check_gradient(J, Rr0, seed=9, nprobe=2):
        assert rel < TOL_GEOM, (ana, fd, rel, e)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    import time
    import traceback

    failures = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print("PASS  %-52s %6.1fs" % (t.__name__, time.time() - t0))
        except Exception:
            failures += 1
            print("FAIL  %-52s %6.1fs" % (t.__name__, time.time() - t0))
            traceback.print_exc()
    print("\n%d/%d passed" % (len(TESTS) - failures, len(TESTS)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
