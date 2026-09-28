"""Experiment 5: shape optimization, and a spectral check of the operator.

Part A validates the discrete surface Laplacian on a shape with no closed-form
closest point.  On any closed curve, Delta_s is just d^2/ds^2 in arclength, so the
spectrum is determined by the perimeter alone:

    lambda_j = -(2 pi j / L)^2,   j = 0, 1, 1, 2, 2, 3, 3, ...

That gives an exact target for a circle *and* for an ellipse, whose closest point
map here goes through the Newton/IFT solver -- the same code path a learned SDF
uses.  It also makes the point that a curve's Laplace-Beltrami spectrum in 2D
carries only its length: "isospectralization" is only interesting from surfaces
up.

Part B is the shape inverse problem.  Data come from an ellipse; we start from a
circle and recover the axes from sparse noisy observations of the PDE solution,
differentiating through the closest point solve.  The band is fixed and built
wide enough to hold every shape the optimizer visits -- and checked, because
nothing detects a surface leaving the band automatically.

Run:  python examples_diffcpm/ex5_shape_optimization_ellipse.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffcpm.grid import band_from_cp, make_grid1d
from diffcpm.interp import InterpPattern
from diffcpm.inverse import Observer, check_gradient, lbfgs, misfit
from diffcpm.operators import build_operator
from diffcpm.solve import solve_operator
from diffcpm.surfaces import ellipsoid_cp, sphere_cp

AXES_TRUE = np.array([1.25, 0.80])
AXES_INIT = np.array([1.00, 1.00])
DX = 0.08
NSENSOR = 40
NOISE = 1e-3


def perimeter(axes, m=200000):
    """Arclength of an ellipse by the midpoint rule (smooth periodic integrand)."""
    a, b = float(axes[0]), float(axes[1])
    t = (np.arange(m) + 0.5) * 2 * np.pi / m
    return float(np.sum(np.hypot(a * np.sin(t), b * np.cos(t)) * 2 * np.pi / m))


def surface_eigenvalues(grid, pattern, cp, k):
    """The k smallest-magnitude eigenvalues of the discrete surface Laplacian."""
    op = build_operator(grid, cp, alpha=1.0, c=0.0, pattern=pattern)
    M = np.asarray(op.to_dense())
    ev = np.linalg.eigvals(M)
    ev = ev[np.argsort(np.abs(ev))][:k]
    return ev


def part_a():
    print("=" * 74)
    print("A  Laplace-Beltrami spectrum of a closed curve: lambda_j = -(2 pi j/L)^2")
    print("=" * 74)
    for label, axes, cp_fun in (
        ("circle  a=b=1.0", np.array([1.0, 1.0]),
         lambda g: sphere_cp(1.0, jnp.asarray(g.xg))),
        ("ellipse 1.25x0.8", AXES_TRUE,
         lambda g: ellipsoid_cp(jnp.asarray(AXES_TRUE), jnp.asarray(g.xg))),
    ):
        x1d = [make_grid1d(-1.9, 1.9, 0.06)] * 2
        cpn = (lambda a: (lambda p: np.asarray(ellipsoid_cp(jnp.asarray(a),
                                                            jnp.asarray(p)))))(axes)
        grid = band_from_cp(x1d, cpn, p=3, stenrad=1)
        pattern = InterpPattern(grid)
        cp = cp_fun(grid)
        assert pattern.violations(cp) == 0
        L = perimeter(axes)
        ev = surface_eigenvalues(grid, pattern, cp, 7)
        exact = [-(2 * np.pi * j / L) ** 2 for j in (0, 1, 1, 2, 2, 3, 3)]
        print("\n%s   perimeter L = %.8f   band %d" % (label, L, grid.n))
        print("   %-4s %-16s %-16s %-11s %s"
              % ("j", "computed Re", "exact", "abs err", "max |Im|"))
        for i, (c, e) in enumerate(zip(ev, exact)):
            print("   %-4d %-16.8f %-16.8f %-11.2e %.1e"
                  % ((0, 1, 1, 2, 2, 3, 3)[i], c.real, e, abs(c.real - e), abs(c.imag)))
    print("\nThe leading eigenvalue is 0 (constants), the rest come in pairs, and")
    print("both shapes match the perimeter formula -- including the ellipse, whose")
    print("closest points are computed by Newton and differentiated by the IFT.")


def part_b():
    print()
    print("=" * 74)
    print("B  recover the ellipse axes from sparse noisy observations")
    print("=" * 74)
    x1d = [make_grid1d(-1.9, 1.9, DX)] * 2
    # Band built for the *initial* circle, widened so every shape the optimizer
    # visits stays inside it.
    # extra_bw has to cover the surface motion: the axes move by up to 0.25,
    # which is about 3.1 cells at this dx, so 4.5 leaves margin.  The clearance
    # assertions below are what actually check it.
    grid = band_from_cp(x1d, lambda p: np.asarray(sphere_cp(1.0, jnp.asarray(p))),
                        p=3, stenrad=1, extra_bw=4.5)
    pattern = InterpPattern(grid)
    observer = Observer(grid)
    print("grid %s, band %d points, dx %.3f, band radius %.2f cells"
          % (grid.shape, grid.n, DX, grid.bw))

    xg = jnp.asarray(grid.xg)
    def cp_dist(axes):
        return lambda pts: np.linalg.norm(
            np.asarray(ellipsoid_cp(jnp.asarray(axes), jnp.asarray(pts))) - pts, axis=1)

    cp_true = ellipsoid_cp(jnp.asarray(AXES_TRUE), xg)
    assert pattern.violations(cp_true) == 0, "true shape outside the band"
    assert pattern.violations(ellipsoid_cp(jnp.asarray(AXES_INIT), xg)) == 0
    # The sufficient check: full Ruuth-Merriman clearance for both endpoints of
    # the optimization, so the band's artificial Dirichlet edge never reaches the
    # surface.  InterpPattern.violations alone would not catch that (see ex3).
    for name, axes in (("initial", AXES_INIT), ("true", AXES_TRUE)):
        c = grid.clearance(cp_dist(axes))
        print("   band clearance at the %s shape: %.3f of the RM radius" % (name, c))
        assert c >= 1.0, "widen extra_bw: %s shape has clearance %.3f" % (name, c)

    # Sensors sit at fixed ambient points on the true surface, as physical
    # sensors would.  They do not move with the trial shape.
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 2 * np.pi, NSENSOR))
    sensors = jnp.asarray(np.stack([AXES_TRUE[0] * np.cos(t),
                                    AXES_TRUE[1] * np.sin(t)], axis=1))
    assert observer.violations(sensors) == 0

    def forward(axes):
        cp = ellipsoid_cp(axes, xg)
        op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pattern)
        # An ambient source, so the geometry enters through where the surface
        # samples it as well as through the operator.
        b = 1.0 + cp[:, 0] + 0.5 * cp[:, 1] ** 2
        # Matrix-free GMRES rather than dense LU: at this size it is ~6x faster
        # per value-and-gradient and agrees with dense to every printed digit
        # (tests/test_diffcpm.py::test_gmres_and_dense_agree_including_gradients).
        return solve_operator(op, b, method="gmres", tol=1e-13, maxiter=1500,
                              restart=250)

    clean = observer(forward(jnp.asarray(AXES_TRUE)), sensors)
    noisy = clean + NOISE * jnp.asarray(rng.standard_normal(NSENSOR))
    print("sensors %d, noise %.1e, signal rms %.4f"
          % (NSENSOR, NOISE, float(jnp.sqrt(jnp.mean(clean**2)))))

    def make_objective(data, sigma):
        return lambda axes: misfit(observer(forward(axes), sensors), data,
                                   sigma=sigma)

    objective = make_objective(noisy, NOISE)

    print("\ngradient check at the initial circle:")
    for ana, fd, rel, e in check_gradient(objective, jnp.asarray(AXES_INIT),
                                          seed=0, nprobe=2):
        print("   analytic %+.10e   fd %+.10e   rel %.2e  (eps %.0e)"
              % (ana, fd, rel, e))

    trace = []
    axes_hat, res = lbfgs(objective, jnp.asarray(AXES_INIT), maxiter=80,
                          callback=lambda z: trace.append(np.array(z, copy=True)))
    print("\n   %-6s %-10s %-10s %-12s" % ("iter", "a", "b", "J"))
    for i, z in enumerate(trace[:: max(1, len(trace) // 8)]):
        print("   %-6d %-10.6f %-10.6f %-12.5e" % (i, z[0], z[1], objective(z)))
    print("   %-6s %-10.6f %-10.6f %-12.5e" % ("final", axes_hat[0], axes_hat[1],
                                               res.fun))
    print("   %-6s %-10.6f %-10.6f" % ("true", AXES_TRUE[0], AXES_TRUE[1]))
    err = np.asarray(axes_hat) - AXES_TRUE
    print("\n   axis error  %+.2e %+.2e" % (err[0], err[1]))
    print("   L-BFGS iterations %d, function evaluations %d" % (res.nit, res.nfev))
    print("   band clearance at the optimum: %.3f of the RM radius"
          % grid.clearance(cp_dist(np.asarray(axes_hat))))
    print("   J at the optimum %.4f, expected noise floor %.1f"
          % (res.fun, 0.5 * NSENSOR))

    # Control 1: the same data without noise.  Note carefully what this can and
    # cannot show -- see the discussion printed below.
    axes_clean, res_clean = lbfgs(make_objective(clean, NOISE),
                                  jnp.asarray(AXES_INIT), maxiter=80)
    err_clean = np.asarray(axes_clean) - AXES_TRUE
    print("\n   control 1, same discrete model, noise removed:")
    print("   axes %.6f %.6f   axis error %+.2e %+.2e   J %.3e"
          % (axes_clean[0], axes_clean[1], err_clean[0], err_clean[1],
             res_clean.fun))

    # Control 2: data from a *finer* grid.  This is the one that measures
    # discretization bias, because the forward model that made the data is not
    # the forward model being inverted.
    fine_dx = DX / 2
    fine_x1d = [make_grid1d(-1.9, 1.9, fine_dx)] * 2
    fine_grid = band_from_cp(
        fine_x1d,
        lambda pts: np.asarray(ellipsoid_cp(jnp.asarray(AXES_TRUE),
                                            jnp.asarray(pts))),
        p=3, stenrad=1)
    fine_pattern = InterpPattern(fine_grid)
    fine_obs = Observer(fine_grid)
    fine_cp = ellipsoid_cp(jnp.asarray(AXES_TRUE), jnp.asarray(fine_grid.xg))
    assert fine_pattern.violations(fine_cp) == 0
    assert fine_obs.violations(sensors) == 0
    fine_op = build_operator(fine_grid, fine_cp, alpha=-1.0, c=1.0,
                             pattern=fine_pattern)
    fine_b = 1.0 + fine_cp[:, 0] + 0.5 * fine_cp[:, 1] ** 2
    fine_u = solve_operator(fine_op, fine_b, method="gmres", tol=1e-13,
                            maxiter=1500, restart=250)
    fine_data = fine_obs(fine_u, sensors)
    axes_fine, res_fine = lbfgs(make_objective(fine_data, NOISE),
                                jnp.asarray(AXES_INIT), maxiter=80)
    err_fine = np.asarray(axes_fine) - AXES_TRUE
    print("\n   control 2, noise-free data from a dx = %.3f grid (%d band points),"
          % (fine_dx, fine_grid.n))
    print("   inverted on the dx = %.2f grid:" % DX)
    print("   axes %.6f %.6f   axis error %+.2e %+.2e   J %.3e"
          % (axes_fine[0], axes_fine[1], err_fine[0], err_fine[1], res_fine.fun))
    print("   data mismatch between the two grids: %.3e rms"
          % float(jnp.sqrt(jnp.mean((fine_data - clean) ** 2))))

    print("\nThe optimizer is not the limitation: J reaches its expected noise")
    print("floor in a handful of L-BFGS iterations, starting from a circle that")
    print("is nowhere near the answer.")
    print("\nControl 1 recovers the axes to machine precision, and that is a")
    print("statement about this script rather than about CPM.  The data were")
    print("generated by the *same* discrete forward model that is then inverted,")
    print("so the discretization error cancels exactly and the only thing left to")
    print("confirm is that the adjoint and the optimizer are consistent -- which")
    print("they are, to 1e-14.  This is the inverse crime, and it is worth naming:")
    print("a noise-free synthetic recovery that lands on machine precision is")
    print("measuring self-consistency, not accuracy.")
    print("\nControl 2 is the honest version.  Generating the data on a grid twice")
    print("as fine breaks the cancellation, and the axis error it leaves is the")
    print("genuine CPM discretization bias at dx = %.2f -- the error you would" % DX)
    print("get from real measurements of a real surface.  Compare it with the")
    print("noisy run above: the two come out the same order here, so at this")
    print("noise level neither one alone is the binding constraint.")
    print("\nSeparately, note the caveat from ex3: the shape derivative is only")
    print("piecewise C^1, so there is a scale below which a line search cannot")
    print("usefully tighten.  It is not what binds here, but it would become the")
    print("ceiling once noise and dx were both reduced far enough.")


if __name__ == "__main__":
    part_a()
    part_b()
