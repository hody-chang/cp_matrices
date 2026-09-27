"""Experiment 2: recover a spatially varying diffusivity on a sphere.

Forward problem      a(x) Delta_s u - u = -f    on the unit sphere
Inverse problem      given u at a few noisy sensors, recover a(x)

Unlike experiment 1 this is nonlinear in the unknown: ``a`` multiplies the
operator rather than the right-hand side, so ``dA/dtheta`` is the term that
carries the gradient and the adjoint earns its keep.  Positivity of ``a`` is
imposed by writing ``a = exp(sum_j c_j P_j)`` with ``P_j`` the real spherical
harmonics up to degree 2, so the unknown is 9 coefficients.

The diffusivity is in non-conservative form ``a Delta_s u`` rather than flux form
``div_s(a grad_s u)``.  That is a deliberate simplification: the operator is then
``diag(a) M``, which needs nothing beyond what operators.py already builds.  Flux
form would need a variable-coefficient surface divergence, which CPM can do but
which is a separate piece of work.

Run:  python examples_diffcpm/ex2_diffusivity_recovery_sphere.py
"""

import os
import sys
import time

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
from diffcpm.surfaces import sphere_cp

DX = 0.2
NSENSOR = 60
NOISE = 2e-3
C_TRUE = np.array([0.15, 0.30, -0.20, 0.25, 0.10, -0.15, 0.05, 0.20, -0.10])


def harmonics(pts):
    """Real spherical harmonics up to degree 2, as polynomials, (m, 9)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    one = jnp.ones_like(x)
    return jnp.stack([one, x, y, z, x * y, y * z, z * x,
                      x**2 - y**2, 3 * z**2 - 1.0], axis=1)


def main():
    x1d = [make_grid1d(-1.55, 1.55, DX)] * 3
    grid = band_from_cp(x1d, lambda p: np.asarray(sphere_cp(1.0, jnp.asarray(p))),
                        p=3, stenrad=1)
    pattern = InterpPattern(grid)
    cp = sphere_cp(1.0, jnp.asarray(grid.xg))
    assert pattern.violations(cp) == 0
    print("grid %s, band %d points, dx %.2f" % (grid.shape, grid.n, DX))

    basis = harmonics(cp)  # (n, 9), evaluated at closest points -> normal-constant
    f = 1.0 + cp[:, 2] + 0.5 * cp[:, 0] * cp[:, 1]

    observer = Observer(grid)
    method = "gmres" if grid.n > 2500 else "dense"

    def make_sensors(nsensor, seed):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((nsensor, 3))
        pts = jnp.asarray(v / np.linalg.norm(v, axis=1, keepdims=True))
        assert observer.violations(pts) == 0
        return pts, rng

    def forward(c):
        a = jnp.exp(basis @ c)
        op = build_operator(grid, cp, alpha=a, c=-1.0, pattern=pattern)
        return solve_operator(op, -f, method=method, tol=1e-12, maxiter=800,
                              restart=200)

    a_true = np.exp(np.asarray(basis @ jnp.asarray(C_TRUE)))
    u_true = forward(jnp.asarray(C_TRUE))
    print("solver %s, noise %.1e" % (method, NOISE))
    print("true diffusivity range [%.3f, %.3f]" % (a_true.min(), a_true.max()))

    def make_problem(nsensor, seed=1):
        sensors, rng = make_sensors(nsensor, seed)
        clean = observer(u_true, sensors)
        data = clean + NOISE * jnp.asarray(rng.standard_normal(nsensor))

        def objective(c, reg):
            pred = observer(forward(c), sensors)
            return misfit(pred, data, sigma=NOISE) + 0.5 * reg * jnp.sum(c**2)

        return objective, float(jnp.sqrt(jnp.mean(clean**2)))

    objective, rms = make_problem(NSENSOR)
    print("signal rms %.4f with %d sensors" % (rms, NSENSOR))

    print("\ngradient check (reg = 1.0), 4 random directions in R^9:")
    for ana, fd, rel, e in check_gradient(lambda c: objective(c, 1.0),
                                          jnp.zeros(9), seed=0):
        print("   analytic %+.10e   fd %+.10e   rel %.2e  (eps %.0e)"
              % (ana, fd, rel, e))

    def run(objective, reg):
        t0 = time.time()
        c_hat, res = lbfgs(lambda c: objective(c, reg), jnp.zeros(9), maxiter=300)
        c_hat = np.asarray(c_hat)
        a_hat = np.exp(np.asarray(basis @ jnp.asarray(c_hat)))
        return (c_hat, res, np.linalg.norm(c_hat - C_TRUE),
                np.abs(a_hat - a_true).max() / a_true.max(), time.time() - t0)

    print("\nhow many sensors does it take?  (reg = 1.0)")
    print("   %-9s %-7s %-11s %-11s %-11s" %
          ("sensors", "iters", "J", "||dc||", "max da / max a"))
    best = None
    for nsensor in (8, 15, 30, 60):
        obj, _ = make_problem(nsensor)
        c_hat, res, dc, da, dt = run(obj, 1.0)
        print("   %-9d %-7d %-11.4e %-11.4e %-11.4f  (%.0fs)"
              % (nsensor, res.nit, res.fun, dc, da, dt))
        if nsensor == NSENSOR:
            best = c_hat

    print("\nregularization at %d sensors.  The misfit is weighted by" % NSENSOR)
    print("1/sigma^2 = %.0e, so a weight has to reach that order before it bites."
          % (1.0 / NOISE**2))
    print("   %-9s %-7s %-11s %-11s %-11s" %
          ("reg", "iters", "J", "||dc||", "max da / max a"))
    for reg in (1.0, 1e2, 1e4):
        c_hat, res, dc, da, dt = run(objective, reg)
        print("   %-9.0e %-7d %-11.4e %-11.4e %-11.4f  (%.0fs)"
              % (reg, res.nit, res.fun, dc, da, dt))

    print("\ncoefficients at %d sensors, reg = 1.0" % NSENSOR)
    names = ["1", "x", "y", "z", "xy", "yz", "zx", "x2-y2", "3z2-1"]
    print("   %-10s %s" % ("basis", " ".join("%7s" % n for n in names)))
    print("   %-10s %s" % ("true", " ".join("%+7.3f" % v for v in C_TRUE)))
    print("   %-10s %s" % ("recovered", " ".join("%+7.3f" % v for v in best)))
    print("   %-10s %s" % ("error", " ".join("%7.3f" % v
                                             for v in np.abs(best - C_TRUE))))
    deg = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])
    print("\n   rms coefficient error by degree: " + "  ".join(
        "deg %d: %.4f" % (d, np.sqrt(np.mean((best - C_TRUE)[deg == d] ** 2)))
        for d in (0, 1, 2)))

    print("\nWhat the sensor sweep shows is where the ill-posedness actually")
    print("lives.  With 9 unknowns and 60 noisy sensors the problem is well")
    print("determined and the regularization weight is inert until it is large")
    print("enough to start biasing the answer -- which is why the reg table is")
    print("flat and then degrades rather than showing an optimum.  Cut the")
    print("sensor count towards the number of unknowns and the recovery falls")
    print("apart instead; that is the regime where regularization and an")
    print("identifiability analysis are the substance of the problem rather")
    print("than a formality.")


if __name__ == "__main__":
    main()
