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
    rng = np.random.default_rng(1)
    v = rng.standard_normal((NSENSOR, 3))
    sensors = jnp.asarray(v / np.linalg.norm(v, axis=1, keepdims=True))
    assert observer.violations(sensors) == 0

    method = "gmres" if grid.n > 2500 else "dense"

    def forward(c):
        a = jnp.exp(basis @ c)
        op = build_operator(grid, cp, alpha=a, c=-1.0, pattern=pattern)
        return solve_operator(op, -f, method=method, tol=1e-12, maxiter=800,
                              restart=200)

    u_true = forward(jnp.asarray(C_TRUE))
    clean = observer(u_true, sensors)
    data = clean + NOISE * jnp.asarray(rng.standard_normal(NSENSOR))
    print("solver %s, sensors %d, noise %.1e, signal rms %.4f"
          % (method, NSENSOR, NOISE, float(jnp.sqrt(jnp.mean(clean**2)))))
    a_true = np.exp(np.asarray(basis @ jnp.asarray(C_TRUE)))
    print("true diffusivity range [%.3f, %.3f]" % (a_true.min(), a_true.max()))

    def objective(c, reg):
        pred = observer(forward(c), sensors)
        return misfit(pred, data, sigma=NOISE) + 0.5 * reg * jnp.sum(c**2)

    print("\ngradient check (reg = 1e-2), 4 random directions in R^9:")
    for ana, fd, rel, e in check_gradient(lambda c: objective(c, 1e-2),
                                          jnp.zeros(9), seed=0):
        print("   analytic %+.10e   fd %+.10e   rel %.2e  (eps %.0e)"
              % (ana, fd, rel, e))

    print("\nrecovery")
    print("   %-8s %-8s %-11s %-11s %s"
          % ("reg", "iters", "J", "||dc||", "max |a_hat - a| / max a"))
    for reg in (1e-3, 1e-2, 1e-1):
        t0 = time.time()
        c_hat, res = lbfgs(lambda c: objective(c, reg), jnp.zeros(9), maxiter=300)
        a_hat = np.exp(np.asarray(basis @ c_hat))
        print("   %-8.0e %-8d %-11.4e %-11.4e %.4f     (%.0fs)"
              % (reg, res.nit, res.fun, np.linalg.norm(np.asarray(c_hat) - C_TRUE),
                 np.abs(a_hat - a_true).max() / a_true.max(), time.time() - t0))

    c_hat, _ = lbfgs(lambda c: objective(c, 1e-2), jnp.zeros(9), maxiter=300)
    print("\n   coefficient-by-coefficient (reg = 1e-2)")
    print("   true      %s" % " ".join("%+6.3f" % v for v in C_TRUE))
    print("   recovered %s" % " ".join("%+6.3f" % v for v in np.asarray(c_hat)))

    print("\nThe degree-0 coefficient is the one that is well determined; the")
    print("higher harmonics are recovered less accurately because a smooth")
    print("diffusivity influences a smooth solution only weakly at high degree.")
    print("That is the identifiability structure of the problem, and it is why")
    print("the regularization weight cannot simply be set to zero here.")


if __name__ == "__main__":
    main()
