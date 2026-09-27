"""Experiment 1: recover a source on a circle from sparse noisy sensors.

Forward problem      (-Delta_s + 1) u = b(theta_coeffs)  on a circle of radius R
Inverse problem      given u at a handful of noisy sensors, recover the source

The source is expanded in a Fourier basis, so the map theta -> u is linear and
the objective is quadratic: any discrepancy between the recovered and the true
coefficients is then attributable to noise, sensor sparsity and regularization,
not to optimizer trouble.  That is the point of starting here.

Run:  python examples_diffcpm/ex1_source_recovery_circle.py
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
from diffcpm.surfaces import sphere_cp

R = 1.0
DX = 0.05
NMODES = 6
NSENSOR = 12
NOISE = 1e-3
THETA_TRUE = np.array([0.8, -0.5, 0.35, 0.0, -0.15, 0.05])


def setup():
    x1d = [make_grid1d(-1.6, 1.6, DX)] * 2
    grid = band_from_cp(x1d, lambda p: np.asarray(sphere_cp(R, jnp.asarray(p))),
                        p=3, stenrad=1)
    pattern = InterpPattern(grid)
    cp = sphere_cp(R, jnp.asarray(grid.xg))
    assert pattern.violations(cp) == 0
    th_band = jnp.arctan2(cp[:, 1], cp[:, 0])
    op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pattern)
    return grid, pattern, op, th_band


def source(theta, th_band):
    modes = jnp.arange(1, theta.shape[0] + 1)
    return jnp.sum(theta[None, :] * jnp.cos(modes[None, :] * th_band[:, None]), axis=1)


def main():
    grid, pattern, op, th_band = setup()
    print("grid %s, band %d points, dx %.3f" % (grid.shape, grid.n, DX))

    observer = Observer(grid)
    rng = np.random.default_rng(0)
    sensor_th = jnp.asarray(np.sort(rng.uniform(0, 2 * np.pi, NSENSOR)))
    sensor_pts = R * jnp.stack([jnp.cos(sensor_th), jnp.sin(sensor_th)], axis=1)
    assert observer.violations(sensor_pts) == 0

    def forward(theta):
        b = source(theta, th_band)
        return solve_operator(op, b, method="dense")

    u_true = forward(jnp.asarray(THETA_TRUE))
    clean = observer(u_true, sensor_pts)
    data = clean + NOISE * jnp.asarray(rng.standard_normal(NSENSOR))
    print("sensors %d, noise %.1e, signal rms %.4f"
          % (NSENSOR, NOISE, float(jnp.sqrt(jnp.mean(clean**2)))))

    # Forward check.  On the circle, mode k of the source produces mode k of the
    # solution with amplitude theta_k/(1 + k^2).  Project the *surface* trace of
    # u, not the banded values -- away from the surface u is an extension of the
    # solution only to discretization accuracy, and at the outer band edge it is
    # contaminated by the artificial Dirichlet condition there.
    modes = np.arange(1, NMODES + 1)
    exact_amp = THETA_TRUE / (1.0 + modes**2)
    fine_th = jnp.linspace(0, 2 * np.pi, 512, endpoint=False)
    fine_pts = R * jnp.stack([jnp.cos(fine_th), jnp.sin(fine_th)], axis=1)
    u_surf = observer(u_true, fine_pts)
    got_amp = np.array([2.0 * float(jnp.mean(u_surf * jnp.cos(k * fine_th)))
                        for k in modes])
    print("forward check, solution mode amplitudes (source mode k -> 1/(1+k^2))")
    print("   exact  %s" % " ".join("%+9.6f" % v for v in exact_amp))
    print("   CPM    %s" % " ".join("%+9.6f" % v for v in got_amp))
    print("   err    %s" % " ".join("%9.2e" % v for v in abs(got_amp - exact_amp)))

    def objective(theta, reg):
        pred = observer(forward(theta), sensor_pts)
        return misfit(pred, data) + 0.5 * reg * jnp.sum(theta**2)

    print("\ngradient check on the objective (reg = 1e-6):")
    for ana, fd, rel, e in check_gradient(lambda t: objective(t, 1e-6),
                                          jnp.zeros(NMODES), seed=0):
        print("   analytic %+.10e   fd %+.10e   rel %.2e  (eps %.0e)"
              % (ana, fd, rel, e))

    print("\nrecovery vs regularization weight")
    print("   %-10s %-12s %-12s %s" % ("reg", "J", "||dtheta||", "per-mode error"))
    for reg in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        theta_hat, res = lbfgs(lambda t: objective(t, reg), jnp.zeros(NMODES),
                               maxiter=500)
        err = np.asarray(theta_hat) - THETA_TRUE
        print("   %-10.0e %-12.4e %-12.4e %s"
              % (reg, res.fun, np.linalg.norm(err),
                 np.array2string(err, precision=4, suppress_small=True)))

    print("\nWith %d sensors and %d unknowns the problem is only mildly"
          % (NSENSOR, NMODES))
    print("overdetermined, and the high modes are damped by 1/(1+k^2) before they")
    print("reach the data -- so they are the ones the noise destroys first.  That")
    print("is identifiability, not an optimizer failure: note that reg = 0 still")
    print("drives J to the noise floor while getting the last modes wrong.")


if __name__ == "__main__":
    main()
