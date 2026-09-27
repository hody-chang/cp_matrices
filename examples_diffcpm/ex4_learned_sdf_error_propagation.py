"""Experiment 4: how does a learned SDF's geometric error propagate into a
recovered parameter?

This is the question that connects the layer back to reconstruction pipelines.
A SIREN, DeepSDF or NeuS fit is not an exact distance function: |grad f| != 1, the
zero level set is displaced from the true surface, and sharp features are rounded.
If we then solve an inverse problem on that surface, the recovered physics
inherits an error that has nothing to do with the data.  How much?

Protocol.  The true surface is the unit circle.  Data are generated on the *true*
geometry, noise-free, so that every bit of the recovered-parameter error is
attributable to geometry.  We then fit SIRENs of increasing quality (by training
longer), and for each one:

  * measure the geometric error   max |cp_siren(x) - cp_exact(x)|  over the band,
  * measure the SDF-ness defect   max | |grad f| - 1 |  on the surface,
  * run the source-recovery inverse problem of experiment 1 using that geometry,
  * report the resulting error in the recovered Fourier coefficients.

The baseline row uses the exact geometry, and shows the error floor set by the
discretization and the optimizer rather than by the surface.

Run:  python examples_diffcpm/ex4_learned_sdf_error_propagation.py
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
from diffcpm.inverse import Observer, lbfgs, misfit
from diffcpm.operators import build_operator
from diffcpm.sdf import (adam, cp_level_set, level_set_residual_norm, siren_apply,
                         siren_init)
from diffcpm.solve import solve_operator
from diffcpm.surfaces import sphere_cp

DX = 0.08
NMODES = 4
NSENSOR = 16
THETA_TRUE = np.array([0.8, -0.5, 0.35, 0.15])
TRAIN_STEPS = (150, 400, 1200, 4000)


def fit_siren(steps, key):
    """Regress a SIREN onto the exact circle SDF for ``steps`` Adam steps."""
    params = siren_init(key, 2, (32, 32), w0_first=5.0, w0=5.0)
    xs = jax.random.uniform(jax.random.fold_in(key, 1), (4096, 2),
                            minval=-1.6, maxval=1.6)
    target = jnp.linalg.norm(xs, axis=1) - 1.0

    def loss(p):
        pred = jax.vmap(siren_apply, in_axes=(None, 0))(p, xs)
        return jnp.mean((pred - target) ** 2)

    params, final = adam(jax.value_and_grad(loss), params, steps, lr=3e-3)
    return params, float(final)


def solve_and_recover(grid, pattern, cp, observer, sensors, data):
    """Recover the source coefficients on the geometry described by ``cp``."""
    th = jnp.arctan2(cp[:, 1], cp[:, 0])
    modes = jnp.arange(1, NMODES + 1)
    op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pattern)

    def objective(theta):
        b = jnp.sum(theta[None, :] * jnp.cos(modes[None, :] * th[:, None]), axis=1)
        u = solve_operator(op, b, method="dense")
        return misfit(observer(u, sensors), data)

    theta_hat, res = lbfgs(objective, jnp.zeros(NMODES), maxiter=400)
    return np.asarray(theta_hat), res


def main():
    x1d = [make_grid1d(-1.6, 1.6, DX)] * 2
    # One band, built from the exact circle with room to spare, shared by every
    # geometry below so the comparison isolates the surface and not the band.
    grid = band_from_cp(x1d, lambda p: np.asarray(sphere_cp(1.0, jnp.asarray(p))),
                        p=3, stenrad=1, extra_bw=2.0)
    pattern = InterpPattern(grid)
    observer = Observer(grid)
    print("grid %s, band %d points, dx %.3f" % (grid.shape, grid.n, DX))

    cp_exact = sphere_cp(1.0, jnp.asarray(grid.xg))
    assert pattern.violations(cp_exact) == 0

    rng = np.random.default_rng(0)
    sensor_th = jnp.asarray(np.sort(rng.uniform(0, 2 * np.pi, NSENSOR)))
    sensors = jnp.stack([jnp.cos(sensor_th), jnp.sin(sensor_th)], axis=1)
    assert observer.violations(sensors) == 0

    # noise-free data, generated on the true geometry
    th_e = jnp.arctan2(cp_exact[:, 1], cp_exact[:, 0])
    modes = jnp.arange(1, NMODES + 1)
    b_true = jnp.sum(jnp.asarray(THETA_TRUE)[None, :] *
                     jnp.cos(modes[None, :] * th_e[:, None]), axis=1)
    op_e = build_operator(grid, cp_exact, alpha=-1.0, c=1.0, pattern=pattern)
    data = observer(solve_operator(op_e, b_true, method="dense"), sensors)
    print("noise-free data from the exact circle, rms %.4f"
          % float(jnp.sqrt(jnp.mean(data**2))))

    rows = []
    theta_hat, res = solve_and_recover(grid, pattern, cp_exact, observer, sensors, data)
    rows.append(("exact", 0.0, 0.0, 0.0, np.linalg.norm(theta_hat - THETA_TRUE),
                 res.fun, theta_hat))

    for steps in TRAIN_STEPS:
        params, fit_loss = fit_siren(steps, jax.random.PRNGKey(7))
        cp_s = cp_level_set(siren_apply, params, jnp.asarray(grid.xg))
        fmax, sinmax = level_set_residual_norm(siren_apply, params,
                                               jnp.asarray(grid.xg), cp_s)
        if fmax > 1e-8 or sinmax > 1e-8:
            print("   (steps %5d) Newton did not converge: |f| %.1e, sin %.1e"
                  % (steps, fmax, sinmax))
        viol = pattern.violations(cp_s)
        clear = grid.clearance(
            lambda pts: np.linalg.norm(
                np.asarray(cp_level_set(siren_apply, params, jnp.asarray(pts))) - pts,
                axis=1))
        if clear < 1.0:
            print("   (steps %5d) band clearance only %.1f of the RM radius"
                  % (steps, clear))
        geo = float(jnp.max(jnp.linalg.norm(cp_s - cp_exact, axis=1)))
        gs = jax.vmap(jax.grad(siren_apply, argnums=1), in_axes=(None, 0))(params, cp_s)
        gradef = float(jnp.max(jnp.abs(jnp.linalg.norm(gs, axis=1) - 1.0)))
        if viol:
            rows.append((str(steps), fit_loss, geo, gradef, np.nan, np.nan, None))
            print("   (steps %5d) surface left the band: %d violations" % (steps, viol))
            continue
        theta_hat, res = solve_and_recover(grid, pattern, cp_s, observer, sensors, data)
        rows.append((str(steps), fit_loss, geo, gradef,
                     np.linalg.norm(theta_hat - THETA_TRUE), res.fun, theta_hat))

    print("\n%-8s %-11s %-12s %-14s %-12s %-11s"
          % ("steps", "SDF mse", "geom err", "| |grad f|-1 |", "||dtheta||", "final J"))
    for name, fl, geo, ge, dt, jv, _ in rows:
        print("%-8s %-11.3e %-12.3e %-14.3e %-12.3e %-11.3e"
              % (name, fl, geo, ge, dt, jv))

    print("\nrecovered coefficients")
    print("   true      %s" % " ".join("%+7.4f" % v for v in THETA_TRUE))
    for name, _, geo, _, _, _, th in rows:
        if th is not None:
            print("   %-9s %s   (geom err %.1e)"
                  % (name, " ".join("%+7.4f" % v for v in th), geo))

    good = [(r[2], r[4]) for r in rows[1:] if np.isfinite(r[4]) and r[2] > 0]
    if len(good) >= 2:
        g = np.array(good)
        slope = np.polyfit(np.log(g[:, 0]), np.log(g[:, 1]), 1)[0]
        print("\nlog-log slope of ||dtheta|| against geometric error: %.2f" % slope)
        print("A slope near 1 says the geometric error passes into the recovered")
        print("parameters at first order and is not damped by the inverse problem.")
    print("\nThe 'exact' row is the floor: with noise-free data on the true")
    print("geometry the only error left is discretization and optimizer tolerance.")
    print("Everything above it is paid for by the surface fit, which is the")
    print("quantity a reconstruction pipeline controls -- so this table is the")
    print("exchange rate between reconstruction quality and parameter accuracy.")


if __name__ == "__main__":
    main()
