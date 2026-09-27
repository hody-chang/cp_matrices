"""Experiment 3: does the discrete CPM shape derivative converge to the
continuous one?

This is the numerical-analysis question underneath the whole idea.  It has an
answer here that is not the obvious one, and the answer is the point of the
experiment.

Setup.  On a circle of radius R, Delta_s = R^{-2} d^2/dtheta^2, so

    (-Delta_s + 1) u = cos(k theta)   =>   u = cos(k theta) / (1 + k^2/R^2)

and for J(R) = 1/2 int_Gamma u^2 ds, with ds = R dtheta,

    J(R)     = (pi/2) R q^{-2},                  q = 1 + k^2/R^2
    dJ/dR    = (pi/2) [ q^{-2} + 4 k^2 / (R^2 q^3) ].

Both are closed form, so we can measure the discretization error in the discrete
objective J_h and, separately, in the discrete shape derivative dJ_h/dR obtained
from the CPM adjoint.  The band is held fixed and the surface moves inside it.

Findings:

1.  J_h -> J at a clean second order.

2.  dJ_h/dR -> dJ/dR, but *not* monotonically and not at second order.  The
    pointwise error oscillates: refining the grid can make it worse.

3.  The reason is grid alignment.  The CPM discretization error depends on where
    the surface sits relative to the grid lines; sweeping R across a single cell
    at fixed h shows the derivative error swinging by orders of magnitude.
    Differentiating J_h with respect to a shape parameter differentiates that
    dependence as well as the physics.

4.  Averaging dJ_h/dR over sub-cell grid offsets recovers the second-order rate,
    which confirms the diagnosis: the oscillation is a function of R/h with
    roughly zero mean, superimposed on a clean O(h^2) trend.

That is a concrete, quantified statement of the "gradient correctness" question,
and an argument for the tapered-weight or blended-stencil fix rather than a
reason to distrust the adjoint: the adjoint is the exact derivative of the
discrete objective (see the tests), it is the discrete objective itself whose
error is grid-aligned.

Run:  python examples_diffcpm/ex3_shape_derivative_convergence.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffcpm.grid import band_from_cp, make_grid1d
from diffcpm.interp import InterpPattern, apply_sparse
from diffcpm.operators import build_operator
from diffcpm.solve import solve_operator
from diffcpm.surfaces import sphere_cp

K = 3
HALF = 1.8
MQUAD = 256


def exact_J(R):
    return 0.5 * np.pi * R / (1.0 + K**2 / R**2) ** 2


def exact_dJ(R):
    q = 1.0 + K**2 / R**2
    return 0.5 * np.pi * (1.0 / q**2 + 4 * K**2 / (R**2 * q**3))


def make_objective(dx, shift=0.0, R_band=1.0):
    """J_h(R) on a band built for radius ``R_band``, grid offset by shift*dx."""
    x1d = [make_grid1d(-HALF + shift * dx, HALF + shift * dx, dx)] * 2
    grid = band_from_cp(x1d, lambda p: np.asarray(sphere_cp(R_band, jnp.asarray(p))),
                        p=3, stenrad=1)
    pattern = InterpPattern(grid)
    th = jnp.linspace(0, 2 * np.pi, MQUAD, endpoint=False)
    w = 2 * np.pi / MQUAD

    def J(R):
        cp = sphere_cp(R, jnp.asarray(grid.xg))
        op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pattern)
        b = jnp.cos(K * jnp.arctan2(cp[:, 1], cp[:, 0]))
        u = solve_operator(op, b, method="dense")
        pts = R * jnp.stack([jnp.cos(th), jnp.sin(th)], axis=1)
        cols, vals = pattern.weights(pts)
        return 0.5 * jnp.sum(w * apply_sparse(cols, vals, u) ** 2) * R

    return J, grid, pattern


def part1_refinement():
    print("=" * 72)
    print("1/3  grid refinement at R = 1, k = %d" % K)
    print("=" * 72)
    print("%-8s %-6s %-12s %-7s %-12s %-7s" %
          ("dx", "n", "|J_h - J|", "rate", "|dJ_h - dJ|", "rate"))
    pj = pd = None
    for dx in (0.2, 0.1, 0.05, 0.025, 0.0125):
        J, grid, _ = make_objective(dx)
        ej = abs(float(J(1.0)) - exact_J(1.0))
        ed = abs(float(jax.grad(J)(1.0)) - exact_dJ(1.0))
        rj = "" if pj is None else "%.2f" % np.log2(pj / ej)
        rd = "" if pd is None else "%.2f" % np.log2(pd / ed)
        print("%-8.4f %-6d %-12.4e %-7s %-12.4e %-7s" % (dx, grid.n, ej, rj, ed, rd))
        pj, pd = ej, ed
    print("\nJ_h converges at 2.0.  The derivative does not: the rate column")
    print("alternates sign, so the error is not a clean power of h.")


def part2_grid_alignment():
    dx = 0.1
    print()
    print("=" * 72)
    print("2/3  sweeping R across one cell at fixed dx = %.2f" % dx)
    print("=" * 72)
    J, grid, pattern = make_objective(dx)
    gJ = jax.jit(jax.grad(J))
    print("%-10s %-14s %-12s %-8s" % ("R", "dJ_h/dR", "error", "band ok"))
    for R in np.linspace(1.0, 1.0 + 1.2 * dx, 13):
        viol = pattern.violations(sphere_cp(R, jnp.asarray(grid.xg)))
        print("%-10.5f %-14.9f %+.3e   %s"
              % (R, float(gJ(R)), float(gJ(R)) - exact_dJ(R),
                 "yes" if viol == 0 else "NO (%d)" % viol))
    print("\nThe error grows smoothly as the surface drifts away from the radius")
    print("the band was built for, then breaks down entirely once the surface")
    print("leaves the band -- the last rows flag that explicitly.  Within the")
    print("band the variation is the grid-alignment effect.")


def part3_offset_average(nshift=8):
    print()
    print("=" * 72)
    print("3/3  averaging dJ_h/dR over %d sub-cell grid offsets" % nshift)
    print("=" * 72)
    print("%-8s %-13s %-7s %-12s %-13s %-7s" %
          ("dx", "|mean - dJ|", "rate", "offset spread", "|mean J - J|", "rate"))
    pm = pj = None
    for dx in (0.2, 0.1, 0.05, 0.025):
        ds, js = [], []
        for s in np.arange(nshift) / nshift:
            J, _, _ = make_objective(dx, shift=float(s))
            ds.append(float(jax.grad(J)(1.0)))
            js.append(float(J(1.0)))
        ds, js = np.array(ds), np.array(js)
        em = abs(ds.mean() - exact_dJ(1.0))
        ej = abs(js.mean() - exact_J(1.0))
        rm = "" if pm is None else "%.2f" % np.log2(pm / em)
        rj = "" if pj is None else "%.2f" % np.log2(pj / ej)
        print("%-8.4f %-13.4e %-7s %-12.3e %-13.4e %-7s"
              % (dx, em, rm, ds.max() - ds.min(), ej, rj))
        pm, pj = em, ej
    print("\nThe offset-averaged derivative error converges at roughly second")
    print("order, while the spread across offsets is as large as the mean error")
    print("itself.  That is the signature of an oscillation in R/h with near-zero")
    print("mean, and it is what a principled moving-band treatment would remove.")


if __name__ == "__main__":
    part1_refinement()
    part2_grid_alignment()
    part3_offset_average()
