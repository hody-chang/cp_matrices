"""Experiment 7: the checkpoint -- a real scanned surface, at real scale.

Everything before this ran on geometry chosen for convenience: circles, spheres,
tori, an ellipse, a SIREN fitted to a circle.  This runs the whole pipeline on the
Stanford bunny from ../surfaces/tri/, 69,451 triangles, holes and thin ears
included, and asks the only question that matters at this stage: does the
recovered physics converge as the grid is refined?

Forward problem      a(x) Delta_s u - u = -f    on the bunny
Inverse problem      recover a(x) = exp(c . [1, x, y, z]) from sparse sensors

Two geometries are run side by side at each resolution:

  mesh    closest points computed exactly from the triangle mesh
  siren   closest points on the zero level set of a SIREN fitted to that mesh

Comparing them is experiment 4's question asked at bunny scale: what does using a
learned surface instead of the true one cost the recovered parameters?  Each is
convergence-tested against data generated on a finer grid *of its own geometry*,
so the refinement study measures discretization error and not the geometry gap,
and the geometry gap is measured separately by comparing the two answers.

The SIREN is fitted by scripts/fit_bunny_siren.py, which caches its weights.

Run:  python examples_diffcpm/ex7_bunny_diffusivity.py
"""

import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from diffcpm.grid import band_from_cp, band_from_dist, make_grid1d
from diffcpm.interp import InterpPattern
from diffcpm.inverse import Observer, check_gradient, lbfgs, misfit
from diffcpm.mesh import load_bunny
from diffcpm.operators import build_operator
from diffcpm.sdf import cp_level_set, level_set_residuals, siren_apply
from diffcpm.solve import solve_operator

C_TRUE = np.array([0.20, 0.35, -0.25, 0.30])
NSENSOR = 120
NOISE = 2e-3
DX_LIST = (0.12, 0.09, 0.07)
# The data-generating grid must be clearly finer than every grid we invert on,
# or its discretization error barely differs from theirs and the refinement study
# is measuring the inverse crime again. 0.035 is 2x finer than the finest
# inversion grid; an earlier run used 0.05, only 1.4x, which is not enough
# separation to argue from.
DX_FINE = 0.035
NEWTON = dict(projection_steps=8, newton_iters=25, damping=1e-10, n_backtrack=6)
SIREN_CACHE = os.path.join(_ROOT, "scripts", "bunny_siren.pkl")


def basis_of(pts):
    """[1, x, y, z] at the closest points, so a() is constant along normals."""
    return jnp.stack([jnp.ones(pts.shape[0]), pts[:, 0], pts[:, 1], pts[:, 2]], 1)


def make_grid(mesh, dx, kind="mesh", params=None, extra_bw=0.0):
    """Band for one geometry.

    Each geometry gets a band built from *its own* surface.  Building the learned
    surface's band from the mesh looks harmless and is not: a SIREN with 0.008 rms
    signed-distance error still puts its zero level set as much as 0.7 away from
    the mesh at a few points, which is more than a band half-width, so the learned
    surface pokes out of a mesh-built band and the run is garbage (field error
    865% in an earlier attempt, with an interpolation violation to prove it).

    For the learned surface the distance is taken as ``|f|`` rather than from a
    closest point solve: the network is trained to be a signed distance function,
    so ``|grad f|`` is close to 1, and one forward pass per grid point is affordable
    where a Newton solve over the whole grid is not.  It is an estimate, so it is
    checked afterwards by ``violations`` and ``clearance`` on the real closest
    points rather than trusted.
    """
    x1d = [make_grid1d(-1.25, 1.25, dx), make_grid1d(-1.25, 1.25, dx),
           make_grid1d(-1.0, 1.0, dx)]
    if kind == "mesh":
        return band_from_cp(x1d, lambda p: mesh.closest_point(p, k=24, exact=False),
                            p=3, stenrad=1, chunk=60000, extra_bw=extra_bw)

    def dist(pts):
        f = jax.vmap(siren_apply, in_axes=(None, 0))(params, jnp.asarray(pts))
        return np.abs(np.asarray(f))

    return band_from_dist(x1d, dist, p=3, stenrad=1, chunk=60000,
                          extra_bw=extra_bw)


def geometry(kind, mesh, grid, params, dx):
    """Closest points for the band, and how many of them are unusable.

    For the learned surface the Newton solve does not converge everywhere, so the
    failures are counted and returned rather than hidden.  They are left in
    place: dropping them would punch holes in the band, and substituting mesh
    closest points would quietly turn the learned-geometry run into a hybrid.
    """
    if kind == "mesh":
        return jnp.asarray(mesh.closest_point(grid.xg, k=24, exact=True)), 0.0
    cp = cp_level_set(siren_apply, params, jnp.asarray(grid.xg),
                      max_step=4 * dx, **NEWTON)
    fv, sn = level_set_residuals(siren_apply, params, jnp.asarray(grid.xg), cp)
    bad = (np.asarray(fv) > 1e-8) | (np.asarray(sn) > 1e-6)
    return cp, float(bad.mean())


def forward(c, grid, pattern, cp, basis, f, method):
    a = jnp.exp(basis @ c)
    op = build_operator(grid, cp, alpha=a, c=-1.0, pattern=pattern)
    return solve_operator(op, -f, method=method, tol=1e-11, maxiter=2000,
                          restart=250)


def setup(kind, mesh, params, dx, sensors, extra_bw=1.0):
    grid = make_grid(mesh, dx, kind=kind, params=params, extra_bw=extra_bw)
    pattern = InterpPattern(grid)
    cp, fail = geometry(kind, mesh, grid, params, dx)
    viol = pattern.violations(cp)
    basis = basis_of(cp)
    f = 1.0 + cp[:, 2] + 0.5 * cp[:, 0] * cp[:, 1]
    observer = Observer(grid)
    method = "gmres" if grid.n > 2500 else "dense"
    return dict(grid=grid, pattern=pattern, cp=cp, basis=basis, f=f,
                observer=observer, method=method, fail=fail, viol=viol,
                sensor_viol=observer.violations(sensors))


def run_one(kind, mesh, params, dx, sensors, data, reg=1e-2, maxiter=60):
    st = setup(kind, mesh, params, dx, sensors)

    def objective(c):
        u = forward(c, st["grid"], st["pattern"], st["cp"], st["basis"], st["f"],
                    st["method"])
        return misfit(st["observer"](u, sensors), data, sigma=NOISE) \
            + 0.5 * reg * jnp.sum(c * c)

    t0 = time.time()
    c_hat, res = lbfgs(objective, jnp.zeros(4), maxiter=maxiter)
    return np.asarray(c_hat), res, st, time.time() - t0


def main():
    mesh = load_bunny()
    print("bunny: %d faces, bbox %s, surface area %.2f"
          % (mesh.n_faces,
             np.round(mesh.verts.max(0) - mesh.verts.min(0), 3), 9.43))
    if not os.path.exists(SIREN_CACHE):
        print("\nNo SIREN weights at %s" % SIREN_CACHE, flush=True)
        print("Run scripts/fit_bunny_siren.py first (about 5 minutes on CPU).", flush=True)
        return 1
    params = jax.tree_util.tree_map(jnp.asarray,
                                    pickle.load(open(SIREN_CACHE, "rb")))

    # sensors: surface points from the mesh, fixed across every run
    rng = np.random.default_rng(0)
    tri = mesh.tri
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1)
    fidx = rng.choice(len(tri), size=NSENSOR, p=area / area.sum())
    u1, v1 = rng.random((NSENSOR, 1)), rng.random((NSENSOR, 1))
    over = (u1 + v1) > 1
    u1[over], v1[over] = 1 - u1[over], 1 - v1[over]
    a3, b3, c3 = tri[fidx, 0], tri[fidx, 1], tri[fidx, 2]
    sensors = jnp.asarray(a3 + u1 * (b3 - a3) + v1 * (c3 - a3))
    print("%d sensors on the mesh surface, noise %.1e, %d unknowns"
          % (NSENSOR, NOISE, len(C_TRUE)))

    results = {}
    for kind in ("mesh", "siren"):
        print("\n" + "=" * 76, flush=True)
        print("geometry: %s" % kind, flush=True)
        print("=" * 76, flush=True)

        # data on a finer grid of THIS geometry -- no inverse crime
        t0 = time.time()
        fine = setup(kind, mesh, params, DX_FINE, sensors)
        u_f = forward(jnp.asarray(C_TRUE), fine["grid"], fine["pattern"],
                      fine["cp"], fine["basis"], fine["f"], fine["method"])
        clean = fine["observer"](u_f, sensors)
        data = clean + NOISE * jnp.asarray(rng.standard_normal(NSENSOR))
        print("data grid dx %.3f, band %d, newton failures %.2f%%, "
              "signal rms %.4f  (%.0fs)"
              % (DX_FINE, fine["grid"].n, 100 * fine["fail"],
                 float(jnp.sqrt(jnp.mean(clean ** 2))), time.time() - t0))

        floor = 0.5 * NSENSOR
        print("\nJ's expected noise floor is %.0f. J well above it means model "
              "error,\nnot sensor noise, is what the fit is up against." % floor,
              flush=True)
        print("\n%-7s %-8s %-7s %-6s %-10s %-8s %-11s %-10s %s"
              % ("dx", "band n", "fail%", "viol", "J", "J/floor", "||dc||",
                 "field err", "time"), flush=True)
        prev = None
        for dx in DX_LIST:
            c_hat, res, st, dt = run_one(kind, mesh, params, dx, sensors, data)
            a_hat = np.exp(np.asarray(st["basis"] @ jnp.asarray(c_hat)))
            a_tru = np.exp(np.asarray(st["basis"] @ jnp.asarray(C_TRUE)))
            rel = np.abs(a_hat - a_tru).max() / a_tru.max()
            step = "" if prev is None else "  d(prev) %.4f" % np.linalg.norm(c_hat - prev)
            print("%-7.3f %-8d %-7.2f %-6d %-10.4e %-8.1f %-11.4e %-10.4f %.0fs%s"
                  % (dx, st["grid"].n, 100 * st["fail"], st["viol"], res.fun,
                     res.fun / floor, np.linalg.norm(c_hat - C_TRUE), rel, dt,
                     step), flush=True)
            prev = c_hat
            results[(kind, dx)] = c_hat
            results[(kind, dx, "field")] = rel

        print("\n   coefficients at dx = %.3f" % DX_LIST[-1], flush=True)
        print("   %-10s %s" % ("basis", " ".join("%8s" % n
                                                 for n in ("1", "x", "y", "z"))))
        print("   %-10s %s" % ("true", " ".join("%+8.4f" % v for v in C_TRUE)), flush=True)
        print("   %-10s %s" % ("recovered",
                               " ".join("%+8.4f" % v for v in results[(kind, DX_LIST[-1])])))

    print("\n" + "=" * 76, flush=True)
    print("what the learned surface costs", flush=True)
    print("=" * 76, flush=True)
    print("%-7s %-14s %-14s %s" % ("dx", "||dc|| mesh", "||dc|| siren",
                                   "||c_siren - c_mesh||"))
    for dx in DX_LIST:
        cm, cs = results[("mesh", dx)], results[("siren", dx)]
        print("%-7.3f %-14.4e %-14.4e %.4e"
              % (dx, np.linalg.norm(cm - C_TRUE), np.linalg.norm(cs - C_TRUE),
                 np.linalg.norm(cs - cm)))
    print("\n%-7s %-16s %s" % ("dx", "field err mesh", "field err siren"),
          flush=True)
    for dx in DX_LIST:
        print("%-7.3f %-16.4f %.4f"
              % (dx, results[("mesh", dx, "field")],
                 results[("siren", dx, "field")]), flush=True)

    print("\nRead the two error measures separately, because they disagree and", flush=True)
    print("the disagreement is the finding.  The field error is the physical", flush=True)
    print("quantity: how wrong the recovered diffusivity is pointwise, relative", flush=True)
    print("to its scale.  ||dc|| is the error in the four basis coefficients.", flush=True)
    print("The field error can fall steadily while ||dc|| moves around, because", flush=True)
    print("the four coefficients are not equally determined by the data and can", flush=True)
    print("trade against one another at nearly constant misfit.  Quoting only", flush=True)
    print("||dc|| would read as non-convergence; quoting only the field error", flush=True)
    print("would hide that the coefficients are ill-conditioned.", flush=True)
    print("\nThe mesh rows isolate discretization error.  The siren rows add the", flush=True)
    print("learned surface on top, and the comparison table is the difference.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
