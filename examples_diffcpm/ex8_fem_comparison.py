"""Experiment 8: the CPM layer against a differentiable P1 surface FEM.

This is the comparison a reviewer asks for first, and until now the package
asserted it rather than measuring it.  The baseline is in diffcpm/fem.py:
cotangent stiffness, lumped mass, matrix-free, differentiable with respect to
both the coefficients and the vertex positions, and driven by the *same*
``lax.custom_linear_solve`` adjoint as the CPM layer, so nothing here is a
comparison of solvers.

The PDE is the one both methods discretize identically,

    -Delta_s u + c(x) u = f,

and the inverse problem recovers ``c = exp(theta . [1, x, y, z])`` from sparse
noisy sensors.  A variable diffusivity would have been unfair: FEM has the flux
form ``div(a grad u)`` naturally, while CPM assembles the non-conservative
``a Delta_s u`` cheaply, so the choice of form alone would decide the result.

Part A is a fixed-topology problem where FEM is at its best: the unit sphere,
where a subdivided icosahedron is an excellent mesh.  The question is accuracy
against cost, and FEM should win.

Part B is the case the whole idea rests on: a shape whose topology changes.  FEM
needs a mesh, meshes come from marching squares, and marching squares is not
differentiable -- but the more damaging fact, measured here, is that its output is
not even *continuous* in the shape parameter.

Run:  python examples_diffcpm/ex8_fem_comparison.py
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffcpm.fem import P1Surface, barycentric_observer, observe
from diffcpm.grid import band_from_cp, band_from_dist, make_grid1d
from diffcpm.interp import InterpPattern
from diffcpm.inverse import Observer, check_gradient, lbfgs, misfit
from diffcpm.operators import build_operator
from diffcpm.sdf import cp_level_set
from diffcpm.solve import linear_solve, solve_operator
from diffcpm.surfaces import sphere_cp

THETA_TRUE = np.array([0.20, 0.35, -0.25, 0.30])
NSENSOR = 120
NOISE = 2e-3
GMRES = dict(method="gmres", tol=1e-12, maxiter=800, restart=50)


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------


def icosphere(subdiv):
    t = (1 + np.sqrt(5)) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t],
                  [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1],
                  [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdiv):
        v_list, mid, nf = list(v), {}, []

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = v[a] + v[b]
                mid[k] = len(v_list)
                v_list.append(p / np.linalg.norm(p))
            return mid[k]

        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v, f = np.array(v_list), np.array(nf)
    return v, f


def basis_of(pts):
    return jnp.stack([jnp.ones(pts.shape[0]), pts[:, 0], pts[:, 1], pts[:, 2]], 1)


def sphere_sensors(seed=1):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(NSENSOR, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True), rng


# ---------------------------------------------------------------------------
# part A: fixed topology, where FEM should win
# ---------------------------------------------------------------------------


def cpm_sphere(dx, sensors):
    x1d = [make_grid1d(-1.5, 1.5, dx)] * 3
    grid = band_from_cp(x1d, lambda p: np.asarray(sphere_cp(1.0, jnp.asarray(p))),
                        p=3, stenrad=1, chunk=60000)
    pattern = InterpPattern(grid)
    cp = sphere_cp(1.0, jnp.asarray(grid.xg))
    basis = basis_of(cp)
    f = 1.0 + cp[:, 2] + 0.5 * cp[:, 0] * cp[:, 1]
    observer = Observer(grid)
    assert pattern.violations(cp) == 0 and observer.violations(sensors) == 0

    def forward(theta):
        op = build_operator(grid, cp, alpha=-1.0, c=jnp.exp(basis @ theta),
                            pattern=pattern)
        return solve_operator(op, f, **GMRES)

    return forward, lambda u: observer(u, sensors), grid.n, basis


def fem_sphere(subdiv, sensors):
    v, faces = icosphere(subdiv)
    fe = P1Surface(len(v), faces)
    vj = jnp.asarray(v)
    basis = basis_of(vj)
    f = 1.0 + vj[:, 2] + 0.5 * vj[:, 0] * vj[:, 1]
    cells, weights = barycentric_observer(v, faces, np.asarray(sensors))

    def forward(theta):
        return fe.solve(vj, jnp.exp(basis @ theta), f, **GMRES)

    return forward, lambda u: observe(u, cells, weights), len(v), basis


def part_a():
    print("=" * 78)
    print("A  fixed topology (unit sphere), accuracy against cost")
    print("=" * 78)
    sensors_np, rng = sphere_sensors()
    sensors = jnp.asarray(sensors_np)

    rows = []
    for label, builder, sizes, fine in (
        ("CPM", cpm_sphere, (0.16, 0.12, 0.09, 0.07), 0.035),
        ("P1 FEM", fem_sphere, (2, 3, 4, 5), 6),
    ):
        fwd_f, obs_f, dof_f, _ = builder(fine, sensors)
        clean = obs_f(fwd_f(jnp.asarray(THETA_TRUE)))
        data = clean + NOISE * jnp.asarray(rng.standard_normal(NSENSOR))
        print("\n%s: data from the fine discretization (%d dof), signal rms %.4f"
              % (label, dof_f, float(jnp.sqrt(jnp.mean(clean ** 2)))), flush=True)
        print("%-10s %-9s %-10s %-8s %-11s %-11s %s"
              % ("size", "dof", "J", "J/floor", "||dtheta||", "field err",
                 "s / grad"), flush=True)
        for size in sizes:
            fwd, obs, dof, basis = builder(size, sensors)

            def objective(theta):
                return misfit(obs(fwd(theta)), data, sigma=NOISE) \
                    + 0.5 * 1e-2 * jnp.sum(theta * theta)

            vg = jax.jit(jax.value_and_grad(objective))
            vg(jnp.zeros(4))[0].block_until_ready()
            t0 = time.time()
            for _ in range(3):
                val, g = vg(jnp.zeros(4))
                g.block_until_ready()
            per = (time.time() - t0) / 3

            th, res = lbfgs(objective, jnp.zeros(4), maxiter=60)
            th = np.asarray(th)
            a_hat = np.exp(np.asarray(basis @ jnp.asarray(th)))
            a_tru = np.exp(np.asarray(basis @ jnp.asarray(THETA_TRUE)))
            rows.append((label, dof, np.linalg.norm(th - THETA_TRUE), per))
            print("%-10s %-9d %-10.4e %-8.1f %-11.4e %-11.4f %.3f"
                  % (str(size), dof, res.fun, res.fun / (0.5 * NSENSOR),
                     np.linalg.norm(th - THETA_TRUE),
                     np.abs(a_hat - a_tru).max() / a_tru.max(), per), flush=True)

    print("\nOn a fixed, smooth, well-meshed surface FEM is the better tool.  Match")
    print("the rows by ||dtheta|| rather than by size: at about 1.6e-3, FEM needs")
    print("2,562 unknowns and CPM 7,840; at about 1.1e-3, FEM 10,242 and CPM")
    print("21,676.  So FEM is roughly 3x cheaper in both unknowns and seconds per")
    print("gradient at matched accuracy, and both reach the noise floor.")
    print("\nThe reason is structural: a CPM band in 3D spends unknowns filling a")
    print("shell around the surface while P1 elements put them on it.  3x, not the")
    print("order of magnitude that argument might suggest, because the band is thin")
    print("and CPM's interpolation is higher order than P1.")
    print("\nThe conclusion to draw is that CPM's case cannot rest on cost for a")
    print("fixed smooth surface.  It has to rest on what part B measures.")
    return rows


# ---------------------------------------------------------------------------
# part B: the topology change
# ---------------------------------------------------------------------------

R, KS = 0.6, 0.05
S_MERGE = R + KS * np.log(2.0)


def level_set(s, y):
    d1 = jnp.sqrt((y[0] + s) ** 2 + y[1] ** 2 + 1e-30) - R
    d2 = jnp.sqrt((y[0] - s) ** 2 + y[1] ** 2 + 1e-30) - R
    return -KS * jax.scipy.special.logsumexp(
        jnp.stack([-d1 / KS, -d2 / KS]))


def level_set_np(s, pts):
    d1 = np.sqrt((pts[:, 0] + s) ** 2 + pts[:, 1] ** 2) - R
    d2 = np.sqrt((pts[:, 0] - s) ** 2 + pts[:, 1] ** 2) - R
    a = np.stack([-d1 / KS, -d2 / KS])
    mx = a.max(axis=0)
    return -KS * (mx + np.log(np.exp(a - mx).sum(axis=0)))


def circle_dist(s, pts):
    d1 = np.abs(np.sqrt((pts[:, 0] + s) ** 2 + pts[:, 1] ** 2) - R)
    d2 = np.abs(np.sqrt((pts[:, 0] - s) ** 2 + pts[:, 1] ** 2) - R)
    return np.minimum(d1, d2)


def marching_squares(s, h, half=(1.9, 1.2), tol=1e-9):
    """Extract the zero contour as a vertex/segment mesh.

    This is the step that makes differentiable mesh FEM impossible on a moving
    shape, and it is worth being precise about why.  It is not merely
    non-differentiable: the number of vertices it returns is an integer that
    changes whenever the contour crosses a grid node, so the mesh it produces is
    not even a continuous function of the shape parameter.  There is no
    derivative to approximate, and a finite difference of anything computed on
    top of it straddles a change of dimension.
    """
    xs = np.arange(-half[0], half[0] + 0.5 * h, h)
    ys = np.arange(-half[1], half[1] + 0.5 * h, h)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    F = level_set_np(s, np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)

    verts, vmap, segs = [], {}, []

    def vid(p):
        key = (round(p[0] / tol), round(p[1] / tol))
        if key not in vmap:
            vmap[key] = len(verts)
            verts.append(p)
        return vmap[key]

    def crossing(pa, pb, fa, fb):
        # Clamp t away from the endpoints.  When the contour passes almost
        # exactly through a grid node, t lands at 0 or 1, the crossing coincides
        # with the node, two crossings in one cell can coincide, and the segment
        # has zero length -- whose P1 stiffness is 1/L. That produced NaN for 9 of
        # 13 shapes in the first run of this experiment. This is a defect of the
        # extractor, not of FEM, so it has to be fixed before the comparison means
        # anything.
        t = fa / (fa - fb)
        t = min(max(t, 1e-6), 1.0 - 1e-6)
        return (pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]))

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            c = [(xs[i], ys[j]), (xs[i + 1], ys[j]),
                 (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])]
            fv = [F[i, j], F[i + 1, j], F[i + 1, j + 1], F[i, j + 1]]
            pts = []
            for e in range(4):
                a, b = e, (e + 1) % 4
                if (fv[a] < 0) != (fv[b] < 0):
                    pts.append(crossing(c[a], c[b], fv[a], fv[b]))
            if len(pts) == 2:
                segs.append((vid(pts[0]), vid(pts[1])))
            elif len(pts) == 4:
                # saddle: pair by the cell-centre sign
                mid = 0.25 * sum(fv)
                order = (0, 1, 2, 3) if mid < 0 else (1, 2, 3, 0)
                segs.append((vid(pts[order[0]]), vid(pts[order[1]])))
                segs.append((vid(pts[order[2]]), vid(pts[order[3]])))

    verts = np.array(verts)
    segs = np.array(segs, dtype=np.int64)
    # Drop degenerate and duplicate segments, then renumber to the vertices that
    # survive, so every vertex carries positive lumped mass and the P1 system is
    # nonsingular.
    keep = np.linalg.norm(verts[segs[:, 1]] - verts[segs[:, 0]], axis=1) > 1e-10 * h
    segs = segs[keep]
    segs = np.unique(np.sort(segs, axis=1), axis=0)
    used = np.unique(segs)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[segs]


def polyline_observer(verts, segs, points):
    """Nearest-segment linear interpolation weights, the 1D analogue of barycentric."""
    a, b = verts[segs[:, 0]], verts[segs[:, 1]]
    ab = b - a
    L2 = np.sum(ab * ab, axis=1) + 1e-300
    t = np.clip(np.einsum("pd,sd->ps", points, ab)
                - np.sum(a * ab, axis=1)[None, :], None, None) / L2[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(proj - points[:, None, :], axis=2)
    k = np.argmin(d, axis=1)
    rows = np.arange(len(points))
    tt = t[rows, k]
    cells = segs[k]
    w = np.stack([1.0 - tt, tt], 1)
    return jnp.asarray(cells), jnp.asarray(w)


def fem_objective_at(s, h_mesh, sensors_np, data):
    """J for the FEM pipeline at shape parameter s: remesh, solve, compare."""
    verts, segs = marching_squares(s, h_mesh)
    fe = P1Surface(len(verts), segs)
    vj = jnp.asarray(verts)
    f = 1.0 + vj[:, 0] + 0.25 * vj[:, 1]
    u = fe.solve(vj, jnp.ones(len(verts)), f, **GMRES)
    cells, w = polyline_observer(verts, segs, sensors_np)
    pred = observe(u, cells, w)
    ncomp = _count_components(len(verts), segs)
    return float(misfit(pred, data, sigma=NOISE)), len(verts), ncomp


def _count_components(nv, segs):
    parent = list(range(nv))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in segs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    used = set()
    for a, b in segs:
        used.add(find(a))
    return len(used)


def part_b():
    print()
    print("=" * 78)
    print("B  a shape whose topology changes (merge at s = %.4f)" % S_MERGE)
    print("=" * 78)
    h_mesh = 0.025
    dx = 0.05
    s_true = 0.50

    # CPM side: one fixed band covering every shape
    x1d = [make_grid1d(-1.9, 1.9, dx), make_grid1d(-1.2, 1.2, dx)]
    svals = np.linspace(s_true, 0.80, 9)
    grid = band_from_dist(
        x1d, lambda pts: np.min(np.stack([circle_dist(sv, pts) for sv in svals]), 0),
        p=3, stenrad=1, extra_bw=3.0)
    pattern = InterpPattern(grid)
    observer = Observer(grid)
    newton = dict(projection_steps=12, newton_iters=30, damping=1e-11,
                  n_backtrack=5, max_step=4 * dx)
    xg = jnp.asarray(grid.xg)

    @jax.jit
    def cpm_u(s):
        cp = cp_level_set(level_set, s, xg, **newton)
        op = build_operator(grid, cp, alpha=-1.0, c=1.0, pattern=pattern)
        return solve_operator(op, 1.0 + cp[:, 0] + 0.25 * cp[:, 1], **GMRES)

    rng = np.random.default_rng(0)
    th = np.sort(rng.uniform(0, 2 * np.pi, 48))
    ring = jnp.asarray(np.stack([(R + s_true) * np.cos(th), R * np.sin(th)], 1))
    sensors = jnp.asarray(np.asarray(cp_level_set(level_set, s_true, ring, **newton)))
    sensors_np = np.asarray(sensors)

    # Data from a finer CPM grid, not from the grid we then evaluate on. The first
    # run of this experiment took the data from this same grid, which drove J to
    # exactly 0 at the true shape -- the inverse crime, again, in a file written
    # after it had already been named and fixed elsewhere.
    fine_dx = dx / 2
    fine_x1d = [make_grid1d(-1.9, 1.9, fine_dx), make_grid1d(-1.2, 1.2, fine_dx)]
    fine_grid = band_from_dist(
        fine_x1d, lambda pts: circle_dist(s_true, pts), p=3, stenrad=1, extra_bw=1.0)
    fine_pat = InterpPattern(fine_grid)
    fine_obs = Observer(fine_grid)
    fine_cp = cp_level_set(level_set, s_true, jnp.asarray(fine_grid.xg), **newton)
    fine_op = build_operator(fine_grid, fine_cp, alpha=-1.0, c=1.0, pattern=fine_pat)
    fine_u = solve_operator(
        fine_op, 1.0 + fine_cp[:, 0] + 0.25 * fine_cp[:, 1], **GMRES)
    data = fine_obs(fine_u, sensors) \
        + NOISE * jnp.asarray(rng.standard_normal(sensors.shape[0]))
    print("data from a dx = %.3f grid (band %d) plus %.0e noise"
          % (fine_dx, fine_grid.n, NOISE), flush=True)

    print("CPM band %d points at dx %.3f; FEM remeshes at h %.3f; %d sensors"
          % (grid.n, dx, h_mesh, sensors.shape[0]), flush=True)

    def cpm_J(s):
        return misfit(observer(cpm_u(s), sensors), data, sigma=NOISE)

    print("\nsweep across the merge: what each pipeline sees", flush=True)
    print("%-9s %-7s %-8s %-13s %-13s %s"
          % ("s", "cmpts", "FEM nv", "FEM J", "CPM J", "notes"), flush=True)
    prev_nv = None
    for s in np.linspace(0.80, s_true, 13):
        jf, nv, nc = fem_objective_at(float(s), h_mesh, sensors_np, data)
        jc = float(cpm_J(float(s)))
        note = "" if prev_nv is None or nv == prev_nv else "nv changed by %+d" % (nv - prev_nv)
        print("%-9.4f %-7d %-8d %-13.5e %-13.5e %s"
              % (s, nc, nv, jf, jc, note), flush=True)
        prev_nv = nv

    print("\nNow the derivative, which is the whole question.", flush=True)
    print("\nEach pipeline is asked the same self-contained question: as eps", flush=True)
    print("shrinks, does its own central difference settle on a limit?  That", flush=True)
    print("avoids comparing the two methods' gradient *values*, which differ", flush=True)
    print("because their discretizations differ and would not be a fair contrast.", flush=True)
    for s0, where in ((0.70, "far from the merge"), (0.6347, "at the merge")):
        g_cpm = float(jax.grad(lambda z: cpm_J(z))(s0))
        print("\ns = %.4f, %s" % (s0, where), flush=True)
        print("  CPM exact adjoint gradient: %+.6e" % g_cpm, flush=True)
        print("  %-10s %-18s %-18s %s"
              % ("eps", "FEM fd", "CPM fd", "mesh nv (-eps, +eps)"), flush=True)
        for eps in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
            jp, nvp, _ = fem_objective_at(s0 + eps, h_mesh, sensors_np, data)
            jm, nvm, _ = fem_objective_at(s0 - eps, h_mesh, sensors_np, data)
            fem_fd = (jp - jm) / (2 * eps) if np.isfinite(jp) and np.isfinite(jm) \
                else float("nan")
            cpm_fd = float((cpm_J(s0 + eps) - cpm_J(s0 - eps)) / (2 * eps))
            print("  %-10.0e %-18.6e %-18.6e %d, %d"
                  % (eps, fem_fd, cpm_fd, nvm, nvp), flush=True)

    print("\nRead down the two fd columns.  The CPM quotient settles onto its", flush=True)
    print("exact adjoint gradient, because its objective is smooth in the shape", flush=True)
    print("parameter: the band is fixed and only the interpolation weights move.", flush=True)
    print("\nThe FEM quotient is asked to approximate a derivative of a function", flush=True)
    print("that has none.  Watch the mesh-size column: the vertex count changes", flush=True)
    print("between the two evaluations, so the difference quotient subtracts two", flush=True)
    print("objectives computed on different meshes -- on different numbers of", flush=True)
    print("unknowns.  Shrinking eps does not shrink that, because the mesh changes", flush=True)
    print("by whole vertices however small the step is.  There is no limit to find.", flush=True)
    print("\nAt the merge the failure is categorical rather than noisy: the number", flush=True)
    print("of components changes, so meshes either side are not even in", flush=True)
    print("correspondence, and no notion of moving vertices connects them.", flush=True)
    print("\nWhat is NOT claimed: that FEM cannot differentiate. With connectivity", flush=True)
    print("held fixed it differentiates with respect to vertex positions perfectly", flush=True)
    print("well, and diffcpm/fem.py does so to about 1e-9 against finite", flush=True)
    print("differences -- that is checked, not assumed. Part A shows FEM is also the", flush=True)
    print("cheaper method by roughly 3x when the shape is fixed and well meshed. The", flush=True)
    print("obstacle is re-extraction, and it is the only thing CPM is being argued", flush=True)
    print("to avoid.", flush=True)


if __name__ == "__main__":
    part_a()
    part_b()
