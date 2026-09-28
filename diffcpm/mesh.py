"""Triangle meshes: loading, exact closest points, and signed distance.

This exists to get a *real* scanned surface into the pipeline -- the bunny in
../surfaces/tri/ -- so that the closest point machinery can be exercised on
geometry nobody chose for its convenience.  Two jobs:

  * exact closest points on the mesh, used to build the band (the band is
    structure, so it is fine and preferable to build it from the true geometry);
  * signed distance samples, used as the regression target when fitting a neural
    SDF to the mesh.

Everything here is numpy, called once outside any JAX trace.

Closest points are computed exactly (point-to-triangle, all seven Voronoi
regions), but only against candidate triangles found with a KD-tree on triangle
centroids.  Brute force over 69k triangles times 20k query points is 1.4e9
point-triangle tests; the tree cuts that to a few tens per point.  The candidate
count is a correctness knob, not just a speed one -- see ``closest_point``.
"""

import numpy as np


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def read_ply(path):
    """Read an ASCII PLY with vertex x/y/z and triangular faces.

    Returns ``(verts (nv, 3), faces (nf, 3))``.  Deliberately minimal: it reads
    the files in ../surfaces/tri/ and raises on anything else rather than
    guessing.
    """
    with open(path, "r") as fh:
        if fh.readline().strip() != "ply":
            raise ValueError("not a PLY file: %s" % path)
        n_vert = n_face = None
        fmt = None
        in_vertex = False
        vert_props = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("unexpected end of PLY header")
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if parts[1] == "vertex":
                    n_vert = int(parts[2])
                elif parts[1] == "face":
                    n_face = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                vert_props.append(parts[-1])
            elif parts[0] == "end_header":
                break
        if fmt != "ascii":
            raise ValueError("only ascii PLY is supported, got %r" % fmt)
        if n_vert is None or n_face is None:
            raise ValueError("PLY header lacks vertex or face counts")
        if vert_props[:3] != ["x", "y", "z"]:
            raise ValueError("expected x, y, z first, got %r" % vert_props[:3])

        nprop = len(vert_props)
        verts = np.empty((n_vert, 3))
        for i in range(n_vert):
            vals = fh.readline().split()
            if len(vals) < nprop:
                raise ValueError("short vertex line %d" % i)
            verts[i] = [float(v) for v in vals[:3]]

        faces = np.empty((n_face, 3), dtype=np.int64)
        for i in range(n_face):
            vals = fh.readline().split()
            k = int(vals[0])
            if k != 3:
                raise ValueError("face %d has %d vertices; triangles only" % (i, k))
            faces[i] = [int(v) for v in vals[1:4]]
    return verts, faces


def normalize(verts, radius=1.0):
    """Centre a mesh on its bounding-box centre and scale to a given radius.

    Returns ``(verts_scaled, centre, scale)``.  Scanned meshes come in arbitrary
    units; putting the surface in a known box makes grid spacings comparable
    across examples.
    """
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre = 0.5 * (lo + hi)
    scale = radius / np.max(hi - lo) * 2.0
    return (verts - centre) * scale, centre, scale


# ---------------------------------------------------------------------------
# point to triangle
# ---------------------------------------------------------------------------


def _closest_on_triangles(p, a, b, c):
    """Closest point to ``p`` on each triangle (a, b, c), vectorized.

    ``p`` is (m, 1, 3) or (m, k, 3); ``a``, ``b``, ``c`` are (m, k, 3).  Returns
    (m, k, 3).  This is the standard barycentric region test (Ericson,
    *Real-Time Collision Detection*), covering all three vertices, all three
    edges and the interior, so it is exact rather than a projection onto the
    triangle plane.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.sum(ab * ap, axis=-1)
    d2 = np.sum(ac * ap, axis=-1)
    bp = p - b
    d3 = np.sum(ab * bp, axis=-1)
    d4 = np.sum(ac * bp, axis=-1)
    cp_ = p - c
    d5 = np.sum(ab * cp_, axis=-1)
    d6 = np.sum(ac * cp_, axis=-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    safe = np.where(np.abs(denom) < 1e-300, 1.0, denom)
    v_in = vb / safe
    w_in = vc / safe

    # start from the interior solution, then override each outside region
    v = v_in
    w = w_in

    reg_a = (d1 <= 0) & (d2 <= 0)                    # vertex a
    reg_b = (d3 >= 0) & (d4 <= d3)                   # vertex b
    reg_c = (d6 >= 0) & (d5 <= d6)                   # vertex c
    reg_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)       # edge ab
    reg_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)       # edge ac
    reg_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)  # edge bc

    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0)
        t_ac = np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0)
        t_bc = np.where((d4 - d3) + (d5 - d6) != 0,
                        (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0)

    v = np.where(reg_bc, 1.0 - t_bc, v)
    w = np.where(reg_bc, t_bc, w)
    v = np.where(reg_ac, 0.0, v)
    w = np.where(reg_ac, np.clip(t_ac, 0.0, 1.0), w)
    v = np.where(reg_ab, np.clip(t_ab, 0.0, 1.0), v)
    w = np.where(reg_ab, 0.0, w)
    v = np.where(reg_c, 0.0, v)
    w = np.where(reg_c, 1.0, w)
    v = np.where(reg_b, 1.0, v)
    w = np.where(reg_b, 0.0, w)
    v = np.where(reg_a, 0.0, v)
    w = np.where(reg_a, 0.0, w)

    return a + v[..., None] * ab + w[..., None] * ac


class TriMesh:
    """A triangle mesh with a KD-tree accelerated closest point query."""

    def __init__(self, verts, faces):
        from scipy.spatial import cKDTree

        self.verts = np.ascontiguousarray(verts, dtype=float)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.tri = self.verts[self.faces]  # (nf, 3, 3)
        self.centroids = self.tri.mean(axis=1)
        self._tree = cKDTree(self.centroids)
        # circumradius-ish bound: how far a triangle reaches from its centroid
        self.tri_reach = np.max(
            np.linalg.norm(self.tri - self.centroids[:, None, :], axis=2), axis=1
        )
        self.max_reach = float(self.tri_reach.max())
        e1 = self.tri[:, 1] - self.tri[:, 0]
        e2 = self.tri[:, 2] - self.tri[:, 0]
        n = np.cross(e1, e2)
        self.face_normals = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-300)
        # area-weighted vertex normals, for a more reliable inside/outside sign
        # than the closest face's own normal near edges and vertices
        area = 0.5 * np.linalg.norm(n, axis=1)
        vn = np.zeros_like(self.verts)
        for j in range(3):
            np.add.at(vn, self.faces[:, j], self.face_normals * area[:, None])
        self.vert_normals = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-300)

    @property
    def n_faces(self):
        return self.faces.shape[0]

    def closest_point(self, pts, k=24, chunk=20000, exact=True, report=False):
        """Closest points on the mesh to ``pts`` (m, 3).

        Two passes.  The first takes the ``k`` nearest triangle centroids from a
        KD-tree and finds the best triangle among them.  That alone is a
        heuristic, and the bound that would make it provable is loose: a triangle
        outside the candidate set has centroid distance at least ``dcen[k-1]``,
        so its distance to the query point is at least
        ``dcen[k-1] - max_reach``.  With ``max_reach`` a global bound over a mesh
        of uneven triangle sizes, that test fails for most near-surface points
        even when the answer is right.

        So rather than tune ``k`` and hope, the second pass makes it exact: for
        every point whose candidate set is not provably sufficient, re-query all
        centroids within ``dbest + max_reach`` and take the best over those.
        That ball provably contains every triangle that could win.  ``exact``
        turns the second pass off; ``report`` returns
        ``(cp, fraction_needing_pass_two, fraction_pass_two_changed)``.
        """
        pts = np.ascontiguousarray(pts, dtype=float)
        out = np.empty_like(pts)
        n_unsure = n_changed = 0
        kk = min(k, self.n_faces)
        for s in range(0, len(pts), chunk):
            q = pts[s:s + chunk]
            dcen, idx = self._tree.query(q, k=kk)
            if kk == 1:
                dcen, idx = dcen[:, None], idx[:, None]
            tri = self.tri[idx]  # (mm, k, 3, 3)
            cand = _closest_on_triangles(q[:, None, :], tri[:, :, 0],
                                         tri[:, :, 1], tri[:, :, 2])
            d = np.linalg.norm(cand - q[:, None, :], axis=2)
            best = np.argmin(d, axis=1)
            rows = np.arange(len(q))
            cp = cand[rows, best]
            dbest = d[rows, best]

            unsure = np.flatnonzero(dcen[:, -1] < dbest + self.max_reach)
            n_unsure += len(unsure)
            if exact and len(unsure):
                radii = dbest[unsure] + self.max_reach
                balls = self._tree.query_ball_point(q[unsure], radii)
                for j, faces in zip(unsure, balls):
                    if len(faces) <= kk:
                        continue
                    f = np.asarray(faces, dtype=np.int64)
                    t = self.tri[f]
                    c2 = _closest_on_triangles(q[j][None, :], t[:, 0], t[:, 1],
                                               t[:, 2])
                    d2 = np.linalg.norm(c2 - q[j], axis=1)
                    b2 = int(np.argmin(d2))
                    if d2[b2] < dbest[j] - 1e-15:
                        cp[j] = c2[b2]
                        n_changed += 1
            out[s:s + chunk] = cp
        if report:
            m = max(len(pts), 1)
            return out, n_unsure / m, n_changed / m
        return out

    def closest_point_bruteforce(self, pts, chunk=64):
        """Closest points checked against every triangle.  For validating the tree."""
        pts = np.ascontiguousarray(pts, dtype=float)
        out = np.empty_like(pts)
        for s in range(0, len(pts), chunk):
            q = pts[s:s + chunk]
            cand = _closest_on_triangles(q[:, None, :], self.tri[None, :, 0],
                                         self.tri[None, :, 1], self.tri[None, :, 2])
            d = np.linalg.norm(cand - q[:, None, :], axis=2)
            out[s:s + chunk] = cand[np.arange(len(q)), np.argmin(d, axis=1)]
        return out

    def signed_distance(self, pts, k=24, chunk=20000):
        """Signed distance to the mesh; negative inside.

        The sign comes from an angle-free pseudonormal: the area-weighted vertex
        normals interpolated at the closest point.  Using the closest *face*
        normal instead gives wrong signs on a thin feature whose closest point
        lands on an edge or vertex, which the bunny has plenty of.
        """
        pts = np.ascontiguousarray(pts, dtype=float)
        cp = self.closest_point(pts, k=k, chunk=chunk)
        d = pts - cp
        dist = np.linalg.norm(d, axis=1)

        # barycentric coordinates of cp in its closest triangle, to interpolate
        # the vertex normals; recover the triangle by a second short query
        _, idx = self._tree.query(cp, k=1)
        tri = self.tri[idx]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        n = np.cross(b - a, c - a)
        denom = np.sum(n * n, axis=1) + 1e-300
        u = np.sum(np.cross(b - cp, c - cp) * n, axis=1) / denom
        v = np.sum(np.cross(c - cp, a - cp) * n, axis=1) / denom
        w = 1.0 - u - v
        vn = self.vert_normals[self.faces[idx]]
        nrm = (u[:, None] * vn[:, 0] + v[:, None] * vn[:, 1]
               + w[:, None] * vn[:, 2])
        sign = np.where(np.sum(d * nrm, axis=1) >= 0.0, 1.0, -1.0)
        return sign * dist


def load_bunny(path=None, radius=1.0):
    """Load ../surfaces/tri/bunny.ply, normalized into a box of given radius."""
    import os

    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "surfaces", "tri", "bunny.ply")
    verts, faces = read_ply(path)
    verts, centre, scale = normalize(verts, radius=radius)
    return TriMesh(verts, faces)
