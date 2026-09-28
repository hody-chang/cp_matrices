"""A differentiable P1 surface FEM, as a baseline to compare the CPM layer against.

This is not part of the closest point method layer.  It exists so that the claim
"CPM wins where meshing is the bottleneck" can be measured instead of asserted,
which is the comparison a reviewer asks for first.

What it provides, on a triangulated surface (or a polyline in 2D):

  * the cotangent stiffness matrix for ``-Delta_s`` and a lumped mass matrix,
    applied matrix-free, exactly as the CPM operators are;
  * differentiability with respect to the vertex positions *and* the
    coefficients, using the same ``lax.custom_linear_solve`` adjoint, so the
    comparison is between discretizations and not between solvers;
  * barycentric observation at arbitrary surface points, the FEM counterpart of
    the CPM ``Observer``.

The point of interest is what this baseline *cannot* do.  It is differentiable
with respect to vertex positions only while the connectivity is held fixed.  A
shape change that alters connectivity -- a merge, a split, a change of genus --
requires re-extracting the mesh, and mesh extraction (marching squares or cubes)
is not differentiable and does not even produce a continuous vertex count.  That
is the gap the topology experiment measures.

The PDE is kept in the form both methods discretize identically,

    -Delta_s u + c(x) u = f,

so no part of the comparison rests on one method being asked a question in an
awkward form.  A variable *diffusivity* would favour FEM, which has the flux form
naturally, over the non-conservative ``a Delta_s u`` that CPM assembles cheaply;
recovering a reaction coefficient avoids that asymmetry entirely.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .solve import linear_solve


# ---------------------------------------------------------------------------
# element operators
# ---------------------------------------------------------------------------


def p1_triangle_stiffness(tri):
    """Local stiffness matrices for P1 elements on triangles.

    ``tri`` is (nf, 3, 3): per face, three vertices in 3D.  Returns (nf, 3, 3)
    local matrices and (nf,) areas.

    Uses the standard cotangent form ``K_local = (e_i . e_j) / (4 A)`` with
    ``e_i`` the edge opposite vertex ``i``, which is the P1 stiffness matrix for
    ``-Delta`` on the triangle and is what makes this the classic cotangent
    Laplacian.
    """
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e0 = p2 - p1
    e1 = p0 - p2
    e2 = p1 - p0
    cross = jnp.cross(-e2, e1)          # (p1-p0) x (p2-p0)
    area = 0.5 * jnp.linalg.norm(cross, axis=1)
    e = jnp.stack([e0, e1, e2], axis=1)  # (nf, 3, 3)
    gram = jnp.einsum("fid,fjd->fij", e, e)
    return gram / (4.0 * area[:, None, None] + 1e-300), area


def p1_segment_stiffness(seg):
    """Local stiffness and lengths for P1 elements on segments (a curve in R^d).

    ``seg`` is (ne, 2, d).  For a segment of length L the local stiffness of
    ``-d^2/ds^2`` is ``[[1, -1], [-1, 1]] / L``.
    """
    d = seg[:, 1] - seg[:, 0]
    length = jnp.linalg.norm(d, axis=1)
    base = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
    return base[None, :, :] / (length[:, None, None] + 1e-300), length


class P1Surface:
    """P1 finite elements on a fixed-connectivity mesh.

    ``cells`` is (nc, k) vertex indices, k = 3 for triangles, 2 for segments.
    The connectivity is *static*; the vertex positions are traced, which is
    exactly the scope of differentiable mesh FEM.
    """

    def __init__(self, n_verts, cells):
        self.n = int(n_verts)
        self.cells = np.ascontiguousarray(cells, dtype=np.int64)
        self.k = self.cells.shape[1]
        if self.k not in (2, 3):
            raise ValueError("cells must be segments (k=2) or triangles (k=3)")
        self._cells_j = jnp.asarray(self.cells)

    def assemble(self, verts):
        """Local stiffness, cell measure and lumped mass, from vertex positions."""
        cell_pts = verts[self._cells_j]  # (nc, k, d)
        if self.k == 3:
            kloc, measure = p1_triangle_stiffness(cell_pts)
        else:
            kloc, measure = p1_segment_stiffness(cell_pts)
        mass = jnp.zeros(self.n).at[self._cells_j].add(
            (measure / self.k)[:, None] * jnp.ones((1, self.k)))
        return kloc, measure, mass

    def stiffness_matvec(self, kloc, u):
        """K u, assembled matrix-free by scatter-add over cells."""
        uc = u[self._cells_j]                       # (nc, k)
        contrib = jnp.einsum("cij,cj->ci", kloc, uc)
        return jnp.zeros(self.n, dtype=u.dtype).at[self._cells_j].add(contrib)

    def operator(self, verts, c):
        """The matvec for ``-Delta_s + c``, and the mass matrix used for the rhs.

        Returns ``(matvec, mass)``.  The reaction term is lumped, consistent with
        the mass matrix, so ``c`` enters as ``mass * c * u``.
        """
        kloc, _, mass = self.assemble(verts)

        def matvec(u):
            return self.stiffness_matvec(kloc, u) + mass * c * u

        return matvec, mass

    def solve(self, verts, c, f, **kwargs):
        """Solve ``(-Delta_s + c) u = f`` weakly: ``(K + M c) u = M f``."""
        matvec, mass = self.operator(verts, c)
        return linear_solve(matvec, mass * f, **kwargs)


# ---------------------------------------------------------------------------
# observation at surface points
# ---------------------------------------------------------------------------


def barycentric_observer(verts_np, faces_np, points_np):
    """Static interpolation data for observing a P1 field at fixed surface points.

    Finds, for each point, the nearest cell and its barycentric weights, once,
    from the reference geometry.  Returns ``(cells, weights)`` so the observation
    itself is a differentiable gather-and-weight, matching what the CPM
    ``Observer`` does with interpolation weights.
    """
    from .mesh import TriMesh, _closest_on_triangles

    mesh = TriMesh(verts_np, faces_np)
    _, idx = mesh._tree.query(points_np, k=32)
    tri = mesh.tri[idx]
    cand = _closest_on_triangles(points_np[:, None, :], tri[:, :, 0],
                                 tri[:, :, 1], tri[:, :, 2])
    d = np.linalg.norm(cand - points_np[:, None, :], axis=2)
    best = np.argmin(d, axis=1)
    rows = np.arange(len(points_np))
    fidx = idx[rows, best]
    cp = cand[rows, best]

    a, b, c = (mesh.verts[faces_np[fidx, 0]], mesh.verts[faces_np[fidx, 1]],
               mesh.verts[faces_np[fidx, 2]])
    n = np.cross(b - a, c - a)
    denom = np.sum(n * n, axis=1) + 1e-300
    w0 = np.sum(np.cross(b - cp, c - cp) * n, axis=1) / denom
    w1 = np.sum(np.cross(c - cp, a - cp) * n, axis=1) / denom
    w2 = 1.0 - w0 - w1
    return jnp.asarray(faces_np[fidx]), jnp.asarray(np.stack([w0, w1, w2], 1))


def observe(u, cells, weights):
    return jnp.sum(u[cells] * weights, axis=1)
