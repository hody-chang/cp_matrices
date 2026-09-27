"""Differentiable barycentric-free Lagrange interpolation for CPM.

The extension matrix E of the closest point method interpolates grid data at
the closest points cp(x_i).  Its entries are tensor products of 1D Lagrange
weights, which are *polynomials* in the offset of cp(x_i) inside its stencil
cell.  That polynomial dependence is the whole reason geometry gradients flow:
d(weights)/d(cp) is smooth, and cp depends on the surface.

Two deliberate departures from cp_matrices/LagrangeWeights1D.m:

1.  We evaluate the Lagrange basis in the *nodal* form

        w_j(s) = prod_{m != j} (s - m) / (j - m),      s = (x - x_base)/dx

    rather than the barycentric form ``w_j = c_j/(x-x_j) / sum_k ...``.  The
    barycentric form divides by ``x - x_j``, which is zero when the closest
    point lands exactly on a grid line -- a case that happens routinely for
    symmetric surfaces on symmetric grids.  MATLAB special-cases it with a
    branch returning binary weights; under autodiff that branch would return a
    *constant*, i.e. a wrong (zero) derivative at those points.  The nodal form
    is a polynomial with no removable singularity, so it is exact and smoothly
    differentiable everywhere.  The cost is O(N^2) instead of O(N) per point,
    which is negligible for N = p+1 <= 8.

2.  The base-point index and the local coordinate are computed *together*, from
    a single division, and never re-derived from each other.  This matters more
    than it looks.  The obvious implementation computes

        I = floor((x - relpt)/dx) - (p-1)/2
        s = (x - (relpt + I*dx)) / dx

    which divides by ``dx`` twice and so rounds the same real number twice.
    When the closest point lands on a grid line -- routine for a symmetric
    surface on a symmetric grid -- the two roundings can straddle the integer,
    and then ``I`` and ``s`` describe stencils one cell apart.  The interpolated
    value is then simply wrong, by O(1), at those points.  Worse, whether it
    happens depends on how the compiler reassociates the arithmetic, so the same
    code gives different answers under ``jax.jit`` than eagerly.  Computing
    ``s = t - i0 + shift`` from the same ``t`` and ``i0`` makes the two
    consistent by construction, at any rounding.

    The base index is additionally snapped to the nearest integer within
    ``SNAP_TOL`` cells, purely so the choice of stencil at a grid line is
    deterministic; with the consistency fix in place the interpolated value is
    continuous across that choice either way.

    The index is wrapped in ``stop_gradient``: it is a ``round``/``floor``, so
    its derivative is zero almost everywhere and undefined on cell boundaries.
    Stopping the gradient records that explicitly.  The consequence -- that the
    discrete adjoint ignores stencil-cell crossings -- is the "moving band"
    discontinuity, discussed in diffcpm/README.md.
"""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np


def _nodal_denominators(N):
    """prod_{m != j} (j - m) for j = 0..N-1."""
    d = np.empty(N, dtype=float)
    for j in range(N):
        d[j] = np.prod([j - m for m in range(N) if m != j])
    return d


@functools.partial(jax.jit, static_argnums=1)
def lagrange_weights_1d(s, N):
    """1D Lagrange weights on nodes 0..N-1, evaluated at local coordinate ``s``.

    ``s`` has shape (...,) and is measured in grid cells from the base point.
    Returns shape (..., N).  Weights sum to 1 exactly (up to rounding) and are
    binary when ``s`` is an integer node, without any branching.
    """
    m = jnp.arange(N, dtype=s.dtype)
    diff = s[..., None] - m  # (..., N)
    off_diag = ~np.eye(N, dtype=bool)  # (N, N), [j, m] -> include m in prod for j
    # (..., N_j, N_m) -> product over m
    terms = jnp.where(off_diag, diff[..., None, :], jnp.ones_like(diff)[..., None, :])
    num = jnp.prod(terms, axis=-1)  # (..., N)
    return num / jnp.asarray(_nodal_denominators(N), dtype=s.dtype)


SNAP_TOL = 1e-9  # in units of grid cells


def base_index_and_local(x, p, relpt, dx):
    """Stencil base index and local coordinate, consistent by construction.

    Returns ``(I, s)`` where ``I`` is the 0-based index of the lower corner of
    the stencil hypercube (cp_matrices/findGridInterpBasePt.m, 0-based) and
    ``s = (x - X_base)/dx`` is the offset inside it, in cells.  ``s`` lies in
    ``[shift, shift+1]`` with ``shift = floor(p/2)``, and carries the gradient;
    ``I`` is integer and stopped.  See the module docstring for why these must
    not be computed independently.
    """
    t = (x - relpt) / dx
    shift = p // 2
    if p % 2 == 0:
        i0 = jnp.round(t)
    else:
        tr = jnp.round(t)
        i0 = jnp.where(jnp.abs(t - tr) < SNAP_TOL, tr, jnp.floor(t))
    i0 = jax.lax.stop_gradient(i0)
    I = (i0 - shift).astype(jnp.int32)
    s = t - i0 + shift
    return I, s


def base_index(x, p, relpt, dx):
    """Just the base index.  See :func:`base_index_and_local`."""
    return base_index_and_local(x, p, relpt, dx)[0]


def stencil_offsets(dim, N):
    """(N**dim, dim) integer offsets in C-order, matching ravel_multi_index."""
    return np.array(list(itertools.product(range(N), repeat=dim)), dtype=np.int64)


class InterpPattern:
    """Static (geometry-independent) part of the extension matrix E.

    Holds the grid metadata and the stencil offsets.  Call :meth:`weights` with
    closest points to get the differentiable ``(cols, vals)`` pair.
    """

    def __init__(self, grid, p=None):
        self.grid = grid
        self.p = grid.p if p is None else p
        self.N = self.p + 1
        self.dim = grid.dim
        self.offsets = stencil_offsets(self.dim, self.N)  # (K, dim)
        self.K = self.offsets.shape[0]
        self._relpt = jnp.asarray(grid.relpt)
        self._dx = jnp.asarray(grid.dx)
        self._shape = np.array(grid.shape, dtype=np.int64)
        # C-order strides for ravel_multi_index
        self._strides = jnp.asarray(
            np.concatenate([np.cumprod(self._shape[::-1])[::-1][1:], [1]])
        )
        self._inv = jnp.asarray(grid.inv)
        self._offsets_j = jnp.asarray(self.offsets, dtype=jnp.int32)
        self._shape_j = jnp.asarray(self._shape, dtype=jnp.int32)

    def weights(self, cp):
        """Interpolation weights at closest points ``cp`` of shape (m, dim).

        Returns ``(cols, vals)`` with shapes (m, K) int32 and (m, K) float.
        ``cols`` are *band* indices; entries whose stencil point falls outside
        the band (or off the grid) get column 0 and weight 0, so the matvec is
        well defined.  Use :meth:`violations` to check that never happens.
        """
        cp = jnp.asarray(cp)
        # I and s come from one division; see the module docstring.
        I, s = base_index_and_local(cp, self.p, self._relpt, self._dx)

        w = lagrange_weights_1d(s[:, 0], self.N)  # (m, N)
        for d in range(1, self.dim):
            wd = lagrange_weights_1d(s[:, d], self.N)  # (m, N)
            w = (w[:, :, None] * wd[:, None, :]).reshape(w.shape[0], -1)
        vals = w  # (m, K), C-order matching stencil_offsets

        sub = I[:, None, :] + self._offsets_j[None, :, :]  # (m, K, dim)
        on_grid = jnp.all((sub >= 0) & (sub < self._shape_j), axis=-1)
        sub_c = jnp.clip(sub, 0, self._shape_j - 1)
        flat = jnp.sum(sub_c * self._strides.astype(jnp.int32), axis=-1)
        cols = self._inv[flat].astype(jnp.int32)  # (m, K), -1 if out of band
        ok = on_grid & (cols >= 0)
        cols = jnp.where(ok, cols, 0)
        vals = jnp.where(ok, vals, 0.0)
        return cols, vals

    def violations(self, cp):
        """Number of stencil entries that fall outside the band.

        Nonzero means the band is too narrow for this geometry -- either the
        Ruuth-Merriman estimate was not respected, or the surface has moved out
        of a band that was built for a different shape.
        """
        cp = np.asarray(cp)
        I = np.asarray(base_index(jnp.asarray(cp), self.p, self._relpt, self._dx))
        sub = I[:, None, :] + self.offsets[None, :, :]
        shape = self._shape
        on_grid = np.all((sub >= 0) & (sub < shape), axis=-1)
        sub_c = np.clip(sub, 0, shape - 1)
        flat = np.ravel_multi_index(tuple(sub_c.reshape(-1, self.dim).T), tuple(shape))
        cols = np.asarray(self.grid.inv)[flat].reshape(sub.shape[:-1])
        return int(np.sum(~(on_grid & (cols >= 0))))


def apply_sparse(cols, vals, u):
    """Apply a fixed-pattern sparse operator: ``(A u)_i = sum_k vals[i,k] u[cols[i,k]]``."""
    return jnp.sum(vals * u[cols], axis=1)


def apply_sparse_T(cols, vals, v, n):
    """Apply the transpose of the same operator, into a length-``n`` vector."""
    return jnp.zeros(n, dtype=v.dtype).at[cols].add(vals * v[:, None])
