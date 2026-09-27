"""CPM operators with a fixed sparsity pattern, as JAX-traceable matvecs.

The discrete operator of the closest point method is built from two pieces:

  * ``L``: a standard Cartesian finite-difference Laplacian on the band.  Its
    pattern *and* its values are geometry-independent, so it is plain numpy.
  * ``E``: the closest-point extension (interpolation) matrix.  Its pattern is
    fixed by the band; its values depend differentiably on cp(x_i), hence on
    the surface.

Following Macdonald & Ruuth 2009 (and cp_matrices/lapsharp.m) we do not use the
naive product ``L E``, which has poor spectra for implicit time stepping.  Two
stabilizations are offered:

  ``diagsplit``  M = diag(L) + (L - diag(L)) E        [cp_matrices default]
  ``lapsharp``   M = L E - gamma (I - E),  gamma = 2*dim/dx^2

Both reduce to the surface Laplacian on functions constant along normals.

Out-of-band finite-difference neighbours are dropped, which imposes a Dirichlet
condition at the outer edge of the band.  This matches
examples/example_heat_circle.m and the note in
cp_matrices/laplacian_2d_matrix.m; widen the band (``extra_bw``) to push that
artificial boundary away from the surface.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from .interp import InterpPattern, apply_sparse


def laplacian_pattern(grid, order=2):
    """Centred finite-difference Laplacian over the band.

    Returns ``(cols, vals)``, both (n, S) numpy arrays, with column 0 holding
    the diagonal (self) entry.  Out-of-band neighbours carry weight 0.
    """
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")
    dim, n = grid.dim, grid.n
    dx = grid.dx

    if order == 2:
        taps = [(-1, 1.0), (1, 1.0)]
        self_w = -2.0
    else:
        taps = [(-2, -1.0 / 12.0), (-1, 4.0 / 3.0),
                (1, 4.0 / 3.0), (2, -1.0 / 12.0)]
        self_w = -5.0 / 2.0
        if grid.stenrad < 2:
            raise ValueError(
                "order 4 needs a stencil radius of 2; rebuild the band with "
                "stenrad=2 so the Ruuth-Merriman radius accounts for it"
            )

    cols = [np.arange(n, dtype=np.int64)]
    vals = [np.full(n, sum(self_w / dx[d] ** 2 for d in range(dim)))]
    for d in range(dim):
        for shift, w in taps:
            off = np.zeros(dim, dtype=np.int64)
            off[d] = shift
            c = grid.neighbour_band_index(off)
            ok = c >= 0
            cols.append(np.where(ok, c, 0))
            vals.append(np.where(ok, w / dx[d] ** 2, 0.0))
    return np.stack(cols, axis=1), np.stack(vals, axis=1)


def laplacian_dropped(grid, order=2):
    """How many finite-difference neighbours were dropped at the band edge."""
    cols, vals = laplacian_pattern(grid, order)
    return int(np.sum(vals[:, 1:] == 0.0))


@dataclasses.dataclass
class CPMOperator:
    """The banded CPM operator, as a callable linear map.

    ``matvec(u)`` evaluates

        A u = alpha * M u + c * u

    where ``M`` is the stabilized surface Laplacian and ``alpha``, ``c`` are
    (n,) arrays.  This covers the model problems we care about:

        -Delta_s u + c u = b     ->  alpha = -1
        a Delta_s u - c u = b    ->  alpha = a, c = -c

    Only ``e_vals`` (and optionally ``alpha``, ``c``) carry gradients; the
    pattern arrays are constants.
    """

    e_cols: jnp.ndarray
    e_vals: jnp.ndarray
    l_cols: jnp.ndarray
    l_vals: jnp.ndarray
    n: int
    stabilization: str = "diagsplit"
    gamma: float = 0.0
    alpha: jnp.ndarray = 1.0
    c: jnp.ndarray = 0.0

    def extend(self, u):
        """Closest point extension E u."""
        return apply_sparse(self.e_cols, self.e_vals, u)

    def surface_laplacian(self, u):
        """M u, the stabilized discrete surface Laplacian."""
        Eu = self.extend(u)
        if self.stabilization == "diagsplit":
            diag = self.l_vals[:, 0]
            off = jnp.sum(self.l_vals[:, 1:] * Eu[self.l_cols[:, 1:]], axis=1)
            return diag * u + off
        elif self.stabilization == "lapsharp":
            LEu = apply_sparse(self.l_cols, self.l_vals, Eu)
            return LEu - self.gamma * (u - Eu)
        elif self.stabilization == "none":
            return apply_sparse(self.l_cols, self.l_vals, Eu)
        raise ValueError(f"unknown stabilization {self.stabilization!r}")

    def matvec(self, u):
        return self.alpha * self.surface_laplacian(u) + self.c * u

    def __call__(self, u):
        return self.matvec(u)

    def to_dense(self):
        """Materialize A as a dense (n, n) array.  For tests and eigenvalues."""
        return jax.vmap(self.matvec)(jnp.eye(self.n)).T


def build_operator(grid, cp, *, pattern=None, order=2, stabilization="diagsplit",
                   alpha=1.0, c=0.0, l_pattern=None):
    """Assemble a :class:`CPMOperator` for closest points ``cp``.

    ``cp`` is (n, dim) and may be a traced JAX array -- that is the point.
    ``pattern`` / ``l_pattern`` let callers hoist the static work out of a loop.
    """
    if pattern is None:
        pattern = InterpPattern(grid)
    if l_pattern is None:
        l_pattern = laplacian_pattern(grid, order)
    l_cols, l_vals = l_pattern
    e_cols, e_vals = pattern.weights(cp)
    # gamma = -diag(L) makes 'lapsharp' algebraically identical to 'diagsplit':
    #   diag(L) u + (L - diag(L)) E u = L E u - gamma (I - E) u   when gamma = -diag(L).
    # Taking it from the assembled L rather than from a formula keeps that true
    # for any order and any anisotropic spacing.
    gamma = float(-l_vals[0, 0])
    return CPMOperator(
        e_cols=e_cols,
        e_vals=e_vals,
        l_cols=jnp.asarray(l_cols, dtype=jnp.int32),
        l_vals=jnp.asarray(l_vals),
        n=grid.n,
        stabilization=stabilization,
        gamma=gamma,
        alpha=alpha,
        c=c,
    )
