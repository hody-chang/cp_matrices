"""Banded Cartesian grids for the closest point method.

The band is *static structure*: linear indices into a uniform Cartesian grid,
plus the inverse map from grid index to band index.  It is built once from a
nominal geometry and then held fixed while the geometry moves during an
optimization.  Keeping it fixed is what makes the sparsity pattern of the CPM
operators constant, which is what lets the whole solve be traced by JAX.

Conventions follow ../cp_matrices (MATLAB):

  * grids are ndgrid-style tensor products of 1D vectors ``x1d[0] x x1d[1] ...``
  * linear indices are C-order (``numpy.ravel_multi_index``) over ``shape``
  * the band radius is the Ruuth-Merriman estimate, see ``rm_bandwidth``
"""

import dataclasses
from math import sqrt

import numpy as np


def rm_bandwidth(dim, p, stenrad=1, safety=1.0001):
    """Ruuth-Merriman band radius in units of dx.

    Matches cp_matrices/rm_bandwidth.m:

        bw = safety * sqrt((dim-1)*((p+1)/2)^2 + (stenrad + (p+1)/2)^2)

    A grid point within ``bw*dx`` of the surface has the property that the
    interpolation stencils of all its finite-difference neighbours are
    themselves inside the band, so ``L @ E`` needs no data from outside.
    """
    half = (p + 1) / 2.0
    return safety * sqrt((dim - 1) * half**2 + (stenrad + half) ** 2)


@dataclasses.dataclass(frozen=True)
class BandedGrid:
    """A uniform Cartesian grid together with a fixed band of active points."""

    x1d: tuple  # tuple of 1D numpy arrays, one per dimension
    band: np.ndarray  # (n,) int64 C-order linear indices into the full grid
    inv: np.ndarray  # (prod(shape),) int64, band index or -1
    p: int  # interpolation degree the band was sized for
    stenrad: int  # finite-difference stencil radius the band was sized for
    bw: float  # band radius actually used, in units of dx

    # ---- derived quantities -------------------------------------------------

    @property
    def dim(self):
        return len(self.x1d)

    @property
    def shape(self):
        return tuple(len(v) for v in self.x1d)

    @property
    def n(self):
        return len(self.band)

    @property
    def dx(self):
        """Per-dimension grid spacing, as a (dim,) array."""
        return np.array([v[1] - v[0] for v in self.x1d])

    @property
    def relpt(self):
        """Reference point, the grid corner corresponding to index (0,...,0)."""
        return np.array([v[0] for v in self.x1d])

    @property
    def required_bw(self):
        """The Ruuth-Merriman radius this band *must* have, in units of dx.

        Distinct from ``bw``, which is what the band actually has: ``bw`` may be
        larger by ``extra_bw`` slack, deliberately, so a moving surface stays
        covered.  Clearance is measured against this requirement, not against the
        slack.
        """
        return rm_bandwidth(self.dim, self.p, self.stenrad)

    @property
    def sub(self):
        """(n, dim) integer subscripts of the band points."""
        return np.stack(np.unravel_index(self.band, self.shape), axis=1)

    @property
    def xg(self):
        """(n, dim) coordinates of the band points."""
        return self.relpt + self.sub * self.dx

    def uncovered_near_surface(self, dist_fun, safety=1.0, chunk=200000):
        """Grid points within the band radius of a surface but outside the band.

        ``InterpPattern.violations`` checks only that the interpolation stencils
        of the *band* points are inside the band.  That is necessary but not
        sufficient for a surface that has moved: the finite-difference Laplacian
        drops out-of-band neighbours, imposing a Dirichlet condition at the outer
        edge of the band, and the surface needs clearance from that edge, not
        merely from the point where interpolation fails.

        This is the sufficient check.  It asks the question the band was built to
        answer -- "is every grid point within the Ruuth-Merriman radius of the
        surface in the band?" -- for a *new* surface, given as a distance
        function.  A nonzero count means results near the surface are
        contaminated by the artificial boundary, even if nothing has failed
        outright.  ``safety`` below 1 asks for less clearance than the
        Ruuth-Merriman radius.

        The radius used is ``required_bw``, not ``bw``: a band widened by
        ``extra_bw`` has slack to spend on surface motion, and measuring against
        its own inflated radius would spend the slack on nothing.
        """
        radius = safety * self.required_bw * float(np.max(self.dx))
        return int(np.sum(self._outside_distances(dist_fun, chunk) <= radius))

    def _outside_distances(self, dist_fun, chunk=200000):
        """Distances to the surface, for grid points *not* in the band.

        Computed once and returned as a flat array, because ``dist_fun`` can be a
        neural network evaluated through a Newton solve -- expensive enough that
        ``clearance`` must not call it once per safety level.
        """
        total = int(np.prod(self.shape))
        out = []
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            idx = np.arange(start, stop)
            idx = idx[self.inv[idx] < 0]
            if idx.size == 0:
                continue
            sub = np.stack(np.unravel_index(idx, self.shape), axis=1)
            out.append(np.asarray(dist_fun(self.relpt + sub * self.dx)))
        return np.concatenate(out) if out else np.zeros(0)

    def stale_fraction(self, dist_fun, chunk=200000):
        """Fraction of band points that are too *far* from the surface to be valid.

        The converse of :meth:`clearance`, and the check that matters when one
        fixed band is reused across a family of shapes.  ``clearance`` asks
        whether every grid point near the surface is in the band -- sufficiency.
        This asks whether every point in the band is near the surface -- validity.
        A band can pass one and fail the other, and a band built as the union over
        a family of shapes routinely does.

        A band point farther than the Ruuth-Merriman radius from the surface has
        no meaningful closest point extension: it lies outside the reach the method
        assumes, and its row of the operator is noise.  The consequence is not a
        mild loss of accuracy.  Measured on two circles whose separation is the
        shape parameter, a union band carrying 51.7% stale points produced a
        *spurious* eigenvalue of ``-Delta_s + 1`` that fell to 0.055 at an isolated
        separation, against a true smallest eigenvalue of exactly 1 (the constants
        on each component).  The solution there grew by a factor of 7 and the
        objective by a factor of 27, so the exact adjoint gradient spiked by orders
        of magnitude at that shape.  Rebuilding the band as the intersection over
        the family removed the spurious mode entirely -- and left clearance at
        0.000, i.e. not covering the surface at all.

        That is the moving-band problem in two numbers: over a family whose shapes
        move by more than about a band width, the union is sufficient but invalid
        and the intersection is valid but insufficient.  Check both, and treat
        ``min |eig(A)| < 1`` as evidence of a spurious mode.
        """
        radius = self.required_bw * float(np.max(self.dx))
        d = np.asarray(dist_fun(self.xg))
        return float(np.mean(d > radius))

    def clearance(self, dist_fun, chunk=200000):
        """How much of the Ruuth-Merriman radius the band covers, as a fraction.

        Sufficiency only.  See :meth:`stale_fraction` for the converse, which a
        band reused across several shapes can fail while this passes.

        1.0 means the surface has full clearance: every grid point within the
        Ruuth-Merriman radius of it is in the band.  Less than 1.0 says how far
        out the band does reach, so 0.8 means the artificial Dirichlet edge has
        come inside the outer 20% of the radius the discretization wants.

        Exact rather than a search over levels: the answer is just the distance
        from the surface to the nearest out-of-band grid point, in units of the
        radius, capped at 1.0 since more slack than required is still full
        clearance.
        """
        d = self._outside_distances(dist_fun, chunk)
        if d.size == 0:
            return 1.0
        radius = self.required_bw * float(np.max(self.dx))
        return float(min(1.0, d.min() / radius))

    def neighbour_band_index(self, offset):
        """Band index of each band point shifted by integer ``offset``.

        Returns ``-1`` where the neighbour falls outside the band or off the
        grid.  Used to build finite-difference matrices.
        """
        sub = self.sub + np.asarray(offset, dtype=np.int64)
        shape = np.array(self.shape)
        inside = np.all((sub >= 0) & (sub < shape), axis=1)
        out = np.full(self.n, -1, dtype=np.int64)
        if np.any(inside):
            flat = np.ravel_multi_index(tuple(sub[inside].T), self.shape)
            out[inside] = self.inv[flat]
        return out


def make_grid1d(lo, hi, dx):
    """1D grid vector covering [lo, hi] with spacing (close to) dx.

    The endpoint is adjusted outwards so the spacing is exactly ``dx``.
    """
    m = int(np.ceil((hi - lo) / dx))
    return lo + dx * np.arange(m + 1)


def band_from_dist(x1d, dist_fun, p=3, stenrad=1, safety=1.0001, extra_bw=0.0,
                   chunk=200000):
    """Build a band from a distance function evaluated on the whole grid.

    ``dist_fun`` takes an (m, dim) array of points and returns (m,) distances to
    the surface.  ``extra_bw`` widens the band (in units of dx) beyond the
    Ruuth-Merriman estimate; use it when the geometry will move during an
    optimization so that the moving surface stays inside a fixed band.
    """
    x1d = tuple(np.asarray(v, dtype=float) for v in x1d)
    dim = len(x1d)
    shape = tuple(len(v) for v in x1d)
    dx = np.array([v[1] - v[0] for v in x1d])
    relpt = np.array([v[0] for v in x1d])
    bw = rm_bandwidth(dim, p, stenrad, safety) + extra_bw
    radius = bw * float(np.max(dx))

    total = int(np.prod(shape))
    keep = np.zeros(total, dtype=bool)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        idx = np.arange(start, stop)
        sub = np.stack(np.unravel_index(idx, shape), axis=1)
        pts = relpt + sub * dx
        keep[start:stop] = np.asarray(dist_fun(pts)) <= radius

    band = np.flatnonzero(keep).astype(np.int64)
    if band.size == 0:
        raise ValueError(
            "empty band: the surface does not intersect the grid, or dx is too "
            "coarse relative to the band radius"
        )
    inv = np.full(total, -1, dtype=np.int64)
    inv[band] = np.arange(band.size, dtype=np.int64)
    return BandedGrid(x1d=x1d, band=band, inv=inv, p=p, stenrad=stenrad, bw=bw)


def band_from_cp(x1d, cp_fun, **kwargs):
    """Build a band from a closest point function.

    ``cp_fun`` maps (m, dim) points to (m, dim) closest points; the distance is
    taken as ``|cp(x) - x|``.  This is the honest CPM notion of distance and
    works for a learned SDF whose ``|grad f|`` is not 1.
    """

    def dist_fun(pts):
        cp = np.asarray(cp_fun(pts))
        return np.linalg.norm(cp - pts, axis=1)

    return band_from_dist(x1d, dist_fun, **kwargs)
