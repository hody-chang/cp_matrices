# Python Package Plan

## Goal

Extend the MATLAB Closest Point Method project into a modern Python package,
starting with only the core files needed to build the interpolation matrix `E`
and Laplacian matrix `L`.

The first milestone is not a full port of the repository. It should reproduce
the essential MATLAB workflow used by the heat equation examples:

1. Build a Cartesian embedding grid.
2. Compute closest points on a simple surface.
3. Select a narrow computational band.
4. Build the interpolation matrix.
5. Build the finite-difference Laplacian matrix.

## Initial Scope

Port the smallest useful Closest Point Method core:

- Barycentric Lagrange interpolation weights.
- Interpolation stencil base-point selection.
- Sparse interpolation matrices in 2D, 3D, and general n-D.
- Sparse Laplacian matrices in 2D, 3D, and general n-D.
- Band-to-grid and grid-to-band index helpers.
- Ruuth-Merriman bandwidth helper.
- Minimal closest-point surfaces: circle and sphere.

## MATLAB Reference Files

Use these MATLAB files as the primary references:

- `cp_matrices/interp_matrix.m`
- `cp_matrices/interpn_matrix.m`
- `cp_matrices/interp2_matrix.m`
- `cp_matrices/interp3_matrix.m`
- `cp_matrices/LagrangeWeights1D_vec.m`
- `cp_matrices/findGridInterpBasePt_vec.m`
- `cp_matrices/laplacian_matrix.m`
- `cp_matrices/laplacian_2d_matrix.m`
- `cp_matrices/laplacian_3d_matrix.m`
- `cp_matrices/laplacian_nd_matrix.m`
- `cp_matrices/private/helper_diff_matrix2d.m`
- `cp_matrices/private/helper_diff_matrix3d.m`
- `cp_matrices/private/helper_diff_matrixnd.m`
- `cp_matrices/secondderiv_cen2_nd_matrices.m`
- `cp_matrices/make_invbandmap.m`
- `cp_matrices/rm_bandwidth.m`
- `surfaces/cpCircle.m`
- `surfaces/cpSphere.m`

## Existing Python Code

There is already a `python/` directory, but it is Python 2-era code and should
not be used as the foundation for the new package. Treat it as historical
context only.

Before implementing new Python code:

- Do not depend on existing modules under `python/`.
- Do not modernize the old Python tree as the main approach.
- Only inspect old Python files if a MATLAB behavior is ambiguous and a
  second historical reference would help.
- Prefer direct ports from the MATLAB reference files listed above.
- Avoid carrying over Python 2 syntax, obsolete packaging, or legacy API
  design.

## Proposed Python Package Shape

Create a small package with focused modules:

- `src/cpmatrices/__init__.py`
- `src/cpmatrices/interpolation.py`
- `src/cpmatrices/laplacian.py`
- `src/cpmatrices/bands.py`
- `src/cpmatrices/surfaces.py`
- `src/cpmatrices/grid.py`

Use:

- `numpy` for array operations.
- `scipy.sparse` for sparse matrices.
- `pytest` for tests.
- `pyproject.toml` for packaging.

## Porting Steps

1. Add modern package skeleton.
2. Implement band/index helpers.
3. Implement `lagrange_weights_1d`.
4. Implement interpolation base-point selection.
5. Implement `interp2_matrix`.
6. Implement `interp3_matrix`.
7. Implement `interpn_matrix`.
8. Implement second-order Laplacian builders.
9. Add fourth-order Laplacian support.
10. Add minimal circle and sphere closest-point functions.
11. Add one Python example matching the MATLAB heat-circle workflow.
12. Add tests comparing Python behavior to small MATLAB-derived expected values.

## Important Porting Notes

- MATLAB uses 1-based linear indices; Python must use 0-based indices.
- MATLAB 2D and 3D routines often use `meshgrid` ordering; n-D code uses
  `ndgrid` ordering. The Python API should make ordering explicit.
- Sparse matrix construction should use COO triplets first, then convert to CSR.
- The banded matrix behavior must be preserved: rows are built from `band1`,
  columns are restricted to `band2`.
- Interpolation weights should sum to 1 up to floating-point tolerance.
- The first implementation should prioritize correctness and parity over speed.

## Validation Plan

Start with small, focused tests:

- Lagrange weights at grid nodes.
- Lagrange weights for off-grid interpolation points.
- `interp2_matrix` full matrix versus banded matrix.
- `interp3_matrix` small-grid construction.
- `make_invbandmap` behavior.
- 2D Laplacian equals `Dxx + Dyy`.
- 3D Laplacian equals `Dxx + Dyy + Dzz`.
- One heat-circle smoke test using `E` and `L`.

If MATLAB is available, add parity fixtures generated from the MATLAB reference
functions. If MATLAB is not available, document that validation used translated
expected values only.

## Deferred Work

Do not include these in the first milestone:

- WENO interpolation.
- Grid refinement.
- Normals and orientation.
- CSG surfaces.
- Triangulated surfaces and MEX/Cython acceleration.
- PETSc integration.
- Full MATLAB example parity.
- Python package cleanup unrelated to the closest-point core.

## First Milestone API

The first milestone is successful when this workflow works in Python:

```python
from cpmatrices.interpolation import interp2_matrix
from cpmatrices.laplacian import laplacian_2d_matrix

E = interp2_matrix(x1d, y1d, cpx_band, cpy_band, degree=3, band=band)
L = laplacian_2d_matrix(x1d, y1d, order=2, band1=band, band2=band)
```

The resulting matrices should numerically match the MATLAB implementation for
small 2D circle examples.
