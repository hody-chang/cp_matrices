# diffcpm — a differentiable closest point method layer

A JAX implementation of the closest point method whose solve is differentiable
with respect to both the PDE coefficients and the **geometry**, including the
weights of a neural implicit surface.

It exists to support PDE-constrained optimization directly on implicit
surfaces: given observations of a surface PDE solution, recover a coefficient,
a source, or the shape itself, with no meshing step anywhere in the pipeline.

```
                theta (physics)        phi (geometry, e.g. SIREN weights)
                     |                          |
                     |                    cp(x_i; phi)          <- IFT
                     |                          |
                     v                          v
                 b(theta)                interpolation weights E
                     |                          |
                     +------ A u = b -----------+                <- custom_linear_solve
                              |
                         observe(u)
                              |
                      J = ||observe(u) - data||^2 + reg
```

Reverse-mode differentiation of `J` gives `dJ/dtheta` and `dJ/dphi` at the cost
of one extra linear solve, via the adjoint, with no unrolling.

## Why the closest point method for this

A neural implicit surface changes shape during optimization. Mesh FEM would need
marching cubes (not differentiable, changes topology) then remeshing, each step.
CPM needs only closest-point queries on a fixed Cartesian grid: the band and the
sparsity pattern stay fixed, the operator is assembled from interpolation
weights that are *polynomials* in the closest point, and the geometry gradient
therefore falls out of the same autodiff pass as everything else.

## Layout

| file | contents |
|---|---|
| `grid.py` | banded Cartesian grids, Ruuth–Merriman bandwidth, band construction |
| `interp.py` | differentiable Lagrange interpolation weights, the extension matrix `E` |
| `operators.py` | banded Laplacian `L`, stabilized CPM operator, `alpha*M + c` |
| `solve.py` | `lax.custom_linear_solve` wrapper: dense LU or matrix-free GMRES |
| `sdf.py` | closest points on a level set via Newton + IFT; a small SIREN; Adam |
| `surfaces.py` | circle/sphere/torus closed forms, ellipsoid via the implicit solver |
| `inverse.py` | observation operator, misfit, regularizers, finite-difference checker |
| `tests/test_diffcpm.py` | the verification suite |
| `../examples_diffcpm/` | runnable experiments |

Conventions match the MATLAB `cp_matrices/` in this repository: ndgrid tensor
grids, C-order linear indices, the Ruuth–Merriman band radius of
`rm_bandwidth.m`, the base-point convention of `findGridInterpBasePt.m`, and the
diagonal-split stabilization of `lapsharp.m`.

## How the gradients are obtained

**The linear solve.** `jax.lax.custom_linear_solve` is given the forward matvec,
a solver, and a solver for the transpose. Reverse mode then produces exactly

```
forward   A u = b
adjoint   A^T lam = dJ/du
gradient  dJ/dp = lam^T (db/dp - (dA/dp) u)
```

with the last line assembled by JAX from the VJP of the matvec closure. Any
quantity the matvec depends on — coefficients, sources, interpolation weights,
hence the geometry — gets a gradient with no further code. Memory does not grow
with the iteration count, and the gradient is the exact adjoint of the
*discrete* operator rather than of the iteration that approximated it.
`tests/test_diffcpm.py::test_custom_linear_solve_matches_hand_adjoint` checks
this against the three lines above assembled explicitly, to 1e-11, with no
finite differences involved.

**The geometry.** Closest points on `{f(y; phi) = 0}` satisfy

```
y - x = mu grad f(y),    f(y) = 0
```

which we solve by Newton in `d+1` unknowns and differentiate with
`jax.lax.custom_root`, i.e. by the implicit function theorem at the converged
root. Two details that are easy to get wrong:

* The usual projection `cp(x) = x - f grad f/|grad f|^2`, iterated, is **not** the
  closest point unless `f` is an exact distance function. It follows the gradient
  flow of `f` and lands somewhere on the zero level set. For a learned SDF, where
  `|grad f| != 1`, the difference is an O(SDF error) perturbation of the geometry
  itself. We use it only as a Newton initial guess and then enforce the
  conditions above, which give the true closest point for *any* `f` with the
  right zero set.
* Differentiating the Newton *iterates* would give the gradient of the
  approximation instead of an approximation of the gradient. `custom_root`
  sidesteps this; `sdf.level_set_residual_norm` is provided to confirm the root
  really converged.

## A numerical trap worth knowing about

The obvious way to locate an interpolation stencil is

```
I = floor((x - relpt)/dx) - (p-1)/2
s = (x - (relpt + I*dx)) / dx
```

This divides by `dx` twice, rounding the same real number twice. When a closest
point lands on a grid line — routine for a symmetric surface on a symmetric grid
— the two roundings can straddle the integer, and then `I` and `s` describe
stencils **one cell apart**. The interpolated value is then wrong by O(1) at
those points. Worse, whether it happens depends on how the compiler
reassociates the arithmetic, so the same code gives different answers under
`jax.jit` than eagerly. We hit this as an 85% error in the solution on the unit
circle at `dx = 0.1`, visible only under `jit`.

`interp.base_index_and_local` computes `s = t - i0 + shift` from the same `t` and
`i0`, which makes them consistent at any rounding, and snaps `i0` to a nearby
integer so the stencil choice at a grid line is deterministic. Also,
`lagrange_weights_1d` uses the nodal form `prod_{m!=j}(s-m)/(j-m)` rather than
the barycentric form of `LagrangeWeights1D.m`: the barycentric form divides by
`x - x_j`, which MATLAB special-cases with a branch returning binary weights —
correct for evaluation, but under autodiff that branch returns a *constant* and
so a wrong (zero) derivative exactly at grid lines. The nodal form is a
polynomial with no removable singularity, exact and smoothly differentiable
everywhere, at O(N^2) instead of O(N) for `N = p+1 <= 8`.

## What is verified

`python diffcpm/tests/test_diffcpm.py` (or under pytest). All of the following
are assertions in that file, not claims:

* `E` reproduces polynomials of degree `<= p` at the closest points to 1e-11, and
  `apply_sparse_T` is the exact transpose of `apply_sparse`.
* The forward solve of `(-Delta_s + 1)u = cos(k theta)` on a circle converges at
  second order near the surface.
* `custom_linear_solve` reproduces the hand-assembled adjoint to 1e-11.
* Dense-LU and matrix-free GMRES agree on both value and gradient to ~1e-11.
* `dJ/dtheta` matches central differences to ~1e-9 for a source in a Fourier
  basis and for a nodal reaction field.
* `dJ/d(shape)` matches central differences for a circle radius, ellipse axes
  (through the implicit-solver/IFT path), torus radii, and a 3D sphere radius.
* `dJ/dphi` matches central differences in random SIREN weight-space directions.
* The discrete shape derivative converges, under grid refinement, to the exact
  continuous shape derivative of `J(R) = 1/2 int_Gamma u^2 ds` on a circle.

The `diagsplit` and `lapsharp` stabilizations agree to machine precision for a
second-order Laplacian on an equal-spacing grid, since there
`gamma = -diag(L) = 2 dim/dx^2` makes them algebraically identical; the
unstabilized product `L E` has a positive eigenvalue on the same grid, which is
why neither is optional.

## Known limitations

These are real, and the first two are where the research content of the proposal
actually lies.

**Moving bands / stencil crossings.** The band and the stencil *choice* are
computed under `stop_gradient`. As the surface moves, closest points cross
stencil-cell boundaries and grid points enter and leave the band. With `I` and
`s` consistent the objective is continuous across a crossing, but its derivative
is not: `J` is continuous and only piecewise `C^1` in the shape parameters, with
kinks at a density of roughly `n_band/dx` per unit shape perturbation. This is
directly visible in `inverse.check_gradient`: the finite-difference error
plateaus near 1e-5 for steps of 1e-4 to 1e-5 and then falls to 1e-10 once the
step is small enough that the interval contains no crossing. Consequences:

* Finite-difference verification of geometry gradients must use a step below the
  crossing scale, which is why `check_gradient` sweeps steps and reports the
  best. This is a property of the discretization, not of the checker.
* A gradient-based shape optimizer will see small derivative discontinuities. In
  the examples this does not prevent convergence, but it caps the accuracy a
  line search can usefully ask for. A principled fix — a fixed wide band with
  smoothly tapered weights, or blended stencils near cell boundaries — is not
  implemented here.

**The band is fixed, and sized for the initial shape.** `extra_bw` widens it so a
moving surface stays inside. `InterpPattern.violations` and
`operators.laplacian_dropped` report when that fails; nothing detects it
automatically inside `jit`, where out-of-band stencil entries are silently
zero-weighted. Check `violations == 0` explicitly, as the examples do.

**Band-edge boundary condition.** Out-of-band finite-difference neighbours are
dropped, imposing a Dirichlet condition at the outer edge of the band. This
matches `examples/example_heat_circle.m` and the note in
`laplacian_2d_matrix.m`, and it makes the solution near the *outer band edge*
meaningless — the errors quoted above are deliberately measured near the
surface. Widen the band to push that boundary away.

**Scale.** `solve.py` offers dense LU (exact, `O(n^3)`) and unpreconditioned
restarted GMRES. Neither is the right answer at 3D production sizes; CPM
multigrid (Chen & Macdonald) or a preconditioned Krylov method would be. The
`dense_limit` heuristic in `linear_solve` picks dense below 2500 unknowns.

**Open surfaces** need unsigned distance fields, whose gradients vanish on the
surface, so the Newton system for the closest point degenerates there. Not
handled.

**Ill-posedness.** Nothing here decides how much regularization an inverse
problem needs, or whether a parameter is identifiable from given data. The
examples show the effect but do not resolve it.

## Requirements

`jax`, `jaxlib`, `numpy`; `scipy` for the L-BFGS driver in the examples. Float64
is required — every module assumes `jax.config.update("jax_enable_x64", True)`,
which the tests and examples set.
