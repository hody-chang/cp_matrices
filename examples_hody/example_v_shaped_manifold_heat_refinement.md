# Branch-Aware CPM Heat Refinement on a V-Shaped Manifold

This note documents the current V-shaped-manifold heat-equation experiment
implemented in:

```matlab
example_v_shaped_manifold_heat_refinement_branch_cpm
```

The purpose of the example is to test a Closest Point Method (CPM)
discretization on a two-branch embedded curve with a shared vertex.  The key
idea is that the two arms are treated as separate labeled branches, so the
CPM extension on one arm cannot accidentally interpolate through the other
arm when the V becomes narrow or nearly vertical.  The two labels communicate
only through a special graph-Laplacian update at the common vertex.

Run the example from the repository root with:

```matlab
addpath('examples')
[results, diagnostics] = example_v_shaped_manifold_heat_refinement_branch_cpm();
```

For a shorter diagnostic run:

```matlab
opts = struct( ...
    'anglesDeg', [90 150 170], ...
    'hvals', 1 ./ [40 80 160], ...
    'makePlots', true, ...
    'showDiagnostics', true);
[results, diagnostics] = example_v_shaped_manifold_heat_refinement_branch_cpm(opts);
```

The default parameters are:

```text
anglesDeg    = [90 120 150 170]
hvals        = 1 ./ [200 400 800 1600 3200]
finalTime    = 0.01
kappa        = 1
cfl          = 0.1
interpDegree = 3
lapOrder     = 2
```

## Methodology Summary

The computation is a branch-aware CPM discretization of heat flow on a metric
graph embedded in `R^2`.

1. Split the V into a left branch and a right branch.
2. Build a separate Cartesian grid, closest-point map, narrow band,
   extension matrix, Cartesian Laplacian, and restriction matrix for each
   branch.
3. Assemble each branch CPM Laplacian as

   ```text
   LCPM = R * Lcart * Eband.
   ```

4. Correct the rows near branch endpoints where closest-point endpoint
   clamping would otherwise impose the wrong local behavior.
5. Advance all non-vertex branch nodes with the corrected branch CPM
   operators.
6. Advance the shared vertex with a two-neighbor graph-Laplacian formula.
7. Store the same vertex value on both branch arrays so continuity is imposed
   exactly at every time step.

This is different from the older implementation that first advanced the two
single-branch CPM problems and then averaged the vertex rows.  That earlier
procedure made the shared vertex behave like a Neumann endpoint inside each
single-branch calculation, and this endpoint artifact polluted nearby rows.
The current implementation removes that source of error by replacing
endpoint-affected rows and by excluding the vertex row from the branch CPM
time update.

## Geometry

For each opening angle `angleDeg = alpha`, define

```text
theta = alpha / 2.
```

The embedded V has one shared vertex and two unit-length branches:

```text
left endpoint   p_L = (-cos(theta), sin(theta))
shared vertex   v   = (0, 0)
right endpoint  p_R = ( cos(theta), sin(theta)).
```

The left branch is parameterized by arclength `s in [-1,0]`:

```text
X_L(s) = (s cos(theta), -s sin(theta)).
```

The right branch is parameterized by arclength `s in [0,1]`:

```text
X_R(s) = (s cos(theta), s sin(theta)).
```

Thus `s = 0` is the shared vertex on both branches.  The left array is stored
in MATLAB order

```text
sLeft = [0, -h, -2h, ..., -1]^T,
```

and the right array is stored as

```text
sRight = [0, h, 2h, ..., 1]^T.
```

This ordering makes index `1` the shared vertex on both branches.  The first
non-vertex neighbors used by the graph update are `uLeft(2)` and
`uRight(2)`.

Because each branch is unit speed, the surface Laplacian on either straight
branch is simply the second derivative with respect to arclength:

```text
Delta_Gamma u = d^2 u / ds^2.
```

## PDE and Manufactured Solution

The solved equation is

```text
u_t = kappa * Delta_Gamma u.
```

The manufactured solution is

```text
u_exact(s,t)
  = 11/2 - (9/2) exp(-kappa (pi/2)^2 t) cos(pi (s+1) / 2).
```

The initial condition is the exact solution at `t = 0`:

```text
u(s,0) = 11/2 - (9/2) cos(pi (s+1) / 2).
```

This gives

```text
u(-1,0) = 1,
u( 0,0) = 5.5,
u( 1,0) = 10.
```

The derivative vanishes at the two physical endpoints:

```text
u_s(-1,t) = 0,
u_s( 1,t) = 0.
```

The solution is smooth through the shared vertex when viewed as a function on
the full coordinate interval `[-1,1]`.  Therefore it satisfies both the
continuity condition

```text
u_L(0,t) = u_R(0,t) = u_v(t)
```

and the Kirchhoff/flux condition at the junction.  This makes it a useful
manufactured solution for checking whether the discretization transfers heat
through the vertex without allowing cross-branch shortcutting away from the
vertex.

## Narrow-Band Construction

The narrow band is built separately for each labeled branch.

For one branch, `buildBranch` first constructs a Cartesian tensor-product
grid around that branch only.  The branch bounding box is padded by

```text
padding = max(0.08, (bw + interpDegree + 2) * h)
```

unless a custom padding is passed in `opts.padding`.  The CPM bandwidth is
computed by

```matlab
bw = rm_bandwidth(2, interpDegree, lapOrder / 2);
```

For each Cartesian grid point `(x,y)`, the code calls

```matlab
[cpx, cpy, dist, bdy, t] = cpLineSegment2d(xx, yy, p0, p1);
```

where `(p0,p1)` is the line segment for the current branch.  The narrow band
is then

```matlab
band = find(abs(dist) <= bw * h);
```

Only these band points participate in the branch-local CPM matrices.

The closest-point parameter `t` returned by `cpLineSegment2d` is converted
to the branch arclength coordinate:

```matlab
if side < 0
    sBand = -1 + t(band);   % left branch, endpoint to vertex
else
    sBand = t(band);        % right branch, vertex to endpoint
end
```

This gives each band point a closest point on that same branch only.  No band
point on the left branch is allowed to interpolate from right-branch unknowns,
and no right-branch band point is allowed to interpolate from left-branch
unknowns.

## Extension Matrix on One Branch

The branch extension matrix is

```matlab
Eband = interp1BranchMatrix(sNodes, sBand, interpDegree);
```

It maps branch-node values to values at the closest points of the Cartesian
band.  The interpolation is one-dimensional barycentric Lagrange
interpolation in the branch arclength coordinate `s`.

At band points whose closest point is the shared vertex, the extension row is
overwritten:

```matlab
Eband(vertexMask, :) = 0;
Eband(vertexMask, 1) = 1;
```

Since index `1` is the vertex on both branches, this makes the vertex fiber
use the branch's stored vertex value exactly.  The default `vertexTol = 0`
selects rows whose closest point is the origin up to roundoff; if no such row
is detected, the code keeps the band row whose closest-point coordinate is
nearest to the vertex.

The diagnostic check

```matlab
extensionUsesOnlyOwnBranch(branch)
```

verifies that `Eband` uses only columns belonging to the branch being built,
and that vertex-fiber rows use only the vertex column.

## Laplacian Matrix Built from the Band

For a branch, the Cartesian finite-difference Laplacian on the narrow band is
assembled by

```matlab
Lcart = laplacian_2d_matrix(x1d, y1d, lapOrder, band, band);
```

The restriction/interpolation matrix from band values back to the ordered
branch nodes is

```matlab
[xManifold, yManifold] = branchCoordinates(side, sNodes, angleDeg);
R = interp2_matrix(x1d, y1d, xManifold, yManifold, interpDegree, band);
```

The raw closest-point Laplacian on the branch nodes is then

```matlab
LCPM_raw = R * Lcart * Eband.
```

Conceptually, for a branch-node vector `u`, this applies the CPM sequence:

```text
branch values
  -> extend to Cartesian band by closest-point interpolation
  -> apply Cartesian Laplacian on the band
  -> restrict/interpolate the result back to branch nodes.
```

For a smooth function on a straight unit-speed branch, this approximates
`d^2 u / ds^2`.

## What Happens in a Single-Branch Calculation

If one branch is considered by itself, then the shared vertex is just an
endpoint of that segment.  The closest-point map for a line segment clamps
points beyond either endpoint back to the endpoint.  In CPM this clamping
acts like a homogeneous Neumann endpoint treatment.

That behavior is correct at the physical insulated endpoint, but it is not
correct at the shared vertex of the original V-shaped manifold.  In the
original manifold, the vertex is an interior junction where heat can pass
between the two branches.  Treating it as a single-branch Neumann endpoint
would block that communication.

The current algorithm therefore uses the single-branch CPM matrices only as
branch-local ingredients.  It does not trust the raw single-branch CPM row at
the shared vertex.  The vertex is advanced by a graph-junction update, and
the nearby endpoint-affected rows are replaced as described next.

## Endpoint-Affected Row Correction

The graph update at the vertex is necessary but not sufficient by itself.
The raw CPM matrix near an endpoint is influenced by closest-point clamping
and interpolation support.  Therefore the row immediately at the vertex is
not the only affected row; the first few neighboring rows can also inherit
single-branch endpoint behavior.

The current code fixes this in

```matlab
replaceEndpointAffectedRows
```

for `lapOrder == 2`.  It sets

```matlab
capRows = max(3, interpDegree + lapOrder / 2);
```

and replaces the rows in the endpoint caps.  For rows inside the branch but
close to either endpoint, the replacement stencil is the standard
one-dimensional second difference:

```text
(u_{j-1} - 2 u_j + u_{j+1}) / h^2.
```

In matrix form:

```text
L(j,j-1) =  1 / h^2
L(j,j)   = -2 / h^2
L(j,j+1) =  1 / h^2.
```

At the physical endpoint, the homogeneous Neumann closure is

```text
2 (u_{n-1} - u_n) / h^2,
```

implemented as:

```text
L(n,n-1) =  2 / h^2
L(n,n)   = -2 / h^2.
```

The shared vertex row itself is not used in the branch update.  It is
replaced by the graph update below.

This row correction is the main accuracy change relative to the earlier
study.  The vertex formula alone is a second-order graph stencil, but the
old branch CPM operator still carried endpoint-clamping errors into rows
near the shared vertex.  Replacing the endpoint-affected rows makes the
branch interiors and the graph vertex closure consistent with the intended
one-dimensional metric-graph discretization.

## Time Integration and Vertex Coupling

The time integration is explicit Euler.  The time step is chosen from

```text
dt_initial = cfl * h^2 / kappa
```

and then adjusted so that an integer number of steps lands exactly on
`finalTime`.

At the beginning of each step, the two stored vertex values are averaged:

```matlab
uv = 0.5 * (uLeft(1) + uRight(1));
uLeft(1) = uv;
uRight(1) = uv;
```

This enforces continuity before applying the operators.

The corrected branch CPM operators are applied:

```matlab
lapLeft  = left.LCPM  * uLeft;
lapRight = right.LCPM * uRight;
```

The non-vertex branch nodes are then updated by

```matlab
uLeftNew(2:end)  = uLeft(2:end)  + dt * kappa * lapLeft(2:end);
uRightNew(2:end) = uRight(2:end) + dt * kappa * lapRight(2:end);
```

The shared vertex is updated separately:

```matlab
uvNew = uv + dt * kappa * (uLeft(2) + uRight(2) - 2 * uv) / h^2;
```

Then the same value is written to both branches:

```matlab
uLeftNew(1) = uvNew;
uRightNew(1) = uvNew;
```

In mathematical notation, if `u_{L,1}` and `u_{R,1}` are the first
non-vertex neighbors on the left and right branches, this is

```text
du_v/dt = kappa * (u_{L,1} + u_{R,1} - 2 u_v) / h^2.
```

This is the central second-difference at the vertex on the combined
arclength grid

```text
-1, ..., -h, 0, h, ..., 1.
```

It imposes a single continuous vertex value and gives the discrete
Kirchhoff/flux balance expected for a two-edge graph junction.  At steady
state, the numerator

```text
u_{L,1} + u_{R,1} - 2 u_v
```

vanishes, which is the centered discrete form of matching the two one-sided
slopes at the vertex.

## Error, Mass, and Vertex Diagnostics

After the final time, the code compares against the exact solution on the
branch nodes.  The duplicated vertex is counted once in the infinity norm:

```matlab
uniqueError = [errorLeft; errorRight(2:end)];
finalInfError = max(abs(uniqueError));
```

The reported `FinalL2Error` uses composite-trapezoid weighting on the two
branches without double-counting the shared vertex:

```matlab
sqrt(h * (sum(errorLeft(2:end).^2) ...
        + sum(errorRight(2:end).^2) ...
        + 0.5 * (errorLeft(1)^2 + errorRight(1)^2)))
```

The mass diagnostic also uses a composite trapezoid rule and avoids
double-counting the vertex.  The example prints the reminder:

```text
Equilibrium check: exact insulated equilibrium is 5.5.
```

The vertex continuity diagnostic is

```matlab
abs(uLeft(1) - uRight(1)).
```

It should remain at roundoff because the algorithm writes the same `uvNew`
to both branches.

The flux-jump diagnostic is computed from one-sided differences at the
vertex:

```matlab
uv = 0.5 * (uLeft(1) + uRight(1));
jump = (uLeft(2) - uv) / h - (uv - uRight(2)) / h;
```

This is reported as `MaxAbsFluxJump`.  It is a diagnostic of local
one-sided slope mismatch during the transient calculation, not an additional
constraint solve.

## Output Table

The returned `results` table contains:

```text
AngleDegrees
h
CFL
dt
TimeSteps
FinalInfError
FinalL2Error
FinalInfRate
FinalL2Rate
MassDrift
MaxVertexJump
MaxAbsFluxJump
NoShortcutOK
```

The convergence rates are computed between successive grid spacings:

```text
p_j = log(E_{j-1} / E_j) / log(h_{j-1} / h_j).
```

This formula does not assume the refinement ratio is exactly two.

`NoShortcutOK` confirms that the branch extension matrices used only data
from their own branch and that vertex-fiber rows use only the vertex column.

## Plots

The current implementation uses `tiledlayout` and `nexttile`, not
`subplot`.  Both the figure and axes backgrounds are explicitly set to white:

```matlab
set(fig, 'Color', 'w');
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
```

Figure 1 shows, for the finest grid at each angle, the numerical and exact
solutions as functions of the arclength coordinate `s`.  The shared vertex is
marked separately.

Figure 2 shows final-time convergence by opening angle.  Each tile plots:

```text
L_inf error
L_2 error
O(h) reference line
O(h^2) reference line
```

The `tiledlayout(fig, 'flow', ...)` layout reflows the panels according to
the figure size and avoids the black-background subplot issue encountered in
the older plotting code.

## Interpretation for Reporting

This example should be described as a labeled-branch CPM method with a
metric-graph junction closure.

The important methodological points are:

1. The geometry is embedded in `R^2`, but each branch is discretized by its
   own arclength coordinate.
2. The CPM narrow band is constructed independently for each branch using
   `cpLineSegment2d`.
3. The branch Laplacian is assembled by the standard CPM composition
   `R * Lcart * Eband`.
4. The branch labels prevent cross-branch interpolation away from the shared
   vertex.
5. The shared vertex is not treated as a Neumann boundary of either branch.
   Instead, it is updated with the symmetric two-neighbor graph Laplacian.
6. Rows near endpoints are corrected because raw closest-point endpoint
   clamping contaminates more than the endpoint row itself.
7. Physical endpoints retain homogeneous Neumann behavior.
8. Vertex continuity is imposed strongly by storing the same vertex value on
   both branch arrays after every time step.

The accuracy issue in the earlier study was not the vertex formula itself.
The formula

```text
du_v/dt = kappa * (u_{L,1} + u_{R,1} - 2 u_v) / h^2
```

is the expected second-order graph stencil at a two-branch junction with
equal branch spacing.  The problem was that the raw single-branch CPM
operators still treated the shared vertex as an endpoint and introduced
endpoint-clamping errors in nearby rows.  The current implementation fixes
that by replacing endpoint-affected rows with the intended one-dimensional
graph stencils, while keeping the branch-local CPM construction away from
the endpoint caps.
