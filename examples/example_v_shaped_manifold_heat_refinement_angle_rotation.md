# Angle-Rotation CPM Heat Refinement on a V-Shaped Manifold

This note documents the MATLAB example:

```matlab
example_v_shaped_manifold_heat_refinement_angle_rotation
```

The experiment solves the heat equation on a two-branch V-shaped curve.  The
main purpose is to test a closest point method (CPM) extension rule that
keeps each branch on its own Cartesian grid, but repairs the shared vertex by
rotating vertex-fiber grid points onto the opposite branch.

The implemented method has three important local rules:

1. Interior branch points use the usual closest point extension on their own
   branch.
2. Shared-vertex closest point rows use the angle-rotation rule.
3. Physical endpoints use reflected same-branch ghost values for the
   standard second-order Neumann endpoint closure.

## Running the Example

From the repository root:

```matlab
addpath('examples')
[results, diagnostics] = ...
    example_v_shaped_manifold_heat_refinement_angle_rotation();
```

For a shorter run:

```matlab
opts = struct( ...
    'anglesDeg', [90 150], ...
    'hvals', 1 ./ [40 80 160], ...
    'makePlots', true, ...
    'showProgress', true);
[results, diagnostics] = ...
    example_v_shaped_manifold_heat_refinement_angle_rotation(opts);
```

The example prints progress messages by default.  These messages report the
current angle, grid spacing, matrix-build stage, rotation angles, time-step
count, and several time-integration checkpoints.

The default time stepping path is

```matlab
opts.timeStepper = 'explicitManifold';
```

This uses the CPM matrices to build the sampled spatial operator `R*L*E` on
the branch nodes, then advances the manifold values by explicit Euler with
`dt = cfl*h^2/kappa`.  The older literal Cartesian-band re-extension loop is
still available for comparison:

```matlab
opts.timeStepper = 'explicitBand';
```

## Geometry and Branch Labels

The opening angle is `alpha = angleDeg`, and the code uses

```matlab
halfAngle = 0.5 * angleDeg * pi / 180;
```

The left branch is branch A, stored with `side < 0`:

```matlab
A endpoint:      p0 = [-cos(halfAngle), sin(halfAngle)]
shared vertex:   p1 = [0, 0]
```

The right branch is branch B, stored with `side > 0`:

```matlab
shared vertex:   p0 = [0, 0]
B endpoint:      p1 = [cos(halfAngle), sin(halfAngle)]
```

The branch-node coordinates are

```matlab
sLeft  = (0:-h:-1)';
sRight = (0:h:1)';
```

so index `1` is the shared vertex for both branch-node arrays.  The
embedding used for plotting and restriction is

```matlab
A: X_A(s) = (s cos(halfAngle), -s sin(halfAngle)),  s in [-1, 0]
B: X_B(s) = (s cos(halfAngle),  s sin(halfAngle)),  s in [ 0, 1]
```

## CPM Grid Construction

For each branch, `buildBranch` creates a Cartesian tensor-product grid around
that branch only.  The code calls

```matlab
[cpx, cpy, dist, bdy, t] = cpLineSegment2d(xx, yy, p0, p1);
band = find(abs(dist) <= bw * h);
```

The default long-lived state is stored on the branch nodes.  The code builds
node-to-band extension matrices for the sampled CPM spatial operator.  For
ordinary rows, the node extension matrix interpolates in the branch
coordinate:

```matlab
Nown = interp1BranchMatrix(sNodes, sBand, interpDegree);
```

For the optional `explicitBand` comparison path, the code also builds a
band-to-band closest-point extension matrix:

```matlab
Bown = interp2_matrix(x1d, y1d, cpxBand, cpyBand, interpDegree, band);
```

The Cartesian finite-difference Laplacian is

```matlab
Lcart = laplacian_2d_matrix(x1d, y1d, lapOrder, band, band);
```

The restriction matrix from Cartesian band values back to branch nodes is

```matlab
R = interp2_matrix(x1d, y1d, xManifold, yManifold, interpDegree, band);
```

The sampled manifold spatial operator is assembled from these pieces as
`R*Lcart*N`, with cross-branch angle-rotation blocks at the vertex.

## Vertex and Endpoint Row Sets

The code separates endpoint behavior into two masks:

```matlab
vertexMask = makeVertexMask(cpxBand, cpyBand, sBand, h, vertexTol);
physicalEndpointMask = makePhysicalEndpointMask(side, bdyBand);
```

`vertexMask` selects band rows whose closest point is the shared vertex.
Those are the only rows that receive angle-rotation coupling.

`physicalEndpointMask` selects rows clamped at the non-vertex endpoint:

```matlab
A physical endpoint rows: bdyBand == 1
B physical endpoint rows: bdyBand == 2
```

These rows are not angle-rotated.  They use the physical Neumann rule
described later.

## Inward and Outward Tangents

The tangent calculation is done in `branchTangents`.

For a branch, the code first chooses the physical endpoint:

```matlab
if branch.side < 0
  endpoint = branch.p0;   % A physical endpoint
else
  endpoint = branch.p1;   % B physical endpoint
end
```

The shared vertex is always

```matlab
vertex = [0, 0];
```

The outward tangent points from the shared vertex out toward the physical
endpoint:

```matlab
tangents.outward = unitVector(endpoint - vertex);
```

The inward tangent points from the physical endpoint back toward the shared
vertex:

```matlab
tangents.inward = -tangents.outward;
```

For the V geometry, this gives

```text
A outward = [-cos(halfAngle),  sin(halfAngle)]
A inward  = [ cos(halfAngle), -sin(halfAngle)]

B outward = [ cos(halfAngle),  sin(halfAngle)]
B inward  = [-cos(halfAngle), -sin(halfAngle)]
```

## Rotation Angles

The rotation angles are built in `addAngleRotationMaps`.

For branch A to branch B, the code rotates the outward tangent of A onto the
inward tangent of B:

```matlab
thetaLeftToRight = signedAngle(leftTangents.outward, ...
                               rightTangents.inward);
```

This is the code's `theta_AtoB`.

For branch B to branch A, the code rotates the outward tangent of B onto the
inward tangent of A:

```matlab
thetaRightToLeft = signedAngle(rightTangents.outward, ...
                               leftTangents.inward);
```

This is the code's `theta_BtoA`.

The signed angle helper is

```matlab
crossValue = fromVector(1) * toVector(2) - ...
             fromVector(2) * toVector(1);
dotValue = dot(fromVector, toVector);
theta = atan2(crossValue, dotValue);
```

Thus the sign follows the usual counterclockwise-positive convention.  For
example:

```text
alpha = 90 degrees:   theta_AtoB =  90 degrees, theta_BtoA =  -90 degrees
alpha = 150 degrees:  theta_AtoB = 150 degrees, theta_BtoA = -150 degrees
```

These values are reported in the output table as
`ThetaLeftToRightDeg` and `ThetaRightToLeftDeg`.

## Angle-Rotation Extension at the Vertex

The angle-rotation rows are built in `addAngleRotationMap`.

Suppose the current branch is B and the other branch is A.  The code finds
all B-grid rows whose closest point is the shared vertex:

```matlab
vertexRows = find(branch.vertexMask);
```

For those Cartesian grid points `x`, it rotates the point by `theta_BtoA`
about the shared vertex:

```matlab
[xRot, yRot] = rotatePoints(branch.xBand(vertexRows), ...
                            branch.yBand(vertexRows), theta);
```

The rotation is

```matlab
xRot = cos(theta) * x - sin(theta) * y;
yRot = sin(theta) * x + cos(theta) * y;
```

After rotation, the code finds the closest point on branch A:

```matlab
[targetX, targetY, ~, ~, targetT] = ...
    cpLineSegment2d(xRot, yRot, other.p0, other.p1);
```

Then it builds interpolation matrices from A to the rotated closest points.
The default manifold operator uses a node-to-band block:

```matlab
Nvertex = interp1BranchMatrix(other.sNodes, targetS, interpDegree);
```

The optional band-state comparison path uses a band-to-band block:

```matlab
Bvertex = interp2_matrix(other.x1d, other.y1d, targetX, targetY, ...
                         interpDegree, other.band);
```

Only the vertex rows are filled with these opposite-branch interpolation
weights:

```matlab
rowSelector = sparse(vertexRows, 1:numel(vertexRows), 1, ...
                     nBand, numel(vertexRows));
branch.Nother = rowSelector * Nvertex;
branch.Bother = rowSelector * Bvertex;
```

All non-vertex rows of `Nother` and `Bother` remain zero.  The local check
`angleRotationUsesOnlyVertexFibers` verifies that the cross-branch extension
is restricted to `vertexMask`.

For an initial node-to-band extension, the code uses

```matlab
uLeftBand  = left.Nown  * uLeft  + left.Nother  * uRight;
uRightBand = right.Nown * uRight + right.Nother * uLeft;
```

This means:

```text
normal A rows:       A values from A closest points
normal B rows:       B values from B closest points
A vertex-fiber rows: A values from rotated closest points on B
B vertex-fiber rows: B values from rotated closest points on A
```

## Physical Endpoint Neumann Closure

The angle rotation is not used at the two outer endpoints.  Those endpoints
represent ordinary homogeneous Neumann boundaries.

Rows clamped at the physical endpoint are handled by
`applyPhysicalEndpointNeumannNodeRows` and
`applyPhysicalEndpointNeumannBandRows`.  For a physical endpoint `p`, the
code reflects each endpoint-fiber grid point through `p`:

```matlab
xReflect = 2 * endpoint(1) - xBand(mask);
yReflect = 2 * endpoint(2) - yBand(mask);
```

It then closest-point maps the reflected point back to the same branch:

```matlab
[targetX, targetY] = cpLineSegment2d(xReflect, yReflect, p0, p1);
```

The corresponding interpolation rows replace the raw endpoint-clamped
extension rows.  In the node-to-band map this is

```matlab
targetS = branchParameter(side, targetT);
Ereflect = interp1BranchMatrix(sNodes, targetS, interpDegree);
Nown(mask, :) = Ereflect;
```

and in the band-to-band map this is

```matlab
Ereflect = interp2_matrix(x1d, y1d, targetX, targetY, ...
                          interpDegree, band);
Bown(mask, :) = Ereflect;
```

This gives the Cartesian centered finite-difference stencil the usual
second-order Neumann ghost value at the physical endpoint, instead of the
lower-order raw closest-point endpoint clamping.

## Time-Stepping Algorithm

The default path, `timeStepper = 'explicitManifold'`, treats the CPM
construction as a method-of-lines spatial discretization on the branch
nodes.  The branch-node update is

```matlab
u_M^{n+1} = u_M^n + dt * kappa * R * Lcart * N * u_M^n.
```

The operator blocks are assembled once:

```matlab
ALL = left.R  * (left.Lcart  * left.Nown);
ALR = left.R  * (left.Lcart  * left.Nother);
ARR = right.R * (right.Lcart * right.Nown);
ARL = right.R * (right.Lcart * right.Nother);
```

Then each explicit Euler step applies

```matlab
lapLeft  = ALL * uLeft  + ALR * uRight;
lapRight = ARR * uRight + ARL * uLeft;

uLeftNew  = uLeft  + dt * kappa * lapLeft;
uRightNew = uRight + dt * kappa * lapRight;
```

The shared vertex row is overwritten by the graph-junction formula:

```matlab
uvNew = uv + dt * kappa * ...
    (uLeft(2) + uRight(2) - 2 * uv) / h^2;
```

This path still uses the explicit heat-equation stability scaling

```matlab
dt = cfl * h^2 / kappa;
```

but it does not apply the interpolation projection `R*N` to the old solution
at every time step.  This distinction matters at very small `h`.  For
example, `h = 1/3200`, `finalTime = 0.01`, and `cfl = 0.1` require about
1,024,000 explicit steps.  If the code repeatedly computes

```matlab
u_M^{n+1} = R * (N*u_M^n + dt * Lcart * N*u_M^n),
```

then the nominal identity part is actually `R*N`, not the exact identity.
The tiny interpolation/projection error is applied once per explicit step,
and at `h = 1/3200` that can dominate the discretization error.  The
`explicitManifold` update instead uses

```matlab
u_M^{n+1} = u_M^n + dt * R * Lcart * N*u_M^n,
```

so the identity part is exact on the branch nodes.

For comparison, `timeStepper = 'explicitBand'` runs the literal Cartesian
band loop:

1. Extend branch-node values to Cartesian band values.
2. Apply the Cartesian finite-difference heat step on each branch grid.
3. Re-extend the updated Cartesian band values using `Bown` and `Bother`.
4. Sample branch-node values for the vertex update and diagnostics.

This is useful as a diagnostic, but it is more sensitive to repeated
re-extension/projection error in very fine explicit runs.

## Diagnostics and Plots

The output table includes:

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
ThetaLeftToRightDeg
ThetaRightToLeftDeg
AngleRotationOK
```

The diagnostic struct stores the final solution and error curve data:

```matlab
diagnostic.sLeft
diagnostic.sRight
diagnostic.uLeft
diagnostic.uRight
diagnostic.exactLeft
diagnostic.exactRight
diagnostic.errorS
diagnostic.finalAbsError
```

The figures are:

1. Final numerical and exact solution versus `s`.
2. Convergence of final-time `L_inf` and `L_2` errors versus `h`.
3. Final-time pointwise error `|u_h - u|` versus `s`.

Figure 3 uses the finest grid available for each tested angle.

## Interpretation

The angle-rotation step locally straightens the shared vertex by aligning
one branch's outward tangent with the other branch's inward tangent.  This
prevents the shared vertex from behaving like a single-branch Neumann
endpoint while still avoiding cross-branch interpolation away from the
vertex.

The physical endpoints are separate from this mechanism.  They are true
Neumann boundaries, so the code uses reflected same-branch ghost values
there.  This distinction is important: applying angle rotation only at the
shared vertex and using second-order Neumann reflection at the physical
endpoints is what gives the observed higher-order convergence in the smooth
test cases.
