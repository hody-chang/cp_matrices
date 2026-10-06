# The angle rotation glue for a two-branch iCPM junction

Reference implementation: `example_double_convex_lens_curvature_convergence.m`.

This note explains how the angle rotation method is applied in that experiment, in the
order a reader has to reconstruct it, and lists the conventions and guards that make it
work. The same recipe transfers to any planar geometry built from two smooth branches
meeting transversally at a shared vertex.

## 1. The problem the glue solves

The manifold is a closed curve assembled from two circular arcs that share the vertices
$V_{top}=(0,H)$ and $V_{bottom}=(0,-H)$. It is not smooth at the vertices, so a single
closest-point extension is not available there: the medial axis of the closed curve runs
out of each vertex along the wedge bisector, and a single-band CPM stencil straddling that
bisector would couple the two arcs directly.

The experiment therefore solves the shifted problem $u - \Delta_\Gamma u = f$ with one band
per arc. Each band is self-contained except at its endpoint rows, where the extension has
no footpoint of its own to read from. The glue supplies those rows. The angle rotation
method supplies them by rigidly rotating the ghost node about the vertex until the source
branch's outward tangent lines up with the target branch's inward tangent, then projecting
onto the target arc.

## 2. Discretization: one grid, two bands

One Cartesian grid covers the whole lens, built once per refinement level from the union
bounding box (`solve_level`):

```matlab
x1d = (geo.bounds(1) - pad + 0.317*h):h:(geo.bounds(2) + pad + 0.317*h);
y1d = (geo.bounds(3) - pad + 0.211*h):h:(geo.bounds(4) + pad + 0.211*h);
[xx, yy] = meshgrid(x1d, y1d);
```

The irrational shifts `0.317*h` and `0.211*h` keep the vertices and the symmetry axis off
the nodes. `setup_branch` is then called twice on the *same* grid. Each call runs its own
branch `cpArc` over every node, keeps `abs(dist) <= bw*h`, and builds its own
`innerband`, `outerband`, `L`, `E`, `R`, and unknown count `nin`.

Consequences to keep in mind:

- Near each vertex a node lies in both bands and carries **two independent unknowns**, one
  per branch field. The bands overlap geometrically; the unknowns never do.
- `band` indexes the shared global numbering `1:numel(xx)`, and
  `make_invbandmap(numel(xx), innerband)` inverts global index to branch-local unknown.
  The shared grid is what makes the cross-branch block assemblable at all: with per-branch
  grids there is no common column index space, and an extra grid-to-grid transfer would
  contaminate the glue error being measured.
- No shortcut arises from the shared grid, because each band consults only its own
  branch's closest-point map, and a circular arc's cp map is single-valued and smooth on
  any tube of radius below its reach $R$. Here $\mathrm{bw}\,h \le 0.036$ against
  $R \approx 0.475$.

## 3. The glue, step by step

### Step 1: exact outward tangents, indexed by vertex

`make_branch` stores a unit tangent at each endpoint pointing *out of* the arc:

```matlab
t1 = [-sin(angle1) cos(angle1)];
t2 = [-sin(angle1 + span) cos(angle1 + span)];
b.tauout_start = -t1;  b.tauout_end = t2;
```

`t1` and `t2` are the counterclockwise tangents at the two end angles; negating the start
one makes both point outward. The two rows are then permuted so that `b.tauout(iv,:)`
belongs to `geo.vertices(iv,:)` for **both** branches. This permutation is essential: the
branches traverse in opposite directions, so without it the left and right arcs disagree
about which row is the top vertex, and the glue silently pairs the wrong tangents.

### Step 2: the glue rotation

`glue_rot2d` (in `rotations/`) returns the rotation carrying the source's outward tangent
$u = \tau_{out}^{source}$ onto the target's inward tangent $w = -\tau_{out}^{target}$. It
returns that rotation as a $(\cos\theta, \sin\theta)$ pair and never forms the angle:

```matlab
c = u(1)*w(1) + u(2)*w(2);        % cos: the dot product
s = u(1)*w(2) - u(2)*w(1);        % sin: the cross product's e_z component
n = hypot(c, s);  c = c/n;  s = s/n;
R = [c -s; s c];
```

This is Rodrigues' rotation formula $R = I + \sin\theta\,K + (1-\cos\theta)K^2$
specialized to the plane: the rotation axis is $e_z$, whose skew matrix restricted to the
plane is $K = \bigl[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\bigr]$ with
$K^2 = -I$, so the formula collapses to $R = \cos\theta\,I + \sin\theta\,K$. It is the
same construction `angle3D.m` uses in three dimensions, so the 2D and 3D glues are one
construction rather than two.

Outward to inward is the correct pairing: leaving the source arc at the vertex means
entering the target arc. Only four rotations exist per level (2 branches x 2 vertices);
they are computed once, outside the row loop, under `use = vids == iv`.

For this geometry the interior angle at each vertex is 102.8-110.5 degrees, so the glue
rotation is 69.5-77.2 degrees in magnitude. It is the turning angle of the curve at the
junction.

**Why not through an angle.** The earlier form of this step was

```matlab
angle = atan2(-tauout_target(2), -tauout_target(1)) - ...
        atan2(tauout_source(2), tauout_source(1));
angle = atan2(sin(angle), cos(angle));            % rewrap to (-pi, pi]
R = [cos(angle) -sin(angle); sin(angle) cos(angle)];
```

which reaches the same matrix through six library transcendental calls where the
construction above uses none; the angle is an intermediate that `cos` and `sin`
immediately undo. `example_ellipse_cut_rotation_construction.m` measures the difference on
the cut ellipse. In brief:

- **Accuracy.** Against a rotation known to full precision, $\|R - R_{exact}\|$ comes out
  two to seven times larger for the angle-first path at every angle sampled over a full
  turn, four times larger at the median; worst case over that sweep,
  $1.2\times 10^{-15}$ against $2.8\times 10^{-16}$. Both stay within a few units in the
  last place, so this does not reach the solution: on the cut ellipse the two
  constructions move the computed $u$ by at most $4\times 10^{-13}$ relative, against a
  tangent-estimator error larger by a factor of at least $5\times 10^{10}$.
- **No branch cut.** The difference of two `atan2` values lands in $(-2\pi, 2\pi]$ and has
  to be wrapped back, so the stored datum is discontinuous at $\pm\pi$ even though the
  rotation it stands for is not. A $(\cos, \sin)$ pair has no such seam, which is what
  lets `rot2d_err` compare two glues through their relative rotation instead of through a
  wrapped difference of angles.
- **Exact inverses.** Swapping the two tangents gives the other branch's glue at the same
  vertex. Under Rodrigues the cosine comes out bit for bit the same and the sine bit for
  bit negated, so the two branches' rotations are *exact* transposes; the angle-first path
  satisfies that only to within rounding.
- **Reproducibility.** IEEE 754 pins down multiply, add, divide and `hypot` but says
  nothing about `atan2`, `sin` or `cos`, so the angle-first matrix can differ in its last
  bits between platforms, libm versions and MATLAB releases. Dot, cross and `hypot`
  cannot.
- **Cost.** About 2.4x faster in the measurement above, which does not matter: four
  rotations per level.

### Step 3: rotate the grid node, not its closest point

```matlab
rows = find(br(source).vid ~= 0);                     % cap rows
xq   = [br(source).xout(rows), br(source).yout(rows)];% Cartesian grid nodes
[cc, ss] = glue_rot2d(tauout_source, tauout_target);
qmap(use,:) = rotate_about2d(xq(use,:), geo.vertices(iv,:), cc, ss);
```

`rotate_about2d` takes the pair, not an angle, for the same reason: an angle argument would
have to be turned back into a cosine and a sine at every call site.

`vid` is set in `setup_branch` by matching `cpxout/cpyout` against the vertex list to
`100*eps`; a row with `bdyout ~= 0` and `vid == 0` is an error. The rotated object is the
**grid node**, not its projection. That is what makes the map a rigid continuation: both
the tangential overhang past the vertex and the normal offset are carried across intact.
Rotating the closest point instead would only reparametrize and would lose the normal
coordinate.

### Step 4: re-project, rebuild the row

```matlab
[cpx, cpy] = br(target).geo.cpf(qmap(:,1), qmap(:,2));
[Ei, Ej, Es] = interp2_matrix(x1d, y1d, cpx, cpy, p);
jj = br(target).inv_inner(Ej);
if any(jj == 0), error('a cross-branch interpolation stencil leaves the inner band'); end
Eb{source,target} = sparse(rows(Ei), jj, Es, br(source).nout, br(target).nin);
Eself = Eb{source,source};  Eself(rows,:) = 0;  Eb{source,source} = Eself;
```

The rotated point is projected with the *target's* `cpArc`, the interpolation stencil is
built there, and the cap row is removed from the diagonal block entirely. After both
directions are routed, the block operator is

```matlab
Eblk = [Eb{1,1} Eb{1,2}; Eb{2,1} Eb{2,2}];
Lblk = blkdiag(br(1).L, br(2).L);  Rblk = blkdiag(br(1).R, br(2).R);
M    = lapsharp_unordered(Lblk, Eblk, Rblk);
```

and the unknown vector is the concatenation `[u1; u2]`.

### Step 5: make the forcing consistent with the routed row

```matlab
ts{source}(rows) = br(target).geo.parfun(cpx, cpy);
```

This is the step most easily missed. A routed row now represents a point at arclength
$\xi$ *past* the vertex on the target branch, so $f$ at that row must be evaluated at the
continued global arclength on the target branch. Leaving `ts` at the source value tests
the rotation against an inconsistent right-hand side and corrupts the measured rate.

## 4. Variants provided

| `glue_type` | tangent source | what it matches at the vertex |
| --- | --- | --- |
| `rotation` | analytic `tauout` | tangent only |
| `rotation_dk2` | `cp_tangent_dk2` estimate | tangent only |
| `arclength` | analytic `tauout` | tangent and curvature |

`cp_tangent_dk2` exists because cap rows have `CP == V` exactly, so a plain difference of
closest points returns nothing. It reflects the node through its own closest point,
re-projects, and takes a one-sided second-order difference:

```matlab
[b1x, b1y] = br.geo.cpf(2*cx - x, 2*cy - y);
[b2x, b2y] = br.geo.cpf(3*cx - 2*x, 3*cy - 2*y);
dvx = 1.5*cx - 2*b1x + 0.5*b2x;   % and likewise dvy
```

normalizes, and averages over the cap. The rotation machinery downstream is identical;
only the angle is estimated.

`map_by_arclength` is the control. Instead of a rigid rotation it converts the node's
angular offset about the source centre to arclength $\xi = R_{source}\,\theta$ and replays
that same $\xi$ on the target circle as $\alpha = \xi / R_{target}$. Its sign conventions
come from the outward source tangent and the inward target tangent, and it asserts
$\xi \ge 0$, which holds because the cap region is exactly the angular sector beyond the
branch's end angle.

## 5. The claim being tested

A single rotation angle can only match the tangent. When $\kappa_L \ne \kappa_R$ the
rotated ghost node sits $O(\Delta\kappa\,\xi^2)$ off the target arc, so the glue picks up a
first-order term. `fit_mixed_order` fits $e/h = C_1 + C_2 h$ by two-column nonnegative
least squares over the pre-floor levels, and the third subplot plots $C_1$ against
$\Delta\kappa$: $C_1$ should grow with the curvature jump and vanish as
$R_{right} \to R_{left}$, while the arclength control stays at rate 2 throughout.

## 6. Conventions and traps

1. **Vertex row ordering.** `tauout` rows must follow `geo.vertices` for every branch,
   independent of traversal direction. See the `norm(vstart - [0 abs(vstart(2))])` test in
   `make_branch`.
2. **Outward to inward, not outward to outward.** Off by $\pi$ otherwise, which still
   produces a plausible-looking band and a wrong answer. `glue_rot2d` negates the target
   tangent itself, so pass both tangents outward; `rot2d_from_to` is the unnegated form for
   the cases where the caller already has the two directions the way it wants them.
3. **Rotate nodes, not closest points.**
4. **Reparametrize the forcing for routed rows.**
5. **Zero the diagonal block on routed rows** before adding the off-diagonal block, or the
   row sum becomes 2.
6. **The cap region extends past the other branch.** The $V_{top}$ cap is the half-disc of
   radius $\mathrm{bw}\,h$ bounded by the *normal* line at the vertex, on the `tauout` side,
   i.e. directions within 90 degrees of $\tau_{out}$. The other branch leaves the vertex at
   the glue angle from $\tau_{out}$, so the cap overshoots it by
   (interior angle $-$ 90 degrees), here 12.8-20.5 degrees. Some source cap nodes therefore lie
   inside the lens on the far side of the target arc. This is harmless - their glued value
   is read at small positive $\xi$ - but it is where the two ghost layers interpenetrate and
   where the $C_1 h$ error is deposited.
7. **Band separation is not asserted.** Away from the vertices the two arcs approach to
   0.075, while $2\,\mathrm{bw}\,h = 0.072$ at the coarsest active level $h = 10^{-2}$.
   Correctness of the two-band operator does not depend on separation, but at
   $h = 0.02$ (the commented-out `hvals`) the bands overlap along their entire length, and
   any diagnostic assuming two geometrically distinct tubes would be wrong. If separation
   matters to a future variant, assert `bw*h` against the inter-arc gap in `solve_level`.

## 7. Guards already in the script

- `nnz(Ltemp) ~= sten*s.nin` - the Laplacian stencil stays inside the initial band.
- inner band contained in outer band.
- `any(jj == 0)` on the cross-branch stencil and on the surface-error stencil - every
  interpolation stays inside the target's inner band.
- `max(abs(sum(Eblk,2) - 1)) <= 1e-10` - the rotated stencils are still a partition of
  unity, so the glue is an interpolation.
- all three glue types must route the **same** endpoint-row set at each $h$; comparing
  rates across different row sets would be meaningless.
- non-finite checks after the solve and on the surface error.

## 8. Porting the method to a new geometry

1. Give every branch a `cpf`, a global-arclength `parfun`, a `pointfun`, and a `tauout`
   whose rows are indexed by the shared vertex list.
2. Build one grid over the union bounding box; build one band per branch on it.
3. Identify cap rows by matching the branch closest point against the vertex list, and
   assert that every boundary-clamped row matches a vertex.
4. Per (source branch, vertex): one glue angle, rotate that vertex's cap nodes, project
   with the target `cpf`, write the off-diagonal block, zero the diagonal rows, and
   overwrite the routed rows' arclength parameter.
5. Keep the row-sum and stencil-containment assertions; they catch every convention error
   in section 6 except the forcing reparametrization, which shows up only as a lost rate.

## 9. Note on three dimensions

Everything here needs only a scalar angle because both tangents lie in the grid plane. At
a surface edge in $\mathbb{R}^3$ the glue must rotate a frame rather than an angle, and the
planar argument does not carry over.

## 10. Three-dimensional case: the half-twisted rectangular tube

Reference implementation: `3D_surface/example_half_twisted_rectangular_tube_rotation_convergence.m`,
called as `example_half_twisted_rectangular_tube_rotation_convergence(hvals)`. `hvals` is
optional; the default list is set in the script and may change. Supporting files:
`surfaces/halfTwistedRectTubeFrame.m` (parametrization and analytic derivatives),
`surfaces/cpHalfTwistedRectTube.m` (Euclidean closest point of one strip, with edge flag
`bdy`), and `3D_surface/halfTwistedRectTubeMMS.m` (manufactured solution).

### Geometry and branches

With $R=1$, $a=0.45$, $b=0.20$,
$F(\theta,p)=(R+u)(\cos\theta,\sin\theta,0)+v\,(0,0,1)$,
$u=p_1\cos\frac\theta2-p_2\sin\frac\theta2$, $v=p_1\sin\frac\theta2+p_2\cos\frac\theta2$.
Branch A (wide) uses $p=(s,b)$, $s\in[-a,a]$; branch B (narrow) uses $p=(a,s)$,
$s\in[-b,b]$. Each strip covers its face pair once with $\theta$ 4π-periodic. The strips
meet along two closed curves:

- upper (red): A at $(\theta,+a)$ equals B at $(\theta,+b)$;
- lower (blue): A at $(\theta,-a)$ equals B at $(\theta+2\pi,-b)$ (mod 4π), and the
  inverse map uses the same shift.

Metric: $E=|F_s|^2=1$, $H=F_s\cdot F_\theta=-b/2$ (A) or $+a/2$ (B),
$G=(R+u)^2+(p_1^2+p_2^2)/4$, $D=G-H^2$. The outward conormal derivative at $s=\pm W$ is
$\operatorname{sign}(s)\sqrt{G/D}\,(u_s-\tfrac HG u_\theta)$.

### Manufactured solution

On each strip, $u$ is a cubic Hermite interpolant in $s$ between edge traces and edge
slopes that vary smoothly in $\theta$. Both strips use the same traces on each curve. The
slopes are $u_s=\tfrac HG w_\theta+\operatorname{sign}(s)\sqrt{D/G}\,g$, where $g$ is a
prescribed outward flux that is nonzero and depends on $\theta$. The two strips receive
opposite values of $g$, so the outgoing conormal fluxes sum to zero on both curves. The
right-hand side $f=u-\Delta_S u$ is analytic. It uses the divergence form
$\Delta_S u=\frac1{\sqrt D}\big[\partial_s\frac{Gu_s-Hu_\theta}{\sqrt D}+\partial_\theta\frac{-Hu_s+u_\theta}{\sqrt D}\big]$,
with every $\theta$-derivative of the slopes computed analytically. No finite differences or
symbolic toolbox are used. The MMS returns `[u,f,us,ut]` for the checks.

### Glue

An outer node whose own-branch closest point lies on an edge is routed. In 3D the glue
rotates the grid node about the exact edge tangent $\tau=F_\theta/|F_\theta|$ through the
edge closest point. The signed rotation angle carries the source outward conormal onto
minus the target outward conormal, so the source's outward direction maps to the target's
inward direction. The faces are not exactly perpendicular because of the twist, so this
angle varies along the curve. The routed row interpolates the target branch at the closest
point of the rotated node, and its right-hand side is the target $f$ there.

Both methods use the analytic edge frame for the rotation axis: the exact $\tau$ through the
edge closest point. They differ only in the conormals that set the angle.

- `exact`: both conormals are analytic.
- `dk2`: both conormals are estimated pointwise. Each estimate is a second-order closest
  point difference $1.5c-2\,cp(c+v)+0.5\,cp(c+2v)$, projected off the exact $\tau$ and
  normalized. The source estimate reflects the original routed grid node
  ($v=c-x$). Rays that are unusable (too short, degenerate, hitting an edge, jumping to
  the opposite sheet, or estimating an inward direction) are probed again from
  $c+h\,\eta_s$. The target estimate always uses the probe $c_t+h\,\eta_t$.
  The exact $\tau$ supplies the rotation axis and the projection plane; analytic
  conormals place the probes and check outward orientation. Both conormal directions
  used in the rotation angle come from closest-point differences. Thus `dk2` is not
  a geometry-free estimator. An unusable probe raises an error; there is no replacement
  of an estimated conormal by the exact one.

### Band and routing closure

Bands and routed rows are built once per level and shared by both methods, so `exact` and
`dk2` solve on identical bands with identical routed rows. Each branch's inner band
contains the interpolation stencils of:

- its initial-band closest points;
- every surface error sample and corner-curve trace sample;
- the target closest points of rows routed into it by either rotation.

Enlarging an inner band enlarges the outer band, which can create new routed rows. Routing
and banding are therefore iterated until no new support appears, for at most
`P.max_closure_iter` sweeps. All support must lie in the conservative initial band;
otherwise the run errors. The sweep count is reported as `closure_iter`.

### Reported quantities

For each level and method the script reports:

- the surface sup error, sampled on a uniform $(\theta,s)$ grid of each strip that includes
  both edge curves;
- the trace discrepancy of the two branches on each curve;
- the maximum `dk2` angle error;
- extension row sums, the linear residual, routed-row counts per branch and curve, and the
  number of degenerate source rows.

Rates use the actual $h$ ratios, plus a least-squares fit. Results are saved to
`figs/half_twisted_rectangular_tube_rotation_convergence.png` and `.mat`.

### Observed results

These come from one run with `hvals = 1./[25 30 40]`. Routed rows occurred on both
branches, on both corner curves, in both directions. Closure took 2, 1 and 1 sweeps. The
relative residual was below $10^{-10}$ and the row-sum error below $1.6\times10^{-15}$ on
every level.

| $h$ | `exact` sup error | rate | `dk2` sup error | rate |
| --- | --- | --- | --- | --- |
| 1/25 | 8.5293e-2 | – | 8.5201e-2 | – |
| 1/30 | 6.3814e-2 | 1.59 | 6.3876e-2 | 1.58 |
| 1/40 | 4.5560e-2 | 1.17 | 4.5584e-2 | 1.17 |

Least-squares fitted rates: `exact` 1.320, `dk2` 1.317.

On these levels the two rotations are indistinguishable. These are three measured levels,
not an asymptotic order. The rates are not second order on the tested levels, and no
guaranteed $O(h)$ (or other) order is claimed from theory.
