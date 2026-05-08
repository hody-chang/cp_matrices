# Vertex-Coupled V-Shaped Manifold Heat-Equation Test

This experiment tests a vertex-coupled branch-aware closest point method
(CPM) heat solver on a family of V-shaped curves whose two branches become
nearly vertical and close to one another as the angle approaches
\(180^\circ\).

The MATLAB entry point is:

```matlab
results = example_v_shaped_manifold_heat_refinement();
```

Run it from the repository root with:

```matlab
addpath('examples')
results = example_v_shaped_manifold_heat_refinement();
```

This solves the two V arms as labeled line-segment CPM problems.  Each arm
keeps its own cubic stencil support away from the vertex.  The two labels
are allowed to communicate only through band values whose closest point is
the shared vertex.

## Geometry

For each opening angle \(\alpha\), the curve is

$$
\Gamma_\alpha =
\left\{
X_\alpha(s) = \left(s, m_\alpha |s|\right) : s \in [-1, 1]
\right\},
\qquad
m_\alpha = \tan\left(\frac{\alpha}{2}\right).
$$

The default script currently tests:

$$
\alpha \in \{90^\circ, 120^\circ, 150^\circ, 170^\circ, 175^\circ\}.
$$

With this parameterization, \(\alpha \to 180^\circ\) means
\(m_\alpha \to \infty\), so the branches become increasingly vertical.

The point \(s = 0\) is the V vertex.  The finite endpoints \(s = \pm 1\) are
handled by closest-point endpoint clamping, which corresponds to the
natural homogeneous Neumann endpoint behavior for this heat-equation test.

## PDE

The solved equation is

$$
u_t = \Delta_\Gamma u.
$$

On either branch of the V, arclength satisfies

$$
d\ell = \sqrt{1 + m_\alpha^2}\,ds,
$$

so the surface Laplacian is

$$
\Delta_\Gamma u = \frac{1}{1 + m_\alpha^2} u_{ss}.
$$

The default manufactured solution is a connected hot-left/cold-right mode:

$$
u(s,t) =
\frac{1}{2}
-
\frac{1}{2}
\exp\left(
-
\frac{
\left(\pi/2\right)^2
}{
1 + m_\alpha^2
}
t
\right)
\sin\left(\frac{\pi s}{2}\right).
$$

At \(t = 0\), this gives a hot left endpoint and a cold right endpoint:

$$
u(-1,0) = 1,
\qquad
u(1,0) = 0.
$$

It is compatible with homogeneous Neumann conditions at the two physical
endpoints:

$$
u_s(-1,t) = 0,
\qquad
u_s(1,t) = 0.
$$

It is also smooth through the vertex as a function of the global branch
parameter \(s\), so it tests whether heat can pass from one labeled branch
to the other through \(s = 0\).

## Discretization

The default experiment uses Cartesian grid spacings

$$
h \in \left\{
\frac{1}{100},
\frac{1}{200},
\frac{1}{400}
\right\}.
$$

For each angle and grid spacing, the V is split into its left and right
line-segment branches.  A separate Cartesian grid, CPM band, closest-point
map, interpolation matrix, and finite-difference Laplacian are built for
each labeled branch.

Away from the shared vertex, the method uses branch-local heat steps:

$$
\widetilde{u}_L^{n+1}
=
E_L
\left(
u_L^n + \Delta t L_L u_L^n
\right),
\qquad
\widetilde{u}_R^{n+1}
=
E_R
\left(
u_R^n + \Delta t L_R u_R^n
\right).
$$

The branch labels are then coupled only at rows whose closest point is the
shared vertex.  Let those row sets be

$$
\mathcal{V}_L
=
\left\{
i : \operatorname{cp}_L(x_i) = (0,0)
\right\},
\qquad
\mathcal{V}_R
=
\left\{
i : \operatorname{cp}_R(x_i) = (0,0)
\right\}.
$$

The shared vertex value is computed by averaging the two branch-side vertex
means:

$$
u_V^{n+1}
=
\frac{1}{2}
\left(
\operatorname{mean}_{i\in\mathcal{V}_L}
\widetilde{u}_{L,i}^{n+1}
+
\operatorname{mean}_{i\in\mathcal{V}_R}
\widetilde{u}_{R,i}^{n+1}
\right).
$$

Then the vertex rows on both labels are reset to this value:

$$
u_{L,i}^{n+1} = u_V^{n+1}
\quad
\text{for } i\in\mathcal{V}_L,
\qquad
u_{R,i}^{n+1} = u_V^{n+1}
\quad
\text{for } i\in\mathcal{V}_R.
$$

By default, the vertex sets use only rows whose closest point is exactly the
shared vertex.  If `vertexTol` is set positive, the code instead uses the
slightly enlarged condition

$$
|s_{\operatorname{cp}}| \le \texttt{vertexTol}\,h.
$$

The final time is:

$$
T = 0.01.
$$

The time step is chosen as

$$
\Delta t = 0.2h^2,
$$

then adjusted so an integer number of steps lands exactly on \(T\).

## Error Measurements

The numerical solution is interpolated back to dense surface sample points
\(X_\alpha(s_k)\).

The global relative infinity-norm error is

$$
E_{\mathrm{global,rel}}(h,\alpha)
=
\frac{
\max_k
\left|
u_h\left(X_\alpha(s_k)\right)
-
u\left(X_\alpha(s_k),T\right)
\right|
}{
\max_k
\left|
u\left(X_\alpha(s_k),T\right)
\right|
}.
$$

The script also reports the global sampled relative RMS-style error:

$$
\mathrm{RelGlobalL2}
=
\frac{
\sqrt{\operatorname{mean}\left(\mathrm{error}^2\right)}
}{
\sqrt{\operatorname{mean}\left(u_{\mathrm{exact}}^2\right)}
}.
$$

Observed rates are computed from infinity-norm errors:

$$
p(h_j,\alpha)
=
\frac{
\log\left(E(h_{j-1},\alpha) / E(h_j,\alpha)\right)
}{
\log\left(h_{j-1}/h_j\right)
}.
$$

This formula does not assume that the refinement ratio is exactly \(2\).

## Output

The returned MATLAB table contains:

```text
AngleDegrees
h
CFL
InterpDegree
TimeSteps
RelGlobalInf
RelGlobalL2
RelGlobalInfRate
```

The script opens two MATLAB figures directly and does not save them.

Figure 1 shows the finest-grid x-y pointwise error on the computational
band, using \(h = 1/400\), with one panel for each opening angle.  The
plotted quantity is

$$
\left|u_h - u_{\mathrm{exact}}\right|.
$$

The panels have the same physical width and the same x-limits, while the
y-limits are chosen separately for each angle.  This makes the near-vertical
cases readable without letting the \(175^\circ\) y-range collapse the other
panels.

Figure 2 plots convergence curves in separate panels, one panel for each
opening angle.  Each panel shows both the absolute global infinity-norm
error

$$
E_{\mathrm{global,abs}}(h,\alpha)
:=
\max_k
\left|
u_h\left(X_\alpha(s_k)\right)
-
u\left(X_\alpha(s_k),T\right)
\right|.
$$

and the absolute global sampled RMS-style error

$$
E_{2,\mathrm{global,abs}}(h,\alpha)
:=
\sqrt{
\operatorname{mean}
\left(
\left(
u_h-u_{\mathrm{exact}}
\right)^2
\right)
}.
$$

In every panel, circles mark \(E_{\infty,\mathrm{abs}}\), squares mark
\(E_{2,\mathrm{global,abs}}\), and the white dotted line is the \(O(h)\)
reference anchored at that angle's coarsest-grid infinity-norm error.

## Interpretation

This test checks whether the near-vertical V failure can be avoided while
still allowing physically meaningful heat transfer through the shared
vertex.  The branch labels prevent cross-branch shortcutting away from the
vertex, while the vertex coupling allows the hot left branch and cold right
branch to equilibrate through \(s = 0\).

The corresponding conclusion should therefore be phrased cautiously:

> With branch labels that prevent cross-branch interpolation away from the
> vertex, and with explicit vertex-only coupling, the CPM heat-equation
> solution error converges at approximately first order for all tested
> opening angles, including nearly vertical V branches.

It does not prove that the error constant is independent of angle.  To make
that stronger claim, one would also need to study whether

$$
\frac{E(h,\alpha)}{h}
$$

stays bounded as \(\alpha \to 180^\circ\).
