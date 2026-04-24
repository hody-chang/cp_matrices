# Configuration Summary for CSG Heat Examples

This document summarizes the effective configuration used by:

- `examples/example_heat_csg_circle_boolean_ops.m`
- `examples/example_heat_csg_sphere_boolean_ops.m`

Both scripts solve the surface heat equation

```matlab
u_t = Delta_s u + f
```

with a manufactured solution (MMS) on three constructive solid geometry (CSG) manifolds built from two overlapping primitives:

- `union`
- `intersection`
- `difference`

## Shared high-level structure

Both examples follow the same workflow:

1. Define two overlapping primitives with the same `radius` and center offset `shift`.
2. Build three CSG trees: union, intersection, and difference.
3. Create a Cartesian embedding grid.
4. Build a narrow computational band using `rm_bandwidth(...)` and `cpCSG(...)`.
5. Assemble closest-point interpolation and Cartesian Laplacian matrices.
6. Advance the heat equation with explicit Euler time stepping:

```matlab
unew = u + dt*(L*u + forcing);
u = E*unew;
```

7. Interpolate to plotting points and compare against the manufactured exact solution.

## 1. Circle example

File: `examples/example_heat_csg_circle_boolean_ops.m`

### Geometry and CSG setup

| Setting | Value | Notes |
| --- | --- | --- |
| Primitive type | circles | 2D closest-point geometry |
| `radius` | `1` | Both circles use the same radius |
| `shift` | `0.75` | Centers are shifted left/right along the x-axis |
| `cen1` | `[-shift 0]` | Left circle center |
| `cen2` | `[ shift 0]` | Right circle center |
| Primitive CP maps | `cpCircle(...)` | Wrapped in `csgLeaf(..., 2)` |
| CSG operations | `csgUnion`, `csgIntersection`, `csgDifference` | Evaluated in a loop over `trees` |
| Operation names | `{'union', 'intersection', 'difference'}` | Used for branching and plot titles |

### Embedding grid and discretization

| Setting | Value |
| --- | --- |
| `dx` | `0.004` |
| `x1d` | `(-2.2:dx:2.2)'` |
| `y1d` | `(-1.8:dx:1.8)'` |
| Grid | `[xx, yy] = meshgrid(x1d, y1d)` |
| Dimension `dim` | `2` |
| Interpolation degree `p` | `3` |
| FD order `order` | `2` |
| Bandwidth | `bw = rm_bandwidth(dim, p, order/2)` |
| Active band | `dist <= bw*dx` |

### Time stepping

| Setting | Value |
| --- | --- |
| Final time `Tf` | `0.2` |
| Initial timestep formula | `dt = 0.2*dx^2` |
| Step count | `numtimesteps = ceil(Tf/dt)` |
| Adjusted timestep | `dt = Tf / numtimesteps` |
| Scheme | Explicit Euler with closest-point extension each step |

Because `dx = 0.004`, the script starts from `dt = 3.2e-6`, then recomputes `dt` so the final time is hit exactly.

### Linear operators

| Operator | Construction |
| --- | --- |
| Closest-point interpolation | `E = interp2_matrix(x1d, y1d, cpxb, cpyb, p, band)` |
| Cartesian Laplacian | `L = laplacian_2d_matrix(x1d, y1d, order, band)` |

### Initial condition and forcing

The solution is parameterized by curve arclength `s` along the current CSG curve. The helper routines:

- split the CSG curve into two arcs,
- compute local arc coordinates,
- use a quartic profile with zero slope at arc junctions,
- apply a time factor `exp(-t)`.

Key helper functions:

- `manufactured_solution(s, t, arc_length1, arc_length2)`
- `manufactured_forcing(s, t, arc_length1, arc_length2)`
- `local_arc_coordinate(...)`

Profile definition:

```matlab
xi = eta ./ arc_length;
profile = 1 - 4*xi.^2 .* (1 - xi).^2;
u = exp(-t) .* profile;
```

### Operation-specific curve configuration

The overlap angle is

```matlab
alpha = acos(shift / radius)
```

With `radius = 1` and `shift = 0.75`, this is the shared geometric parameter controlling arc lengths and parameterizations.

#### `union`

| Setting | Value |
| --- | --- |
| Arc 1 length | `2*pi - 2*alpha` |
| Arc 2 length | `2*pi - 2*alpha` |
| Total curve length | `4*pi - 4*alpha` |
| Plot parameterization | two outer arcs, one from each circle |

#### `intersection`

| Setting | Value |
| --- | --- |
| Arc 1 length | `2*alpha` |
| Arc 2 length | `2*alpha` |
| Total curve length | `4*alpha` |
| Plot parameterization | two inner overlap arcs |

#### `difference`

| Setting | Value |
| --- | --- |
| Arc 1 length | `2*pi - 2*alpha` |
| Arc 2 length | `2*alpha` |
| Total curve length | `2*pi` |
| Plot parameterization | outer left-circle arc plus inner right-circle seam arc |

### Plotting and output

| Figure | Content |
| --- | --- |
| Figure 1 | Embedded solution in the `x-y` plane |
| Figure 2 | Solution vs. normalized curve parameter `theta_plot = 2*pi*s_plot/total_length` |
| Figure 3 | Pointwise error along the curve |

Additional plot settings:

- `nplot = 400` curve samples
- white figure background: `set(gcf, 'color', 'w')`
- Figure 1 axis limits: `[-2.2 2.2 -1.8 1.8]`
- overlay of the exact CSG curve in black

Console output per operation:

```matlab
fprintf('%s: %d band points, %d timesteps, max_err=%g\n', ...)
```

## 2. Sphere example

File: `examples/example_heat_csg_sphere_boolean_ops.m`

### Geometry and CSG setup

| Setting | Value | Notes |
| --- | --- | --- |
| Primitive type | spheres | 3D closest-point geometry |
| `radius` | `1` | Both spheres use the same radius |
| `shift` | `0.75` | Centers are shifted left/right along the x-axis |
| `alpha` | `acos(shift / radius)` | Cap opening angle at the CSG seam |
| `seam_radius` | `sqrt(radius^2 - shift^2)` | Radius of the seam circle |
| `reference_theta_span` | `pi - alpha` | Used to scale MMS amplitude by patch size |
| `cen1` | `[-shift 0 0]` | Left sphere center |
| `cen2` | `[ shift 0 0]` | Right sphere center |
| Primitive CP maps | `cpSphere(...)` | Wrapped in `csgLeaf(..., 3)` |
| CSG operations | `csgUnion`, `csgIntersection`, `csgDifference` | Evaluated in a loop over `trees` |
| Operation names | `{'union', 'intersection', 'difference'}` | Used for branching and plot titles |

### Embedding grid and discretization

| Setting | Value |
| --- | --- |
| `dx` | `0.05` |
| `x1d` | `(-2.6:dx:2.6)'` |
| `y1d` | `(-2.0:dx:2.0)'` |
| `z1d` | `y1d` |
| Grid | `[xx, yy, zz] = meshgrid(x1d, y1d, z1d)` |
| Dimension `dim` | `3` |
| Interpolation degree `p` | `3` |
| FD order `order` | `2` |
| Bandwidth | `bw = rm_bandwidth(dim, p, order/2)` |
| Active band | `dist <= bw*dx` |

### Time stepping

| Setting | Value |
| --- | --- |
| Final time `Tf` | `0.1` |
| Initial timestep formula | `dt = 0.1*dx^2` |
| Step count | `numtimesteps = ceil(Tf/dt)` |
| Adjusted timestep | `dt = Tf / numtimesteps` |
| Scheme | Explicit Euler with closest-point extension each step |

Because `dx = 0.05`, the script starts from `dt = 2.5e-4`, then adjusts `dt` to land exactly on `Tf`.

### Linear operators

| Operator | Construction |
| --- | --- |
| Closest-point interpolation | `E = interp3_matrix(x1d, y1d, z1d, cpxb, cpyb, cpzb, p, band)` |
| Cartesian Laplacian | `L = laplacian_3d_matrix(x1d, y1d, z1d, order, band, band)` |

### Initial condition and forcing

The manufactured solution is axisymmetric on each spherical patch and uses the polar angle `theta` measured relative to the local patch axis.

Key helper functions:

- `manufactured_solution(theta, theta_start, theta_end, reference_theta_span, t)`
- `manufactured_forcing(theta, theta_start, theta_end, reference_theta_span, radius, t)`
- `classify_surface_points(...)`

The patch-local coordinate is

```matlab
xi = (theta - theta_start) ./ theta_span;
amp = (theta_span ./ reference_theta_span).^2;
```

and the profile is

```matlab
profile = 1 - amp .* 4 .* xi.^2 .* (1 - xi).^2;
u = exp(-t) .* profile;
```

The amplitude factor `amp` reduces stiffness on smaller caps, especially for the intersection geometry.

### Operation-specific patch configuration

Each boolean surface is represented as two spherical patches with:

- `center`
- `axis_sign`
- `theta_start`
- `theta_end`

Patch metadata comes from `boolean_surface_patches(...)`.

#### `union`

| Patch | Center | `axis_sign` | `theta_start` | `theta_end` |
| --- | --- | --- | --- | --- |
| 1 | `cen1` | `1` | `alpha` | `pi` |
| 2 | `cen2` | `-1` | `alpha` | `pi` |

Interpretation: two large exterior spherical patches joined at the seam circle.

#### `intersection`

| Patch | Center | `axis_sign` | `theta_start` | `theta_end` |
| --- | --- | --- | --- | --- |
| 1 | `cen1` | `1` | `0` | `alpha` |
| 2 | `cen2` | `-1` | `0` | `alpha` |

Interpretation: two small overlap caps.

#### `difference`

| Patch | Center | `axis_sign` | `theta_start` | `theta_end` |
| --- | --- | --- | --- | --- |
| 1 | `cen1` | `1` | `alpha` | `pi` |
| 2 | `cen2` | `-1` | `0` | `alpha` |

Interpretation: one large left exterior patch plus one right-side seam cap.

### Surface sampling and plotting configuration

| Setting | Value |
| --- | --- |
| `ntheta` | `48` |
| `nphi` | `64` |
| `nmeridian` | `300` |
| `numops` | `numel(trees)` |

Sampling helpers:

- `sample_surface_patches(...)`
- `flatten_surface_samples(...)`
- `boolean_meridian_parameterization(...)`
- `axisymmetric_patch_coords(...)`

Figures:

| Figure | Content |
| --- | --- |
| Figure 1 | Surface solution rendered with `surf(...)` on sampled spherical patches |
| Figure 2 | Solution along a representative meridian |
| Figure 3 | Error along that meridian |

Additional plot settings:

- seam circle plotted with `phi_seam = linspace(0, 2*pi, 200)`
- Figure 1 axis limits: `[-2.2 2.2 -1.6 1.6 -1.6 1.6]`
- view angle: `view([-35 24])`
- interpolated shading: `shading interp`
- white figure background: `set(gcf, 'color', 'w')`

Console output per operation:

```matlab
fprintf('%s: %d band points, %d timesteps, max_err=%g\n', ...)
```

## Main differences between the two scripts

| Topic | Circle example | Sphere example |
| --- | --- | --- |
| Geometry dimension | 2D curves | 3D surfaces |
| Primitive CP function | `cpCircle` | `cpSphere` |
| Grid spacing | `dx = 0.004` | `dx = 0.05` |
| Final time | `Tf = 0.2` | `Tf = 0.1` |
| Timestep prefactor | `0.2*dx^2` | `0.1*dx^2` |
| Interpolation matrix | `interp2_matrix` | `interp3_matrix` |
| Laplacian matrix | `laplacian_2d_matrix` | `laplacian_3d_matrix` |
| MMS coordinate | arc length along piecewise curve | polar angle on spherical patches |
| Plot samples | `nplot = 400` | `ntheta = 48`, `nphi = 64`, `nmeridian = 300` |
| Figure 1 visualization | embedded planar band plot | 3D `surf(...)` patch rendering |

## How the manufactured exact solution is obtained

Both scripts use the method of manufactured solutions: they first choose a smooth exact solution on each CSG component, then compute the forcing term `f` so that

```matlab
u_t = Delta_s u + f
```

is satisfied exactly.

### Circle example

For `example_heat_csg_circle_boolean_ops.m`, the exact solution is built arc-by-arc along the boolean curve.

1. Each CSG curve is split into two arcs.
2. A global arclength parameter `s` is converted into a local arc coordinate `eta`.
3. That local coordinate is normalized to

```matlab
xi = eta ./ arc_length;
```

4. On each arc, the spatial profile is chosen as

```matlab
profile = 1 - 4*xi.^2 .* (1 - xi).^2;
```

5. The time dependence is

```matlab
u(s,t) = exp(-t) * profile.
```

This quartic profile has zero slope at the arc endpoints, which makes the solution join smoothly at the CSG seam points. The forcing function in `manufactured_forcing(...)` is then obtained by differentiating this exact solution in time and along the curve so that the PDE holds exactly on each arc.

### Sphere example

For `example_heat_csg_sphere_boolean_ops.m`, the exact solution is built patch-by-patch on the spherical surface pieces.

1. Each boolean surface is represented by two spherical patches.
2. Each closest point is classified onto one patch and assigned a local polar angle `theta`.
3. That local angular coordinate is normalized over the patch span:

```matlab
theta_span = theta_end - theta_start;
xi = (theta - theta_start) ./ theta_span;
```

4. A patch-dependent amplitude scaling is introduced:

```matlab
amp = (theta_span ./ reference_theta_span).^2;
```

5. The spatial profile is

```matlab
profile = 1 - amp .* 4 .* xi.^2 .* (1 - xi).^2;
```

6. The time dependence is again

```matlab
u(theta,t) = exp(-t) * profile.
```

This choice makes the solution axisymmetric on each spherical patch and gives zero meridional slope at poles and seam circles, so the exact solution stays continuous across the CSG joins. The forcing in `manufactured_forcing(...)` is derived from the time derivative and the axisymmetric surface Laplacian on the sphere:

```matlab
Delta_s u = (u_thetatheta + cos(theta)./sin(theta).*u_theta) / radius^2
```

with a special pole treatment when `sin(theta)` is very small.

## Practical interpretation

If you want to adjust the behavior of these examples, the main knobs are:

- geometry overlap: `shift`
- primitive size: `radius`
- grid resolution: `dx`
- interpolation order: `p`
- finite-difference order: `order`
- final time: `Tf`
- timestep scale: the coefficient in `dt = c*dx^2`
- plot density: `nplot` for curves, or `ntheta`, `nphi`, `nmeridian` for surfaces

The rest of each script is mostly operation-specific parameterization and MMS bookkeeping needed to generate exact solutions and error plots for the three CSG geometries.
