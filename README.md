cpSurf: Surface Computing using the Closest Point Method
========================================================
Matlab and Python implementations of the Closest Point Method.

For more information and publications about the Closest Point Method, see

http://people.maths.ox.ac.uk/macdonald/closestpoint


# Status

This is work in process, so the usual disclaimers apply.

TODO: start a list of contributors

TODO: add a reference to cite if one uses this

# 2026 examples

`examples_2026/` groups examples by manifold type and embedding dimension:

| Folder | Examples |
| --- | --- |
| `2D_curve/` | Circle, V-shaped line, planar cusp, double-convex lens, curvature mismatch, and ellipse experiments |
| `3D_curve/` | Cusp unfolded in three-dimensional space |
| `3D_surface/` | Glued sphere caps, flipped ellipsoid caps, and half-twisted rectangular tube |

Geometry-specific experiment helpers live with their examples. Added geometry
routines absent from [cbm755/cp_matrices](https://github.com/cbm755/cp_matrices)
live in `examples_2026/surfaces/`: `cpEllipseArc.m`,
`cpHalfTwistedRectTube.m`, and `halfTwistedRectTubeFrame.m`. Upstream geometry
routines remain in the repository's top-level `surfaces/`.

Figures and saved results remain shared in `examples_2026/figs/`; the theory
notes remain in `examples_2026/angle_rotation_glue.md`.

From the repository root, add all example subfolders to the MATLAB path:

```matlab
addpath(genpath('examples_2026'));
```

Scripts resolve library and output paths relative to their own files, rather
than the current working directory.

# Geometry preview

Run `examples_2026/3D_surface/example_half_twisted_rectangular_tube_geometry.m` in MATLAB
to view the half-twisted rectangular tube with `R = 1`, `a = 0.45`, and
`b = 0.20`. The figure shows the complete surface, the wide face pair, and
the narrow face pair, with both closed edge curves in red and blue.
Each face pair is sampled once over `theta` in `[0,4*pi]`; opposite faces
exchange after `2*pi`. The script saves
`examples_2026/figs/half_twisted_rectangular_tube_geometry.png` and does not
run a convergence experiment.

# Narrow-band visualization

Run the plotting-only function from the repository root:

```matlab
addpath('examples_2026/3D_surface');
[fig, bands] = example_half_twisted_rectangular_tube_narrow_band(1/25);
```

It shows the manifold and the actual operator outer-band grid points:
blue for branch A, orange for branch B, and magenta for nodes routed
through a branch rotation. Rotation colour takes priority; remaining
nodes shared by A and B have a blue dot and an orange ring.
The plot uses `halfTwistedRectTubeBands.m`, the same band builder as the
convergence experiment, without an elliptic solve. It saves
`examples_2026/figs/half_twisted_rectangular_tube_narrow_band.png`.

# Half-twisted tube convergence experiment

Run `examples_2026/3D_surface/example_half_twisted_rectangular_tube_rotation_convergence.m`
in MATLAB, optionally with a vector of grid spacings. From the repository root:

    matlab -batch "addpath('examples_2026/3D_surface'); example_half_twisted_rectangular_tube_rotation_convergence"
    matlab -batch "addpath('examples_2026/3D_surface'); example_half_twisted_rectangular_tube_rotation_convergence(1./[25 30 40])"

It splits the tube into its wide and narrow strips along both closed edge
curves and solves `u - Lap_S u = f` with the manufactured solution
`examples_2026/3D_surface/halfTwistedRectTubeMMS.m`. It compares the exact iCPM glue
rotation with the `d_k2` estimated rotation. The script prints and returns
surface sup errors per level, rates computed from the actual `h` ratios, and a
least-squares fitted rate. It saves
`examples_2026/figs/half_twisted_rectangular_tube_rotation_convergence.png`
and `.mat`. See section 10 of `examples_2026/angle_rotation_glue.md`.


# Other people's code

readply:
    MATLAB functions to read and write 3D data PLY files
    by Pascal Getreuer, 2004.  License: unknown.
    This code doesn't seem to work in Octave but I haven't
    tried to fix it.

ba_interp:
    fast matlab interpolation (not currently included in this
    code, but very useful).

surfaces:
    Some surface triangulations are included in surfaces/tri/
    Each should have a README file associated with it.
    These are mostly from Aim@Shape and probably have their own licenses.


