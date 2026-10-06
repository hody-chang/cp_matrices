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
| `2D_curve/` | Circle, V-shaped line, planar cusp, double-convex lens, curvature mismatch, and ellipse experiments, including the Rodrigues-against-`atan2` comparison for the glue rotation |
| `3D_curve/` | Cusp unfolded in three-dimensional space |
| `3D_surface/` | Glued sphere caps, flipped ellipsoid caps, and half-twisted rectangular tube |

Geometry-specific experiment helpers live with their examples. Added geometry
routines absent from [cbm755/cp_matrices](https://github.com/cbm755/cp_matrices)
live in `examples_2026/surfaces/`: `cpEllipseArc.m`,
`cpHalfTwistedRectTube.m`, and `halfTwistedRectTubeFrame.m`. Upstream geometry
routines remain in the repository's top-level `surfaces/`.

The glue rotation at a branch junction is built in `examples_2026/rotations/`,
shared by every example that glues two branches:

| File | What it does |
| --- | --- |
| `rot2d_from_to.m` | the plane rotation carrying one direction onto another, by Rodrigues' formula: the cosine from a dot product and the sine from a cross product, returned as a `(cos, sin)` pair with no angle formed |
| `glue_rot2d.m` | the same thing with the junction's convention applied, outward tangent onto the other branch's inward tangent |
| `rotate_about2d.m` | applies a `(cos, sin)` pair about a junction point |
| `rot2d_err.m` | the angle and matrix error of a rotation against a reference, through their relative rotation rather than a wrapped difference of angles |

Nothing stores a glue angle any more. Angles are derived from the pairs for
printing only, which is why `rotate_about2d` and `rot2d_err` take the pairs:
an angle argument would have to be turned back into a cosine and a sine at
every call site, which is the round trip `rot2d_from_to` exists to avoid.
`examples_2026/2D_curve/example_ellipse_cut_rotation_construction.m` measures
this construction against the earlier `atan2`-first one on the cut ellipse --
accuracy against a rotation known in closed form, the identities a rotation
should satisfy exactly, how far the difference propagates into the solution
(not far: it is rounding), and cost.

Figures and saved results remain shared in `examples_2026/figs/`; the theory
notes remain in `examples_2026/angle_rotation_glue.md`, whose section 3 covers
the glue rotation step by step.

From the repository root, add all example subfolders to the MATLAB path:

```matlab
addpath(genpath('examples_2026'));
```

Scripts resolve library and output paths relative to their own files, rather
than the current working directory.

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


