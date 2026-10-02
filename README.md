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


