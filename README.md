cpSurf: Surface Computing using the Closest Point Method
========================================================
Matlab and Python implementations of the Closest Point Method.

For more information and publications about the Closest Point Method, see

http://people.maths.ox.ac.uk/macdonald/closestpoint


# Differentiable CPM (JAX)

`diffcpm/` is a Python 3 / JAX implementation of the closest point method whose
solve is differentiable with respect to the PDE coefficients *and* the geometry,
including the weights of a neural implicit surface. It is for PDE-constrained
optimization and inverse problems on implicit surfaces, with no meshing step.
Runnable experiments are in `examples_diffcpm/`. See `diffcpm/README.md` for what
it does, what is verified, and what its limitations are. It follows the
conventions of the MATLAB `cp_matrices/` but is otherwise independent of it, and
of the older Python 2 code in `python/`.


# Status

This is work in process, so the usual disclaimers apply.

TODO: start a list of contributors

TODO: add a reference to cite if one uses this


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


