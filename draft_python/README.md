# Draft Python Closest Point Matrices

This is a Python 3 draft of the core Closest Point Method matrix builders.
It is intentionally separate from the legacy `python/` directory.

The first target is parity with the MATLAB routines that build:

- interpolation matrices `E`
- finite-difference Laplacian matrices `L`
- band/index helpers
- simple closest-point surfaces

Python APIs use 0-based indices for bands.
