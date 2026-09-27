# examples_diffcpm

Runnable experiments for the `../diffcpm` differentiable closest point method
layer. Each script is self-contained, prints its own findings, needs no
arguments, and runs on CPU.

| script | what it shows | runtime |
|---|---|---|
| `ex1_source_recovery_circle.py` | recover a source on a circle from 12 noisy sensors; regularization vs. identifiability | ~2 min |
| `ex2_diffusivity_recovery_sphere.py` | recover a spatially varying diffusivity on a sphere — nonlinear in the unknown, so `dA/dtheta` carries the gradient | ~10 min |
| `ex3_shape_derivative_convergence.py` | does the discrete CPM shape derivative converge to the continuous one? measured against a closed form | ~15 min |
| `ex4_learned_sdf_error_propagation.py` | exchange rate between SIREN fit quality and recovered-parameter error | ~10 min |
| `ex5_shape_optimization_ellipse.py` | spectrum of the discrete surface Laplacian vs. the exact perimeter formula; then recover ellipse axes by shape optimization | ~15 min |

Suggested reading order: `ex1` (does the layer work), `ex3` (is the geometry
gradient the right one), `ex5` (does shape optimization run), then `ex2` and
`ex4` for the two application shapes.

The headline results are summarised in `../diffcpm/README.md`; `ex3` is the one
with a result that is not what you would guess.

Runtimes are for a single CPU core with float64 and dense LU solves, chosen for
verifiability rather than speed. They are dominated by `jnp.linalg.solve` on
matrices of a few thousand unknowns and by L-BFGS iteration counts.
