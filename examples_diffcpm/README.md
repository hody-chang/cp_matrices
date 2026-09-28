# examples_diffcpm

Runnable experiments for the `../diffcpm` differentiable closest point method
layer. Each script is self-contained, prints its own findings, needs no
arguments, and runs on CPU.

| script | what it shows | runtime |
|---|---|---|
| `ex1_source_recovery_circle.py` | recover a source on a circle from 12 noisy sensors; regularization vs. identifiability | ~4 min |
| `ex2_diffusivity_recovery_sphere.py` | recover a spatially varying diffusivity on a sphere — nonlinear in the unknown, so `dA/dtheta` carries the gradient; sensor-count sweep | ~8 min |
| `ex3_shape_derivative_convergence.py` | does the discrete CPM shape derivative converge to the continuous one? measured against a closed form | ~12 min |
| `ex4_learned_sdf_error_propagation.py` | exchange rate between SIREN fit quality and recovered-parameter error | ~6 min |
| `ex5_shape_optimization_ellipse.py` | spectrum of the discrete surface Laplacian vs. the exact perimeter formula; then recover ellipse axes by shape optimization, with a cross-resolution control that measures the real discretization bias | ~6 min |

Suggested reading order: `ex1` (does the layer work), `ex3` (is the geometry
gradient the right one), `ex5` (does shape optimization run), then `ex2` and
`ex4` for the two application shapes.

The headline results are summarised in `../diffcpm/README.md`. Two are not what
you would guess: `ex3` finds that the discrete shape derivative converges more
slowly than the objective it differentiates, and `ex5` shows why a noise-free
synthetic recovery that hits machine precision is measuring self-consistency
rather than accuracy.

Runtimes are rough, measured with three scripts sharing four CPU cores, in
float64. They are dominated by linear solves on matrices of one to a few thousand
unknowns, by L-BFGS iteration counts, and by XLA compilation — each fresh
objective closure recompiles the whole forward-and-adjoint graph, which is a
noticeable share of the shorter runs.
