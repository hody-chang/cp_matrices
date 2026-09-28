"""Differentiable linear solves via the adjoint, not by unrolling.

``jax.lax.custom_linear_solve`` is the whole trick: give it the forward matvec,
a solver, and a solver for the transpose, and reverse-mode differentiation
produces exactly the textbook adjoint recipe

    forward   A u = b
    adjoint   A^T lam = dJ/du
    gradient  dJ/dp = lam^T (db/dp - (dA/dp) u)

The last line is assembled by JAX from the VJP of the matvec closure, so any
quantity the matvec depends on -- source terms, coefficients, interpolation
weights and hence the geometry -- gets a gradient with no extra code.  Nothing
is unrolled, so the memory cost is independent of the number of Krylov
iterations, and the gradient is the exact adjoint of the *discrete* operator
rather than of the iteration that approximated it.
"""

import jax
import jax.numpy as jnp


def _dense_solve(matvec, b):
    n = b.shape[0]
    A = jax.vmap(matvec)(jnp.eye(n, dtype=b.dtype)).T
    return jnp.linalg.solve(A, b)


def _gmres_solve(matvec, b, tol, atol, restart, maxiter):
    x, _ = jax.scipy.sparse.linalg.gmres(
        matvec, b, tol=tol, atol=atol, restart=restart, maxiter=maxiter,
        solve_method="batched",
    )
    return x


def linear_solve(matvec, b, *, method="auto", tol=1e-12, atol=0.0, restart=50,
                 maxiter=400, dense_limit=2500):
    """Solve ``A u = b`` with a solve that differentiates by the adjoint.

    ``method``:
      ``dense``  materialize A and use LU.  Exact; use for verification and for
                 small problems.  Cost O(n^3), and O(n) matvecs to assemble.
      ``gmres``  matrix-free restarted GMRES.  A is nonsymmetric (``E`` is), so
                 CG is not an option.
      ``auto``   dense below ``dense_limit`` unknowns, GMRES above.

    The default ``dense_limit`` is conservative on purpose: dense LU cannot fail
    to converge, which is what you want from the path the tests compare against.
    It is not the fast path -- on a 2D band of ~1000 unknowns GMRES is around 6x
    quicker per value-and-gradient, and agrees to every printed digit -- so pass
    ``method="gmres"`` explicitly once the gradients have been checked.

    **``restart`` is paid in full on every call.**  ``solve_method="batched"``
    builds the entire restart-sized Krylov basis whether or not the residual
    already fell after twenty vectors, so ``restart`` is a cost rather than a
    budget.  Measured on a 2,137-point 2D band: ``restart=250`` cost 147 s per
    forward solve and 270 s per value-and-gradient, against 3.3 s and 6.6 s at
    ``restart=50``, for the same 1.8e-13 residual.  A factor of 44 for nothing,
    and it silently dominated every example in this package until it was measured.
    Raise ``maxiter``, not ``restart``, when a problem needs more iterations;
    raise ``restart`` only if the outer iteration stalls, which is the
    convergence that restarting gives up.
    """
    n = b.shape[0]
    if method == "auto":
        method = "dense" if n <= dense_limit else "gmres"

    if method == "dense":
        solve = _dense_solve
    elif method == "gmres":
        def solve(mv, rhs):
            return _gmres_solve(mv, rhs, tol, atol, restart, maxiter)
    else:
        raise ValueError(f"unknown method {method!r}")

    return jax.lax.custom_linear_solve(
        matvec, b, solve, transpose_solve=solve, symmetric=False
    )


def solve_operator(op, b, **kwargs):
    """Solve ``op @ u = b`` for a :class:`~diffcpm.operators.CPMOperator`."""
    return linear_solve(op.matvec, b, **kwargs)
