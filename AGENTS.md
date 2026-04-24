# AGENTS.md

## Scope

This repository should be treated as a MATLAB project.

- Use MATLAB for all code changes, examples, tests, and documentation updates.
- Do not add new Python features, ports, or maintenance work.
- Ignore the `python/` directory unless the user explicitly asks to inspect historical code for reference.

## Project Purpose

This codebase implements the Closest Point Method for numerical PDEs on curves and surfaces.

The main MATLAB areas are:

- `cp_matrices/`: sparse matrix builders, interpolation operators, finite-difference operators, refinement utilities, and MATLAB unit tests.
- `surfaces/`: closest-point surface definitions, parameterizations, triangulation helpers, and surface-focused tests.
- `examples/`: runnable MATLAB examples for heat, advection, eigenvalue, biharmonic, reaction-diffusion, and boundary-condition problems.

## Primary Entry Points

When getting oriented, start here:

- `README.md`
- `cp_matrices/README.txt`
- `surfaces/README.txt`
- `examples/setupPaths.m`
- `cp_matrices/setupPaths.m`

Useful MATLAB test runners:

- `cp_matrices/run_unit_tests.m`
- `surfaces/run_unit_tests.m`

## Working Rules

- Prefer modifying existing MATLAB workflows rather than introducing new abstractions.
- Preserve the current directory structure and naming conventions, which are MATLAB-oriented and function-per-file.
- Keep compatibility with the repo's existing MATLAB style unless the user asks for a broader refactor.
- Be careful with path setup code and relative-directory assumptions; several scripts expect to be run from specific folders.
- Do not remove or rewrite legacy code unless it is necessary for the requested task.
- Do not modify the current files under `cp_matrices/` by default. Treat that directory as read-only unless the user explicitly asks for a change there.

## Git Workflow

- Do not work directly on `main`.
- Use the `ai-edit` branch for changes made during this project.
- Do not merge `ai-edit` into `main` unless the user explicitly instructs you to do so.
- If branch work is needed and the current branch is unclear, pause and verify before taking git actions that affect branch state.

## Validation

When validating MATLAB changes, prefer:

1. Running the smallest relevant example or test first.
2. Running `cp_matrices/run_unit_tests.m` for matrix/operator changes.
3. Running `surfaces/run_unit_tests.m` for geometry/surface changes.

If MATLAB execution is not available, state clearly what was not run.

## Out of Scope By Default

- Python package cleanup or modernization
- PETSc or Cython work under `python/`
- Cross-language parity changes between MATLAB and Python

Only do those if the user explicitly requests them.
