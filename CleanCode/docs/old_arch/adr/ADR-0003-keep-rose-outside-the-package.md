# ADR-0003: Keep ROSE outside the package

- **Status:** Accepted
- **Date:** 2026-09-01
- **Evidence:** Commit `289777e`; `lrom_git/CleanCode/benchmarks/rose.py`; `lrom_git/CleanCode/README.md` § “Reproducing the notebooks”; `lrom_git/CleanCode/docs/architecture.json`

## Context

ROSE supplies an external full-order comparison, not the implementation of the
learned emulator. Importing it into `lrom` would couple the reusable package to
a benchmark dependency and blur the boundary between the model under study and
its external reference.

## Decision

Keep the ROSE adapter and cached reference arrays under `benchmarks/`. Use ROSE
only as an external full-order comparison from benchmark or notebook code;
never place ROSE integration inside the `lrom` package.

## Consequences

The `lrom` source and public API remain independent of ROSE, and comparisons
stay methodologically visible. Rebuilding reference arrays remains a benchmark
workflow rather than a package API; callers cannot obtain ROSE results through
`lrom`. Adapter and cache provenance must be maintained separately from package
code.
