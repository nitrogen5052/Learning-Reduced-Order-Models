# LROM Versioning And Naming

## Package versions

| Version | Milestone | Where |
|---------|-----------|-------|
| **1.x** | Wavefunction emulation (current: 1.1.0, single-file package) | `lrom/` |
| **2.0** | Cross sections — parked, needs fixes | `lrom_legacy/v2_0/` |
| **3.0** | Global Koning-Delaroche emulator (future) | — |

Minor digits are fixes/restructures within a milestone (1.1.0 = 1.0 physics
in the simplified single-file architecture). The next notebook milestone is
never a minor bump of the previous one.

`1.0` is not a valid Python identifier, so parked package snapshots use
module names like `v2_0` (`lrom_legacy.v2_0`).

## Benchmark naming (law since 2026-07-13)

**Benchmark notebooks are named after the legacy notebook they recreate**:
`benchmark_02.ipynb` recreates
`scientific_archive/legacy_code/Legacy_benchmark/notebooks/02_lrom_method_walkthrough.ipynb`.
A future cross-section benchmark would be `benchmark_03.ipynb` (legacy 03),
and so on. This replaces the earlier `Benchmark_<version>` naming.

## Milestone flow

The project goal (see `_agents/backlog/paper-results-roadmap.md`, distilled
from the Paper Results Map) is a series of paper notebooks. Each package
milestone is validated against the corresponding archived legacy notebook
before its paper notebook is finalized. Version 1 physics is currently being
re-verified against `benchmark_02` — that is why 2.0 is parked.

Known 2.0 fixes before it resumes (recorded in `lrom_legacy/__init__.py`):
potential predictors carry no spin-orbit parameter information, and the CAT
per-sample timing methodology is too noisy.

## Deliberate design decisions (do not "fix")

- **One-file package.** `lrom/__init__.py` deliberately holds the whole
  version-1 workflow; do not split it back into modules.
- **The ROSE comparisons in benchmark notebooks run the public `rose`
  package natively** — inline `InteractionEIMSpace`, base solver, and
  `ScatteringAmplitudeEmulator.from_train(...)` in notebook cells, with
  ROSE's native free-reference basis. Coefficients are compared only within
  one basis convention; wavefunctions and errors compare directly.
- **No plotting wrappers** in benchmark notebooks; all Matplotlib inline
  (enforced by the contract tests).
- **Display floors** are display-only clipping (`DIFFERENCE_FLOOR = 1e-5`);
  underlying arrays are untouched.
- EIM `training_info` bounds include testing rows; changing this is a
  scientific decision reserved to the user.
