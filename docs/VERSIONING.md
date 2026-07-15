# LROM Versioning And Naming

## Package versions

| Version | Milestone | Where |
|---------|-----------|-------|
| **2.0** | Cross sections + wavefunctions (current, single-file package) | `lrom/` |
| **1.2** | Validated wavefunction-only snapshot | `lrom_legacy/v1_2/` |
| **2.0 (first attempt)** | Parked parts donor | `lrom_legacy/v2_0/` |
| **3.0** | Global Koning-Delaroche emulator (future) | — |

Minor digits are fixes/restructures within a milestone (1.1.0 = 1.0 physics
in the simplified single-file architecture). The next notebook milestone is
never a minor bump of the previous one.

`1.0` is not a valid Python identifier, so parked package snapshots use
module names like `v2_0` (`lrom_legacy.v2_0`).

## Benchmark naming (law since 2026-07-13)

**Benchmark notebooks are named after the legacy notebook they recreate**:
`benchmark_notebooks/1.0/benchmark_02.ipynb` recreates
`scientific_archive/legacy_code/Legacy_benchmark/notebooks/02_lrom_method_walkthrough.ipynb`.
`benchmark_notebooks/2.0/benchmark_03.ipynb` recreates legacy 03 (cross-section
CAT). Benchmarks live under `notebooks/benchmark_notebooks/<version>/`;
`benchmark_notebooks/2.0/benchmark_01.ipynb` additionally proves 2.0
reproduces notebook 01 and matches the `v1_2` snapshot bit-for-bit.

## Milestone flow

The project goal (see `.agents/backlog/paper-results-roadmap.md`, distilled
from the Paper Results Map) is a series of paper notebooks. Each package
milestone is validated against the corresponding archived legacy notebook
before its paper notebook is finalized. Version 2.0 fixed the first attempt's flaws: spin-orbit-AWARE potential
predictors (maxvol over stacked central + spin-orbit profiles with per-block
variance normalization) and min-of-repeats CAT timing with a cached
S-matrix assembler.

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
