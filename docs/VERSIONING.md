# LROM Versioning Scheme

## The rule

**Notebook milestone N ↔ version N.0.** This is the naming law for the whole
project — record-keeping agents must apply it to every future notebook:

| Old label | Version | Milestone |
|-----------|---------|-----------|
| N1        | **1.0** | Wavefunction emulation, legacy-benchmark validation (validated) |
| N2        | **2.0** | Cross sections (current) |
| N3        | **3.0** | Global Koning-Delaroche emulator across isotopes/energies |

Small fixes within a milestone increment the minor digit (1.1, 1.2, …).
Do **not** label the next notebook's work as a minor bump of the previous
one — Notebook 2 work is 2.0, never 1.x.

In Python, where `1.0` is not a valid identifier, version modules are named
`v1_0`, `v2_0`, …: the 1.0 snapshot imports as `lrom_legacy.v1_0`.

## Milestone flow

The project goal (see `scientific_archive/references/Paper Results Map.pdf`,
distilled in `_agents/backlog/paper-results-roadmap.md`) is a series of paper
notebooks. Each package milestone is validated against the corresponding
archived legacy notebook before the paper notebook is produced:

1. **1.0 (validated)**: `notebooks/Benchmark_1.0.ipynb` recreates the archived
   `scientific_archive/legacy_code/Legacy_benchmark/notebooks/02_lrom_method_walkthrough.ipynb`
   through the `lrom_legacy.v1_0` public API without copying legacy code.
   Spec: `_agents/superpowers/specs/2026-06-30-benchmark-n1-notebook-design.md`.
   Verified: figures match legacy (violin medians agree on all shared
   categories); notebook 01 satisfies the Paper Results Map and executes
   end-to-end.
2. **2.0 (current)**: the active `lrom` package (2.0.0) adds cross-section
   observables (`observable="cross_section"`, S-matrices, spin-orbit
   channels) with the physics fixes applied (ROSE `KD_simple` sign
   convention with positive `Wv`/`Wd`; `Rso` derived from the target via
   `full_woods_saxon_central(target_a=...)`; `sph_harm` shim angle order).
   `notebooks/Benchmark_2.0.ipynb` recreates the archived
   `03_cross_section_cat_comparison.ipynb` live (no caches).
   Spec: `_agents/superpowers/specs/2026-07-12-benchmark-2-0-notebook-design.md`.

## Deliberate design decisions (do not "fix")

- **The ROSE comparisons in benchmark notebooks run the public `rose`
  package natively** — inline `InteractionEIMSpace`, base solver, and
  `ScatteringAmplitudeEmulator.from_train(...)` in notebook cells, with
  ROSE's native free-reference basis. The apples-to-oranges basis
  conventions are intentional: coefficients are compared only within each
  convention; observables (wavefunctions, cross sections) compare directly.
- **No plotting wrappers.** Every figure cell contains its complete inline
  Matplotlib code. Enforced by the benchmark contract tests.
- **Display floors** are display-only clipping; underlying arrays are
  untouched (`DIFFERENCE_FLOOR = 1e-5` for 1.0 figures, `1e-6` for 2.0
  cross-section error figures).
- EIM `training_info` bounds include testing rows; changing this is a
  scientific decision reserved to the user.
