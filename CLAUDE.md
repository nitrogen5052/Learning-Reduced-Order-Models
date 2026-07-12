# LROM_Project — Agent Orientation

Read this before changing anything in this repository.

## Versioning law (do not deviate)

Notebook milestone N ↔ version N.0. "N1" = **1.0** (current, being validated),
"N2" = **2.0** (cross sections, not started on `main`), "N3" = **3.0**.
Minor fixes within a milestone are 1.1, 1.2, …; the next notebook is never a
minor bump. Python modules encode versions as `v1_0`, `v2_0`
(`lrom_legacy.v1_0` is the 1.0 snapshot). Full rules: `docs/VERSIONING.md`.

## Current state

- `main` holds the version 2.0 line: the `lrom` package (2.0.0) with
  cross-section observables (`observable="cross_section"`, S-matrices,
  spin-orbit channels, physics-sign fixes), validated by
  `notebooks/Benchmark_2.0.ipynb` against the archived legacy CAT notebook
  (`scientific_archive/.../03_cross_section_cat_comparison.ipynb`).
- Version 1.0 (validated) remains importable as `lrom_legacy.v1_0`; the 1.0
  benchmark notebooks (`Benchmark_1.0.ipynb`, `Benchmark_01.ipynb`,
  notebook 01) are retained as the wavefunction record.
- 2026-07-12: the original working copy under iCloud-synced
  `~/Documents/Documents-Agent/LROM_Project` suffered mass file eviction
  (dataless placeholders) that destroyed its local `.git`; this repository
  was reconstructed from GitHub plus surviving working files. Local commit
  history between 2026-06-15 and 2026-07-12 exists only as the snapshot
  commit and the `_memory` daily notes. **Keep this repo outside
  iCloud-synced paths and push to GitHub after every milestone.**

## Layout

- `lrom/`, `lrom_bench/`, `lrom_legacy/`, `notebooks/`, `docs/`, `tests/`,
  `scientific_archive/`, `outputs/` — researcher-facing.
- `_agents/` — **agent working area** (plans, specs, backlog, validation
  notebooks, generation scripts). See `_agents/README.md`. Put new agent
  plans/specs/scratch there, not in the root or `docs/`.
- `scientific_archive/` is read-only reference material. Never modify it.

## Deliberate decisions — do not "fix"

- Benchmark_1.0's ROSE comparison runs the public `rose` package natively,
  inline in notebook cells, with ROSE's free-reference basis while the LROM
  uses a central-reference basis. This apples-to-oranges setup is intentional
  (spec: `_agents/superpowers/specs/2026-06-30-benchmark-n1-notebook-design.md`).
- No plotting wrapper functions in benchmark notebooks; all Matplotlib inline.
- `DIFFERENCE_FLOOR = 1e-5` display floor in difference figures.
- EIM `training_info` bounds include testing rows; changing this is a
  scientific decision reserved to the user.

## Workflow expectations

- Run `python -m pytest -q` before and after changes; keep the suite green.
- The workspace-level memory protocol (`../CLAUDE.md`,
  `../_memory/Context/active-state.md`) applies here too.
