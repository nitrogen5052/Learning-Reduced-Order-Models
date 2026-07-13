# Simplify To Single-File v1 Package Implementation Plan

> Written with the superpowers `writing-plans` skill. Executed with
> `verification-before-completion`: every claim below is backed by a fresh
> command run before commit.

**Goal:** Return version 1 (wavefunction emulation) as the dominant `lrom`
package, consolidated into ONE file exposing one object with `.sampling`,
`.train`, `.predict`; park version 2.0 in `lrom_legacy/v2_0`; delete the
confusing extras; rename benchmarks after the legacy notebook they recreate.

**User requirements (2026-07-13):**
1. Simplify the project directory as much as possible.
2. `lrom` = v1 physics, single-file if possible (one object: `.sampling`,
   `.train`, `.predict`); otherwise max physics with the simplest
   architecture in that shape.
3. Save 2.0 in legacy with a note that it needs fixes; benchmark 1.0 physics
   still needs verification.
4. Delete new/confusing notebooks (02 skeleton, Benchmark_2.0) and
   Benchmark_01 + `lrom_bench` (purpose unclear to the user).
5. Benchmarks named for the legacy notebook they recreate:
   `benchmark_02.ipynb` recreates legacy `02_lrom_method_walkthrough.ipynb`.
6. benchmark_02: no wrapper functions, package-driven, simple comments for a
   physics-literate reader, violin plot styled like the legacy notebook.
7. The package must recreate notebook 01 and benchmark_02.

## File structure after the change

- `lrom/__init__.py` — the entire package: errors, potentials, state,
  sampling designs, ROSE FOM boundary, centered SVD basis, predictors,
  RF-LROM fit/solve, training engine (incl. shared-basis ROSE RBM needed by
  notebook 01), portable artifacts, and the public `LROM` object + `load()`.
  Version `1.1.0` (1.0 physics, simplified architecture).
- `lrom_legacy/v2_0/` — verbatim snapshot of the 2.0 package (cross
  sections). Docstring notes the open fixes (spin-orbit-blind predictors,
  noisy CAT timing).
- `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb` (regenerated,
  executed) and `notebooks/benchmark_02.ipynb` (renamed, package-driven,
  legacy-style violin, executed).
- Deleted: `lrom_bench/`, `lrom_legacy/v1_0/`, `notebooks/Benchmark_01.ipynb`,
  `notebooks/Benchmark_2.0.ipynb`, `notebooks/02_lrom_method_walkthrough.ipynb`,
  9 lrom_bench test files, `test_v1_0_rose_boundary.py`,
  `test_benchmark_2_0_notebook.py`.

## Tasks

1. [ ] Snapshot 2.0: `git mv lrom lrom_legacy/v2_0`; update
   `lrom_legacy/__init__.py`; add fix-me note. Verify `import lrom_legacy.v2_0`.
2. [ ] Write single-file `lrom/__init__.py` by consolidating the v1-line
   modules (strip all 2.0 cross-section code). Verify import + smoke train.
3. [ ] Delete `lrom_bench`, `v1_0`, extra notebooks, and their tests; update
   surviving test imports (`from lrom import ...` instead of submodules);
   update `pyproject.toml` (version 1.1.0, drop `lrom_bench*`).
4. [ ] Run the full suite; fix until green. Commit.
5. [ ] Rename `Benchmark_1.0.ipynb` → `benchmark_02.ipynb`; retarget to
   `import lrom`; add concise physics-literate comments; restyle the violin
   to the legacy five-category split (ROSE LS floor, LROM LS floor, ROSE
   ROM, linear LROM, predictor LROM; log axis, median markers, 1e-5 display
   floor). Update the contract test. Execute end-to-end.
6. [ ] Regenerate notebook 01 from the generator; execute end-to-end.
7. [ ] Update README, CLAUDE.md, docs/VERSIONING.md (benchmark naming law:
   benchmark_<legacy notebook>; package versions unchanged).
8. [ ] Full suite green; commit; push to GitHub (user's standing rule).
