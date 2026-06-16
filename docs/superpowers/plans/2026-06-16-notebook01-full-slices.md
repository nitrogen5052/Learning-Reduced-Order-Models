# Notebook 01 Full Slices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all Notebook 1 implementation slices so the notebook contains executable Vv-only and Vv/Rv/av RBM-vs-LROM sections.

**Architecture:** Add small ROSE/FOM helper boundaries only where Notebook 1 needs them. Keep plotting code and scientific sequencing visible in notebook cells. Do not add package plotting functions or a one-call notebook workflow function.

**Tech Stack:** Python 3.11+, NumPy, SciPy, nbformat, pytest, numba through nuclear-rose, nuclear-rose imported as `rose`, Matplotlib inside notebook cells only.

---

## Task 1: Real Woods-Saxon Problem Boundary

**Files:**
- Modify: `lrom_bench/rose_fom.py`
- Modify: `tests/test_rose_fom_real_ws.py`

Steps:

- [ ] Add failing tests for:
  - `real_woods_saxon_interaction` matching `real_woods_saxon_potential`.
  - `make_real_ws_problem` returning `rho_mesh`, `r_mesh`, and finite wavefunctions for a tiny `l=0` problem.
  - `make_real_ws_custom_basis` and `make_real_ws_rbe` returning coefficients and RBM wavefunctions with expected shapes.
- [ ] Run `python -m pytest tests/test_rose_fom_real_ws.py -v` and confirm failure for missing helpers.
- [ ] Implement only these helpers:
  - `real_woods_saxon_interaction`
  - `RealWSProblem`
  - `make_real_ws_problem`
  - `make_real_ws_custom_basis`
  - `make_real_ws_rbe`
- [ ] Run `python -m pytest tests/test_rose_fom_import.py tests/test_rose_fom_real_ws.py -v` and confirm pass.
- [ ] Commit with `git commit -m "Add real Woods-Saxon problem boundary"`.

## Task 2: Full Notebook 1 Generator Cells

**Files:**
- Modify: `scripts/generate_notebook01.py`
- Modify: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`

Steps:

- [ ] Replace placeholder cells with visible notebook code for:
  - Vv-only wavefunction solves.
  - Vv-only basis and LS floor.
  - Vv-only ROSE/RBM coefficients and RF-LROM coefficients.
  - Vv-only wavefunction error comparison.
  - Three-parameter samples and rainbow summaries.
  - Raw-parameter LROM diagnostic.
  - Potential predictor pack and maxvol point visualization.
  - Three-parameter performance table.
- [ ] Keep all plotting code inline in cells.
- [ ] Regenerate with `python scripts/generate_notebook01.py`.
- [ ] Verify headings, stable IDs, visible plotting, and absence of `plot_notebook` or `run_notebook01`.
- [ ] Commit with `git commit -m "Fill Notebook 1 RBM LROM sections"`.

## Task 3: Verification

**Files:**
- No new files.

Steps:

- [ ] Run `python -m pytest -v`.
- [ ] Run `python scripts/generate_notebook01.py` twice and confirm no Notebook 1 diff.
- [ ] Run a reduced direct Notebook 1 smoke using smaller `Notebook01Config` values to execute Vv-only and three-parameter package calls.
- [ ] Run `git status --short` and report only remaining unrelated workspace migration noise.
