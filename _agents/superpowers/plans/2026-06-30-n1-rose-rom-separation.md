(Recovered 2026-07-13 from Codex session rollouts after iCloud eviction destroyed the original file.)

# N1 And ROSE ROM Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all ROSE-ROM evaluation and state from `lrom_legacy.N1`, then make `Benchmark_N1.ipynb` run an independent public ROSE emulator with every Matplotlib figure written inline.

**Architecture:** N1 continues using ROSE for full-order sampling and centered-basis construction, but its training state contains only basis, predictors, RF-LROM, LS/FOM/LROM results, and diagnostics. The notebook constructs comparison ROSE reduced emulators from N1's physical solver state and shared sample rows, while keeping all plotting code visible in each figure cell.

**Tech Stack:** Python, nuclear-rose, NumPy, Matplotlib, Jupyter/nbformat, pytest

---

### Task 1: Lock The N1 Boundary With Failing Tests

**Files:**
- Create: `tests/test_n1_rose_boundary.py`
- Modify: `tests/test_benchmark_n1_notebook.py`

- [ ] Assert that an N1 emulator has no `rose_rbm` property, `TrainingState` has no `rose_rbm` field, and `TestingResults`/`TestingCase` have no `rose` field.
- [ ] Assert that N1 result coefficients and metrics contain exactly `ls` and `lrom`.
- [ ] Assert that the notebook contains `import rose` and `import lrom_legacy.N1 as n1`, contains no plotting-function definitions, and retains all 16 figure markers.
- [ ] Run the focused tests and verify they fail against the current implementation.
- [ ] Commit the failing tests.

### Task 2: Remove ROSE ROM From N1

**Files:**
- Modify: `lrom_legacy/N1/state.py`
- Modify: `lrom_legacy/N1/training.py`
- Modify: `lrom_legacy/N1/emulator.py`
- Modify: `lrom_legacy/N1/artifacts.py`

- [ ] Replace `_shared_rose_basis()` with a basis-only builder that constructs `rose.basis.CustomBasis` only long enough to copy `phi0`, vectors, radius, and singular values into `BasisState`.
- [ ] Remove `RoseRBMState`, `TrainingState.rose_rbm`, `LROM.rose_rbm`, `TestingResults.rose`, and `TestingCase.rose`.
- [ ] Simplify `_evaluate()` to calculate only FOM references, LS projections/reconstructions, LROM coefficients/reconstructions, and `ls`/`lrom` diagnostics.
- [ ] Remove serialized `rose_rbm` placeholders.
- [ ] Run the focused N1 boundary tests and existing N1/package tests.
- [ ] Commit the package-boundary change.

### Task 3: Rewrite Benchmark N1 Computation And Plots

**Files:**
- Modify: `notebooks/Benchmark_N1.ipynb`
- Modify: `tests/test_benchmark_n1_notebook.py`

- [ ] Import public `rose` separately from `lrom_legacy.N1`.
- [ ] For each N1 study, construct a notebook-owned `rose.basis.CustomBasis` and `rose.reduced_basis_emulator.ReducedBasisEmulator` from the matching N1 FOM solver, training wavefunctions, central wavefunction, mesh, and sample rows.
- [ ] Store notebook-local ROSE coefficients, wavefunctions, and relative errors without attaching them to N1.
- [ ] Delete every plotting helper definition, including `plot_rainbow`, coefficient/error scan functions, and `split_violin`.
- [ ] Expand the complete Matplotlib construction into each of the 16 marked figure cells.
- [ ] Update labels so `N1 LROM` and separately owned `ROSE ROM` are unambiguous.
- [ ] Run the notebook structural and compilation tests.
- [ ] Commit the notebook rewrite.

### Task 4: Execute And Verify

**Files:**
- Modify: `notebooks/Benchmark_N1.ipynb` with executed outputs

- [ ] Execute the notebook with a 1200-second timeout.
- [ ] Verify 16 PNG outputs, no cell errors, and all code cells executed.
- [ ] Run focused tests and the full clean-branch test suite.
- [ ] Visually inspect representative coefficient, predictor, and violin figures.
- [ ] Commit verified executed outputs.
