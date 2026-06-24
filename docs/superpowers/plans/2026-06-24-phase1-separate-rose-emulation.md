# Phase 1 Separate ROSE Emulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict package-owned ROSE usage to full-order sampling and run the complete ROSE reduced-emulator comparison independently inside Notebook 1.

**Architecture:** Replace ROSE `CustomBasis`/`ReducedBasisEmulator` usage in `lrom.train()` with the package's numeric centered-SVD basis. Remove ROSE comparison state from the package. Notebook 1 directly imports ROSE, rebuilds the same FOM cases, constructs its own zero-potential-reference basis/emulator, and compares its wavefunctions to LROM on identical parameter rows.

**Tech Stack:** Python 3.11-3.13, NumPy, SciPy, numba, nuclear-rose, Matplotlib, pytest, nbformat, Jupyter

---

### Task 1: Remove ROSE Reduced Emulation from `lrom.train()`

**Files:**
- Modify: `lrom/state.py`
- Modify: `lrom/emulator.py`
- Modify: `lrom/training.py`
- Modify: `lrom/artifacts.py`
- Modify: `tests/test_lrom_lifecycle.py`
- Modify: `tests/test_lrom_training.py`
- Modify: `tests/test_lrom_artifacts.py`

- [ ] **Step 1: Write failing package-boundary tests**

Update training expectations to contain only `ls` and `lrom`, assert the public
object has no `rose_rbm`, and monkeypatch ROSE reduced-emulator construction to
raise if `train()` attempts to use it:

```python
assert not hasattr(emulator, "rose_rbm")
assert set(emulator.testing_results.coefficients) == {"ls", "lrom"}
assert set(emulator.testing_results.metrics["relative_l2"][0]) == {"ls", "lrom"}
assert set(emulator.training_results.metrics["pointwise_absolute"][0]) == {
    "ls",
    "lrom",
}
```

Update lifecycle fake `TrainingState` constructors and artifact expectations so
no package result contains ROSE state.

- [ ] **Step 2: Run focused tests and confirm existing ROSE state fails them**

Run:

```bash
pytest tests/test_lrom_training.py tests/test_lrom_lifecycle.py tests/test_lrom_artifacts.py -q
```

Expected: failures because `rose_rbm`, ROSE result fields, and package-owned ROSE
reduced emulation still exist.

- [ ] **Step 3: Simplify scientific state**

Remove `RoseRBMState`, `TrainingState.rose_rbm`, `TestingResults.rose`, and
`TestingCase.rose`. Remove `LROM.rose_rbm` and the ROSE field assembled by
`testing_case()`.

Update artifact reconstruction and fake trainers to construct the simplified
`TrainingState`.

- [ ] **Step 4: Build LROM's basis numerically**

Replace `_shared_rose_basis` with the existing generic builder:

```python
from .basis import build_basis, project_coordinates, reconstruct

def _lrom_bases(*, emulator, basis_size: int) -> dict[int, BasisState]:
    samples = emulator.samples
    return {
        channel: build_basis(
            phi0=samples.central_wavefunctions[channel],
            snapshots=samples.training_wavefunctions[channel],
            radius=samples.mesh.radius,
            basis_size=basis_size,
        )
        for channel in emulator.partial_waves
    }
```

Remove `_import_rose` from `lrom/training.py`. `_evaluate` calculates only LS
and LROM coordinates/wavefunctions/metrics. Keep ROSE-backed FOM logic unchanged
inside `lrom/fom.py`.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
pytest tests/test_lrom_training.py tests/test_lrom_lifecycle.py tests/test_lrom_artifacts.py -q
```

Expected: all tests pass.

Commit:

```bash
git add lrom/state.py lrom/emulator.py lrom/training.py lrom/artifacts.py tests/test_lrom_lifecycle.py tests/test_lrom_training.py tests/test_lrom_artifacts.py
git commit -m "Separate ROSE emulation from LROM training"
```

### Task 2: Direct Independent ROSE Workflow in Notebook 1

**Files:**
- Modify: `scripts/generate_notebook01.py`
- Modify: `tests/test_notebook01_generation.py`
- Modify: `tests/test_notebook01_lrom_flow.py`

- [ ] **Step 1: Write failing notebook-boundary tests**

Require the setup to import ROSE directly after the SciPy compatibility alias,
and require explicit ROSE construction/evaluation calls:

```python
assert "import scipy.special" in source
assert "scipy.special.sph_harm_y" in source
assert "import rose" in source
assert "rose.InteractionEIMSpace(" in source
assert "rose.SchroedingerEquation.make_base_solver(" in source
assert "rose.basis.CustomBasis(" in source
assert "rose.reduced_basis_emulator.ReducedBasisEmulator(" in source
assert "rose_model.coefficients(row)" in source
assert "rose_model.emulate_wave_function(row)" in source
assert "maximum independent FOM difference" in source
```

Also assert the generator never reads `emulator.rose_rbm` or
`results.rose` from the package.

- [ ] **Step 2: Run notebook tests and verify the direct workflow is absent**

Run:

```bash
pytest tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py -q
```

Expected: failures because the notebook currently consumes package-owned ROSE
results.

- [ ] **Step 3: Add direct ROSE startup and local workflow helpers**

The setup cell installs the spherical-harmonic compatibility alias before:

```python
import rose
```

Add a notebook-local Woods-Saxon interaction and a visible
`run_independent_rose(*, emulator, basis_size, eim_basis_size)` function. The
function:

1. reads the exact training/testing parameter rows and physical/internal meshes
   from the LROM sample state;
2. calls `rose.kinematics`, `rose.InteractionEIMSpace`, and
   `rose.SchroedingerEquation.make_base_solver`;
3. independently solves all training/testing rows;
4. sets only `Vv=0` in a central parameter copy and solves the zero-potential
   reference;
5. constructs `rose.basis.CustomBasis` and
   `rose.reduced_basis_emulator.ReducedBasisEmulator`;
6. evaluates native ROSE and ROSE-LS coefficients, reconstructed wavefunctions,
   pointwise errors, and relative-L2 metrics;
7. returns a `SimpleNamespace` containing only notebook-local ROSE results.

Use the same solver tolerances, rho domain, EIM size, and parameter bounds as the
LROM sampling call. Print the maximum absolute difference between independent
full solutions and `emulator.samples.training/testing_wavefunctions[0]`.

- [ ] **Step 4: Run the independent workflow for both studies**

After each LROM object samples and trains, run:

```python
vv_rose = run_independent_rose(
    emulator=vv_emulator,
    basis_size=4,
    eim_basis_size=8,
)
ws3_rose = run_independent_rose(
    emulator=ws3_emulator,
    basis_size=4,
    eim_basis_size=8,
)
```

No ROSE result is written back into either LROM object.

### Task 3: Compare Distinct Coordinates and Common Wavefunctions

**Files:**
- Modify: `scripts/generate_notebook01.py`
- Modify: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`
- Modify: `tests/test_notebook01_lrom_flow.py`

- [ ] **Step 1: Write failing figure-semantics tests**

Require separate LROM and ROSE coefficient coordinate labels, native LS errors,
and notebook-local ROSE wavefunction/violin inputs:

```python
assert "LROM central-reference coordinates" in source
assert "ROSE zero-potential-reference coordinates" in source
assert "|LROM LS - LROM|" in source
assert "|ROSE LS - ROSE|" in source
assert "vv_rose.testing_wavefunctions" in source
assert "ws3_rose.testing_wavefunctions" in source
assert "ws3_rose.training_relative_l2" in source
assert "ws3_rose.testing_relative_l2" in source
```

- [ ] **Step 2: Run tests and confirm the old shared-coordinate figures fail**

Run:

```bash
pytest tests/test_notebook01_lrom_flow.py -q
```

Expected: failures because coefficient origins are not separated and plots still
use package-owned ROSE arrays.

- [ ] **Step 3: Rebuild coefficient figures by native coordinate system**

For each of the first two Vv coefficients, show:

- LROM LS versus LROM in the central-reference basis, with
  `abs(LROM LS - LROM)`;
- ROSE-native LS versus ROSE in the zero-potential-reference basis, with
  `abs(ROSE LS - ROSE)`.

Retain extrapolation shading. Apply the same two-coordinate-system structure to
the ws3 first-coordinate comparison. Never subtract a ROSE coordinate from a
LROM coordinate.

- [ ] **Step 4: Rebuild wavefunction and violin comparisons**

Representative wavefunction plots take ROSE arrays from `vv_rose`/`ws3_rose`
and LROM/LS arrays from the emulator. Pointwise differences use the common full
testing solution. The split violin uses LROM-object metrics for LS/LROM and
notebook-local independent ROSE metrics for ROSE.

- [ ] **Step 5: Regenerate, test, and execute Notebook 1**

Run:

```bash
python scripts/generate_notebook01.py
cp notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb /tmp/notebook01-rose-first.ipynb
python scripts/generate_notebook01.py
cmp /tmp/notebook01-rose-first.ipynb notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
pytest tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py -q
pytest -q
jupyter nbconvert --to notebook --execute notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb --output /tmp/notebook01-rose-executed.ipynb --ExecutePreprocessor.timeout=3600
```

Expected: deterministic generation, all tests pass, direct ROSE full-solution
parity is printed, and the notebook executes with zero cell errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_notebook01.py notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
git commit -m "Run ROSE comparison independently in Notebook 1"
```

### Task 4: Final Verification and Local Integration

**Files:**
- Verify only

- [ ] **Step 1: Run final checks**

```bash
pytest -q
git diff --check
git status --short --branch
```

Expected: all feature-branch tests pass with a clean worktree.

- [ ] **Step 2: Merge locally**

Fast-forward the feature branch into local `main`, rerun the complete suite on
`main`, then remove the project-local worktree and merged feature branch. Preserve
the known unrelated Notebook 2 files in the main checkout.
