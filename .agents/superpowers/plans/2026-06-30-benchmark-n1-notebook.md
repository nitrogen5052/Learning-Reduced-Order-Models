(Recovered 2026-07-13 from Codex session rollouts after iCloud eviction destroyed the original file.)

# Benchmark N1 Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone `notebooks/Benchmark_N1.ipynb` that uses only `lrom_legacy.N1` to reconstruct all 16 figures from the archived Notebook 02 scientific workflow.

**Architecture:** The notebook owns its configuration, N1 model construction, derived diagnostic arrays, and Matplotlib plotting. A focused structural test reads the notebook as JSON and enforces provenance, import isolation, section order, and the approved figure contract without introducing a generator or plotting helper script.

**Tech Stack:** Python, Jupyter `nbformat`, NumPy, Matplotlib, `lrom_legacy.N1`, pytest

---

## File Structure

- Create `notebooks/Benchmark_N1.ipynb`: the only implementation artifact; contains all N1 studies and plotting cells.
- Create `tests/test_benchmark_n1_notebook.py`: structural and compilation contract for the notebook.
- Do not modify `lrom_legacy/N1/`, the archived notebook, or any notebook generator/helper script.

### Task 1: Lock The Notebook Contract

**Files:**
- Create: `tests/test_benchmark_n1_notebook.py`
- Test: `tests/test_benchmark_n1_notebook.py`

- [ ] **Step 1: Write the failing notebook contract test**

```python
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Benchmark_N1.ipynb"
SOURCE_NOTEBOOK = (
    "scientific_archive/legacy_code/Legacy_benchmark/notebooks/"
    "02_lrom_method_walkthrough.ipynb"
)
FIGURES = {
    "vv-rainbow",
    "vv-coefficients",
    "vv-wavefunction-errors",
    "rv-rainbow",
    "rv-coefficients",
    "rv-wavefunction-errors",
    "broad-coefficients",
    "broad-coefficient-errors",
    "broad-wavefunction-errors",
    "predictor-points",
    "predictor-rainbows",
    "predictor-values",
    "multiparameter-coefficients",
    "multiparameter-coefficient-errors",
    "multiparameter-wavefunction-errors",
    "multiparameter-violins",
}


def notebook_text() -> str:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_benchmark_n1_notebook_contract() -> None:
    assert NOTEBOOK.exists()
    text = notebook_text()
    assert SOURCE_NOTEBOOK in text
    assert "import lrom_legacy.N1 as n1" in text
    assert "import lrom\n" not in text
    assert "import lrom_bench" not in text
    assert "import rose" not in text
    assert "lrom_demo" not in text
    for figure in FIGURES:
        assert f"# FIGURE: {figure}" in text


def test_benchmark_n1_code_cells_compile() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"Benchmark_N1 cell {index}", "exec")
```

- [ ] **Step 2: Run the test and verify the missing notebook failure**

Run:

```bash
pytest -q tests/test_benchmark_n1_notebook.py
```

Expected: FAIL because `notebooks/Benchmark_N1.ipynb` does not exist.

- [ ] **Step 3: Commit the failing contract**

```bash
git add tests/test_benchmark_n1_notebook.py
git commit -m "Test Benchmark N1 notebook contract"
```

### Task 2: Create The Notebook Foundation And One-Parameter Studies

**Files:**
- Create: `notebooks/Benchmark_N1.ipynb`
- Test: `tests/test_benchmark_n1_notebook.py`

- [ ] **Step 1: Create the notebook introduction and import cell**

The first markdown cell must contain:

```markdown
# Benchmark N1. Reconstructing the Legacy LROM Method Walkthrough

This notebook uses `lrom_legacy.N1` to scientifically reconstruct the figures
from `scientific_archive/legacy_code/Legacy_benchmark/notebooks/02_lrom_method_walkthrough.ipynb`.
It does not import or execute the archived notebook's helper code.
```

The import cell must contain:

```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "lrom_legacy" / "N1").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lrom_legacy.N1 as n1

plt.rcParams.update(
    {"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25}
)
```

- [ ] **Step 2: Add deterministic N1 configuration**

Use one shared physical configuration:

```python
TARGET = (40, 20)
PROJECTILE = (1, 0)
LAB_ENERGY = 14.1
BASIS_SIZE = 4
EIM_BASIS_SIZE = 8
MESH_SIZE = 800

central_probe = n1.LROM(
    target=TARGET,
    projectile=PROJECTILE,
    lab_energy=LAB_ENERGY,
    l=0,
    potential="ws_3",
)
CENTRAL = dict(central_probe.central_parameters)
V0, R0, A0 = (CENTRAL[name] for name in ("Vv", "Rv", "av"))
```

- [ ] **Step 3: Add the Vv-only study and its three figures**

Construct `vv_emulator = n1.LROM(..., potential="ws_1")`, sample explicit
training and testing grids equivalent to `0.90–1.10 V0` and `0.65–1.35 V0`,
and train with `basis_size=4`, parameter predictors, and one predictor.

The three plotting cells must be labeled:

```python
# FIGURE: vv-rainbow
# FIGURE: vv-coefficients
# FIGURE: vv-wavefunction-errors
```

Use `vv_emulator.samples`, `vv_emulator.training_results`,
`vv_emulator.testing_results`, and `vv_emulator.basis[0]` directly. Plot FOM
wavefunctions and real Woods-Saxon potentials in the rainbow; plot LS, ROSE,
and N1 LROM coefficients in the coordinate figure; plot relative-L2
wavefunction errors on a logarithmic axis.

- [ ] **Step 4: Add the Rv-only study and its three figures**

Construct a separate `rv_emulator` using `0.94–1.06 R0` training and
`0.70–1.30 R0` testing grids. Label the plotting cells:

```python
# FIGURE: rv-rainbow
# FIGURE: rv-coefficients
# FIGURE: rv-wavefunction-errors
```

Use the same panel layout, method styles, basis size, and error definitions as
the Vv study, changing only the physical scan variable.

- [ ] **Step 5: Run the structural and compilation tests**

Run:

```bash
pytest -q tests/test_benchmark_n1_notebook.py
```

Expected: FAIL only for the ten figure markers not yet implemented; the
notebook must parse and all existing code cells must compile.

- [ ] **Step 6: Commit the one-parameter notebook studies**

```bash
git add notebooks/Benchmark_N1.ipynb
git commit -m "Add Benchmark N1 one-parameter studies"
```

### Task 3: Add Broad-Box And Predictor Diagnostics

**Files:**
- Modify: `notebooks/Benchmark_N1.ipynb`
- Test: `tests/test_benchmark_n1_notebook.py`

- [ ] **Step 1: Add the deterministic broad Vv/Rv design**

Use seed `1204`, 70 training cases inside `±0.22 V0` and `±0.20 R0`, and two
43-point one-at-a-time testing scans spanning `±0.75 V0` and `±0.65 R0`.
Represent these rows with explicit `training_grid` and `testing_grid` values
accepted by `n1.LROM.sampling()`, preserving stable case ordering.

- [ ] **Step 2: Train the raw-parameter broad-box model**

Train `broad_emulator` with:

```python
broad_emulator.train(
    basis_size=BASIS_SIZE,
    predictor="parameters",
    predictor_count=2,
)
```

Add figures labeled:

```python
# FIGURE: broad-coefficients
# FIGURE: broad-coefficient-errors
# FIGURE: broad-wavefunction-errors
```

The plots show each one-at-a-time scan separately, compare LS targets against
N1 LROM coordinates, and show LS-floor, ROSE-ROM, and raw-parameter N1 LROM
wavefunction errors.

- [ ] **Step 3: Train the potential-predictor model**

Create `predictor_emulator` with the identical broad training and testing rows,
then call:

```python
predictor_emulator.train(
    basis_size=BASIS_SIZE,
    predictor="potential",
    predictor_count=5,
)
```

Add figures labeled:

```python
# FIGURE: predictor-points
# FIGURE: predictor-rainbows
# FIGURE: predictor-values
```

Use `predictor_emulator.predictors.selected_radii`, `central_values`,
`training_features`, and `testing_features` directly. The predictor-points
figure overlays selected radii on the central potential; the rainbow figure
overlays them on Vv and Rv potential scans; the predictor-values figure shows
all five normalized features along each scan.

- [ ] **Step 4: Run the notebook contract**

Run:

```bash
pytest -q tests/test_benchmark_n1_notebook.py
```

Expected: FAIL only for the four remaining multiparameter figure markers.

- [ ] **Step 5: Commit the broad-box and predictor diagnostics**

```bash
git add notebooks/Benchmark_N1.ipynb
git commit -m "Add Benchmark N1 predictor diagnostics"
```

### Task 4: Add Multiparameter Comparisons And Violin Summary

**Files:**
- Modify: `notebooks/Benchmark_N1.ipynb`
- Test: `tests/test_benchmark_n1_notebook.py`

- [ ] **Step 1: Add coefficient trajectory and error comparisons**

Use the raw-parameter coordinates from `broad_emulator.testing_results` and
potential-predictor coordinates from `predictor_emulator.testing_results`.
Add:

```python
# FIGURE: multiparameter-coefficients
# FIGURE: multiparameter-coefficient-errors
```

For each Vv and Rv scan, plot LS targets in black, raw-parameter N1 LROM with
dashed lines, and potential-predictor N1 LROM with dotted lines. Plot absolute
coordinate errors on logarithmic axes in the paired error figure.

- [ ] **Step 2: Add the multiparameter wavefunction-error figure**

Add:

```python
# FIGURE: multiparameter-wavefunction-errors
```

For each scan, plot the N1 LS floor, N1 ROSE ROM, raw-parameter N1 LROM, and
potential-predictor N1 LROM relative-L2 errors using the corresponding
`metrics["relative_l2"][0]` arrays.

- [ ] **Step 3: Add the split train/test violin figure**

Add:

```python
# FIGURE: multiparameter-violins
```

Plot split violins for LS floor, ROSE ROM, raw-parameter LROM, and
potential-predictor LROM. Use blue left halves for training, orange right
halves for testing, diamond median markers, a logarithmic y-axis, and a
`1e-5` display floor. Remove central rows from the distributions before
plotting.

- [ ] **Step 4: Run the complete structural contract**

Run:

```bash
pytest -q tests/test_benchmark_n1_notebook.py
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the completed figure set**

```bash
git add notebooks/Benchmark_N1.ipynb
git commit -m "Complete Benchmark N1 figure reconstruction"
```

### Task 5: Execute And Verify The Notebook

**Files:**
- Modify: `notebooks/Benchmark_N1.ipynb` only if execution records outputs
- Test: `tests/test_benchmark_n1_notebook.py`

- [ ] **Step 1: Compile every code cell independently**

Run:

```bash
python -c 'import nbformat; p="notebooks/Benchmark_N1.ipynb"; n=nbformat.read(p, 4); cells=[c for c in n.cells if c.cell_type=="code"]; [compile(c.source, f"{p} cell {i}", "exec") for i,c in enumerate(cells)]; print("compiled", len(cells), "code cells")'
```

Expected: exits zero and prints the code-cell count.

- [ ] **Step 2: Execute the notebook**

Run:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Benchmark_N1.ipynb --ExecutePreprocessor.timeout=1200
```

Expected: exits zero with all 16 figures rendered. If the local ROSE runtime is
missing or execution exceeds the timeout, preserve the unexecuted notebook and
report the exact failure without weakening the structural checks.

- [ ] **Step 3: Re-run focused and neighboring notebook tests**

Run:

```bash
pytest -q \\n+  tests/test_benchmark_n1_notebook.py \\n+  tests/test_notebook01_generation.py \\n+  tests/test_notebook01_lrom_flow.py
```

Expected: all selected tests pass.

- [ ] **Step 4: Verify repository scope**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only `Benchmark_N1.ipynb` execution output, if any, and the intended
test file differ from the task's committed checkpoints; unrelated pre-existing
working-tree changes remain untouched.

- [ ] **Step 5: Commit executed outputs when execution succeeds**

```bash
git add notebooks/Benchmark_N1.ipynb tests/test_benchmark_n1_notebook.py
git commit -m "Verify Benchmark N1 notebook"
```
