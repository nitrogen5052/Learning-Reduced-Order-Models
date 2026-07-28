# Benchmark Helper Pipelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated notebook-local ROSE and least-squares machinery with a small `benchmark_helper.py` API while preserving the validated science and keeping ROSE outside the LROM package.

**Architecture:** Rename the existing helper and replace its `SimpleNamespace` function with two context-holding benchmark objects. Each object exposes independent `run_rose()` and `run_ls()` methods plus a `run()` method that delegates to those same paths; typed flat dataclasses carry raw NumPy arrays and existing result dictionaries back to the notebooks.

**Tech Stack:** Python 3.12, NumPy, Numba, ROSE, parked/public LROM modules, dataclasses, Jupyter notebooks, pytest, Ruff.

## Global Constraints

- Modify only `notebooks/benchmark_helper.py`, Notebook 01, Notebook 02, Benchmark 03, focused tests, architecture documentation, and memory handoff files.
- Do not modify public `lrom`, `lrom_legacy.v1_2`, `lrom_legacy.v2_0`, Benchmark 01, Benchmark 02, or the scientific archive.
- Preserve Notebook 01/02 shared exact snapshots and ROSE's native free-reference `CustomBasis`; do not replace this with `ScatteringAmplitudeEmulator.from_train()`.
- Preserve exact ordered training/testing rows, \(l=0,\ldots,3\), ±20% ranges, physical radius in fm, basis/compression grids, angle grids, timing definitions, alpha selections, figures, and numerical results outside timing noise.
- ROSE-only execution may not construct LS state; LS-only execution may not construct ROSE state.
- Combined methods must call the public independent methods rather than duplicate their implementations.
- Keep helper results flat: dataclasses, NumPy arrays, and plain dictionaries only.
- Do not add dependencies, factories, abstract base classes, or configuration hierarchies.
- Leave the existing untracked `tmp/` directory untouched.

---

### Task 1: Build the minimal benchmark-helper API

**Files:**

- Create: `tests/test_benchmark_helper.py`
- Rename: `notebooks/rose_helper.py` to `notebooks/benchmark_helper.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True)
class RoseWavefunctionResult:
    emulator: Any
    basis: Any
    interaction: Any
    training_rows: np.ndarray
    testing_rows: np.ndarray
    training_coefficients: np.ndarray
    testing_coefficients: np.ndarray
    training_wavefunctions: np.ndarray | None
    testing_wavefunctions: np.ndarray


@dataclass(frozen=True)
class LSWavefunctionResult:
    training_coefficients: np.ndarray
    testing_coefficients: np.ndarray
    training_wavefunctions: np.ndarray
    testing_wavefunctions: np.ndarray


@dataclass(frozen=True)
class WavefunctionBenchmarkResult:
    rose: RoseWavefunctionResult
    ls: LSWavefunctionResult


@dataclass(frozen=True)
class RoseCrossSectionResult:
    emulators: dict[tuple[int, int], Any]
    fom_training_cross_sections: np.ndarray
    fom_testing_cross_sections: np.ndarray
    results: dict[tuple[int, int], dict[str, np.ndarray]]


@dataclass(frozen=True)
class LSCrossSectionResult:
    results: dict[int, dict[str, np.ndarray]]


@dataclass(frozen=True)
class CrossSectionBenchmarkResult:
    rose: RoseCrossSectionResult
    ls: LSCrossSectionResult
```

- Produces:

```python
class WavefunctionBenchmark:
    def __init__(
        self,
        emulator: Any,
        center: Mapping[str, float],
    ) -> None: ...

    def run_rose(
        self,
        basis_size: int,
        eim_basis_size: int = 8,
        include_training_wavefunctions: bool = True,
    ) -> RoseWavefunctionResult: ...

    def run_ls(self) -> LSWavefunctionResult: ...

    def run(
        self,
        basis_size: int,
        eim_basis_size: int = 8,
        include_training_wavefunctions: bool = True,
    ) -> WavefunctionBenchmarkResult: ...
```

- Produces:

```python
class CrossSectionBenchmark:
    def __init__(
        self,
        emulator: Any,
        training_rows: np.ndarray,
        testing_rows: np.ndarray,
        angles_degrees: np.ndarray,
        denominator_floor: float,
        timing_repeats: int,
        timing_inner_loops: int,
    ) -> None: ...

    def run_rose(
        self,
        basis_sizes: tuple[int, ...],
        eim_sizes: tuple[int, ...],
    ) -> RoseCrossSectionResult: ...

    def run_ls(
        self,
        basis_sizes: tuple[int, ...],
        fom_training_cross_sections: np.ndarray,
        fom_testing_cross_sections: np.ndarray,
        predictor_count: int,
    ) -> LSCrossSectionResult: ...

    def run(
        self,
        basis_sizes: tuple[int, ...],
        eim_sizes: tuple[int, ...],
        ls_predictor_count: int,
    ) -> CrossSectionBenchmarkResult: ...
```

- Produces reusable error utilities:

```python
def pointwise_relative_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float,
) -> np.ndarray: ...


def summarize_relative_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float,
) -> dict[str, np.ndarray]: ...
```

- Consumes public v1.2 `least_squares_baseline()` for wavefunctions and parked-v2 `project_coordinates()` / `_cross_section_prediction()` for the explicit cross-section LS oracle.

- [ ] **Step 1: Record the protected baseline**

Run:

```bash
git status --short
shasum -a 256 notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb | shasum -a 256
find ../scientific_archive -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256
python -m pytest -q
```

Expected:

- only the existing `?? tmp/` is untracked;
- Notebook 01 combined digest is `f8197e6c0a111d5192d6039fe0e629e3958f2831a832554e545be43b809d5c44`;
- scientific archive digest is `afe80b7cf267fad0c3265bd18a7e023f966eb5fd4a655123887ee559c0061213`;
- the current suite passes.

- [ ] **Step 2: Write failing helper-API and delegation tests**

Create `tests/test_benchmark_helper.py` with:

```python
import ast
from dataclasses import fields
from pathlib import Path
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "notebooks" / "benchmark_helper.py"


def helper_functions_and_classes() -> tuple[set[str], set[str]]:
    tree = ast.parse(HELPER.read_text())
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    classes = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    return functions, classes


def test_benchmark_helper_public_surface() -> None:
    functions, classes = helper_functions_and_classes()
    assert {
        "pointwise_relative_error",
        "summarize_relative_error",
    } <= functions
    assert {
        "RoseWavefunctionResult",
        "LSWavefunctionResult",
        "WavefunctionBenchmarkResult",
        "RoseCrossSectionResult",
        "LSCrossSectionResult",
        "CrossSectionBenchmarkResult",
        "WavefunctionBenchmark",
        "CrossSectionBenchmark",
    } <= classes


def test_result_types_are_flat() -> None:
    import benchmark_helper

    assert [field.name for field in fields(
        benchmark_helper.CrossSectionBenchmarkResult
    )] == ["rose", "ls"]
    assert [field.name for field in fields(
        benchmark_helper.WavefunctionBenchmarkResult
    )] == ["rose", "ls"]


def test_wavefunction_combined_run_delegates(monkeypatch) -> None:
    import benchmark_helper

    benchmark = benchmark_helper.WavefunctionBenchmark(
        emulator=Mock(),
        center={"Vv": 1.0},
    )
    rose_result = Mock()
    ls_result = Mock()
    run_rose = Mock(return_value=rose_result)
    run_ls = Mock(return_value=ls_result)
    monkeypatch.setattr(benchmark, "run_rose", run_rose)
    monkeypatch.setattr(benchmark, "run_ls", run_ls)

    result = benchmark.run(
        basis_size=4,
        eim_basis_size=8,
        include_training_wavefunctions=False,
    )

    run_rose.assert_called_once_with(
        basis_size=4,
        eim_basis_size=8,
        include_training_wavefunctions=False,
    )
    run_ls.assert_called_once_with()
    assert result.rose is rose_result
    assert result.ls is ls_result


def test_cross_section_combined_run_delegates(monkeypatch) -> None:
    import benchmark_helper

    benchmark = benchmark_helper.CrossSectionBenchmark(
        emulator=Mock(),
        training_rows=np.empty((2, 1)),
        testing_rows=np.empty((1, 1)),
        angles_degrees=np.array([30.0]),
        denominator_floor=1e-8,
        timing_repeats=3,
        timing_inner_loops=20,
    )
    rose_result = Mock(
        fom_training_cross_sections=np.ones((2, 1)),
        fom_testing_cross_sections=np.ones((1, 1)),
    )
    ls_result = Mock()
    run_rose = Mock(return_value=rose_result)
    run_ls = Mock(return_value=ls_result)
    monkeypatch.setattr(benchmark, "run_rose", run_rose)
    monkeypatch.setattr(benchmark, "run_ls", run_ls)

    result = benchmark.run(
        basis_sizes=(4, 6),
        eim_sizes=(4, 8),
        ls_predictor_count=4,
    )

    run_rose.assert_called_once_with(
        basis_sizes=(4, 6),
        eim_sizes=(4, 8),
    )
    run_ls.assert_called_once_with(
        basis_sizes=(4, 6),
        fom_training_cross_sections=rose_result.fom_training_cross_sections,
        fom_testing_cross_sections=rose_result.fom_testing_cross_sections,
        predictor_count=4,
    )
    assert result.rose is rose_result
    assert result.ls is ls_result
```

Add this import setup before importing the helper:

```python
import sys

NOTEBOOKS = ROOT / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
```

- [ ] **Step 3: Run the API tests and verify failure**

Run:

```bash
python -m pytest -q tests/test_benchmark_helper.py
```

Expected: failure because `benchmark_helper.py` and its types do not exist.

- [ ] **Step 4: Rename the helper and implement the flat types**

Use `apply_patch` to move `notebooks/rose_helper.py` to
`notebooks/benchmark_helper.py`. Replace `SimpleNamespace` with the six frozen
dataclasses from the Interfaces section. Group imports at the top:

```python
from collections.abc import Mapping
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from numba import njit
import rose

import lrom as lrom_v1
import lrom_legacy.v2_0 as lrom_v2
```

Keep `rose_real_woods_saxon()` and add the current Notebook 02 full
Woods-Saxon central/spin-orbit callbacks as Numba functions. Do not retain
`SimpleNamespace`, lambda-based coefficient evaluators, or cell-style comments.

- [ ] **Step 5: Implement the wavefunction object in ROSE Guide order**

Implement `WavefunctionBenchmark` with these private methods in source order:

```python
def _interaction_space(
    self,
    eim_basis_size: int,
) -> Any: ...

def _free_reference(self) -> np.ndarray: ...

def _custom_basis(
    self,
    basis_size: int,
) -> Any: ...

def _rose_emulator(
    self,
    interaction: Any,
    basis: Any,
) -> Any: ...
```

`_interaction_space()` retains the current deterministic bounds derived from
the center, training rows, and testing rows. `_custom_basis()` uses exact
shared LROM training wavefunctions and `phi_free`; it must not call
`ScatteringAmplitudeEmulator.from_train()`.

Implement `run_rose()`, `run_ls()`, and delegating `run()` with the exact public
signatures in the Interfaces section.

- [ ] **Step 6: Add independent-path tests**

Append:

```python
def test_wavefunction_ls_does_not_construct_rose(monkeypatch) -> None:
    import benchmark_helper

    emulator = Mock()
    emulator.basis = {0: np.eye(2)}
    emulator.samples.training_wavefunctions = {0: np.ones((2, 2))}
    emulator.samples.testing_wavefunctions = {0: np.ones((1, 2))}
    baseline = Mock(
        side_effect=[
            (np.ones((2, 2)), np.ones((2, 2))),
            (np.ones((1, 2)), np.ones((1, 2))),
        ]
    )
    monkeypatch.setattr(
        benchmark_helper.lrom_v1,
        "least_squares_baseline",
        baseline,
    )
    monkeypatch.setattr(
        benchmark_helper.rose,
        "InteractionEIMSpace",
        Mock(side_effect=AssertionError("ROSE constructed during LS")),
    )

    result = benchmark_helper.WavefunctionBenchmark(
        emulator,
        {"Vv": 1.0},
    ).run_ls()

    assert result.testing_wavefunctions.shape == (1, 2)
    assert baseline.call_count == 2
```

Add a ROSE-only delegation test that monkeypatches the object's private ROSE
stages to return small fakes and patches `lrom_v1.least_squares_baseline` to
raise if called. Assert `run_rose()` returns without calling LS.

- [ ] **Step 7: Implement the cross-section object**

Implement these private stages in ROSE Guide order:

```python
def _interaction_space(self, eim_size: int) -> Any: ...
def _base_solver(self) -> Any: ...
def _free_reference(self, ell: int) -> np.ndarray: ...
def _custom_bases(self, interaction: Any, basis_size: int) -> list[list[Any]]: ...
def _scattering_emulator(
    self,
    interaction: Any,
    bases: list[list[Any]],
) -> Any: ...
def _exact_smatrix(self, emulator: Any, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
def _emulated_smatrix(self, emulator: Any, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
def _cross_section(
    self,
    emulator: Any,
    row: np.ndarray,
    splus: np.ndarray,
    sminus: np.ndarray,
) -> np.ndarray: ...
```

Move the current Notebook 02 logic without changing:

- explicit ordered EIM training rows;
- `CustomBasis` options;
- all-channel \(l=0,\ldots,l_{\max}\) assembly;
- `s_0=6*pi`;
- angle cache validation;
- warmup/minimum timing procedure;
- error formulas and result keys.

Implement `run_rose()` around those stages.

Implement `run_ls()` by training each requested basis with:

```python
emulator.train(
    basis_size=basis_size,
    predictor="effective-interaction",
    predictor_count=predictor_count,
    observable="cross_section",
    angles_degrees=self.angles_degrees,
)
```

Then project exact training/testing wavefunctions and call
`lrom_v2._cross_section_prediction()` exactly as the current notebook does.
Implement delegating `run()` last.

- [ ] **Step 8: Add cross-section independence and metric tests**

Append:

```python
def test_cross_section_ls_does_not_construct_rose(monkeypatch) -> None:
    import benchmark_helper

    benchmark = benchmark_helper.CrossSectionBenchmark(
        emulator=Mock(),
        training_rows=np.empty((2, 1)),
        testing_rows=np.empty((1, 1)),
        angles_degrees=np.array([30.0]),
        denominator_floor=1e-8,
        timing_repeats=3,
        timing_inner_loops=20,
    )
    monkeypatch.setattr(
        benchmark_helper.rose,
        "InteractionEIMSpace",
        Mock(side_effect=AssertionError("ROSE constructed during LS")),
    )
    monkeypatch.setattr(
        benchmark,
        "_run_ls_for_basis",
        Mock(return_value={
            "train_xs": np.ones((2, 1)),
            "test_xs": np.ones((1, 1)),
            "train_median_over_angle_error": np.zeros(2),
            "test_median_over_angle_error": np.zeros(1),
            "train_maximum_over_angle_error": np.zeros(2),
            "test_maximum_over_angle_error": np.zeros(1),
        }),
    )

    result = benchmark.run_ls(
        basis_sizes=(4,),
        fom_training_cross_sections=np.ones((2, 1)),
        fom_testing_cross_sections=np.ones((1, 1)),
        predictor_count=4,
    )

    assert set(result.results) == {4}
```

Append direct error-utility checks:

```python
def test_relative_error_summary_has_one_median_metric() -> None:
    import benchmark_helper

    reference = np.array([[2.0, 4.0]])
    predicted = np.array([[1.0, 6.0]])
    summary = benchmark_helper.summarize_relative_error(
        predicted,
        reference,
        1e-8,
    )

    assert set(summary) == {
        "median_over_angle_error",
        "maximum_over_angle_error",
    }
    np.testing.assert_allclose(
        summary["median_over_angle_error"],
        [0.5],
    )
```

- [ ] **Step 9: Run helper tests and lint**

Run:

```bash
python -m pytest -q tests/test_benchmark_helper.py
python -m ruff check notebooks/benchmark_helper.py tests/test_benchmark_helper.py
```

Expected: all helper tests pass and Ruff reports no findings.

- [ ] **Step 10: Commit Task 1**

Run:

```bash
git add notebooks/rose_helper.py notebooks/benchmark_helper.py tests/test_benchmark_helper.py
git commit -m "refactor: add benchmark helper pipelines"
```

---

### Task 2: Migrate Notebook 01 wavefunction comparisons

**Files:**

- Modify: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`
- Modify: `tests/test_benchmark_helper.py`

**Interfaces:**

- Consumes `WavefunctionBenchmark.run()` and its typed ROSE/LS results.
- Preserves all existing Notebook 01 variable arrays needed by figures and
  case selection.

- [ ] **Step 1: Add a failing Notebook 01 structure contract**

Append:

```python
import json


NOTEBOOK_01 = ROOT / "notebooks" / "01_rbm_vs_lrom_single_wavefunction.ipynb"


def notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def test_notebook01_uses_benchmark_helper() -> None:
    text = notebook_text(NOTEBOOK_01)
    assert "import benchmark_helper" in text
    assert "WavefunctionBenchmark(" in text
    assert ".run(" in text
    assert "import rose_helper" not in text
    assert "least_squares_baseline(" not in text
    assert "InteractionEIMSpace(" not in text
    assert "CustomBasis(" not in text
```

- [ ] **Step 2: Run the contract and verify failure**

Run:

```bash
python -m pytest -q tests/test_benchmark_helper.py::test_notebook01_uses_benchmark_helper
```

Expected: failure because Notebook 01 still imports `rose_helper` and calls LS
directly.

- [ ] **Step 3: Replace the Vv pipeline with one combined call**

In the first import cell, replace `import rose_helper` with:

```python
import benchmark_helper
```

Replace the Vv direct LS calls and `rose_helper.build_rose()` call with:

```python
vv_benchmark = benchmark_helper.WavefunctionBenchmark(
    emulator=vv_emulator,
    center=vv_center,
)
vv_comparison = vv_benchmark.run(
    basis_size=BASIS_SIZE,
    eim_basis_size=8,
    include_training_wavefunctions=False,
)

vv_rose = vv_comparison.rose
vv_ls_train_coefficients = vv_comparison.ls.training_coefficients
vv_ls_coefficients = vv_comparison.ls.testing_coefficients
vv_ls_train_wavefunctions = vv_comparison.ls.training_wavefunctions
vv_ls_wavefunctions = vv_comparison.ls.testing_wavefunctions
```

Update later references from the current helper's `wf_test`, `coefficients`,
and related fields to the typed names. Do not change plotting expressions or
selected-case logic.

- [ ] **Step 4: Replace the three-parameter pipeline**

Use the same pattern:

```python
ws3_benchmark = benchmark_helper.WavefunctionBenchmark(
    emulator=ws3_emulator,
    center=ws3_center,
)
ws3_comparison = ws3_benchmark.run(
    basis_size=BASIS_SIZE,
    eim_basis_size=8,
)
```

Map the typed result arrays to the existing `ws3_*` names used downstream.
Remove every direct `least_squares_baseline()` and old `build_rose()` call.

- [ ] **Step 5: Compile Notebook 01 cells and run the contract**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb")
notebook = json.loads(path.read_text())
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        compile(
            "".join(cell.get("source", [])),
            f"{path.name} cell {index}",
            "exec",
        )
print("Notebook 01 cells compile")
PY
python -m pytest -q tests/test_benchmark_helper.py
```

Expected: all cells compile and helper contracts pass.

- [ ] **Step 6: Execute Notebook 01**

Run with local-kernel permission:

```bash
MPLCONFIGDIR=/tmp/lrom-mpl-cache jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
```

Expected: nbconvert exits 0.

- [ ] **Step 7: Verify Notebook 01 outputs**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb")
notebook = json.loads(path.read_text())
code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
errors = [
    output
    for cell in code
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
images = [
    output
    for cell in code
    for output in cell.get("outputs", [])
    if "image/png" in output.get("data", {})
]
assert not errors
assert len(images) == 10
text = "\n".join(
    "".join(output.get("data", {}).get("text/plain", []))
    for cell in code
    for output in cell.get("outputs", [])
)
for case_id in (
    "test-0011",
    "test-0033",
    "test-0037",
    "test-0071",
    "test-0007",
    "test-0044",
):
    assert case_id in text
print(f"Notebook 01: {len(code)} cells, 0 errors, 10 figures")
PY
```

Visually inspect the ten stored figures. Confirm physical-radius axes and
unchanged LS/LROM/ROSE method ordering.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb tests/test_benchmark_helper.py
git commit -m "refactor(notebook01): use benchmark pipelines"
```

---

### Task 3: Migrate Notebook 02 cross-section comparisons

**Files:**

- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
- Modify: `tests/test_notebook02_cross_sections.py`

**Interfaces:**

- Consumes `CrossSectionBenchmark.run_rose()` and `run_ls()` independently.
- Consumes `benchmark_helper.summarize_relative_error()` for the remaining
  notebook-owned LROM evaluation.
- Preserves Notebook 02's LROM grid, alpha selection, tables, and five figures.

- [ ] **Step 1: Replace the old experiment-function test with a failing helper boundary**

Replace `test_notebook02_experiment_functions_are_extracted()` with:

```python
def test_notebook02_uses_benchmark_helper_boundary() -> None:
    text = notebook_text(NOTEBOOK_02)
    assert "import benchmark_helper" in text
    assert "CrossSectionBenchmark(" in text
    assert ".run_rose(" in text
    assert ".run_ls(" in text
    for implementation_detail in (
        "rose.InteractionEIMSpace(",
        "rose.basis.CustomBasis(",
        "def build_rose_bases",
        "def build_rose_emulator",
        "def build_rose_emulators",
        "def exact_smatrix_all_channels",
        "def emulated_smatrix_all_channels",
        "def cross_section_from_smatrix",
        "def evaluate_fom_cross_sections",
        "def evaluate_ls_oracle",
        "def evaluate_rose_grid",
        "lrom.project_coordinates(",
        "lrom._cross_section_prediction(",
    ):
        assert implementation_detail not in text
```

Update scientific-core and timing assertions so ROSE/LS implementation details
are required in `notebooks/benchmark_helper.py`, while Notebook 02 is required
to contain the high-level calls and unchanged physical constants.

- [ ] **Step 2: Run the boundary test and verify failure**

Run:

```bash
python -m pytest -q \
  tests/test_notebook02_cross_sections.py::test_notebook02_uses_benchmark_helper_boundary
```

Expected: failure because Notebook 02 still defines the pipelines locally.

- [ ] **Step 3: Consolidate imports and delete moved utility definitions**

In Notebook 02's first code cell:

- add `import benchmark_helper`;
- remove direct `rose` and `njit` imports if no remaining notebook cell uses
  them;
- remove `Callable` if only moved ROSE timing used it.

Delete notebook definitions now owned by the helper:

- `pointwise_relative_error`;
- `summarize_relative_error`;
- ROSE timing;
- full Woods-Saxon ROSE callbacks;
- all ROSE basis/emulator/\(S\)-matrix/cross-section functions;
- FOM evaluation;
- LS oracle evaluation;
- ROSE-grid evaluation.

Retain sampling, LROM timing, LROM grid evaluation, presentation functions, and
tables. Replace error utility calls with
`benchmark_helper.summarize_relative_error()` or
`benchmark_helper.pointwise_relative_error()`.

- [ ] **Step 4: Add the short independent ROSE and LS calls**

Replace the moved sequential experiment code with:

```python
cross_section_benchmark = benchmark_helper.CrossSectionBenchmark(
    emulator=emulator,
    training_rows=train_rows,
    testing_rows=test_rows,
    angles_degrees=ANGLES_DEG,
    denominator_floor=ERROR_DENOMINATOR_FLOOR,
    timing_repeats=TIMING_REPEATS,
    timing_inner_loops=TIMING_INNER_LOOPS,
)
rose_comparison = cross_section_benchmark.run_rose(
    basis_sizes=BASIS_SIZES,
    eim_sizes=ROSE_EIM_SIZES,
)
ls_comparison = cross_section_benchmark.run_ls(
    basis_sizes=BASIS_SIZES,
    fom_training_cross_sections=rose_comparison.fom_training_cross_sections,
    fom_testing_cross_sections=rose_comparison.fom_testing_cross_sections,
    predictor_count=LROM_PREDICTOR_COUNTS[0],
)

rose_emulators = rose_comparison.emulators
fom_train_xs = rose_comparison.fom_training_cross_sections
fom_test_xs = rose_comparison.fom_testing_cross_sections
rose_results = rose_comparison.results
ls_results = ls_comparison.results
```

Keep old-v2 and archive-method LROM evaluation visibly separate. Because
`run_ls()` retrains the stateful emulator, run the notebook's LROM grid after
the helper calls so `default_predictor` and final LROM results retain their
existing meaning.

- [ ] **Step 5: Keep presentation calls unchanged**

Update only the pointwise error call inside
`cross_section_error_figure()`:

```python
errors = benchmark_helper.pointwise_relative_error(
    values[index],
    fom_xs[index],
    denominator_floor,
)
```

Do not move alpha selection, figures, tables, or validation presentation into
the helper.

- [ ] **Step 6: Run focused structural tests**

Run:

```bash
python -m pytest -q \
  tests/test_benchmark_helper.py \
  tests/test_notebook02_cross_sections.py::test_notebook02_uses_benchmark_helper_boundary \
  tests/test_notebook02_cross_sections.py::test_notebook02_code_cells_compile
python -m ruff check \
  notebooks/benchmark_helper.py \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb \
  tests/test_benchmark_helper.py \
  tests/test_notebook02_cross_sections.py \
  --ignore E402
```

Expected: all focused tests pass and Ruff reports no findings.

- [ ] **Step 7: Execute Notebook 02**

Run with local-kernel permission:

```bash
MPLCONFIGDIR=/tmp/lrom-mpl-cache jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb
```

Expected: nbconvert exits 0.

- [ ] **Step 8: Verify Notebook 02 scientific parity**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("notebooks/02_rose_vs_lrom_cross_sections.ipynb")
notebook = json.loads(path.read_text())
code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
errors = [
    output
    for cell in code
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
images = [
    output
    for cell in code
    for output in cell.get("outputs", [])
    if "image/png" in output.get("data", {})
]
assert not errors
assert len(images) == 5
text = "\n".join(
    "".join(output.get("data", {}).get("text/plain", []))
    for cell in code
    for output in cell.get("outputs", [])
)
for expected in (
    "test-0002",
    "test-0039",
    "test-0019",
    "0.043529",
    "0.000948",
    "0.028713",
):
    assert expected in text
print(f"Notebook 02: {len(code)} cells, 0 errors, 5 figures")
PY
```

Visually inspect all five figures and confirm that predictor radii use `r [fm]`
and the CAT plot still distinguishes ROSE squares from LROM circles.

- [ ] **Step 9: Commit Task 3**

Run:

```bash
git add notebooks/02_rose_vs_lrom_cross_sections.ipynb tests/test_notebook02_cross_sections.py
git commit -m "refactor(notebook02): use benchmark pipelines"
```

---

### Task 4: Migrate Benchmark 03 to the same helper

**Files:**

- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
- Modify: `tests/test_notebook02_cross_sections.py`

**Interfaces:**

- Uses the same `CrossSectionBenchmark` calls as Notebook 02.
- Supplies Benchmark 03's reduced/full profile rows, \(l_{\max}\), timing
  controls, and deterministic grids through the existing emulator state.

- [ ] **Step 1: Replace the Benchmark 03 implementation-detail contract**

Update `test_benchmark03_notebook02_profile_contract()`:

```python
assert "import benchmark_helper" in text
assert "CrossSectionBenchmark(" in text
assert ".run_rose(" in text
assert ".run_ls(" in text
for implementation_detail in (
    "rose.InteractionEIMSpace(",
    "rose.basis.CustomBasis(",
    "def exact_smatrix_all_channels",
    "def emulated_smatrix_all_channels",
    "def cross_section_from_smatrix",
    "lrom.project_coordinates(",
    "lrom._cross_section_prediction(",
):
    assert implementation_detail not in text
```

Retain assertions for profile selection, \(l_{\max}\), half-width, grids,
figures, alpha naming, timing constants, and `effective-interaction`.

- [ ] **Step 2: Run the benchmark contract and verify failure**

Run:

```bash
python -m pytest -q \
  tests/test_notebook02_cross_sections.py::test_benchmark03_notebook02_profile_contract
```

Expected: failure because Benchmark 03 still contains copied ROSE/LS code.

- [ ] **Step 3: Replace copied code with helper calls**

Apply the same import cleanup and pipeline deletion as Notebook 02. Construct:

```python
cross_section_benchmark = benchmark_helper.CrossSectionBenchmark(
    emulator=emulator,
    training_rows=train_rows,
    testing_rows=test_rows,
    angles_degrees=ANGLES_DEG,
    denominator_floor=ERROR_DENOMINATOR_FLOOR,
    timing_repeats=TIMING_REPEATS,
    timing_inner_loops=TIMING_INNER_LOOPS,
)
rose_comparison = cross_section_benchmark.run_rose(
    basis_sizes=BASIS_SIZES,
    eim_sizes=ROSE_EIM_SIZES,
)
ls_comparison = cross_section_benchmark.run_ls(
    basis_sizes=BASIS_SIZES,
    fom_training_cross_sections=rose_comparison.fom_training_cross_sections,
    fom_testing_cross_sections=rose_comparison.fom_testing_cross_sections,
    predictor_count=LROM_PREDICTOR_COUNTS[0],
)
```

Preserve Benchmark 03's reduced/full profile switch and optional
`LROM_BENCHMARK_L_MAX` values 3 and 10.

- [ ] **Step 4: Run compile, contract, and lint checks**

Run:

```bash
python -m pytest -q tests/test_notebook02_cross_sections.py
python -m ruff check \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb \
  tests/test_notebook02_cross_sections.py \
  --ignore E402
```

Expected: the full notebook contract file passes and Ruff reports no findings.

- [ ] **Step 5: Execute reduced Benchmark 03**

Run with local-kernel permission:

```bash
MPLCONFIGDIR=/tmp/lrom-mpl-cache jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
```

Expected: nbconvert exits 0.

- [ ] **Step 6: Verify Benchmark 03 outputs**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb")
notebook = json.loads(path.read_text())
code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
errors = [
    output
    for cell in code
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
images = [
    output
    for cell in code
    for output in cell.get("outputs", [])
    if "image/png" in output.get("data", {})
]
assert not errors
assert len(images) == 5
text = "\n".join(
    "".join(output.get("data", {}).get("text/plain", []))
    for cell in code
    for output in cell.get("outputs", [])
)
for selection in (
    "alpha selection A",
    "alpha selection B",
    "alpha selection C",
):
    assert selection in text
print(f"Benchmark 03: {len(code)} cells, 0 errors, 5 figures")
PY
```

Visually inspect all five stored figures.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb tests/test_notebook02_cross_sections.py
git commit -m "refactor(benchmark03): reuse benchmark pipelines"
```

---

### Task 5: Final parity, documentation, and handoff

**Files:**

- Modify: `../docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify: `../../_memory/Daily Notes/2026-07-27.md`
- Modify: `../../_memory/Context/active-state.md`

**Interfaces:**

- Documents the notebook/helper boundary and exact unchanged science.
- Produces final test, lint, figure, protected-hash, and repository-state
  evidence.

- [ ] **Step 1: Run the complete project verification**

Run:

```bash
python -m pytest -q
python -m ruff check \
  notebooks/benchmark_helper.py \
  notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb \
  tests \
  --ignore E402
```

Expected: all tests pass and Ruff reports no findings.

- [ ] **Step 2: Verify the helper reduction and independent APIs**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = (
    Path("notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb"),
    Path("notebooks/02_rose_vs_lrom_cross_sections.ipynb"),
    Path("notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb"),
)
for path in paths:
    notebook = json.loads(path.read_text())
    text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "benchmark_helper" in text
    assert "rose_helper" not in text

for path in paths[1:]:
    notebook = json.loads(path.read_text())
    text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    for detail in (
        "InteractionEIMSpace(",
        "CustomBasis(",
        "def exact_smatrix_all_channels",
        "def evaluate_ls_oracle",
    ):
        assert detail not in text

helper = Path("notebooks/benchmark_helper.py").read_text()
for stage in (
    "InteractionEIMSpace(",
    "make_base_solver(",
    "phi_free(",
    "CustomBasis(",
    "ScatteringAmplitudeEmulator(",
    "calculate_xs(",
):
    assert stage in helper
print("benchmark helper boundary verified")
PY
```

- [ ] **Step 3: Recheck protected files**

Run:

```bash
find ../scientific_archive -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256
git diff --name-only f23df4f..HEAD
```

Expected:

- scientific archive digest remains
  `afe80b7cf267fad0c3265bd18a7e023f966eb5fd4a655123887ee559c0061213`;
- no LROM package file, Benchmark 01, Benchmark 02, or archive file appears in
  the changed-file list.

Notebook 01's digest is expected to change because its pipeline calls and
stored execution are intentionally migrated; its scientific outputs, case IDs,
and figure count are the parity gate.

- [ ] **Step 4: Update architecture documentation**

Append a concise “Notebook benchmark helper boundary” section to
`../docs/LROM_ARCHITECTURE_UNDERSTANDING.md` recording:

- `benchmark_helper.py` is notebook support, not package API;
- ROSE and LS run independently;
- ROSE Guide construction stages are preserved;
- shared exact snapshots/free-reference bases remain the validated correction;
- the three migrated notebooks retain only high-level pipeline calls;
- Benchmark 01/02 and both LROM package streams remain unchanged;
- final execution/test/figure evidence.

- [ ] **Step 5: Update memory last**

Append a `[LROM_Project]` entry to
`../../_memory/Daily Notes/2026-07-27.md` and update
`../../_memory/Context/active-state.md` with:

- helper rename and public object/method names;
- migrated notebooks;
- scientific parity results;
- test/lint/execution evidence;
- unchanged protected files;
- pending blockers, if any;
- no push unless separately authorized.

- [ ] **Step 6: Commit the documentation milestone**

From `lrom_git`, commit only tracked repository documentation if modified
inside the repository:

```bash
git status --short
git add docs notebooks tests
git commit -m "docs: record benchmark helper migration"
```

Do not add `tmp/`. If all repository changes were already committed in Tasks
1--4, do not create an empty commit.

- [ ] **Step 7: Report the exact change**

Report:

- the two benchmark objects and independent/combined calls;
- which machinery left each notebook;
- the deliberate deviation from archive `from_train()` and why;
- test, execution, figure, and parity evidence;
- files explicitly unchanged;
- local commit IDs and push status.

