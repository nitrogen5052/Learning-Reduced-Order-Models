# Notebook 02 Clean-Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Notebook 02 into a self-contained, functions-first teaching notebook with consolidated imports, explicit inputs and outputs, no inspection printing, and unchanged scientific results.

**Architecture:** Keep every helper inside Notebook 02, but separate reusable numerical utilities, sampled-study construction, notebook-owned ROSE infrastructure, experiment evaluation, and presentation into distinct code cells. Functions receive study arrays, models, grids, timing controls, and floors explicitly; short top-level execution blocks connect those functions in scientific order.

**Tech Stack:** Python 3.12, Jupyter notebook JSON, NumPy, Pandas, Matplotlib, Numba, ROSE, parked `lrom_legacy.v2_0`, pytest, Ruff, nbconvert.

## Global Constraints

- Work in `/Users/Kitkat/Documents/Documents-Agent/LROM_Project/lrom_git`.
- Modify Notebook 02 and its contract test; do not modify public v1.2, parked-v2 package code, Notebook 01, Benchmark 03, or the scientific archive.
- Keep all extracted helpers inside `notebooks/02_rose_vs_lrom_cross_sections.ipynb`.
- Put every import in the first code cell and remove all temporary `print(...)` and `.head()` inspection output.
- Use standard Python/NumPy annotations on extracted helpers; every study dependency must be an explicit parameter rather than an implicit notebook global.
- Keep concise scientific assertions at the top-level execution boundary; do not add defensive validation branches inside helpers.
- Preserve \(^{40}\mathrm{Ca}(n,n)\), 14.1 MeV, \(l=0,\ldots,3\), the full ten-parameter Woods-Saxon interaction, mesh 600, seed 1204, 200/100 rows, closed ±20% alpha ranges, angles 1 through 179 degrees, and all basis/compression grids.
- Keep ROSE bases, EIMs, prediction, and timing notebook-owned and separate from the LROM package.
- Preserve the five existing figures and their order. Use “alpha selection” in notebook prose and output.
- Remove the tautological `rose_train_rows`/`rose_test_rows` aliases and the duplicate `median_error`/`cat_error` metric.
- Leave the existing untracked `tmp/` directory untouched and out of every commit.
- Use `apply_patch` for source edits and `python -m pytest`, not the standalone `pytest` entry point.

---

### Task 1: Lock the clean notebook shell and numerical utilities

**Files:**

- Modify: `tests/test_notebook02_cross_sections.py`
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb` code cells 0 through 3

**Interfaces:**

- Consumes: existing `code_sources(path)` and `notebook_text(path)` test helpers.
- Produces:
  - `parameter_dicts(rows: np.ndarray, parameter_names: tuple[str, ...]) -> list[dict[str, float]]`
  - `pointwise_relative_error(predicted: np.ndarray, reference: np.ndarray, denominator_floor: float) -> np.ndarray`
  - `summarize_relative_error(predicted: np.ndarray, reference: np.ndarray, denominator_floor: float) -> dict[str, np.ndarray]`
  - `time_lrom_predictions(model: Any, cases: list[dict[str, float]], repeats: int, inner_loops: int) -> np.ndarray`
  - `time_rose_predictions(model: Any, rows: np.ndarray, repeats: int, inner_loops: int, smatrix_function: Callable, cross_section_function: Callable, angles_rad: np.ndarray) -> np.ndarray`
  - `build_sampled_study(...) -> tuple[Any, dict[str, float], dict[str, tuple[float, float]], tuple[str, ...], np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]`

- [ ] **Step 1: Add AST helpers and the failing clean-shell contract**

Add `import ast` to the top of `tests/test_notebook02_cross_sections.py`, then add:

```python
def notebook_functions(path: Path) -> dict[str, ast.FunctionDef]:
    functions = {}
    for source in code_sources(path):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef):
                functions[node.name] = node
    return functions


def test_notebook02_clean_shell_contract() -> None:
    sources = code_sources(NOTEBOOK_02)
    for source in sources[1:]:
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(ast.parse(source))
        )

    calls = [
        node
        for source in sources
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    ]
    assert not any(
        isinstance(call.func, ast.Name) and call.func.id == "print"
        for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "head"
        for call in calls
    )

    text = notebook_text(NOTEBOOK_02)
    assert "rose_train_rows" not in text
    assert "rose_test_rows" not in text

    expected = {
        "parameter_dicts",
        "pointwise_relative_error",
        "summarize_relative_error",
        "time_lrom_predictions",
        "time_rose_predictions",
        "build_sampled_study",
    }
    functions = notebook_functions(NOTEBOOK_02)
    assert expected <= functions.keys()
    for name in expected:
        function = functions[name]
        assert function.returns is not None
        assert all(argument.annotation is not None for argument in function.args.args)
```

- [ ] **Step 2: Run the contract and verify the intended failure**

Run:

```bash
python -m pytest -q tests/test_notebook02_cross_sections.py::test_notebook02_clean_shell_contract
```

Expected: FAIL because Cell 0 contains `print`, the ROSE row aliases remain, and the extracted function names do not yet exist.

- [ ] **Step 3: Consolidate the import cell**

Replace code cell 0 with one import block containing:

```python
from collections.abc import Callable
from pathlib import Path
from typing import Any
import platform
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from numba import njit

ROOT = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "lrom_legacy").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scipy.special
if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
    scipy.special.sph_harm = (
        lambda m, n, theta, phi: scipy.special.sph_harm_y(n, m, phi, theta)
    )

import rose
import lrom_legacy.v2_0 as lrom

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25})
```

Do not replace the removed version print elsewhere. Cell 11 will record
`lrom.__version__` in the validation table.

- [ ] **Step 4: Replace Cell 2 utilities with explicit functional interfaces**

Use these implementations:

```python
def parameter_dicts(
    rows: np.ndarray,
    parameter_names: tuple[str, ...],
) -> list[dict[str, float]]:
    return [
        {name: float(value) for name, value in zip(parameter_names, row)}
        for row in rows
    ]


def pointwise_relative_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float,
) -> np.ndarray:
    denominator = np.maximum(np.abs(reference), denominator_floor)
    return np.abs(predicted - reference) / denominator


def summarize_relative_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float,
) -> dict[str, np.ndarray]:
    pointwise = pointwise_relative_error(
        predicted,
        reference,
        denominator_floor,
    )
    return {
        "median_over_angle_error": np.median(pointwise, axis=1),
        "maximum_over_angle_error": np.max(pointwise, axis=1),
    }


def time_lrom_predictions(
    model: Any,
    cases: list[dict[str, float]],
    repeats: int,
    inner_loops: int,
) -> np.ndarray:
    model.predict(parameters=cases[:1], reconstruct_wavefunctions=False)
    seconds = []
    for case in cases:
        measurements = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(inner_loops):
                model.predict(
                    parameters=case,
                    reconstruct_wavefunctions=False,
                )
            measurements.append(
                (time.perf_counter_ns() - start) / (1e9 * inner_loops)
            )
        seconds.append(min(measurements))
    return np.asarray(seconds)


def time_rose_predictions(
    model: Any,
    rows: np.ndarray,
    repeats: int,
    inner_loops: int,
    smatrix_function: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]],
    cross_section_function: Callable[
        [Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
    ],
    angles_rad: np.ndarray,
) -> np.ndarray:
    warm_splus, warm_sminus = smatrix_function(model, rows[0])
    cross_section_function(
        model,
        rows[0],
        warm_splus,
        warm_sminus,
        angles_rad,
    )
    seconds = []
    for row in rows:
        measurements = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(inner_loops):
                splus, sminus = smatrix_function(model, row)
                cross_section_function(
                    model,
                    row,
                    splus,
                    sminus,
                    angles_rad,
                )
            measurements.append(
                (time.perf_counter_ns() - start) / (1e9 * inner_loops)
            )
        seconds.append(min(measurements))
    return np.asarray(seconds)
```

- [ ] **Step 5: Extract sampled-study construction**

Define in Cell 3:

```python
def build_sampled_study(
    target: tuple[int, int],
    projectile: tuple[int, int],
    lab_energy: float,
    l_max: int,
    half_width: float,
    training_size: int,
    testing_size: int,
    mesh_size: int,
    seed: int,
) -> tuple[
    Any,
    dict[str, float],
    dict[str, tuple[float, float]],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
]:
    emulator = lrom.LROM(
        target=target,
        projectile=projectile,
        lab_energy=lab_energy,
        l=tuple(range(l_max + 1)),
        potential="full_woods-saxon",
    )
    central = dict(emulator.central_parameters)
    ranges = {
        name: tuple(
            sorted(
                (
                    (1.0 - half_width) * value,
                    (1.0 + half_width) * value,
                )
            )
        )
        for name, value in central.items()
    }
    emulator.sampling(
        training_ranges=ranges,
        testing_ranges=ranges,
        training_size=training_size,
        testing_size=testing_size,
        mesh_size=mesh_size,
        strategy="latin_hypercube",
        seed=seed,
        high_fidelity_solver="runge_kutta",
        solver_options={"rk_tols": (1e-9, 1e-9)},
    )
    design = emulator.samples.design
    return (
        emulator,
        central,
        ranges,
        emulator.parameter_names,
        design.training.values.copy(),
        design.testing.values.copy(),
        design.training.case_ids,
        design.testing.case_ids,
    )
```

Call it with the Cell 1 constants. Keep only top-level shape, non-overlap, and
range assertions. Remove both ROSE row aliases and pass `train_rows` directly
to later ROSE construction.

- [ ] **Step 6: Update current downstream names without changing algorithms**

In later cells replace:

```text
rows_as_parameter_dicts -> parameter_dicts
minimum_lrom_times -> time_lrom_predictions
minimum_rose_times -> time_rose_predictions
error_summaries -> summarize_relative_error
train_median_error -> train_median_over_angle_error
test_median_error -> test_median_over_angle_error
train_cat_error -> train_median_over_angle_error
test_cat_error -> test_median_over_angle_error
```

Pass `ERROR_DENOMINATOR_FLOOR`, timing repeat counts, callbacks, and
`ANGLES_RAD` explicitly at each call.

- [ ] **Step 7: Run the focused shell and compile tests**

Run:

```bash
python -m pytest -q \
  tests/test_notebook02_cross_sections.py::test_notebook02_clean_shell_contract \
  tests/test_notebook02_cross_sections.py::test_notebook02_code_cells_compile
```

Expected: 2 passed.

- [ ] **Step 8: Commit Task 1**

```bash
git add tests/test_notebook02_cross_sections.py notebooks/02_rose_vs_lrom_cross_sections.ipynb
git commit -m "refactor(notebook02): extract study utilities"
```

---

### Task 2: Extract notebook-owned ROSE and experiment evaluation

**Files:**

- Modify: `tests/test_notebook02_cross_sections.py`
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb` code cells 5 and 6

**Interfaces:**

- Consumes: Task 1 error/timing functions and sampled study values.
- Produces:
  - `build_rose_bases(interaction, n_phi, emulator, rho_mesh) -> list[list[Any]]`
  - `build_rose_emulator(interaction, n_phi, emulator, rho_mesh, l_max, angles_rad, s0) -> Any`
  - `exact_smatrix_all_channels(sae, row) -> tuple[np.ndarray, np.ndarray]`
  - `emulated_smatrix_all_channels(sae, row) -> tuple[np.ndarray, np.ndarray]`
  - `cross_section_from_smatrix(sae, parameters, splus, sminus, angles_rad) -> np.ndarray`
  - `build_rose_emulators(...) -> dict[tuple[int, int], Any]`
  - `evaluate_fom_cross_sections(...) -> tuple[np.ndarray, np.ndarray]`
  - `evaluate_old_lrom(...) -> dict[tuple[int, int], dict[str, np.ndarray]]`
  - `evaluate_ls_oracle(...) -> dict[str, np.ndarray]`
  - `evaluate_lrom_grid(...) -> tuple[dict[tuple[int, int], dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]], dict[Any, Any]]`
  - `evaluate_rose_grid(...) -> dict[tuple[int, int], dict[str, np.ndarray]]`

- [ ] **Step 1: Add the failing experiment-function contract**

Add:

```python
def test_notebook02_experiment_functions_are_extracted() -> None:
    expected = {
        "build_rose_bases",
        "build_rose_emulator",
        "build_rose_emulators",
        "exact_smatrix_all_channels",
        "emulated_smatrix_all_channels",
        "cross_section_from_smatrix",
        "evaluate_fom_cross_sections",
        "evaluate_old_lrom",
        "evaluate_ls_oracle",
        "evaluate_lrom_grid",
        "evaluate_rose_grid",
    }
    functions = notebook_functions(NOTEBOOK_02)
    assert expected <= functions.keys()
    for name in expected:
        function = functions[name]
        assert function.returns is not None
        assert all(argument.annotation is not None for argument in function.args.args)
```

- [ ] **Step 2: Run the new contract and verify failure**

Run:

```bash
python -m pytest -q tests/test_notebook02_cross_sections.py::test_notebook02_experiment_functions_are_extracted
```

Expected: FAIL because the five `evaluate_*` functions and
`build_rose_emulators` do not yet exist and existing ROSE helpers lack
annotations.

- [ ] **Step 3: Make every ROSE helper dependency explicit**

Retain the current Numba potential formulas, then use:

```python
def build_rose_bases(
    interaction: Any,
    n_phi: int,
    emulator: Any,
    rho_mesh: np.ndarray,
) -> list[list[Any]]:
    bases = []
    for ell, interaction_row in enumerate(interaction.interactions):
        row_bases = []
        for spin_index, _ in enumerate(interaction_row):
            key = ell if len(interaction_row) == 1 else (ell, spin_index)
            model = emulator.samples.full_order_models[key]
            free_reference = np.asarray(
                [
                    rose.free_solutions.phi_free(
                        float(s),
                        ell,
                        emulator.kinematics.eta,
                    )
                    for s in rho_mesh
                ],
                dtype=np.complex128,
            )
            row_bases.append(
                rose.basis.CustomBasis(
                    solutions=np.asarray(
                        emulator.samples.training_wavefunctions[key],
                        dtype=np.complex128,
                    ).T.copy(),
                    phi_0=free_reference,
                    rho_mesh=rho_mesh,
                    n_basis=n_phi,
                    solver=model.solver,
                    subtract_phi0=True,
                    use_svd=True,
                    center=False,
                    scale=False,
                )
            )
        bases.append(row_bases)
    return bases
```

`build_rose_emulator` must receive and pass `emulator`, `rho_mesh`, `l_max`,
`angles_rad`, and `s0`. `cross_section_from_smatrix` must receive
`angles_rad`, compare against the cached grid, and return `.dsdo`.

- [ ] **Step 4: Extract construction of the ROSE configuration grid**

Implement:

```python
def build_rose_emulators(
    basis_sizes: tuple[int, ...],
    eim_sizes: tuple[int, ...],
    l_max: int,
    parameter_count: int,
    emulator: Any,
    training_rows: np.ndarray,
    rho_mesh: np.ndarray,
    angles_rad: np.ndarray,
    s0: float,
) -> dict[tuple[int, int], Any]:
    emulators = {}
    for n_phi in basis_sizes:
        for n_u in eim_sizes:
            interaction = rose.InteractionEIMSpace(
                l_max=l_max,
                coordinate_space_potential=bench_full_ws,
                spin_orbit_term=bench_full_ws_so,
                n_theta=parameter_count,
                mu=emulator.kinematics.mu,
                energy=emulator.kinematics.e_com,
                is_complex=True,
                training_info=training_rows,
                explicit_training=True,
                n_basis=n_u,
                rho_mesh=rho_mesh,
            )
            emulators[(n_phi, n_u)] = build_rose_emulator(
                interaction,
                n_phi,
                emulator,
                rho_mesh,
                l_max,
                angles_rad,
                s0,
            )
    return emulators
```

- [ ] **Step 5: Extract exact FOM evaluation**

Implement:

```python
def evaluate_fom_cross_sections(
    reference_sae: Any,
    training_rows: np.ndarray,
    testing_rows: np.ndarray,
    angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    def evaluate(rows: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                cross_section_from_smatrix(
                    reference_sae,
                    row,
                    *exact_smatrix_all_channels(reference_sae, row),
                    angles_rad,
                )
                for row in rows
            ]
        )

    return evaluate(training_rows), evaluate(testing_rows)
```

Keep the output-shape and finite/nonnegative checks after the function call,
not inside this helper.

- [ ] **Step 6: Extract old-v2, LS, LROM-grid, and ROSE-grid evaluators**

Use one consistent result schema:

```python
{
    "train_xs": np.ndarray,
    "test_xs": np.ndarray,
    "train_median_over_angle_error": np.ndarray,
    "test_median_over_angle_error": np.ndarray,
    "train_maximum_over_angle_error": np.ndarray,
    "test_maximum_over_angle_error": np.ndarray,
    "test_seconds": np.ndarray,
}
```

`evaluate_old_lrom` trains the shared-potential default, times test cases, and
returns `{(basis_size, predictor_count): result}`.

`evaluate_ls_oracle` accepts the currently trained emulator, training/testing
rows, FOM arrays, and denominator floor. It projects exact wavefunctions with
`lrom.project_coordinates`, calls `lrom._cross_section_prediction`, and returns
the schema without `test_seconds`.

`evaluate_lrom_grid` loops over `basis_sizes` and `predictor_counts`, trains
`predictor="effective-interaction"`, times/predicts both row sets, builds new
result dictionaries, calls `evaluate_ls_oracle` once per basis size when
`predictor_count == predictor_counts[0]`, and returns LROM results, LS results,
and the default predictor mapping.

`evaluate_rose_grid` loops over the supplied emulator mapping, times with the
Task 1 ROSE timer, predicts each row through
`emulated_smatrix_all_channels`/`cross_section_from_smatrix`, and returns the
same result schema.

Every one of these signatures must explicitly accept:

```text
rows/cases, basis/compression grids, angles, FOM arrays,
denominator floor, timing repeats, timing inner loops
```

No helper may write `old_v2_results`, `lrom_results`, `ls_results`, or
`rose_results` as a global.

- [ ] **Step 7: Replace the Cell 6 sequential execution with short calls**

The execution block must read in this order:

```python
rose_emulators = build_rose_emulators(...)
reference_sae = rose_emulators[(BASIS_SIZES[0], ROSE_EIM_SIZES[0])]
fom_train_xs, fom_test_xs = evaluate_fom_cross_sections(...)
train_cases = parameter_dicts(train_rows, parameter_names)
test_cases = parameter_dicts(test_rows, parameter_names)
old_v2_results = evaluate_old_lrom(...)
lrom_results, ls_results, default_predictor = evaluate_lrom_grid(...)
archive_lrom_results = lrom_results
rose_results = evaluate_rose_grid(...)
```

Keep the existing configuration-set, finite-result, predictor-identity, and
old-v2 improvement assertions after these calls, updated to
`test_median_over_angle_error`.

- [ ] **Step 8: Run focused experiment and neighboring contracts**

Run:

```bash
python -m pytest -q \
  tests/test_notebook02_cross_sections.py::test_notebook02_experiment_functions_are_extracted \
  tests/test_notebook02_cross_sections.py::test_notebook02_scientific_core_contract \
  tests/test_notebook02_cross_sections.py::test_notebook02_code_cells_compile
```

Expected: 3 passed after updating the scientific-core assertions to require
`training_info=training_rows`, the new function names, and
`median_over_angle_error`, while leaving Benchmark 03 assertions unchanged.

- [ ] **Step 9: Commit Task 2**

```bash
git add tests/test_notebook02_cross_sections.py notebooks/02_rose_vs_lrom_cross_sections.ipynb
git commit -m "refactor(notebook02): extract comparison workflows"
```

---

### Task 3: Extract presentation functions and remove metric duplication

**Files:**

- Modify: `tests/test_notebook02_cross_sections.py`
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb` code cells 4 and 7 through 11

**Interfaces:**

- Consumes: Task 2 result schemas.
- Produces:
  - `predictor_radius_figure(...) -> tuple[dict[Any, Any], pd.DataFrame, Figure]`
  - `select_alpha_cases(lrom_error, rose_error, count) -> list[int]`
  - `representative_cross_section_figure(...) -> Figure`
  - `cross_section_error_figure(...) -> Figure`
  - `error_distribution_figure(...) -> Figure`
  - `summary_results_table(...) -> pd.DataFrame`
  - `accuracy_time_figure(...) -> Figure`
  - `validation_results_table(...) -> pd.DataFrame`

- [ ] **Step 1: Add the failing presentation-function contract**

Add:

```python
def test_notebook02_presentation_functions_are_extracted() -> None:
    expected = {
        "predictor_radius_figure",
        "select_alpha_cases",
        "representative_cross_section_figure",
        "cross_section_error_figure",
        "error_distribution_figure",
        "summary_results_table",
        "accuracy_time_figure",
        "validation_results_table",
    }
    functions = notebook_functions(NOTEBOOK_02)
    assert expected <= functions.keys()
    for name in expected:
        function = functions[name]
        assert function.returns is not None
        assert all(argument.annotation is not None for argument in function.args.args)
```

Update `test_notebook02_results_contract` by replacing:

```python
assert "def plot_" not in text
```

with:

```python
for function_name in (
    "predictor_radius_figure",
    "representative_cross_section_figure",
    "cross_section_error_figure",
    "error_distribution_figure",
    "accuracy_time_figure",
):
    assert f"def {function_name}" in text
```

- [ ] **Step 2: Run the presentation contract and verify failure**

Run:

```bash
python -m pytest -q tests/test_notebook02_cross_sections.py::test_notebook02_presentation_functions_are_extracted
```

Expected: FAIL because the presentation functions do not yet exist.

- [ ] **Step 3: Extract predictor selection and its figure**

`predictor_radius_figure` must receive:

```text
emulator, central, ranges, parameter_names, training_rows, testing_rows,
predictor_count, minimum_radius, l_max
```

It constructs the current predictor mapping and representative potential rows,
draws the same four axes and physical-radius lines, then returns:

```python
return predictors, pd.DataFrame(predictor_rows), fig
```

The execution block calls the function, displays the figure, and displays the
table. Keep `# FIGURE: potential-predictor-rainbows`.

- [ ] **Step 4: Extract alpha selection and representative figures**

Implement the selection function:

```python
def select_alpha_cases(
    lrom_error: np.ndarray,
    rose_error: np.ndarray,
    count: int,
) -> list[int]:
    lrom_rank = np.argsort(np.argsort(lrom_error))
    rose_rank = np.argsort(np.argsort(rose_error))
    ordered = np.argsort(0.5 * (lrom_rank + rose_rank))
    positions = np.linspace(0, len(ordered) - 1, count + 2, dtype=int)[1:-1]
    return [int(ordered[position]) for position in positions]
```

The two representative figure functions receive `angles_deg`, selected
indices/labels, test IDs, FOM arrays, and default LS/ROSE/LROM cross-section
arrays. The error figure additionally receives `denominator_floor` and
`plotting_floor`. Both return `Figure` and preserve the existing legend,
line-style, log-scale, axes, and marker comments.

- [ ] **Step 5: Extract the distribution and summary table**

`error_distribution_figure` receives default LS/ROSE/LROM result dictionaries
and `plotting_floor`; it reads only
`train_median_over_angle_error` and `test_median_over_angle_error`.

`summary_results_table` receives both result mappings and returns columns:

```text
method
n_phi
compression symbol
compression
median pointwise test error
median maximum-over-angle error
median online time [s]
```

Do not retain both `median test error` and `median pointwise CAT error`.

- [ ] **Step 6: Extract CAT and validation presentation**

`accuracy_time_figure` receives result mappings, basis sizes, compression
values, plotting floor, error reference `0.10`, and throughput
`1_000_000`. It uses `test_median_over_angle_error` for the vertical
coordinate and returns the current figure.

`validation_results_table` receives all values explicitly, including
`python_version`, `platform_name`, and `lrom_version`, and returns the final
DataFrame. It reports the old/default archive metric from
`test_median_over_angle_error`.

- [ ] **Step 7: Update the short presentation calls**

Use `plt.show()` only after each returned figure:

```python
figure = representative_cross_section_figure(...)
plt.show()
```

Leave result tables as the final expression or use `display(...)` only for the
two tables intentionally paired with figures. Do not add printing or `.head()`.

- [ ] **Step 8: Run presentation and complete focused tests**

Run:

```bash
python -m pytest -q tests/test_notebook02_cross_sections.py
```

Expected: all Notebook 02 and unchanged Benchmark 03 contract tests pass.

- [ ] **Step 9: Run Ruff on source-bearing artifacts**

Run:

```bash
python -m ruff check tests/test_notebook02_cross_sections.py \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb --ignore E402
```

Expected: no findings. If Ruff identifies a genuine notebook undefined name,
fix the explicit data flow rather than suppressing `F821`.

- [ ] **Step 10: Commit Task 3**

```bash
git add tests/test_notebook02_cross_sections.py notebooks/02_rose_vs_lrom_cross_sections.ipynb
git commit -m "refactor(notebook02): extract presentation functions"
```

---

### Task 4: Execute, compare, inspect, and document the cleaned notebook

**Files:**

- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb` stored execution outputs
- Modify: `../docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify: `../../_memory/Daily Notes/2026-07-27.md`
- Modify: `../../_memory/Context/active-state.md`

**Interfaces:**

- Consumes: cleaned Notebook 02 and baseline commit `a8fc707`.
- Produces: executed notebook with 12/12 code cells, zero errors, five stored PNG figures, parity evidence, and project handoff.

- [ ] **Step 1: Record protected pre-execution digests**

Run:

```bash
shasum -a 256 notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb | shasum -a 256
find ../scientific_archive -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256
```

Expected protected digests:

```text
f8197e6c0a111d5192d6039fe0e629e3958f2831a832554e545be43b809d5c44
afe80b7cf267fad0c3265bd18a7e023f966eb5fd4a655123887ee559c0061213
```

- [ ] **Step 2: Execute Notebook 02 in place**

Run:

```bash
MPLCONFIGDIR=/tmp/lrom-mpl-cache jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb
```

Jupyter requires permission to open localhost kernel sockets. Expected:
nbconvert exits 0 and writes the notebook.

- [ ] **Step 3: Verify executed-cell and figure counts**

Read the notebook JSON and assert:

```python
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
errors = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
images = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if "image/png" in output.get("data", {})
]
assert len(code_cells) == 12
assert all(cell["execution_count"] is not None for cell in code_cells)
assert not errors
assert len(images) == 5
```

- [ ] **Step 4: Compare scientific outputs with `a8fc707`**

Load the current notebook and:

```bash
git show a8fc707:notebooks/02_rose_vs_lrom_cross_sections.ipynb
```

Compare:

- alpha table values and IDs exactly;
- predictor-radius table exactly;
- the single retained median-over-angle result column against either old
  duplicate median column;
- maximum-over-angle results exactly at displayed precision;
- validation physics/configuration rows exactly;
- old-v2 default error `0.043529`;
- archive LROM default error `0.000948`.

Do not require timing equality.

- [ ] **Step 5: Extract and visually inspect all five PNGs**

Extract the stored images to a new `mktemp -d` directory under `/tmp`, then
inspect:

1. channel predictor radii;
2. representative cross sections;
3. representative errors;
4. split train/test distributions;
5. accuracy versus time.

Verify labels remain in physical radius `r [fm]` where applicable and no title,
legend, or axis is clipped.

- [ ] **Step 6: Run final focused and full verification**

Run:

```bash
python -m pytest -q tests/test_notebook02_cross_sections.py
python -m pytest -q
python -m ruff check lrom lrom_legacy tests
```

Expected: every command exits 0.

- [ ] **Step 7: Recheck protected digests**

Repeat Step 1 and require exact equality with both protected hashes.

- [ ] **Step 8: Update the architecture note**

Add a Notebook 02 structure-cleanup change record stating:

- all imports are consolidated;
- utilities, sampling, ROSE, LROM/LS evaluation, and presentation are explicit
  notebook-local functions;
- duplicated median metrics and tautological ROSE aliases were removed;
- ROSE remains notebook-owned;
- scientific outputs and five figures are unchanged outside timing variation;
- package code, Notebook 01, Benchmark 03, and archive files are unchanged.

- [ ] **Step 9: Commit executed notebook and architecture evidence**

Commit the notebook inside `lrom_git`:

```bash
git add notebooks/02_rose_vs_lrom_cross_sections.ipynb
git commit -m "docs(notebook02): execute cleaned comparison"
```

The architecture note is adjacent to the git checkout and is recorded in the
required memory handoff rather than this repository commit.

- [ ] **Step 10: Update required memory logs last**

Append a `[LROM_Project]` entry to
`../../_memory/Daily Notes/2026-07-27.md` and prepend a concise handoff under
`Last Session Handoff` in `../../_memory/Context/active-state.md`. Record
commits, modified files, test counts, execution/figure counts, numerical
parity, protected hashes, and any remaining blocker.

- [ ] **Step 11: Confirm final git scope**

Run:

```bash
git status --short
git log --oneline -8
```

Expected: only the pre-existing untracked `tmp/` remains; no package,
Benchmark 03, Notebook 01, or archive file appears.
