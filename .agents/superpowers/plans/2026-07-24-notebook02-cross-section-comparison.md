# Notebook 02 ROSE-LROM Cross-Section Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute Notebook 02’s shared-sample ROSE/LROM cross-section study and adapt benchmark 03 into its reduced/full reproducibility benchmark.

**Architecture:** Notebook 02 is the authoritative full 200/100 scientific artifact with sparse user-owned prose. Benchmark 03 runs the same physics and Cartesian configuration grid, defaults to a determined 120/30 validation profile, and switches to the notebook profile through `LROM_BENCHMARK_PROFILE=full`. Both notebooks keep method-native bases separate, use exact shared parameter rows, and perform all plot assembly inline.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pandas, Matplotlib, Numba, nuclear-rose, `lrom_legacy.v2_0`, Jupyter/nbformat, pytest.

## Global Constraints

- Work from `/Users/Kitkat/Documents/Documents-Agent/LROM_Project/lrom_git`.
- Create `notebooks/02_rose_vs_lrom_cross_sections.ipynb`.
- Modify `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`.
- Do not modify `lrom/__init__.py`, `lrom_legacy/v1_2/__init__.py`, `lrom_legacy/v2_0/__init__.py`, Notebook 01, or scientific archive files.
- Import `lrom_legacy.v2_0` explicitly; public `import lrom` remains v1.2.
- Use \(^{40}\mathrm{Ca}(n,n)\), `LAB_ENERGY = 14.1`, `L_MAX = 3`, `MESH_SIZE = 600`, and angles 1 through 179 degrees inclusive.
- Use one deterministic v2.0 sampling seed, `SEED = 1204`; v2.0 internally spawns independent training/testing streams.
- Use the same closed ±20% ranges and exact ordered parameter rows for ROSE and LROM.
- Notebook/full profile uses 200 training and 100 testing rows.
- Reduced benchmark profile uses 120 training and 30 testing rows.
- Use `BASIS_SIZES = (4, 6, 8)`, `ROSE_EIM_SIZES = (4, 8, 12)`, and `LROM_PREDICTOR_COUNTS = (4, 8, 12)` as a full Cartesian grid.
- Compare ROSE and LROM as equal-basis only when `n_phi` matches.
- Label \(n_U\) and \(K\) as analogous compression controls, not identical objects.
- Show FOM, `LS-projected cross section`, ROSE, and predictor LROM; never show linear LROM.
- The LS result is a wavefunction-space oracle propagated to the observable, not a guaranteed cross-section floor.
- Use physical radius \(r\) in fm in every spatial result.
- CAT accuracy is maximum stabilized pointwise relative error over angle, one point per sample and configuration.
- Online timing excludes training, EIM construction, high-fidelity solves, plotting, and I/O.
- Notebook Markdown at delivery is limited to the title and nine approved section headings.
- Do not add a persistent notebook generator or plotting wrapper.
- Defer interactive HTML export.
- Commit only scoped files. Push only after the user explicitly authorizes the configured external remote.

---

### Task 1: Lock and create the sparse Notebook 02 shell

**Files:**
- Create: `tests/test_notebook02_cross_sections.py`
- Create: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`

**Interfaces:**
- Consumes: approved headings and numerical constants from the design spec.
- Produces: a valid nbformat v4 notebook with compilable import/configuration cells; `load_notebook(path: Path) -> dict` and `notebook_text(path: Path) -> str` test helpers.

- [ ] **Step 1: Write the failing shell contract**

Create `tests/test_notebook02_cross_sections.py` with:

```python
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_02 = ROOT / "notebooks" / "02_rose_vs_lrom_cross_sections.ipynb"
BENCHMARK_03 = (
    ROOT / "notebooks" / "benchmark_notebooks" / "2.0" / "benchmark_03.ipynb"
)

HEADINGS = (
    "# 02. ROSE and LROM Cross-Section Comparison",
    "## 1. Physical Problem and Optical-Potential Parameters",
    "## 2. Shared Training and Testing Samples",
    "## 3. Potential Variation and LROM Predictor Locations",
    "## 4. Equal-Basis ROSE and LROM Emulators",
    "## 5. Representative Cross-Section Predictions",
    "## 6. Cross-Section Error Distributions",
    "## 7. Basis and Operator-Size Comparison",
    "## 8. Accuracy Versus Online Time",
    "## 9. Validation Summary",
)


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def notebook_text(path: Path) -> str:
    notebook = load_notebook(path)
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def code_sources(path: Path) -> list[str]:
    notebook = load_notebook(path)
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_notebook02_shell_contract() -> None:
    assert NOTEBOOK_02.exists()
    text = notebook_text(NOTEBOOK_02)
    for heading in HEADINGS:
        assert heading in text
    assert "import lrom_legacy.v2_0 as lrom" in text
    assert "TARGET = (40, 20)" in text
    assert "PROJECTILE = (1, 0)" in text
    assert "LAB_ENERGY = 14.1" in text
    assert "L_MAX = 3" in text
    assert "MESH_SIZE = 600" in text
    assert "N_TRAIN = 200" in text
    assert "N_TEST = 100" in text
    assert "HALF_WIDTH = 0.20" in text
    assert "BASIS_SIZES = (4, 6, 8)" in text
    assert "ROSE_EIM_SIZES = (4, 8, 12)" in text
    assert "LROM_PREDICTOR_COUNTS = (4, 8, 12)" in text
    assert "SEED = 1204" in text


def test_notebook02_code_cells_compile() -> None:
    for index, source in enumerate(code_sources(NOTEBOOK_02)):
        compile(source, f"{NOTEBOOK_02.name} code cell {index}", "exec")
```

- [ ] **Step 2: Run the shell contract and observe the expected failure**

Run:

```bash
python -m pytest tests/test_notebook02_cross_sections.py::test_notebook02_shell_contract -v
```

Expected: FAIL because `notebooks/02_rose_vs_lrom_cross_sections.ipynb` does not exist.

- [ ] **Step 3: Create the notebook shell with exact cells**

Use nbformat in a one-time temporary editing script. Do not commit the script. The notebook cells, in order, must be:

```python
# Markdown
# 02. ROSE and LROM Cross-Section Comparison
```

```python
# Code
from pathlib import Path
import platform
import sys
import time

import matplotlib.pyplot as plt
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
print("lrom version:", lrom.__version__)
```

```python
# Markdown
## 1. Physical Problem and Optical-Potential Parameters
```

```python
# Code
TARGET = (40, 20)
PROJECTILE = (1, 0)
LAB_ENERGY = 14.1
L_MAX = 3
MESH_SIZE = 600
N_TRAIN = 200
N_TEST = 100
HALF_WIDTH = 0.20
ANGLES_DEG = np.arange(1.0, 180.0, 1.0)
ANGLES_RAD = np.deg2rad(ANGLES_DEG)
BASIS_SIZES = (4, 6, 8)
ROSE_EIM_SIZES = (4, 8, 12)
LROM_PREDICTOR_COUNTS = (4, 8, 12)
DEFAULT_BASIS_SIZE = 6
DEFAULT_COMPRESSION_SIZE = 8
TIMING_REPEATS = 3
SEED = 1204
ERROR_DENOMINATOR_FLOOR = 1e-12
PLOTTING_FLOOR = 1e-6
```

Add the remaining eight approved Markdown headings as separate cells in their approved order. Leave code gaps between them for later tasks.

- [ ] **Step 4: Run the complete shell tests**

Run:

```bash
python -m pytest tests/test_notebook02_cross_sections.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit the shell**

```bash
git add tests/test_notebook02_cross_sections.py notebooks/02_rose_vs_lrom_cross_sections.ipynb
git commit -m "test: lock Notebook 02 scientific shell"
```

---

### Task 2: Implement shared sampling, ROSE/LROM grids, LS propagation, and metrics

**Files:**
- Modify: `tests/test_notebook02_cross_sections.py`
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`

**Interfaces:**
- Consumes: v2.0 `LROM.sampling`, `LROM.train`, `LROM.predict`, `project_coordinates`, and `_cross_section_prediction`; ROSE `InteractionEIMSpace` and `ScatteringAmplitudeEmulator.from_train`.
- Produces:
  - `train_rows: np.ndarray` with shape `(200, 10)`;
  - `test_rows: np.ndarray` with shape `(100, 10)`;
  - `fom_train_xs: np.ndarray` with shape `(200, 179)`;
  - `fom_test_xs: np.ndarray` with shape `(100, 179)`;
  - `lrom_results[(n_phi, k)]`, `rose_results[(n_phi, n_u)]`, and `ls_results[n_phi]`;
  - each result contains `train_xs`, `test_xs`, `train_median_error`, `test_median_error`, `train_cat_error`, `test_cat_error`, and per-sample `test_seconds`.

- [ ] **Step 1: Extend the contract with scientific-core assertions**

Append:

```python
def test_notebook02_scientific_core_contract() -> None:
    text = notebook_text(NOTEBOOK_02)
    assert 'potential="full_woods-saxon"' in text
    assert "l=tuple(range(L_MAX + 1))" in text
    assert 'strategy="latin_hypercube"' in text
    assert "seed=SEED" in text
    assert "np.array_equal(train_rows, rose_train_rows)" in text
    assert "np.array_equal(test_rows, rose_test_rows)" in text
    assert "rose.InteractionEIMSpace(" in text
    assert "training_info=rose_train_rows" in text
    assert "explicit_training=True" in text
    assert "rose.ScatteringAmplitudeEmulator.from_train(" in text
    assert "lrom.project_coordinates(" in text
    assert "lrom._cross_section_prediction(" in text
    assert "LS-projected cross section" in text
    assert "linear LROM" not in text
    assert "np.max(pointwise_relative_error" in text
```

- [ ] **Step 2: Run the new test and observe the expected failure**

Run:

```bash
python -m pytest tests/test_notebook02_cross_sections.py::test_notebook02_scientific_core_contract -v
```

Expected: FAIL because the scientific cells are absent.

- [ ] **Step 3: Add shared sampling immediately after Section 2**

Add this code cell:

```python
emulator = lrom.LROM(
    target=TARGET,
    projectile=PROJECTILE,
    lab_energy=LAB_ENERGY,
    l=tuple(range(L_MAX + 1)),
    potential="full_woods-saxon",
)
central = dict(emulator.central_parameters)
ranges = {
    name: tuple(sorted(((1.0 - HALF_WIDTH) * value, (1.0 + HALF_WIDTH) * value)))
    for name, value in central.items()
}
emulator.sampling(
    training_ranges=ranges,
    testing_ranges=ranges,
    training_size=N_TRAIN,
    testing_size=N_TEST,
    mesh_size=MESH_SIZE,
    strategy="latin_hypercube",
    seed=SEED,
    high_fidelity_solver="runge_kutta",
    solver_options={"rk_tols": (1e-9, 1e-9)},
)
parameter_names = emulator.parameter_names
train_rows = emulator.samples.design.training.values.copy()
test_rows = emulator.samples.design.testing.values.copy()
train_ids = emulator.samples.design.training.case_ids
test_ids = emulator.samples.design.testing.case_ids

assert train_rows.shape == (N_TRAIN, len(parameter_names))
assert test_rows.shape == (N_TEST, len(parameter_names))
assert not any(np.array_equal(a, b) for a in train_rows for b in test_rows)
for column, name in enumerate(parameter_names):
    lower, upper = ranges[name]
    assert np.all((lower <= train_rows[:, column]) & (train_rows[:, column] <= upper))
    assert np.all((lower <= test_rows[:, column]) & (test_rows[:, column] <= upper))

rose_train_rows = train_rows
rose_test_rows = test_rows
assert np.array_equal(train_rows, rose_train_rows)
assert np.array_equal(test_rows, rose_test_rows)
```

- [ ] **Step 4: Add flat metric and timing helpers after the sampling cell**

```python
def rows_as_parameter_dicts(rows):
    return [
        {name: float(value) for name, value in zip(parameter_names, row)}
        for row in rows
    ]


def pointwise_relative_error(predicted, reference):
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    denominator = np.maximum(np.abs(reference), ERROR_DENOMINATOR_FLOOR)
    return np.abs(predicted - reference) / denominator


def error_summaries(predicted, reference):
    pointwise = pointwise_relative_error(predicted, reference)
    return {
        "median_error": np.median(pointwise, axis=1),
        "cat_error": np.max(pointwise_relative_error(predicted, reference), axis=1),
    }


def minimum_lrom_times(model, cases):
    model.predict(parameters=cases[:1])
    seconds = []
    for case in cases:
        repeats = []
        for _ in range(TIMING_REPEATS):
            start = time.perf_counter()
            model.predict(parameters=case)
            repeats.append(time.perf_counter() - start)
        seconds.append(min(repeats))
    return np.asarray(seconds)


def minimum_rose_times(model, rows):
    model.emulate_dsdo(rows[0])
    seconds = []
    for row in rows:
        repeats = []
        for _ in range(TIMING_REPEATS):
            start = time.perf_counter()
            model.emulate_dsdo(row)
            repeats.append(time.perf_counter() - start)
        seconds.append(min(repeats))
    return np.asarray(seconds)
```

- [ ] **Step 5: Add the notebook-owned ROSE interaction and grid after Section 4**

```python
@njit
def bench_ws(r, radius, diffuseness):
    return 1.0 / (1.0 + np.exp((r - radius) / diffuseness))


@njit
def bench_ws_prime(r, radius, diffuseness):
    ex = np.exp((r - radius) / diffuseness)
    return -(ex / diffuseness) / (1.0 + ex) ** 2


@njit
def bench_full_ws(r, alpha):
    vv, wv, wd, _vso, rv, rd, _rso, av, ad, _aso = alpha
    return (
        -vv * bench_ws(r, rv, av)
        - 1j * wv * bench_ws(r, rv, av)
        + 4j * ad * wd * bench_ws_prime(r, rd, ad)
    )


@njit
def bench_full_ws_so(r, alpha, ldots):
    _vv, _wv, _wd, vso, _rv, _rd, rso, _av, _ad, aso = alpha
    return vso / 139.57039**2 * ldots * bench_ws_prime(r, rso, aso) / r


rho_mesh = emulator.samples.mesh.rho
radius_mesh = emulator.samples.mesh.radius
rose_solver = rose.SchroedingerEquation.make_base_solver(
    s_0=6 * np.pi,
    rk_tols=[1e-9, 1e-9],
    domain=np.array([rho_mesh[0], rho_mesh[-1]]),
)
rose_emulators = {}
for n_phi in BASIS_SIZES:
    for n_u in ROSE_EIM_SIZES:
        interaction = rose.InteractionEIMSpace(
            l_max=L_MAX,
            coordinate_space_potential=bench_full_ws,
            spin_orbit_term=bench_full_ws_so,
            n_theta=len(parameter_names),
            mu=emulator.kinematics.mu,
            energy=emulator.kinematics.e_com,
            is_complex=True,
            training_info=rose_train_rows,
            explicit_training=True,
            n_basis=n_u,
            rho_mesh=rho_mesh,
        )
        rose_emulators[(n_phi, n_u)] = rose.ScatteringAmplitudeEmulator.from_train(
            interaction,
            rose_train_rows,
            base_solver=rose_solver,
            l_max=L_MAX,
            angles=ANGLES_RAD,
            n_basis=n_phi,
            use_svd=True,
            scale=False,
            s_mesh=rho_mesh,
            Smatrix_abs_tol=1e-8,
        )

reference_sae = rose_emulators[(BASIS_SIZES[0], ROSE_EIM_SIZES[0])]
fom_train_xs = np.asarray([reference_sae.exact_dsdo(row) for row in train_rows])
fom_test_xs = np.asarray([reference_sae.exact_dsdo(row) for row in test_rows])
assert fom_train_xs.shape == (N_TRAIN, ANGLES_DEG.size)
assert fom_test_xs.shape == (N_TEST, ANGLES_DEG.size)
assert np.all(np.isfinite(fom_train_xs)) and np.all(fom_train_xs >= -1e-12)
assert np.all(np.isfinite(fom_test_xs)) and np.all(fom_test_xs >= -1e-12)
```

- [ ] **Step 6: Add the LROM/LS Cartesian evaluation after the ROSE cell**

```python
train_cases = rows_as_parameter_dicts(train_rows)
test_cases = rows_as_parameter_dicts(test_rows)
lrom_results = {}
ls_results = {}
default_predictor = None

for n_phi in BASIS_SIZES:
    for predictor_count in LROM_PREDICTOR_COUNTS:
        emulator.train(
            basis_size=n_phi,
            predictor="potential",
            predictor_count=predictor_count,
            observable="cross_section",
            angles_degrees=ANGLES_DEG,
        )
        test_seconds = minimum_lrom_times(emulator, test_cases)
        emulator.predict(parameters=train_cases)
        train_xs = emulator.predictions.cross_sections.values.copy()
        emulator.predict(parameters=test_cases)
        test_xs = emulator.predictions.cross_sections.values.copy()
        train_summary = error_summaries(train_xs, fom_train_xs)
        test_summary = error_summaries(test_xs, fom_test_xs)
        lrom_results[(n_phi, predictor_count)] = {
            "train_xs": train_xs,
            "test_xs": test_xs,
            "train_median_error": train_summary["median_error"],
            "test_median_error": test_summary["median_error"],
            "train_cat_error": train_summary["cat_error"],
            "test_cat_error": test_summary["cat_error"],
            "test_seconds": test_seconds,
        }

        if (n_phi, predictor_count) == (
            DEFAULT_BASIS_SIZE,
            DEFAULT_COMPRESSION_SIZE,
        ):
            default_predictor = emulator.predictors

        if predictor_count == LROM_PREDICTOR_COUNTS[0]:
            ls_train_coordinates = {
                channel: lrom.project_coordinates(
                    basis=emulator.basis[channel],
                    wavefunctions=emulator.samples.training_wavefunctions[channel],
                )
                for channel in emulator.basis
            }
            ls_test_coordinates = {
                channel: lrom.project_coordinates(
                    basis=emulator.basis[channel],
                    wavefunctions=emulator.samples.testing_wavefunctions[channel],
                )
                for channel in emulator.basis
            }
            _, ls_train_state = lrom._cross_section_prediction(
                emulator=emulator,
                values=train_rows,
                coefficients=ls_train_coordinates,
            )
            _, ls_test_state = lrom._cross_section_prediction(
                emulator=emulator,
                values=test_rows,
                coefficients=ls_test_coordinates,
            )
            ls_train_summary = error_summaries(ls_train_state.values, fom_train_xs)
            ls_test_summary = error_summaries(ls_test_state.values, fom_test_xs)
            ls_results[n_phi] = {
                "train_xs": ls_train_state.values.copy(),
                "test_xs": ls_test_state.values.copy(),
                "train_median_error": ls_train_summary["median_error"],
                "test_median_error": ls_test_summary["median_error"],
                "train_cat_error": ls_train_summary["cat_error"],
                "test_cat_error": ls_test_summary["cat_error"],
            }

assert default_predictor is not None
```

- [ ] **Step 7: Add the ROSE grid evaluation**

```python
rose_results = {}
for config, rose_emulator in rose_emulators.items():
    test_seconds = minimum_rose_times(rose_emulator, test_rows)
    train_xs = np.asarray([rose_emulator.emulate_dsdo(row) for row in train_rows])
    test_xs = np.asarray([rose_emulator.emulate_dsdo(row) for row in test_rows])
    train_summary = error_summaries(train_xs, fom_train_xs)
    test_summary = error_summaries(test_xs, fom_test_xs)
    rose_results[config] = {
        "train_xs": train_xs,
        "test_xs": test_xs,
        "train_median_error": train_summary["median_error"],
        "test_median_error": test_summary["median_error"],
        "train_cat_error": train_summary["cat_error"],
        "test_cat_error": test_summary["cat_error"],
        "test_seconds": test_seconds,
    }

for collection in (lrom_results, rose_results, ls_results):
    for result in collection.values():
        for values in result.values():
            assert np.all(np.isfinite(values))

assert set(lrom_results) == {
    (n_phi, predictor_count)
    for n_phi in BASIS_SIZES
    for predictor_count in LROM_PREDICTOR_COUNTS
}
assert set(rose_results) == {
    (n_phi, n_u)
    for n_phi in BASIS_SIZES
    for n_u in ROSE_EIM_SIZES
}
```

- [ ] **Step 8: Run core contracts**

Run:

```bash
python -m pytest tests/test_notebook02_cross_sections.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 9: Commit the scientific core**

```bash
git add tests/test_notebook02_cross_sections.py notebooks/02_rose_vs_lrom_cross_sections.ipynb
git commit -m "feat: add Notebook 02 cross-section core"
```

---

### Task 3: Add paper-map figures, aligned tables, and inline validation

**Files:**
- Modify: `tests/test_notebook02_cross_sections.py`
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`

**Interfaces:**
- Consumes: result dictionaries from Task 2 and physical `radius_mesh`.
- Produces: figure markers `potential-predictor-rainbows`, `representative-cross-sections`, `cross-section-errors`, `error-violins`, and `cat-plot`; `summary_table: pandas.DataFrame`.

- [ ] **Step 1: Add failing figure and validation contracts**

Append:

```python
def test_notebook02_results_contract() -> None:
    text = notebook_text(NOTEBOOK_02)
    for marker in (
        "potential-predictor-rainbows",
        "representative-cross-sections",
        "cross-section-errors",
        "error-violins",
        "cat-plot",
        "validation-summary",
    ):
        assert f"# FIGURE: {marker}" in text or f"# TABLE: {marker}" in text
    assert "selected_radii" in text
    assert "radius_mesh" in text
    assert "combined_rank" in text
    assert "25th percentile" in text
    assert "50th percentile" in text
    assert "75th percentile" in text
    assert "one million evaluations/hour" in text
    assert "10% maximum relative error" in text
    assert "def plot_" not in text
```

Run:

```bash
python -m pytest tests/test_notebook02_cross_sections.py::test_notebook02_results_contract -v
```

Expected: FAIL because figure cells are absent.

- [ ] **Step 2: Add the physical potential/predictor figure after Section 3**

Use the central row plus 12 test rows at equally spaced standardized-distance ranks:

```python
# FIGURE: potential-predictor-rainbows
center_vector = np.asarray([central[name] for name in parameter_names])
range_scale = np.asarray([ranges[name][1] - ranges[name][0] for name in parameter_names])
standardized_distance = np.linalg.norm(
    (test_rows - center_vector[np.newaxis, :]) / range_scale[np.newaxis, :],
    axis=1,
)
ranked = np.argsort(standardized_distance)
rainbow_indices = ranked[
    np.linspace(0, len(ranked) - 1, 12, dtype=int)
]
rainbow_rows = np.vstack([center_vector, test_rows[rainbow_indices]])

fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
for row in rainbow_rows:
    central_part = lrom.full_woods_saxon(radius_mesh, row)
    spin_orbit_part = lrom.full_woods_saxon_spin_orbit(radius_mesh, row)
    axes[0, 0].plot(radius_mesh, central_part.real, alpha=0.55)
    axes[0, 1].plot(radius_mesh, central_part.imag, alpha=0.55)
    axes[1, 0].plot(radius_mesh, central_part.real, alpha=0.55)
    axes[1, 1].plot(
        radius_mesh,
        (central_part + 1.5 * spin_orbit_part).real,
        alpha=0.55,
    )

component_colors = {0: "tab:orange", 1: "tab:purple"}
for radius, component in zip(
    default_predictor.selected_radii,
    default_predictor.selected_components,
):
    color = component_colors[int(component)]
    for ax in axes.flat:
        ax.axvline(radius, color=color, alpha=0.7, lw=1.0)

axes[0, 0].set_title("l=0 central: real")
axes[0, 1].set_title("l=0 central: imaginary")
axes[1, 0].set_title("l=3 central: real")
axes[1, 1].set_title("l=3, j=l+1/2: real")
for ax in axes[1]:
    ax.set_xlabel("r [fm]")
for ax in axes[:, 0]:
    ax.set_ylabel("potential [MeV]")
fig.suptitle("Full Woods-Saxon variation and LROM predictor radii")
fig.tight_layout()
plt.show()

assert np.all(default_predictor.selected_radii >= radius_mesh.min())
assert np.all(default_predictor.selected_radii <= radius_mesh.max())
```

- [ ] **Step 3: Add aligned representative selection and cross sections after Section 5**

```python
default_key = (DEFAULT_BASIS_SIZE, DEFAULT_COMPRESSION_SIZE)
lrom_rank = np.argsort(np.argsort(lrom_results[default_key]["test_cat_error"]))
rose_rank = np.argsort(np.argsort(rose_results[default_key]["test_cat_error"]))
combined_rank = 0.5 * (lrom_rank + rose_rank)
ordered_combined = np.argsort(combined_rank)
quantile_positions = (0.25, 0.50, 0.75)
selected_indices = [
    int(ordered_combined[round(q * (len(ordered_combined) - 1))])
    for q in quantile_positions
]
assert len(set(selected_indices)) == 3
selected_labels = ("25th percentile", "50th percentile", "75th percentile")

# FIGURE: representative-cross-sections
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), sharey=True)
for ax, index, label in zip(axes, selected_indices, selected_labels):
    ax.semilogy(ANGLES_DEG, fom_test_xs[index], color="black", label="FOM")
    ax.semilogy(
        ANGLES_DEG,
        ls_results[DEFAULT_BASIS_SIZE]["test_xs"][index],
        color="tab:blue",
        label="LS-projected cross section",
    )
    ax.semilogy(
        ANGLES_DEG,
        rose_results[default_key]["test_xs"][index],
        "--",
        color="tab:red",
        label="ROSE",
    )
    ax.semilogy(
        ANGLES_DEG,
        lrom_results[default_key]["test_xs"][index],
        ":",
        color="#E6AB02",
        lw=2.2,
        label="LROM",
    )
    ax.set_title(f"{label}: {test_ids[index]}")
    ax.set_xlabel("angle [deg]")
axes[0].set_ylabel(r"d$\sigma$/d$\Omega$ [mb/sr]")
axes[0].legend()
fig.tight_layout()
plt.show()
```

- [ ] **Step 4: Add aligned pointwise errors**

```python
# FIGURE: cross-section-errors
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0), sharey=True)
for ax, index, label in zip(axes, selected_indices, selected_labels):
    for values, method, color, style in (
        (
            ls_results[DEFAULT_BASIS_SIZE]["test_xs"],
            "LS-projected cross section",
            "tab:blue",
            "-",
        ),
        (rose_results[default_key]["test_xs"], "ROSE", "tab:red", "--"),
        (lrom_results[default_key]["test_xs"], "LROM", "#E6AB02", ":"),
    ):
        errors = pointwise_relative_error(values[index], fom_test_xs[index])
        ax.semilogy(
            ANGLES_DEG,
            np.maximum(errors, PLOTTING_FLOOR),
            style,
            color=color,
            label=method,
        )
    ax.set_title(f"{label}: {test_ids[index]}")
    ax.set_xlabel("angle [deg]")
    ax.set_ylim(bottom=PLOTTING_FLOOR)
axes[0].set_ylabel("relative cross-section error")
axes[0].legend()
fig.tight_layout()
plt.show()
```

- [ ] **Step 5: Add the default split violins after Section 6**

```python
# FIGURE: error-violins
categories = (
    ("LS-projected", ls_results[DEFAULT_BASIS_SIZE]),
    ("ROSE", rose_results[default_key]),
    ("LROM", lrom_results[default_key]),
)
positions = np.arange(1, len(categories) + 1)
fig, ax = plt.subplots(figsize=(8.0, 4.8))
for center_pos, (label, result) in zip(positions, categories):
    for side, key, color, alpha in (
        (-1, "train_median_error", "tab:blue", 0.75),
        (+1, "test_median_error", "tab:orange", 0.55),
    ):
        values = np.log10(np.clip(result[key], PLOTTING_FLOOR, None))
        parts = ax.violinplot(
            [values],
            positions=[center_pos],
            widths=0.85,
            showextrema=False,
        )
        body = parts["bodies"][0]
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = (
            np.minimum(vertices[:, 0], center_pos)
            if side < 0
            else np.maximum(vertices[:, 0], center_pos)
        )
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(alpha)
        ax.scatter(
            center_pos + 0.07 * side,
            np.median(values),
            marker="D",
            color=color,
            edgecolor="black",
            zorder=6,
        )
ax.set_xticks(positions, [label for label, _ in categories])
ax.set_ylabel("log10 median pointwise relative error")
ax.legend(
    handles=[
        Patch(facecolor="tab:blue", alpha=0.75, edgecolor="black", label="train"),
        Patch(facecolor="tab:orange", alpha=0.55, edgecolor="black", label="test"),
        Line2D([], [], marker="D", linestyle="None", color="0.3", label="median"),
    ]
)
fig.tight_layout()
plt.show()
```

- [ ] **Step 6: Add Cartesian summary tables after Section 7**

```python
summary_rows = []
for (n_phi, compression), result in rose_results.items():
    summary_rows.append(
        {
            "method": "ROSE",
            "n_phi": n_phi,
            "compression": compression,
            "median test error": float(np.median(result["test_median_error"])),
            "median CAT error": float(np.median(result["test_cat_error"])),
            "median online time [s]": float(np.median(result["test_seconds"])),
        }
    )
for (n_phi, compression), result in lrom_results.items():
    summary_rows.append(
        {
            "method": "LROM",
            "n_phi": n_phi,
            "compression": compression,
            "median test error": float(np.median(result["test_median_error"])),
            "median CAT error": float(np.median(result["test_cat_error"])),
            "median online time [s]": float(np.median(result["test_seconds"])),
        }
    )
summary_table = pd.DataFrame(summary_rows).sort_values(
    ["method", "n_phi", "compression"]
)
summary_table
```

- [ ] **Step 7: Add the per-sample CAT cloud after Section 8**

```python
# FIGURE: cat-plot
fig, ax = plt.subplots(figsize=(9.0, 5.4))
colors = {4: "tab:blue", 6: "tab:green", 8: "tab:red"}
for (n_phi, n_u), result in rose_results.items():
    ax.scatter(
        result["test_seconds"],
        result["test_cat_error"],
        marker="s",
        s=20,
        alpha=0.35,
        color=colors[n_phi],
        label=f"ROSE n_phi={n_phi}, n_U={n_u}",
    )
for (n_phi, predictor_count), result in lrom_results.items():
    ax.scatter(
        result["test_seconds"],
        result["test_cat_error"],
        marker="o",
        s=20,
        alpha=0.35,
        facecolors="none",
        edgecolors=colors[n_phi],
        label=f"LROM n_phi={n_phi}, K={predictor_count}",
    )
ax.axhline(0.10, color="0.25", linestyle="--", label="10% maximum relative error")
ax.axvline(1.0 / (1_000_000 / 3600), color="0.45", linestyle=":")
ax.text(
    1.0 / (1_000_000 / 3600),
    0.12,
    "one million evaluations/hour",
    rotation=90,
    va="bottom",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("online time per sample [s]")
ax.set_ylabel("maximum relative error over angle")
ax.set_title("Computational Accuracy versus Time")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
plt.show()
```

- [ ] **Step 8: Add the validation summary after Section 9**

```python
# TABLE: validation-summary
validation_summary = pd.DataFrame(
    [
        {"check": "training rows", "value": train_rows.shape},
        {"check": "testing rows", "value": test_rows.shape},
        {"check": "partial waves", "value": tuple(range(L_MAX + 1))},
        {"check": "parameter half-width", "value": HALF_WIDTH},
        {"check": "basis sizes", "value": BASIS_SIZES},
        {"check": "ROSE EIM sizes", "value": ROSE_EIM_SIZES},
        {"check": "LROM predictor counts", "value": LROM_PREDICTOR_COUNTS},
        {"check": "Python", "value": sys.version.split()[0]},
        {"check": "platform", "value": platform.platform()},
    ]
)
validation_summary
```

- [ ] **Step 9: Run contracts and commit**

Run:

```bash
python -m pytest tests/test_notebook02_cross_sections.py -v
```

Expected: 4 tests PASS.

Commit:

```bash
git add tests/test_notebook02_cross_sections.py notebooks/02_rose_vs_lrom_cross_sections.ipynb
git commit -m "feat: add Notebook 02 result map figures"
```

---

### Task 4: Adapt benchmark 03 to the reduced/full profiles

**Files:**
- Modify: `tests/test_notebook02_cross_sections.py`
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`

**Interfaces:**
- Consumes: Notebook 02 constants, metric definitions, model loops, and figure markers.
- Produces: `PROFILE: str`, `N_TRAIN: int`, and `N_TEST: int` selected from `LROM_BENCHMARK_PROFILE`; reduced mode `(120, 30)` and full mode `(200, 100)`.

- [ ] **Step 1: Add failing benchmark contracts**

Append:

```python
def test_benchmark03_notebook02_profile_contract() -> None:
    assert BENCHMARK_03.exists()
    text = notebook_text(BENCHMARK_03)
    assert 'os.environ.get("LROM_BENCHMARK_PROFILE", "reduced")' in text
    assert '"reduced": (120, 30)' in text
    assert '"full": (200, 100)' in text
    assert "HALF_WIDTH = 0.20" in text
    assert "L_MAX = 3" in text
    assert "BASIS_SIZES = (4, 6, 8)" in text
    assert "ROSE_EIM_SIZES = (4, 8, 12)" in text
    assert "LROM_PREDICTOR_COUNTS = (4, 8, 12)" in text
    assert "training_info=rose_train_rows" in text
    assert "explicit_training=True" in text
    assert "np.max(pointwise_relative_error" in text
    assert "test_seconds" in text
    assert "LS-projected cross section" in text
    assert "linear LROM" not in text


def test_benchmark03_code_cells_compile() -> None:
    for index, source in enumerate(code_sources(BENCHMARK_03)):
        compile(source, f"{BENCHMARK_03.name} code cell {index}", "exec")
```

Run:

```bash
python -m pytest tests/test_notebook02_cross_sections.py::test_benchmark03_notebook02_profile_contract -v
```

Expected: FAIL because benchmark 03 still uses the legacy 30/20, ±10%/±15%, and \(l_{\max}=6\) profile.

- [ ] **Step 2: Mechanically derive benchmark 03 from the authoritative notebook cells**

Create `/private/tmp/adapt_notebook02_benchmark.py` with the exact one-time
transformation below, run it, and do not add it to git:

```python
import copy
import json
from pathlib import Path


root = Path("/Users/Kitkat/Documents/Documents-Agent/LROM_Project/lrom_git")
source_path = root / "notebooks" / "02_rose_vs_lrom_cross_sections.ipynb"
target_path = (
    root / "notebooks" / "benchmark_notebooks" / "2.0" / "benchmark_03.ipynb"
)
notebook = json.loads(source_path.read_text())
benchmark = copy.deepcopy(notebook)
benchmark["cells"][0]["source"] = [
    "# benchmark_03. Notebook 02 Cross-Section Reproducibility\n",
    "\n",
    "This benchmark runs the Notebook 02 ROSE/LROM cross-section experiment ",
    "with a determined reduced profile by default. Set ",
    "`LROM_BENCHMARK_PROFILE=full` to reproduce the complete 200/100 ",
    "notebook profile.\n",
]

for cell in benchmark["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if "from pathlib import Path" in source:
        source = source.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\nimport os\n",
            1,
        )
        cell["source"] = source.splitlines(keepends=True)
    if source.startswith("TARGET = (40, 20)"):
        replacement = '''TARGET = (40, 20)
PROJECTILE = (1, 0)
LAB_ENERGY = 14.1
L_MAX = 3
MESH_SIZE = 600
HALF_WIDTH = 0.20
ANGLES_DEG = np.arange(1.0, 180.0, 1.0)
ANGLES_RAD = np.deg2rad(ANGLES_DEG)
BASIS_SIZES = (4, 6, 8)
ROSE_EIM_SIZES = (4, 8, 12)
LROM_PREDICTOR_COUNTS = (4, 8, 12)
DEFAULT_BASIS_SIZE = 6
DEFAULT_COMPRESSION_SIZE = 8
TIMING_REPEATS = 3
SEED = 1204
ERROR_DENOMINATOR_FLOOR = 1e-12
PLOTTING_FLOOR = 1e-6

PROFILES = {
    "reduced": (120, 30),
    "full": (200, 100),
}
PROFILE = os.environ.get("LROM_BENCHMARK_PROFILE", "reduced")
if PROFILE not in PROFILES:
    raise ValueError("LROM_BENCHMARK_PROFILE must be 'reduced' or 'full'")
N_TRAIN, N_TEST = PROFILES[PROFILE]
print(f"benchmark profile: {PROFILE} ({N_TRAIN} train, {N_TEST} test)")
'''
        cell["source"] = replacement.splitlines(keepends=True)

target_path.write_text(json.dumps(benchmark, indent=1) + "\n")
```

Run:

```bash
python /private/tmp/adapt_notebook02_benchmark.py
```

Expected: benchmark 03 contains the complete Notebook 02 scientific and
figure cells, the benchmark title, and the reduced/full profile switch. It
does not import Notebook 02 or a plotting helper.

- [ ] **Step 3: Run benchmark contracts**

Run:

```bash
python -m pytest tests/test_notebook02_cross_sections.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 4: Commit benchmark adaptation**

```bash
git add tests/test_notebook02_cross_sections.py notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
git commit -m "feat: align benchmark 03 with Notebook 02"
```

---

### Task 5: Execute reduced validation and diagnose before the full run

**Files:**
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb` only if reduced execution reveals a notebook defect.
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb` only if the same defect applies to the full artifact.

**Interfaces:**
- Consumes: completed notebooks and focused contracts.
- Produces: an executed reduced benchmark with embedded outputs and zero error outputs.

- [ ] **Step 1: Record protected-file hashes**

Run:

```bash
shasum -a 256 \
  lrom/__init__.py \
  lrom_legacy/v1_2/__init__.py \
  lrom_legacy/v2_0/__init__.py \
  notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb \
  > /private/tmp/notebook02-protected-before.sha256
```

Expected: four hashes written.

- [ ] **Step 2: Execute benchmark 03 in reduced mode**

Run:

```bash
LROM_BENCHMARK_PROFILE=reduced python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
```

Expected: exit code 0 and all cells executed.

- [ ] **Step 3: Verify notebook outputs contain no errors**

Run:

```bash
python -c 'import json, pathlib; p=pathlib.Path("notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb"); n=json.loads(p.read_text()); errors=[o for c in n["cells"] if c["cell_type"]=="code" for o in c.get("outputs",[]) if o.get("output_type")=="error"]; executed=[c for c in n["cells"] if c["cell_type"]=="code" and c.get("execution_count") is not None]; print(f"executed={len(executed)} errors={len(errors)}"); assert not errors'
```

Expected: `errors=0`.

- [ ] **Step 4: Run focused tests**

```bash
python -m pytest tests/test_notebook02_cross_sections.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Diagnose any reduced-profile failure without restructuring package code**

If Steps 2–4 fail, use `superpowers:systematic-debugging`. Determine whether the cause is notebook alignment, unsupported configuration, numerical nonfiniteness, or an actual v2.0 defect. Fix notebook-local issues test-first. Stop and request approval before modifying any protected package file.

- [ ] **Step 6: Commit reduced executed evidence**

```bash
git add notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb tests/test_notebook02_cross_sections.py
git commit -m "test: execute reduced Notebook 02 benchmark"
```

---

### Task 6: Execute the full notebook and full benchmark, inspect figures, and close the milestone

**Files:**
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
- Modify: `/Users/Kitkat/Documents/Documents-Agent/LROM_Project/docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify: `/Users/Kitkat/Documents/Documents-Agent/_memory/Daily Notes/2026-07-24.md`
- Modify: `/Users/Kitkat/Documents/Documents-Agent/_memory/Context/active-state.md`

**Interfaces:**
- Consumes: validated reduced profile.
- Produces: executed full Notebook 02, executed full benchmark 03, figure QA evidence, unchanged protected hashes, updated methodology documentation, and a clean local commit.

- [ ] **Step 1: Execute Notebook 02 at 200/100**

Run:

```bash
python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=14400 \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb
```

Expected: exit code 0.

- [ ] **Step 2: Execute benchmark 03 once in full mode**

Run:

```bash
LROM_BENCHMARK_PROFILE=full python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=14400 \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
```

Expected: exit code 0.

- [ ] **Step 3: Check both executed notebooks for errors**

Run:

```bash
python -c 'import json, pathlib; paths=[pathlib.Path("notebooks/02_rose_vs_lrom_cross_sections.ipynb"),pathlib.Path("notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb")]; errors={str(p):[o for c in json.loads(p.read_text())["cells"] for o in c.get("outputs",[]) if o.get("output_type")=="error"] for p in paths}; print({k:len(v) for k,v in errors.items()}); assert all(not value for value in errors.values())'
```

Expected: each notebook reports `errors=0`.

- [ ] **Step 4: Inspect all figures**

Extract each embedded `image/png` output to a temporary directory and inspect every image with the available image viewer. Confirm:

- predictor markers lie on the physical \(r\) [fm] axis;
- \(l=0\) and \(l=3\) titles are correct;
- FOM, LS-projected, ROSE, and LROM curves use consistent labels;
- the same test ID appears across each representative and error panel;
- split violins are not clipped;
- CAT axes are logarithmic and reference lines are labeled;
- legends remain readable with 18 configuration clouds;
- no output overlaps, black squares, or unreadable text.

Expected: zero visual defects. Fix and re-execute affected cells when a defect is found.

- [ ] **Step 5: Restore benchmark 03 to its default reduced embedded outputs**

After recording the full-profile validation values for the handoff, run:

```bash
LROM_BENCHMARK_PROFILE=reduced python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
```

Expected: exit code 0. The committed benchmark opens with outputs matching
its default `reduced` profile; Notebook 02 retains the full 200/100 outputs.

- [ ] **Step 6: Run focused and available full tests**

Run:

```bash
python -m pytest tests/test_notebook02_cross_sections.py -v
```

Expected: 6 tests PASS.

If `/Users/Kitkat/Documents/Documents-Agent/LROM_Project/tests` remains the available broader suite, run:

```bash
python -m pytest /Users/Kitkat/Documents/Documents-Agent/LROM_Project/tests -q
```

Expected: all available tests PASS.

- [ ] **Step 7: Verify protected files are unchanged**

Run:

```bash
shasum -a 256 -c /private/tmp/notebook02-protected-before.sha256
```

Expected: all four paths report `OK`.

- [ ] **Step 8: Update the architecture understanding**

Append a dated Notebook 02 section to
`/Users/Kitkat/Documents/Documents-Agent/LROM_Project/docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
covering these exact facts:

- v2.0 remains explicitly imported and parked;
- exact parameter rows are shared between ROSE and LROM;
- ROSE uses a free-reference basis plus notebook-owned EIM;
- LROM uses central-reference bases plus maxvol potential predictors;
- equal-basis means equal \(n_\phi\), while \(n_U\) and \(K\) are not identical;
- LS projection is a wavefunction oracle whose propagated cross section is not an observable floor;
- CAT uses maximum-over-angle error and per-sample online timing;
- no package code changed.

- [ ] **Step 9: Update memory handoff files**

Append a `[LROM_Project]` entry to the daily note and update active state with:

- modified files;
- executed cell/error counts;
- reduced and full profile results;
- selected representative test IDs;
- test totals;
- figure inspection outcome;
- protected hash outcome;
- current local commit and whether push remains unauthorized.

- [ ] **Step 10: Commit the completed notebook milestone**

From `lrom_git`, commit only repository files:

```bash
git add \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb \
  tests/test_notebook02_cross_sections.py
git commit -m "feat: complete Notebook 02 cross-section comparison"
```

Do not include parent documentation or memory files in this repository commit.

- [ ] **Step 11: Run final repository checks**

```bash
git status --short --branch
git log -5 --oneline
```

Expected: no uncommitted repository changes and the Notebook 02 completion commit at `HEAD`. Do not push until the user explicitly authorizes egress to the configured GitHub remote.
