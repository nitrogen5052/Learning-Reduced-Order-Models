# Notebook 02 Flat Paper-CAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite Notebook 02 as direct, spaced cell code and replace its median configuration CAT markers with paper-faithful per-alpha clusters using solid LROM circles.

**Architecture:** Keep ROSE and LS inside the existing notebook-owned `benchmark_helper.CrossSectionBenchmark`, while leaving parked-v2 LROM training, timing, and prediction directly visible in Notebook 02. Remove every notebook-local function and annotation-only import; preserve the existing scientific arrays and figures except for the approved CAT aggregation change.

**Tech Stack:** Jupyter Notebook JSON, Python 3.12, NumPy, pandas, Matplotlib, `benchmark_helper`, `lrom_legacy.v2_0`, pytest, Ruff.

## Global Constraints

- Work directly in `LROM_Project/lrom_git`.
- Modify only Notebook 02, its focused contract tests, architecture documentation, and memory handoff files.
- Do not modify `lrom/`, `lrom_legacy/`, `notebooks/benchmark_helper.py`, Notebook 01, Benchmark 01, Benchmark 02, Benchmark 03, or `scientific_archive/`.
- Keep the existing ten Markdown headings exactly; add no Markdown prose.
- Notebook 02 code cells must contain no function definitions and no comment tokens.
- Use blank lines and smaller sequential cells to expose the Paper Results Map flow.
- Retain ROSE and LS as separate `run_rose()` and `run_ls()` calls.
- Retain \(l=0,\ldots,3\), plus/minus 20% ranges, 200/100 rows, seed 1204, the existing basis/compression grid, and physical radius in fm.
- Plot one CAT point per held-out alpha per configuration.
- Use solid squares for ROSE and solid circles for LROM.
- CAT y-values are per-alpha `test_maximum_over_angle_error`, matching Figure 5 of the ROSE paper.
- Preserve all non-CAT scientific values, alpha selections, and four other figures.
- Do not stage, edit, remove, or inspect the contents of the existing untracked `tmp/`.

---

### Task 1: Replace function-oriented contracts with flat-notebook and paper-CAT contracts

**Files:**
- Modify: `tests/test_notebook02_cross_sections.py`
- Test: `tests/test_notebook02_cross_sections.py`

**Interfaces:**
- Consumes: `NOTEBOOK_02`, `load_notebook()`, `notebook_text()`, and `code_sources()`.
- Produces: source contracts that require zero notebook-local functions, minimal imports, comment-free code cells, direct benchmark calls, and per-alpha CAT arrays.

- [ ] **Step 1: Add token support and replace the clean-shell function expectations**

Add standard-library imports:

```python
import io
import tokenize
```

Replace `test_notebook02_clean_shell_contract()` with:

```python
def test_notebook02_flat_cell_contract() -> None:
    sources = code_sources(NOTEBOOK_02)

    for source in sources[1:]:
        tree = ast.parse(source)
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(tree)
        )

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for source in sources
        for node in ast.walk(ast.parse(source))
    )

    for source in sources:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        assert not any(token.type == tokenize.COMMENT for token in tokens)

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

    imports = sources[0]
    for removed in (
        "from typing import Any",
        "import platform",
        "from matplotlib.figure import Figure",
        "from matplotlib.lines import Line2D",
        "from matplotlib.patches import Patch",
        "import scipy.special",
    ):
        assert removed not in imports

    for retained in (
        "from pathlib import Path",
        "import sys",
        "import time",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "import benchmark_helper",
        "import lrom_legacy.v2_0 as lrom",
    ):
        assert retained in imports
```

Delete `test_notebook02_presentation_functions_are_extracted()`. Its function
requirements directly contradict the approved flat-cell design.

- [ ] **Step 2: Add a CAT cell helper in the test file**

Add:

```python
def cat_source(path: Path) -> str:
    matches = [
        source
        for source in code_sources(path)
        if "Computational Accuracy versus Time" in source
    ]
    assert len(matches) == 1
    return matches[0]
```

- [ ] **Step 3: Add the paper-faithful CAT contract**

Add:

```python
def test_notebook02_cat_uses_per_alpha_paper_encoding() -> None:
    source = cat_source(NOTEBOOK_02)

    assert 'result["test_seconds"]' in source
    assert 'result["test_maximum_over_angle_error"]' in source
    assert 'marker="s"' in source
    assert 'marker="o"' in source
    assert "color=colors[n_phi]" in source
    assert 'facecolors="none"' not in source
    assert 'edgecolors=colors[n_phi]' not in source
    assert "float(np.median(result[\"test_seconds\"]))" not in source
    assert (
        'np.median(result["test_median_over_angle_error"])'
        not in source
    )
    assert 'ax.set_ylabel("maximum relative error over angle")' in source
    assert 'label="ROSE"' in source
    assert 'label="LROM"' in source
```

- [ ] **Step 4: Update existing Notebook 02 result contracts**

In `test_notebook02_results_contract()`:

- keep the six figure/table marker checks only for Benchmark 03, whose comments
  remain unchanged;
- for Notebook 02, identify figures by their title/axis strings;
- remove all presentation-function definition expectations;
- keep alpha selection A/B/C, `display(alpha_cases)`, physical-radius, plotting
  floor, target-zone, and legend placement checks.

Use these direct-cell checks:

```python
for marker in (
    "Channel effective-interaction predictor radii",
    "relative cross-section error",
    "log10 median pointwise relative error",
    "Computational Accuracy versus Time",
):
    assert marker in text

assert "alpha_cases = pd.DataFrame(" in text
assert "display(alpha_cases)" in text
assert "one million evaluations/hour" in text
assert "axes[0].set_ylim(bottom=PLOTTING_FLOOR)" in text
assert "compression_sizes = {4: 16, 8: 28, 12: 44}" in text
assert "bbox_to_anchor=(1.02, 1.0)" in text
```

In `test_notebook_timing_reuses_initialized_online_paths()`, change the
Notebook 02 timing assertion from:

```python
assert "1e9 * inner_loops" in notebook_text_02
```

to:

```python
assert "1e9 * TIMING_INNER_LOOPS" in notebook_text_02
```

Leave Benchmark 03's `1e9 * inner_loops` assertion unchanged.

- [ ] **Step 5: Run the new contracts and confirm the expected red state**

Run:

```bash
python -m pytest -q \
  tests/test_notebook02_cross_sections.py::test_notebook02_flat_cell_contract \
  tests/test_notebook02_cross_sections.py::test_notebook02_cat_uses_per_alpha_paper_encoding
```

Expected: FAIL because Notebook 02 still contains 16 function definitions,
annotation-only imports, code comments, median CAT coordinates, and open LROM
circles.

- [ ] **Step 6: Commit the red contracts**

```bash
git add tests/test_notebook02_cross_sections.py
git commit -m "test(notebook02): require flat paper cat flow"
```

---

### Task 2: Flatten Notebook 02 into direct sequential cells

**Files:**
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
- Test: `tests/test_notebook02_cross_sections.py`

**Interfaces:**
- Consumes: unchanged `benchmark_helper.CrossSectionBenchmark`, parked-v2
  `LROM`, and current deterministic sample/result arrays.
- Produces: a comment-free Notebook 02 with no local functions and direct
  variables `rose_results`, `ls_results`, `old_v2_results`, `lrom_results`,
  `alpha_cases`, and `summary_table`.

- [ ] **Step 1: Replace the import cell with the minimal imports**

Replace code cell `e3486daf` with:

```python
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "lrom_legacy").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_helper
import lrom_legacy.v2_0 as lrom

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25})
```

Do not move Matplotlib, NumPy, pandas, timing, or path logic into either LROM
package. They are notebook presentation/runtime concerns.

- [ ] **Step 2: Keep constants direct and move sampling into its own cell**

Keep code cell `1cf258dc` as the constants-only cell.

Replace code cell `b0c47244` with direct construction and sampling:

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
```

Replace code cell `6d7d0d1a` with direct extraction and checks:

```python
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

train_cases = [
    {name: float(value) for name, value in zip(parameter_names, row)}
    for row in train_rows
]
test_cases = [
    {name: float(value) for name, value in zip(parameter_names, row)}
    for row in test_rows
]

LS_LABEL = "LS-projected cross section"
```

- [ ] **Step 3: Flatten the predictor-radius figure**

Replace code cell `109814a8` with the body of the current
`predictor_radius_figure()` followed by its current call-site assertions and
display. Apply this exact name mapping while deindenting:

| Current function name | Direct notebook name |
|---|---|
| `training_rows` | `train_rows` |
| `testing_rows` | `test_rows` |
| `predictor_count` | `DEFAULT_COMPRESSION_SIZE` |
| `minimum_radius` | `0.5` |
| `l_max` | `L_MAX` |
| `predictors` | `section3_predictor` |

The cell must end with:

```python
assert all(
    np.all(state.selected_radii >= 0.5)
    for state in section3_predictor.values()
)

plt.show()
display(predictor_radius_table)
```

Remove the `# FIGURE` comment. Keep all radii plotted against the physical
`radius_mesh` in fm.

- [ ] **Step 4: Put ROSE and LS calls alone in the equal-basis cell**

Replace code cell `5ddd34b5` with:

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
reference_sae = rose_emulators[(BASIS_SIZES[0], ROSE_EIM_SIZES[0])]
fom_train_xs = rose_comparison.fom_training_cross_sections
fom_test_xs = rose_comparison.fom_testing_cross_sections
rose_results = rose_comparison.results
ls_results = ls_comparison.results
default_key = (DEFAULT_BASIS_SIZE, DEFAULT_COMPRESSION_SIZE)
```

- [ ] **Step 5: Put old-v2 LROM evaluation directly in the next cell**

Use code cell `af265b1e` for old-v2 training, timing, prediction, and errors.
Use the current `evaluate_old_lrom()` operations directly and replace
`time_lrom_predictions()` with this explicit timing block:

```python
emulator.train(
    basis_size=DEFAULT_BASIS_SIZE,
    predictor="potential",
    predictor_count=DEFAULT_COMPRESSION_SIZE,
    observable="cross_section",
    angles_degrees=ANGLES_DEG,
)

emulator.predict(parameters=test_cases[:1], reconstruct_wavefunctions=False)
old_v2_seconds = []

for case in test_cases:
    measurements = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter_ns()
        for _ in range(TIMING_INNER_LOOPS):
            emulator.predict(parameters=case, reconstruct_wavefunctions=False)
        measurements.append(
            (time.perf_counter_ns() - start)
            / (1e9 * TIMING_INNER_LOOPS)
        )
    old_v2_seconds.append(min(measurements))

old_v2_seconds = np.asarray(old_v2_seconds)
emulator.predict(parameters=test_cases, reconstruct_wavefunctions=False)
old_v2_xs = emulator.predictions.cross_sections.values.copy()
old_v2_error = benchmark_helper.summarize_relative_error(
    old_v2_xs,
    fom_test_xs,
    ERROR_DENOMINATOR_FLOOR,
)

old_v2_results = {
    default_key: {
        "test_xs": old_v2_xs,
        "test_median_over_angle_error": old_v2_error[
            "median_over_angle_error"
        ],
        "test_maximum_over_angle_error": old_v2_error[
            "maximum_over_angle_error"
        ],
        "test_seconds": old_v2_seconds,
    }
}
```

- [ ] **Step 6: Add one direct cell for the archive LROM grid**

Insert a code cell after `af265b1e`. It must contain the current
`evaluate_lrom_grid()` loop directly, with:

```python
lrom_results = {}
default_predictor = None

for n_phi in BASIS_SIZES:
    for predictor_count in LROM_PREDICTOR_COUNTS:
        emulator.train(
            basis_size=n_phi,
            predictor="effective-interaction",
            predictor_count=predictor_count,
            observable="cross_section",
            angles_degrees=ANGLES_DEG,
        )

        emulator.predict(
            parameters=test_cases[:1],
            reconstruct_wavefunctions=False,
        )
        test_seconds = []

        for case in test_cases:
            measurements = []
            for _ in range(TIMING_REPEATS):
                start = time.perf_counter_ns()
                for _ in range(TIMING_INNER_LOOPS):
                    emulator.predict(
                        parameters=case,
                        reconstruct_wavefunctions=False,
                    )
                measurements.append(
                    (time.perf_counter_ns() - start)
                    / (1e9 * TIMING_INNER_LOOPS)
                )
            test_seconds.append(min(measurements))

        emulator.predict(
            parameters=train_cases,
            reconstruct_wavefunctions=False,
        )
        train_xs = emulator.predictions.cross_sections.values.copy()

        emulator.predict(
            parameters=test_cases,
            reconstruct_wavefunctions=False,
        )
        test_xs = emulator.predictions.cross_sections.values.copy()

        train_error = benchmark_helper.summarize_relative_error(
            train_xs,
            fom_train_xs,
            ERROR_DENOMINATOR_FLOOR,
        )
        test_error = benchmark_helper.summarize_relative_error(
            test_xs,
            fom_test_xs,
            ERROR_DENOMINATOR_FLOOR,
        )

        lrom_results[(n_phi, predictor_count)] = {
            "train_xs": train_xs,
            "test_xs": test_xs,
            "train_median_over_angle_error": train_error[
                "median_over_angle_error"
            ],
            "test_median_over_angle_error": test_error[
                "median_over_angle_error"
            ],
            "train_maximum_over_angle_error": train_error[
                "maximum_over_angle_error"
            ],
            "test_maximum_over_angle_error": test_error[
                "maximum_over_angle_error"
            ],
            "test_seconds": np.asarray(test_seconds),
        }

        if (n_phi, predictor_count) == default_key:
            default_predictor = emulator.predictors

archive_lrom_results = lrom_results
```

End this cell with the existing finite-value, result-key, predictor-index, FOM
shape, partial-wave, and old-v2 improvement assertions from code cell
`af265b1e`. Remove no scientific assertion.

- [ ] **Step 7: Flatten alpha selection and all non-CAT presentation cells**

Replace each current presentation function with its body in the corresponding
section, deindented and using the current global variable names:

| Section | Direct values |
|---|---|
| Alpha selection | `lrom_results[default_key]`, `rose_results[default_key]`, `test_ids`, `test_rows`, `parameter_names` |
| Representative cross sections | `fom_test_xs`, default LS/ROSE/LROM `test_xs`, `ANGLES_DEG` |
| Pointwise errors | `benchmark_helper.pointwise_relative_error()`, `ERROR_DENOMINATOR_FLOOR`, `PLOTTING_FLOOR` |
| Split distributions | default LS/ROSE/LROM `train_median_over_angle_error` and `test_median_over_angle_error` |
| Summary table | all ROSE/LROM configurations and their median arrays |
| Validation table | direct shapes, constants, versions, and default error arrays |

Split the current combined representative cell so alpha selection and
cross-section curves remain under Section 5, while pointwise error curves and
split distributions are separate code cells under Section 6.

Build legends without `Line2D` or `Patch`. For the split distribution legend:

```python
ax.scatter([], [], marker="s", color="tab:blue", label="train")
ax.scatter([], [], marker="s", color="tab:orange", label="test")
ax.scatter([], [], marker="D", color="0.3", label="median")
ax.legend()
```

Remove all `# FIGURE` and `# TABLE` comments. Preserve the exact existing
titles, labels, colors, plotting floors, and displayed tables.

- [ ] **Step 8: Implement the paper-faithful CAT cell**

Replace the current `accuracy_time_figure()` definition and call with:

```python
fig, ax = plt.subplots(figsize=(12.5, 5.4))
colors = {4: "tab:blue", 6: "tab:green", 8: "tab:red"}
compression_sizes = {4: 16, 8: 28, 12: 44}

for (n_phi, n_u), result in rose_results.items():
    ax.scatter(
        result["test_seconds"],
        np.maximum(
            result["test_maximum_over_angle_error"],
            PLOTTING_FLOOR,
        ),
        marker="s",
        s=compression_sizes[n_u],
        alpha=0.18,
        color=colors[n_phi],
    )

for (n_phi, predictor_count), result in lrom_results.items():
    ax.scatter(
        result["test_seconds"],
        np.maximum(
            result["test_maximum_over_angle_error"],
            PLOTTING_FLOOR,
        ),
        marker="o",
        s=compression_sizes[predictor_count],
        alpha=0.18,
        color=colors[n_phi],
    )

ax.axhline(0.10, color="0.25", linestyle="--")
ax.axvline(3600 / 1_000_000, color="0.45", linestyle=":")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("online time per sample [s]")
ax.set_ylabel("maximum relative error over angle")
ax.set_title("Computational Accuracy versus Time")

ax.scatter([], [], marker="s", color="black", label="ROSE")
ax.scatter([], [], marker="o", color="black", label="LROM")

for n_phi, color in colors.items():
    ax.scatter(
        [],
        [],
        marker="o",
        color=color,
        label=f"n_phi={n_phi}",
    )

for compression, size in compression_sizes.items():
    ax.scatter(
        [],
        [],
        marker="o",
        s=size,
        color="0.6",
        label=f"compression control={compression}",
    )

ax.plot(
    [],
    [],
    color="0.25",
    linestyle="--",
    label="0.10 maximum relative error",
)
ax.plot(
    [],
    [],
    color="0.45",
    linestyle=":",
    label="one million evaluations/hour",
)

ax.legend(
    bbox_to_anchor=(1.02, 1.0),
    loc="upper left",
    fontsize=8,
)
fig.subplots_adjust(right=0.72)
plt.show()
```

This creates 900 ROSE and 900 LROM points for the 100-row full test profile.
No timing or error arrays are recalculated for the figure.

- [ ] **Step 9: Clear stale execution state**

For every modified or inserted code cell:

- set `execution_count` to `null`;
- set `outputs` to `[]`;
- remove stale `metadata.execution`;
- retain the existing notebook kernel and language metadata.

- [ ] **Step 10: Run structural contracts and Ruff**

Run:

```bash
python -m pytest -q tests/test_notebook02_cross_sections.py
python -m ruff check tests/test_notebook02_cross_sections.py
```

Expected: all focused contracts pass and Ruff reports no findings.

- [ ] **Step 11: Confirm source simplification**

Run:

```bash
jq -r '[.cells[] | select(.cell_type == "code") | .source[]] | join("")' \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb | rg '^def '
```

Expected: no output and exit status 1 because no function definitions remain.

Run:

```bash
jq -r '[.cells[] | select(.cell_type == "code") | .source[]] | join("")' \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb | \
  python -c 'import io,sys,tokenize; print(sum(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(sys.stdin.read()).readline)))'
```

Expected: `0`.

- [ ] **Step 12: Commit the flat source**

```bash
git add notebooks/02_rose_vs_lrom_cross_sections.ipynb tests/test_notebook02_cross_sections.py
git commit -m "refactor(notebook02): flatten paper results flow"
```

---

### Task 3: Execute the full study and validate the five figures

**Files:**
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
- Test: `tests/test_notebook02_cross_sections.py`

**Interfaces:**
- Consumes: the flat Notebook 02 source from Task 2.
- Produces: stored full-profile outputs with per-alpha CAT clusters and
  unchanged non-CAT science.

- [ ] **Step 1: Execute Notebook 02**

Run:

```bash
MPLCONFIGDIR=/tmp/lrom-mpl-cache \
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb
```

Expected: exit status 0 and a final `Writing ...` line.

- [ ] **Step 2: Check execution and output counts**

Run:

```bash
jq '{
  code_cells: ([.cells[] | select(.cell_type == "code")] | length),
  executed_code_cells: ([.cells[] | select(
    .cell_type == "code" and .execution_count != null
  )] | length),
  error_outputs: ([.cells[].outputs[]? | select(
    .output_type == "error"
  )] | length),
  png_outputs: ([.cells[].outputs[]? | select(
    .data["image/png"]?
  )] | length)
}' notebooks/02_rose_vs_lrom_cross_sections.ipynb
```

Expected: executed code cells equal total code cells, `error_outputs` is 0,
and `png_outputs` is 5.

- [ ] **Step 3: Check scientific parity**

Run:

```bash
rg -n \
  'test-0002|test-0039|test-0019|0\.043529|0\.000948|0\.028713' \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb
```

Expected: all three alpha IDs and all three default errors are present.

Inspect the summary output and confirm every non-timing accuracy value matches
the pre-change executed notebook. Timing may vary.

- [ ] **Step 4: Render notebook figures**

Create a disposable directory and render into it:

```bash
figure_dir=$(mktemp -d /tmp/lrom-notebook02-flat-figs.XXXXXX)
jupyter nbconvert \
  --to markdown \
  --output-dir "$figure_dir" \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb
find "$figure_dir/02_rose_vs_lrom_cross_sections_files" \
  -maxdepth 1 -type f -name '*.png' -print | sort
```

Expected: five PNG file paths.

- [ ] **Step 5: Inspect all five figures**

Use `view_image` on:

1. predictor-radius figure;
2. representative cross sections;
3. representative pointwise errors;
4. split train/test error distributions;
5. CAT plot.

Acceptance criteria for CAT:

- ROSE points are solid squares;
- LROM points are solid circles with visible color;
- each configuration appears as a cluster, not one point;
- overlap remains legible at alpha 0.18;
- axes are logarithmic;
- y-axis says `maximum relative error over angle`;
- 10% and one-million-per-hour guides are visible;
- legend is outside the data region.

- [ ] **Step 6: Run focused tests after execution**

Run:

```bash
python -m pytest -q \
  tests/test_benchmark_helper.py \
  tests/test_notebook02_cross_sections.py
python -m ruff check \
  notebooks/benchmark_helper.py \
  tests/test_benchmark_helper.py \
  tests/test_notebook02_cross_sections.py
```

Expected: all tests pass and Ruff reports no findings.

- [ ] **Step 7: Commit stored execution**

```bash
git add notebooks/02_rose_vs_lrom_cross_sections.ipynb
git commit -m "docs(notebook02): store paper cat execution"
```

---

### Task 4: Verify protected boundaries and update the maintained explanation

**Files:**
- Modify: `../docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify: `_memory/Daily Notes/2026-07-28.md`
- Modify: `_memory/Context/active-state.md`
- Test: full repository suite

**Interfaces:**
- Consumes: final committed Notebook 02 and tests.
- Produces: final verification evidence and user-facing methodology record.

- [ ] **Step 1: Run the full suite**

Run:

```bash
python -m pytest -q
python -m ruff check lrom lrom_legacy tests tools notebooks/benchmark_helper.py
```

Expected: all tests pass and Ruff reports no findings.

- [ ] **Step 2: Verify protected paths**

Run:

```bash
git diff --quiet d8830c0 -- \
  lrom \
  lrom_legacy \
  notebooks/benchmark_helper.py \
  notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb \
  notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb \
  notebooks/benchmark_notebooks/2.0/benchmark_01.ipynb \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
```

Expected: exit status 0.

Run:

```bash
test "$(
  rg --files -0 ../scientific_archive |
  sort -z |
  xargs -0 shasum -a 256 |
  shasum -a 256 |
  awk '{print $1}'
)" = "afe80b7cf267fad0c3265bd18a7e023f966eb5fd4a655123887ee559c0061213"
```

Expected: exit status 0, proving the scientific archive digest is unchanged.

- [ ] **Step 3: Update the architecture explanation**

In `../docs/LROM_ARCHITECTURE_UNDERSTANDING.md`, add a change record with:

- Notebook 02 now has no local function definitions;
- direct cells expose sampling, benchmark calls, parked-v2 LROM evaluation,
  alpha selection, figures, and tables;
- imports were reduced without moving analysis dependencies into LROM;
- ROSE and LS remain separate helper calls;
- CAT now matches the paper's one-alpha-per-point interpretation;
- y is per-alpha maximum-over-angle error;
- ROSE squares and LROM solid circles encode method;
- color encodes \(n_\phi\), size encodes \(n_U\) or \(K\);
- non-CAT scientific values and protected files remain unchanged.

- [ ] **Step 4: Update memory last**

Append a `[LROM_Project]` entry to
`_memory/Daily Notes/2026-07-28.md` and prepend the final handoff under
`## Last Session Handoff` in `_memory/Context/active-state.md`.

Record:

- final commit IDs;
- code/execution/error/figure counts;
- alpha IDs and default values;
- per-alpha CAT semantics;
- full test and Ruff results;
- protected paths unchanged;
- no push attempted;
- existing untracked `tmp/` untouched.

Do not run additional filesystem commands after the memory update.

---

## Final Acceptance Checklist

- [ ] Notebook 02 has no local functions.
- [ ] Notebook 02 code cells have no comments.
- [ ] Existing Markdown headings are unchanged.
- [ ] Removed imports are absent and no new dependency is added.
- [ ] ROSE and LS use separate helper calls.
- [ ] LROM remains a separate explicit notebook workflow.
- [ ] CAT contains one point per held-out alpha for every configuration.
- [ ] LROM markers are solid circles.
- [ ] CAT y-values are maximum-over-angle errors.
- [ ] All five figures render and pass visual inspection.
- [ ] Alpha IDs and non-CAT scientific values are unchanged.
- [ ] Full tests and Ruff pass.
- [ ] Protected paths are unchanged.
- [ ] Architecture and memory documentation are current.
