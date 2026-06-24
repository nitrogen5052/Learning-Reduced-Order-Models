# Explicit Sampling Grids and Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept user-supplied named parameter grids in `LROM.sampling()`, retain training diagnostics, and show absolute differences plus split training/testing violins in Notebook 1.

**Architecture:** Add an explicit-grid design builder beside the existing range-based builder and select between them inside the keyword-only `LROM.sampling()` facade. Factor model evaluation into one internal path used for training and testing datasets. Keep all figure construction explicit in the Notebook 1 generator.

**Tech Stack:** Python 3.11-3.13, NumPy, SciPy, Matplotlib, nuclear-rose, pytest, nbformat, Jupyter

---

### Task 1: Named Explicit Parameter Grids

**Files:**
- Modify: `lrom/sampling.py`
- Modify: `lrom/emulator.py`
- Modify: `tests/test_lrom_sampling.py`
- Modify: `tests/test_lrom_lifecycle.py`

- [ ] **Step 1: Write failing explicit-grid tests**

Add sampling tests that prove row alignment, stable IDs, central filling, and
validation:

```python
def test_explicit_grids_are_row_aligned_and_fill_central_values() -> None:
    design = create_explicit_sampling_design(
        parameter_names=("Vv", "Rv", "av"),
        sampleable_names=("Vv", "Rv", "av"),
        central={"Vv": 50.0, "Rv": 4.0, "av": 0.65},
        training_grid={"Vv": [45.0, 50.0, 55.0]},
        testing_grid={"Vv": [42.0, 58.0]},
    )
    assert design.strategy == "explicit_grid"
    assert design.training.case_ids == ("train-0000", "train-0001", "train-0002")
    assert np.allclose(design.training.values[:, 0], [45.0, 50.0, 55.0])
    assert np.all(design.training.values[:, 1] == 4.0)
    assert np.all(design.testing.values[:, 2] == 0.65)


@pytest.mark.parametrize(
    "training_grid, testing_grid, message",
    [
        ({"Vv": [45.0, 50.0], "Rv": [4.0]}, {"Vv": [50.0]}, "equal length"),
        ({"bad": [1.0]}, {"bad": [2.0]}, "unknown parameter"),
        ({"Vv": [np.nan]}, {"Vv": [50.0]}, "finite"),
        ({"Vv": []}, {"Vv": [50.0]}, "at least one"),
    ],
)
def test_explicit_grid_validation(training_grid, testing_grid, message) -> None:
    with pytest.raises(LROMSamplingError, match=message):
        create_explicit_sampling_design(
            parameter_names=("Vv", "Rv", "av"),
            sampleable_names=("Vv", "Rv", "av"),
            central={"Vv": 50.0, "Rv": 4.0, "av": 0.65},
            training_grid=training_grid,
            testing_grid=testing_grid,
        )
```

Add lifecycle tests showing `LROM.sampling(training_grid=..., testing_grid=...)`
reaches the FOM provider and that incomplete or mixed modes raise
`LROMSamplingError`.

- [ ] **Step 2: Run tests and confirm missing explicit-grid behavior**

Run:

```bash
pytest tests/test_lrom_sampling.py tests/test_lrom_lifecycle.py -q
```

Expected: collection or call failures because `create_explicit_sampling_design`
and the grid arguments do not exist.

- [ ] **Step 3: Implement explicit grid validation and conversion**

Add a keyword-only builder in `lrom/sampling.py`:

```python
def create_explicit_sampling_design(
    *,
    parameter_names: tuple[str, ...],
    sampleable_names: tuple[str, ...],
    central: Mapping[str, float],
    training_grid: Mapping[str, Sequence[float]],
    testing_grid: Mapping[str, Sequence[float]],
) -> SamplingDesign:
    training = _explicit_cases(
        prefix="train",
        grid=training_grid,
        parameter_names=parameter_names,
        sampleable_names=sampleable_names,
        central=central,
    )
    testing = _explicit_cases(
        prefix="test",
        grid=testing_grid,
        parameter_names=parameter_names,
        sampleable_names=sampleable_names,
        central=central,
    )
    if set(training_grid) != set(testing_grid):
        raise LROMSamplingError(
            "training_grid and testing_grid must use the same parameter names"
        )
    return SamplingDesign(
        training=training,
        testing=testing,
        strategy="explicit_grid",
        seed=None,
    )
```

`_explicit_cases` validates a non-empty mapping, named sampleable keys,
one-dimensional finite columns, equal positive column lengths, then starts from
`_base_values` so omitted columns retain central values.

- [ ] **Step 4: Route the public sampling facade by mode**

Change the keyword-only signature to optional generated inputs plus grid inputs:

```python
def sampling(
    self,
    *,
    training_ranges: Mapping[str, tuple[float, float]] | None = None,
    testing_ranges: Mapping[str, tuple[float, float]] | None = None,
    training_size: int | None = None,
    testing_size: int | None = None,
    training_grid: Mapping[str, Sequence[float]] | None = None,
    testing_grid: Mapping[str, Sequence[float]] | None = None,
    mesh_size: int = 900,
    radial_domain: tuple[float, float] | None = None,
    strategy: str | None = None,
    seed: int | None = None,
    eim_basis_size: int = 8,
    solver_options: Mapping[str, object] | None = None,
) -> None:
```

When either grid is present, require both and reject range, size, strategy, and
seed arguments. Otherwise require all four generated inputs and pass
`strategy or "latin_hypercube"` to `create_sampling_design`.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
pytest tests/test_lrom_sampling.py tests/test_lrom_lifecycle.py -q
```

Expected: all tests pass.

Commit:

```bash
git add lrom/sampling.py lrom/emulator.py tests/test_lrom_sampling.py tests/test_lrom_lifecycle.py
git commit -m "Add explicit named sampling grids"
```

### Task 2: Training and Testing Evaluation State

**Files:**
- Modify: `lrom/state.py`
- Modify: `lrom/emulator.py`
- Modify: `lrom/training.py`
- Modify: `lrom/artifacts.py`
- Modify: `tests/test_lrom_training.py`
- Modify: `tests/test_lrom_artifacts.py`

- [ ] **Step 1: Write failing dual-dataset diagnostics tests**

After training a small real emulator, assert:

```python
assert emulator.training_results.high_fidelity[0].shape == (training_size, mesh_size)
assert emulator.testing_results.high_fidelity[0].shape == (testing_size, mesh_size)
for results, size in (
    (emulator.training_results, training_size),
    (emulator.testing_results, testing_size),
):
    for method in ("rose", "lrom", "ls"):
        assert results.coefficients[method][0].shape == (size, basis_size)
        assert results.metrics["relative_l2"][0][method].shape == (size,)
        assert results.metrics["pointwise_absolute"][0][method].shape == (
            size,
            mesh_size,
        )
```

Extend artifact tests to assert a loaded inference-only model has
`training_results is None` and still predicts normally.

- [ ] **Step 2: Run tests and confirm training results are absent**

Run:

```bash
pytest tests/test_lrom_training.py tests/test_lrom_artifacts.py -q
```

Expected: failures because `LROM.training_results` and training evaluation state
do not exist.

- [ ] **Step 3: Extend state and public access**

Add `training_results: Any = None` as the final field of `TrainingState`, and add:

```python
@property
def training_results(self) -> Any:
    return None if self._training_state is None else self._training_state.training_results
```

Keep the final default so existing fake trainers and artifact reconstruction
remain compatible.

- [ ] **Step 4: Evaluate both fitted datasets through one helper**

In `lrom/training.py`, factor the repeated reconstruction and metrics work into:

```python
def _evaluate(
    *,
    emulator,
    bases,
    rose_states,
    rf_models,
    wavefunctions,
    parameter_values,
    predictor_features,
) -> TestingResults:
```

For every channel, calculate LS coordinates, LROM coordinates, ROSE coordinates,
all four wavefunction sets, relative-L2 arrays, and pointwise absolute arrays.
Return metrics with both keys:

```python
metrics={
    "relative_l2": relative_metrics,
    "pointwise_absolute": pointwise_metrics,
}
```

Call `_evaluate` once with training arrays/features and once with testing
arrays/features. Store both results and retain
`testing_errors=testing_results.metrics["pointwise_absolute"]` for compatibility.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
pytest tests/test_lrom_training.py tests/test_lrom_artifacts.py -q
```

Expected: all tests pass.

Commit:

```bash
git add lrom/state.py lrom/emulator.py lrom/training.py lrom/artifacts.py tests/test_lrom_training.py tests/test_lrom_artifacts.py
git commit -m "Store training and testing diagnostics"
```

### Task 3: Notebook 1 Absolute Differences and Split Violins

**Files:**
- Modify: `scripts/generate_notebook01.py`
- Modify: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`
- Modify: `tests/test_notebook01_lrom_flow.py`

- [ ] **Step 1: Write failing notebook contracts**

Require coefficient scatter calls, `axvspan` extrapolation bands,
`np.abs(LS - method)` coefficient errors, two-row wavefunction figures, and both
training/testing violin metrics:

```python
assert ".scatter(" in text
assert "axvspan(vv_test.min(), vv_train_low" in text
assert "axvspan(vv_train_high, vv_test.max()" in text
assert "np.abs(ls_coefficients - method_coefficients)" in text
assert "np.abs(vv_case.high_fidelity[0] - vv_case.lrom[0])" in text
assert "np.abs(case.high_fidelity[0] - case.rose[0])" in text
assert "ws3_emulator.training_results.metrics" in text
assert "ws3_emulator.testing_results.metrics" in text
assert "training_violin" in text
assert "testing_violin" in text
```

- [ ] **Step 2: Run tests and confirm the figure contracts fail**

Run:

```bash
pytest tests/test_notebook01_lrom_flow.py -q
```

Expected: failures against the current line plots, single-row comparisons, and
testing-only violins.

- [ ] **Step 3: Implement comparison-plus-error figures**

For both representative wavefunctions, create `plt.subplots(2, 1, sharex=True)`.
The upper axis plots `Re(phi)`. The lower axis plots clipped pointwise absolute
differences for LS, LROM, and ROSE and uses a logarithmic y scale.

Use scatter points for Vv coefficients. For both coefficient panels, shade
`[vv_test.min(), vv_train_low]` and `[vv_train_high, vv_test.max()]`, then plot
`abs(LS - LROM)` and `abs(LS - ROSE)` below. Apply the same comparison/error
pair to the ws3 coefficient-versus-case figure.

- [ ] **Step 4: Implement split training/testing violins**

Build log-error arrays from both result sets:

```python
training_metrics = ws3_emulator.training_results.metrics["relative_l2"][0]
testing_metrics = ws3_emulator.testing_results.metrics["relative_l2"][0]
training_values = [np.log10(np.maximum(training_metrics[m], 1e-16)) for m in methods]
testing_values = [np.log10(np.maximum(testing_metrics[m], 1e-16)) for m in methods]
training_violin = ax.violinplot(training_values, positions=positions, showmedians=True)
testing_violin = ax.violinplot(testing_values, positions=positions, showmedians=True)
```

For each training body, clamp x vertices to the left of its center; for each
testing body, clamp x vertices to the right. Use consistent method colors and a
two-patch legend for training/testing halves.

- [ ] **Step 5: Regenerate deterministically and test**

Run:

```bash
python scripts/generate_notebook01.py
cp notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb /tmp/notebook01-grid-first.ipynb
python scripts/generate_notebook01.py
cmp /tmp/notebook01-grid-first.ipynb notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
pytest tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py -q
pytest -q
```

Expected: deterministic output and all tests pass.

- [ ] **Step 6: Execute Notebook 1 and inspect outputs**

Run:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb --output /tmp/notebook01-grid-executed.ipynb --ExecutePreprocessor.timeout=3600
```

Expected: zero cell errors and all comparison/error/violin figures render.

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_notebook01.py notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb tests/test_notebook01_lrom_flow.py
git commit -m "Expand Notebook 1 comparison diagnostics"
```

### Task 4: Final Verification and Local Integration

**Files:**
- Verify only

- [ ] **Step 1: Run final verification**

```bash
pytest -q
git diff --check
git status --short --branch
```

Expected: all tests pass, no whitespace errors, and only the known unrelated
Notebook 2 work remains dirty in the main checkout.

- [ ] **Step 2: Merge locally**

Fast-forward the verified feature branch into local `main`, rerun the complete
suite on `main`, then remove the project-local worktree and merged feature branch.
