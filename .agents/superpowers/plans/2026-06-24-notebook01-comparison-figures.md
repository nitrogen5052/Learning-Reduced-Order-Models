# Notebook 1 Comparison Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing Vv coefficient and single-wavefunction comparisons and replace the final ws3 error-lines figure with a relative-L2 violin comparison.

**Architecture:** Keep all plotting explicit in the Notebook 1 generator and consume only state already exposed by `vv_emulator` and `ws3_emulator`. Regenerate the checked-in notebook deterministically; do not modify the `lrom` package or Notebook 2.

**Tech Stack:** Python 3.11-3.13, NumPy, Matplotlib, nbformat, pytest, Jupyter

---

### Task 1: Specify the Three Figure Contracts

**Files:**
- Modify: `tests/test_notebook01_lrom_flow.py`

- [ ] **Step 1: Write failing structural tests**

Add tests that require the first two coordinate columns, the central Vv test case,
all four wavefunction sources, and the violin metric:

```python
def test_vv_coefficients_compare_first_two_basis_coordinates() -> None:
    text = source()
    assert "for coefficient_index in range(2)" in text
    assert 'coefficients["ls"][0][:, coefficient_index]' in text
    assert 'coefficients["rose"][0][:, coefficient_index]' in text
    assert 'coefficients["lrom"][0][:, coefficient_index]' in text


def test_vv_central_testing_wavefunction_compares_all_methods() -> None:
    text = source()
    assert "vv_representative_index = len(vv_test) // 2" in text
    assert "vv_emulator.testing_case(case_id=vv_representative_id)" in text
    for method in ("high_fidelity", "ls", "lrom", "rose"):
        assert f"vv_case.{method}[0]" in text


def test_ws3_final_figure_is_relative_l2_violin_comparison() -> None:
    text = source()
    assert 'metrics = ws3_emulator.testing_results.metrics["relative_l2"][0]' in text
    assert "np.log10(np.maximum(metrics[method], 1e-16))" in text
    assert "ax.violinplot(" in text
```

- [ ] **Step 2: Run tests to verify the contract fails**

Run:

```bash
pytest tests/test_notebook01_lrom_flow.py -q
```

Expected: the new coefficient-loop, Vv wavefunction-selection, and violin assertions fail against the existing generator.

### Task 2: Implement and Generate the Figures

**Files:**
- Modify: `scripts/generate_notebook01.py`
- Modify: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`
- Test: `tests/test_notebook01_lrom_flow.py`

- [ ] **Step 1: Replace the one-coordinate panel with two coefficient panels**

Use `vv_emulator.testing_results.coefficients` and plot `Re(a1)` and `Re(a2)`:

```python
coefficients = vv_emulator.testing_results.coefficients
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
for coefficient_index in range(2):
    ax = axes[coefficient_index]
    for method, style in (("ls", "-"), ("rose", "--"), ("lrom", ":")):
        ax.plot(
            vv_test,
            np.real(coefficients[method][0][:, coefficient_index]),
            style,
            label=method.upper(),
        )
    ax.set_ylabel(f"Re(a{coefficient_index + 1})")
axes[-1].set_xlabel("Vv [MeV]")
axes[0].legend()
```

- [ ] **Step 2: Add the central Vv testing-wavefunction comparison**

Select the midpoint of the 41-point linspace and plot the true solution plus all
three approximations:

```python
vv_representative_index = len(vv_test) // 2
vv_representative_id = vv_emulator.samples.design.testing.case_ids[vv_representative_index]
vv_case = vv_emulator.testing_case(case_id=vv_representative_id)
for label, values, style in (
    ("true test solution", vv_case.high_fidelity[0], "-"),
    ("LS", vv_case.ls[0], "-."),
    ("LROM", vv_case.lrom[0], ":"),
    ("ROSE", vv_case.rose[0], "--"),
):
    ax.plot(vv_case.radius, np.real(values), style, label=label)
```

- [ ] **Step 3: Replace the final ws3 line figure with violins**

Use the 81 relative-L2 values per method and plot their base-10 logarithms:

```python
metrics = ws3_emulator.testing_results.metrics["relative_l2"][0]
methods = ("ls", "lrom", "rose")
violin_values = [
    np.log10(np.maximum(metrics[method], 1e-16))
    for method in methods
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
parts = ax.violinplot(violin_values, showmedians=True, showextrema=True)
ax.set_xticks(range(1, 4), [method.upper() for method in methods])
ax.set_ylabel("log10(relative L2 error)")
```

- [ ] **Step 4: Regenerate twice and verify determinism**

Run:

```bash
python scripts/generate_notebook01.py
cp notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb /tmp/notebook01-figures-first.ipynb
python scripts/generate_notebook01.py
cmp /tmp/notebook01-figures-first.ipynb notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
```

Expected: `cmp` exits successfully.

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
pytest tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py -q
pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Execute Notebook 1**

Run:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb --output /tmp/notebook01-figures-executed.ipynb --ExecutePreprocessor.timeout=3600
```

Expected: all code cells execute without an error output and the notebook contains the new figures.

- [ ] **Step 7: Commit only Notebook 1 figure files**

```bash
git add scripts/generate_notebook01.py notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb tests/test_notebook01_lrom_flow.py
git commit -m "Add Notebook 1 scientific comparison figures"
```
