# Notebook 01 Focused Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify Notebook 01 around LROM, replace method-biased single-case plots with three verified matched cases, and consolidate coordinate diagnostics without losing the user's local notebook comments.

**Architecture:** `tools/generate_notebook01.py` remains the only source of notebook structure. Source-contract tests drive each change before the checked-in notebook is regenerated; no package implementation changes are permitted. LS and LROM remain in the central-reference basis, ROSE remains in its free-reference basis, and every wavefunction comparison is aligned through one shared testing-row index.

**Tech Stack:** Python 3, NumPy, Matplotlib, nbformat, Jupyter, pytest, Ruff, public `lrom` v1.2, nuclear-rose.

## Global Constraints

- Work directly on `main` and push every green milestone because the workspace is vulnerable to iCloud eviction.
- Preserve the user's uncommitted edits in `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb` before regeneration.
- Modify no files under `lrom/`, `lrom_legacy/`, or `scientific_archive/`.
- Preserve sample ranges, sample counts `35/41` and `70/81`, seed `1204`, mesh size `800`, `BASIS_SIZE = 4`, predictor choices, and exact Runge-Kutta sampling.
- Keep the public v1.2 lifecycle unchanged.
- Keep all plotting inline; add no plotting wrappers or notebook helper abstractions.
- Use physical radius `r` in femtometers for wavefunction and basis plots.
- Fixed method colors are LS blue, LROM yellow `#E6AB02`, and ROSE red.
- ROSE and LROM/LS coordinates may share a figure but must retain distinct `c_j` and `a_j` labels; never subtract the two conventions.
- The user owns the final sentence-by-sentence prose pass.

---

### Task 1: Preserve User Comments and Shorten the ROSE Blocks

**Files:**
- Modify: `tests/test_notebook01_lrom_flow.py`
- Modify: `tools/generate_notebook01.py`
- Preserve without writing: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`

**Interfaces:**
- Consumes: the user's current notebook comments and `generate_notebook01.notebook_cells()`.
- Produces: generator-owned ROSE blocks with four named stages and no outlier-report tutorial block.

- [ ] **Step 1: Confirm and record the user's current source-only diff**

Run:

```bash
git status --short
git diff --numstat -- notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
```

Expected: the notebook is modified by the user; it is not staged. The untracked advisor backlog and `scientific_archive/ROSE_Guide/` remain untouched.

- [ ] **Step 2: Replace the obsolete ROSE-outlier test with the four-stage comment contract**

Replace `test_notebook01_reports_corrected_rose_outliers_by_named_case` in `tests/test_notebook01_lrom_flow.py` with:

```python
def test_notebook01_rose_blocks_show_only_the_four_required_stages() -> None:
    text = source()

    for phrase in (
        "Assemble the FOM parameter rows used to bound the ROSE EIM",
        "Initialize the notebook-owned EIM interaction",
        "Produce ROSE's free solution and reduced basis",
        "Evaluate ROSE on the same ordered parameter rows",
    ):
        assert text.count(phrase) == 2
    assert "worst corrected ROSE cases" not in text
    assert "ws3_rose_coefficient_norm" not in text
    assert "ws3_rose_worst_indices" not in text
```

Add a separate preservation test:

```python
def test_notebook01_preserves_the_users_explanatory_intent() -> None:
    text = source()

    assert "FOM parameter rows" in text
    assert "EIM interaction" in text
    assert "free solution" in text
    assert "reduced basis" in text
    assert "optional LS floor" in text
```

- [ ] **Step 3: Run the focused tests to verify RED**

Run:

```bash
python -m pytest -q tests/test_notebook01_lrom_flow.py::test_notebook01_rose_blocks_show_only_the_four_required_stages tests/test_notebook01_lrom_flow.py::test_notebook01_preserves_the_users_explanatory_intent
```

Expected: FAIL because the generator still contains the old wording and outlier report.

- [ ] **Step 4: Port the user's comments into both generator ROSE blocks**

Use these exact stage comments in the `Vv` and `ws_3` setup cells, immediately before the corresponding code:

```python
# 1. Assemble the FOM parameter rows used to bound the ROSE EIM.
# 2. Initialize the notebook-owned EIM interaction.
# 3. Produce ROSE's free solution and reduced basis.
# 4. Evaluate ROSE on the same ordered parameter rows as LS and LROM.
```

Change the LS comment in the `ws_3` cell to:

```python
# Compute the optional LS floor explicitly for both sample sets.
```

Delete only the ROSE tutorial diagnostic beginning with
`ws3_rose_coefficient_norm = np.linalg.norm(` and ending immediately before
`print("potential predictor radii [fm]:", ws3_emulator.predictors.selected_radii)`.
Retain that predictor-radius print and every ROSE relative-error array.

Retain `ws3_rose_rel_train`, `ws3_rose_rel_test`, ROSE coefficients, and ROSE wavefunctions because later comparisons consume them.

- [ ] **Step 5: Run focused tests to verify GREEN**

Run:

```bash
python -m pytest -q tests/test_notebook01_lrom_flow.py::test_notebook01_rose_blocks_show_only_the_four_required_stages tests/test_notebook01_lrom_flow.py::test_notebook01_preserves_the_users_explanatory_intent
```

Expected: both Task 1 tests PASS.

- [ ] **Step 6: Commit and push the generator-owned comment milestone**

Run:

```bash
git add tests/test_notebook01_lrom_flow.py tools/generate_notebook01.py
git commit -m "Focus the notebook 01 ROSE explanation"
git push origin main
```

Do not stage the user's notebook yet.

---

### Task 2: Remove Joint-Variation and Singular-Spectrum Figure Bloat

**Files:**
- Modify: `tests/test_notebook01_generation.py`
- Modify: `tests/test_notebook01_lrom_flow.py`
- Modify: `tools/generate_notebook01.py`

**Interfaces:**
- Consumes: rank-four LROM and ROSE basis vectors plus selected potential-predictor radii.
- Produces: one paired basis figure per study and one predictor-radius figure, with no singular spectra or standalone parameter-joint figure.

- [ ] **Step 1: Change generation tests to the compact figure contract**

In `tests/test_notebook01_generation.py`, replace the singular-spectrum assertion in `test_notebook01_preserves_the_project_map_figures_and_methods` with:

```python
    for phrase in (
        "Vv training potentials",
        "High-fidelity training solutions",
        "LROM central-reference basis",
        "ROSE free-reference basis",
        "potential predictor radii",
        '("ls", "blue")',
        '("lrom", "#E6AB02")',
        '("rose", "red")',
    ):
        assert phrase in source
    assert '"orange"' not in source
    assert "normalized singular value" not in source
    assert "snapshot spectrum" not in source.lower()
    assert "Potential-ensemble spectrum" not in source
    assert "joint parameter variation" not in source.lower()
    assert "def plot_" not in source
```

Replace `test_notebook01_pairs_each_basis_with_its_singular_spectrum` in `tests/test_notebook01_lrom_flow.py` with:

```python
def test_notebook01_uses_compact_paired_basis_figures() -> None:
    text = source()

    assert "vv_lrom_singular_values" not in text
    assert "ws3_lrom_singular_values" not in text
    assert ".singular_values" not in text
    assert "normalized singular value" not in text
    assert text.count('fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))') >= 2
    assert "LROM central-reference basis" in text
    assert "ROSE free-reference basis" in text
```

Add:

```python
def test_notebook01_removes_the_standalone_joint_variation_figure() -> None:
    text = source()

    assert "from itertools import combinations" not in text
    assert "joint parameter variation" not in text.lower()
    assert "parameter-pair" not in text.lower()
```

- [ ] **Step 2: Run the compact-figure tests to verify RED**

Run:

```bash
python -m pytest -q tests/test_notebook01_generation.py::test_notebook01_preserves_the_project_map_figures_and_methods tests/test_notebook01_lrom_flow.py::test_notebook01_uses_compact_paired_basis_figures tests/test_notebook01_lrom_flow.py::test_notebook01_removes_the_standalone_joint_variation_figure
```

Expected: FAIL on existing spectrum and pairwise-joint code.

- [ ] **Step 3: Replace each basis-plus-spectrum pair with one paired basis figure**

Use this complete `Vv` basis plotting block:

```python
basis = vv_emulator.basis[0]
fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
for coordinate_index in range(BASIS_SIZE):
    axes[0].plot(
        r,
        np.real(basis.vectors[:, coordinate_index]),
        label=fr"$a_{coordinate_index + 1}$ basis",
    )
    axes[1].plot(
        r,
        np.real(vv_rose_basis.vectors[:, coordinate_index]),
        label=fr"$c_{coordinate_index + 1}$ basis",
    )
axes[0].set(
    xlabel="r [fm]",
    ylabel="Re(basis vector)",
    title="LROM central-reference basis",
)
axes[1].set(
    xlabel="r [fm]",
    ylabel="Re(basis vector)",
    title="ROSE free-reference basis",
)
for ax in axes:
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
```

Use this complete `ws_3` basis plotting block:

```python
ws3_basis = ws3_emulator.basis[0]
fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
for coordinate_index in range(BASIS_SIZE):
    axes[0].plot(
        r3,
        np.real(ws3_basis.vectors[:, coordinate_index]),
        label=fr"$a_{coordinate_index + 1}$ basis",
    )
    axes[1].plot(
        r3,
        np.real(ws3_rose_basis.vectors[:, coordinate_index]),
        label=fr"$c_{coordinate_index + 1}$ basis",
    )
axes[0].set(
    xlabel="r [fm]",
    ylabel="Re(basis vector)",
    title="LROM central-reference basis",
)
axes[1].set(
    xlabel="r [fm]",
    ylabel="Re(basis vector)",
    title="ROSE free-reference basis",
)
for ax in axes:
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
```

- [ ] **Step 4: Remove the standalone joint-variation cell and simplify the predictor-radius cell**

Delete the markdown/code pair that imports or loops over `combinations` and plots the three parameter pairs. Remove `from itertools import combinations` from the setup cell.

Replace the two-panel predictor/spectrum figure with:

```python
r3 = ws3_emulator.mesh.radius
fig, ax = plt.subplots(figsize=(7.4, 4.0))
for potential in ws3_emulator.samples.training_potentials[:20]:
    ax.plot(r3, np.real(potential), color="0.65", alpha=0.20)
for radius_index, selected_radius in enumerate(
    ws3_emulator.predictors.selected_radii, start=1
):
    ax.axvline(
        selected_radius,
        color="purple",
        alpha=0.75,
        label="potential predictor radii" if radius_index == 1 else None,
    )
ax.set(
    xlabel="r [fm]",
    ylabel="V(r) [MeV]",
    title="Potential samples and selected LROM predictor radii",
)
ax.legend()
fig.tight_layout()
plt.show()
```

- [ ] **Step 5: Replace prose that promises deleted figures**

Use this Vv basis introduction:

```markdown
The next figure places the LROM central-reference basis beside the ROSE
free-reference basis. Their coordinates use different reference functions,
so the basis labels remain distinct.
```

Use this potential-predictor introduction:

```markdown
The selected radii show which local potential values become LROM predictor
features. The gray curves are training potentials; the vertical lines are
the six physical radii retained by the predictor.
```

Use this `ws_3` basis introduction:

```markdown
The two rank-four bases are shown together before their coordinates. LROM
and LS use the central-reference basis; ROSE uses its free-reference basis.
```

- [ ] **Step 6: Run focused tests and Ruff**

Run:

```bash
python -m pytest -q tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
python -m ruff check tools/generate_notebook01.py tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
```

Expected: all updated figure-contract tests PASS and Ruff reports `All checks passed!`.

- [ ] **Step 7: Commit and push the figure-reduction milestone**

```bash
git add tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py tools/generate_notebook01.py
git commit -m "Remove notebook 01 figure bloat"
git push origin main
```

---

### Task 3: Consolidate Coordinates into One Figure per Study

**Files:**
- Modify: `tests/test_notebook01_lrom_flow.py`
- Modify: `tools/generate_notebook01.py`

**Interfaces:**
- Consumes: training/testing coordinates for LS, LROM, and ROSE plus ordered physical parameter rows.
- Produces: `vv_coordinate_data` and `ws3_coordinate_data`, each rendered in one `3 x BASIS_SIZE` figure.

- [ ] **Step 1: Replace the old coordinate tests with the combined-layout contract**

Replace `test_vv_coefficients_use_separate_lrom_and_rose_coordinate_figures` with:

```python
def test_vv_coordinates_share_one_method_by_coordinate_figure() -> None:
    text = source()

    assert "vv_coordinate_data = (" in text
    assert 'plt.subplots(3, BASIS_SIZE, figsize=(14.0, 8.0)' in text
    assert '("LS", vv_ls_train_coefficients, vv_ls_coefficients, "blue", "a")' in text
    assert '("LROM", vv_lrom_train_coefficients, vv_lrom_coefficients, "#E6AB02", "a")' in text
    assert '("ROSE", vv_rose_coeff_train, vv_rose_coefficients, "red", "c")' in text
    assert "coordinate difference" not in text.lower()
```

Replace `test_ws3_coefficients_are_separate_and_wavefunctions_keep_absolute_differences` with:

```python
def test_ws3_coordinates_share_one_method_by_coordinate_figure() -> None:
    text = source()

    assert "ws3_coordinate_data = (" in text
    assert 'plt.subplots(3, BASIS_SIZE, figsize=(14.0, 8.4)' in text
    assert '("ROSE", ws3_rose_coeff_train, ws3_rose_coefficients, "red", "c")' in text
    assert "ws3_coordinate_difference" not in text
    assert "ws3_ls_coefficients - ws3_lrom_coefficients" not in text
    assert 'fig.colorbar(test_scatter, ax=axes, label="Rv [fm]")' in text
    assert 'coordinate_symbol = "a"' not in text
```

Update the explicit LS count in `test_notebook01_calls_the_ls_baseline_explicitly` from three to four and require `vv_ls_train_coefficients`.

- [ ] **Step 2: Run coordinate tests to verify RED**

```bash
python -m pytest -q tests/test_notebook01_lrom_flow.py::test_vv_coordinates_share_one_method_by_coordinate_figure tests/test_notebook01_lrom_flow.py::test_ws3_coordinates_share_one_method_by_coordinate_figure tests/test_notebook01_lrom_flow.py::test_notebook01_calls_the_ls_baseline_explicitly
```

Expected: FAIL because old separate and difference figures remain.

- [ ] **Step 3: Add the missing `Vv` training coordinates**

Immediately after the existing testing LS baseline, add:

```python
vv_fom_train = vv_emulator.samples.training_wavefunctions[0]
vv_ls_train_coefficients, _ = lrom.least_squares_baseline(
    basis=vv_emulator.basis[0],
    wavefunctions=vv_fom_train,
)
vv_lrom_train_coefficients = np.asarray(
    vv_emulator.training_results.coefficients["lrom"][0]
)
vv_lrom_coefficients = np.asarray(
    vv_emulator.testing_results.coefficients["lrom"][0]
)
```

After ROSE construction, add:

```python
vv_rose_coeff_train = np.asarray(
    [vv_rose_rbe.coefficients(row) for row in vv_emulator.samples.design.training.values]
)
```

- [ ] **Step 4: Replace all `Vv` coordinate plots with one 3-by-4 figure**

```python
vv_test = vv_emulator.samples.design.testing.values[:, 0]
vv_train = vv_emulator.samples.design.training.values[:, 0]
vv_plot_mask = ~np.isclose(vv_test, Vv0, rtol=0.0, atol=1e-12)
vv_coordinate_data = (
    ("LS", vv_ls_train_coefficients, vv_ls_coefficients, "blue", "a"),
    ("LROM", vv_lrom_train_coefficients, vv_lrom_coefficients, "#E6AB02", "a"),
    ("ROSE", vv_rose_coeff_train, vv_rose_coefficients, "red", "c"),
)
fig, axes = plt.subplots(3, BASIS_SIZE, figsize=(14.0, 8.0), sharex=True)
for method_row, (method, training_coordinates, testing_coordinates, color, symbol) in enumerate(vv_coordinate_data):
    for coordinate_index in range(BASIS_SIZE):
        ax = axes[method_row, coordinate_index]
        ax.scatter(
            vv_train,
            np.real(training_coordinates[:, coordinate_index]),
            facecolors="none",
            edgecolors="black",
            s=24,
            label="training" if coordinate_index == 0 else None,
        )
        ax.scatter(
            vv_test[vv_plot_mask],
            np.real(testing_coordinates[vv_plot_mask, coordinate_index]),
            color=color,
            marker="s",
            s=22,
            alpha=0.75,
            label="testing" if coordinate_index == 0 else None,
        )
        ax.set_ylabel(fr"Re(${symbol}_{coordinate_index + 1}$)")
        ax.set_title(f"{method} coordinate {coordinate_index + 1}", color=color)
        if method_row == 2:
            ax.set_xlabel("Vv [MeV]")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Matched Vv coordinates: rows are basis conventions")
fig.tight_layout()
plt.show()
```

- [ ] **Step 5: Replace all `ws_3` coordinate plots with one 3-by-4 figure**

```python
ws3_rose_coeff_train = np.asarray(
    [ws3_rose_rbe.coefficients(row) for row in ws3_train_rows]
)
ws3_lrom_train_coefficients = np.asarray(
    ws3_emulator.training_results.coefficients["lrom"][0]
)
ws3_lrom_coefficients = np.asarray(
    ws3_emulator.testing_results.coefficients["lrom"][0]
)
ws3_coordinate_data = (
    ("LS", ws3_ls_train_coefficients, ws3_ls_coefficients, "blue", "a"),
    ("LROM", ws3_lrom_train_coefficients, ws3_lrom_coefficients, "#E6AB02", "a"),
    ("ROSE", ws3_rose_coeff_train, ws3_rose_coefficients, "red", "c"),
)
rv_normalization = plt.Normalize(
    ws3_test_rows[:, 1].min(), ws3_test_rows[:, 1].max()
)
fig, axes = plt.subplots(3, BASIS_SIZE, figsize=(14.0, 8.4), sharex=True)
for method_row, (method, training_coordinates, testing_coordinates, color, symbol) in enumerate(ws3_coordinate_data):
    for coordinate_index in range(BASIS_SIZE):
        ax = axes[method_row, coordinate_index]
        ax.scatter(
            ws3_train_rows[:, 0],
            np.real(training_coordinates[:, coordinate_index]),
            facecolors="none",
            edgecolors="black",
            s=22,
            label="training" if coordinate_index == 0 else None,
        )
        test_scatter = ax.scatter(
            ws3_test_rows[:, 0],
            np.real(testing_coordinates[:, coordinate_index]),
            c=ws3_test_rows[:, 1],
            cmap="viridis",
            norm=rv_normalization,
            marker="s",
            s=22,
            alpha=0.78,
        )
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
        ax.set_ylabel(fr"Re(${symbol}_{coordinate_index + 1}$)")
        ax.set_title(f"{method} coordinate {coordinate_index + 1}", color=color)
        if method_row == 2:
            ax.set_xlabel("Vv [MeV]")
axes[0, 0].legend(fontsize=8)
fig.colorbar(test_scatter, ax=axes, label="Rv [fm]")
fig.suptitle("Matched ws_3 coordinates; testing points colored by Rv")
plt.show()
```

- [ ] **Step 6: Apply the fixed method colors to retained aggregate figures**

In the Vv error-family loop, use:

```python
for method, color in (
    ("rose", "red"),
    ("lrom", "#E6AB02"),
    ("ls", "blue"),
):
```

Set its title to `"Vv-only testing errors: ROSE, LROM, and LS"`. In the final
violin cell use:

```python
colors = ("blue", "#E6AB02", "red")
```

- [ ] **Step 7: Run focused tests and commit**

```bash
python -m pytest -q tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
python -m ruff check tools/generate_notebook01.py tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
git add tests/test_notebook01_lrom_flow.py tools/generate_notebook01.py
git commit -m "Combine notebook 01 coordinate figures"
git push origin main
```

Expected: tests and Ruff PASS before commit.

---

### Task 4: Replace Single Cases with Three Method-Neutral Matched Cases

**Files:**
- Modify: `tests/test_notebook01_lrom_flow.py`
- Modify: `tools/generate_notebook01.py`

**Interfaces:**
- Consumes: ordered FOM, LS, LROM, ROSE testing arrays and training/testing parameter rows.
- Produces: `vv_selected_indices: list[int]` and `ws3_selected_indices: list[int]`, each containing three distinct aligned noncentral testing indices.

- [ ] **Step 1: Replace representative-case tests with the selection contract**

Replace `test_vv_central_testing_wavefunction_compares_all_methods` with:

```python
def test_vv_selects_three_aligned_noncentral_cases_without_method_bias() -> None:
    text = source()

    assert "vv_normalized_distance >= 0.25" in text
    assert "vv_training_overlap" in text
    assert "vv_method_ranks" in text
    assert "vv_combined_difficulty = vv_method_ranks.mean(axis=1)" in text
    assert "vv_difficulty_targets = np.array([0.25, 0.50, 0.75])" in text
    assert "assert len(vv_selected_indices) == 3" in text
    assert "assert not vv_training_overlap[selected_index]" in text
    assert "vv_representative_index" not in text
```

Add:

```python
def test_ws3_selects_three_aligned_noncentral_cases_without_method_bias() -> None:
    text = source()

    assert "ws3_normalized_distance >= 0.25" in text
    assert "ws3_training_overlap" in text
    assert "ws3_method_ranks" in text
    assert "ws3_combined_difficulty = ws3_method_ranks.mean(axis=1)" in text
    assert "ws3_difficulty_targets = np.array([0.25, 0.50, 0.75])" in text
    assert "assert len(ws3_selected_indices) == 3" in text
    assert "assert not ws3_training_overlap[selected_index]" in text
    assert "representative_index" not in text
```

Update physical-diagnostic assertions to index the selected arrays rather than `representative_index`.

- [ ] **Step 2: Run the two selector tests to verify RED**

```bash
python -m pytest -q tests/test_notebook01_lrom_flow.py::test_vv_selects_three_aligned_noncentral_cases_without_method_bias tests/test_notebook01_lrom_flow.py::test_ws3_selects_three_aligned_noncentral_cases_without_method_bias
```

Expected: FAIL because the current selectors choose one case.

- [ ] **Step 3: Implement the `Vv` candidate mask, equal-rank score, and three unique selections**

Replace the single Vv selector with:

```python
# Build one error row for each shared testing parameter row.
vv_lrom_relative_l2 = lrom.relative_l2(
    prediction=vv_emulator.testing_results.lrom[0],
    reference=vv_fom_test,
)
vv_rose_relative_l2 = lrom.relative_l2(
    prediction=vv_rose_wavefunctions,
    reference=vv_fom_test,
)
vv_error_table = np.column_stack(
    [vv_ls_relative_l2, vv_lrom_relative_l2, vv_rose_relative_l2]
)

# Exclude near-reference cases and any exact training/testing overlap.
vv_test_rows = vv_emulator.samples.design.testing.values
vv_train_rows = vv_emulator.samples.design.training.values
vv_half_range = 0.5 * (
    vv_testing_ranges["Vv"][1] - vv_testing_ranges["Vv"][0]
)
vv_normalized_distance = np.abs(vv_test_rows[:, 0] - Vv0) / vv_half_range
vv_training_overlap = np.isclose(
    vv_test_rows[:, None, :],
    vv_train_rows[None, :, :],
    rtol=0.0,
    atol=1e-12,
).all(axis=2).any(axis=1)
vv_candidate_mask = (vv_normalized_distance >= 0.25) & ~vv_training_overlap
vv_candidate_indices = np.flatnonzero(vv_candidate_mask)

# Rank each method independently so no method's raw error scale dominates.
vv_candidate_errors = vv_error_table[vv_candidate_indices]
vv_method_ranks = np.empty_like(vv_candidate_errors, dtype=float)
for method_column in range(vv_candidate_errors.shape[1]):
    method_order = np.argsort(
        vv_candidate_errors[:, method_column], kind="stable"
    )
    vv_method_ranks[method_order, method_column] = np.linspace(
        0.0, 1.0, len(method_order)
    )
vv_combined_difficulty = vv_method_ranks.mean(axis=1)

# Select distinct lower, median, and upper combined-difficulty cases.
vv_difficulty_targets = np.array([0.25, 0.50, 0.75])
vv_selected_indices = []
for target in vv_difficulty_targets:
    for local_index in np.argsort(
        np.abs(vv_combined_difficulty - target), kind="stable"
    ):
        selected_index = int(vv_candidate_indices[local_index])
        if selected_index not in vv_selected_indices:
            vv_selected_indices.append(selected_index)
            break
assert len(vv_selected_indices) == 3
```

- [ ] **Step 4: Add Vv row-alignment assertions and the 2-by-3 figure**

```python
vv_case_labels = ("lower", "median", "upper")
fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), sharex=True)
for column, (difficulty_label, selected_index) in enumerate(
    zip(vv_case_labels, vv_selected_indices)
):
    case_id = vv_emulator.samples.design.testing.case_ids[selected_index]
    selected_case = vv_emulator.testing_case(case_id=case_id)
    selected_row = np.array(
        [selected_case.parameters[name] for name in vv_emulator.parameter_names]
    )
    assert np.allclose(selected_row, vv_test_rows[selected_index], rtol=0.0, atol=1e-12)
    assert vv_normalized_distance[selected_index] >= 0.25
    assert not vv_training_overlap[selected_index]
    print(
        difficulty_label,
        case_id,
        dict(selected_case.parameters),
        dict(zip(("LS", "LROM", "ROSE"), vv_error_table[selected_index])),
    )
    axes[0, column].plot(selected_case.radius, np.real(selected_case.high_fidelity[0]), color="black", label="high fidelity")
    axes[0, column].plot(selected_case.radius, np.real(vv_ls_wavefunctions[selected_index]), color="blue", label="LS")
    axes[0, column].plot(selected_case.radius, np.real(selected_case.lrom[0]), color="#E6AB02", linestyle=":", linewidth=2, label="LROM")
    axes[0, column].plot(selected_case.radius, np.real(vv_rose_wavefunctions[selected_index]), color="red", linestyle="--", label="ROSE")
    axes[0, column].set_title(f"{difficulty_label}: {case_id}")
    axes[1, column].plot(selected_case.radius, np.maximum(np.abs(selected_case.high_fidelity[0] - vv_ls_wavefunctions[selected_index]), DISPLAY_ERROR_FLOOR), color="blue")
    axes[1, column].plot(selected_case.radius, np.maximum(np.abs(selected_case.high_fidelity[0] - selected_case.lrom[0]), DISPLAY_ERROR_FLOOR), color="#E6AB02")
    axes[1, column].plot(selected_case.radius, np.maximum(np.abs(selected_case.high_fidelity[0] - vv_rose_wavefunctions[selected_index]), DISPLAY_ERROR_FLOOR), color="red")
    axes[1, column].set_yscale("log")
    axes[1, column].set_xlabel("r [fm]")
axes[0, 0].set_ylabel("Re(phi)")
axes[1, 0].set_ylabel("absolute difference")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Three aligned noncentral Vv testing cases")
fig.tight_layout()
plt.show()
```

- [ ] **Step 5: Implement the complete `ws_3` rank methodology**

```python
ws3_error_table = np.column_stack(
    [ws3_ls_rel_test, lrom_relative, ws3_rose_rel_test]
)
ws3_center_row = np.array(
    [ws3_center[name] for name in ws3_emulator.parameter_names]
)
ws3_testing_half_ranges = np.array(
    [
        0.5 * (ws3_testing_ranges[name][1] - ws3_testing_ranges[name][0])
        for name in ws3_emulator.parameter_names
    ]
)
ws3_normalized_distance = np.linalg.norm(
    (ws3_test_rows - ws3_center_row) / ws3_testing_half_ranges,
    axis=1,
)
ws3_training_overlap = np.isclose(
    ws3_test_rows[:, None, :],
    ws3_train_rows[None, :, :],
    rtol=0.0,
    atol=1e-12,
).all(axis=2).any(axis=1)
ws3_candidate_mask = (ws3_normalized_distance >= 0.25) & ~ws3_training_overlap
ws3_candidate_indices = np.flatnonzero(ws3_candidate_mask)
ws3_candidate_errors = ws3_error_table[ws3_candidate_indices]
ws3_method_ranks = np.empty_like(ws3_candidate_errors, dtype=float)
for method_column in range(ws3_candidate_errors.shape[1]):
    method_order = np.argsort(
        ws3_candidate_errors[:, method_column], kind="stable"
    )
    ws3_method_ranks[method_order, method_column] = np.linspace(
        0.0, 1.0, len(method_order)
    )
ws3_combined_difficulty = ws3_method_ranks.mean(axis=1)
ws3_difficulty_targets = np.array([0.25, 0.50, 0.75])
ws3_selected_indices = []
for target in ws3_difficulty_targets:
    for local_index in np.argsort(
        np.abs(ws3_combined_difficulty - target), kind="stable"
    ):
        selected_index = int(ws3_candidate_indices[local_index])
        if selected_index not in ws3_selected_indices:
            ws3_selected_indices.append(selected_index)
            break
assert len(ws3_selected_indices) == 3
```

Render the `ws_3` cases with this complete block:

```python
ws3_case_labels = ("lower", "median", "upper")
fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), sharex=True)
for column, (difficulty_label, selected_index) in enumerate(
    zip(ws3_case_labels, ws3_selected_indices)
):
    case_id = ws3_emulator.samples.design.testing.case_ids[selected_index]
    selected_case = ws3_emulator.testing_case(case_id=case_id)
    selected_row = np.array(
        [selected_case.parameters[name] for name in ws3_emulator.parameter_names]
    )
    assert np.allclose(
        selected_row,
        ws3_test_rows[selected_index],
        rtol=0.0,
        atol=1e-12,
    )
    assert ws3_normalized_distance[selected_index] >= 0.25
    assert not ws3_training_overlap[selected_index]
    print(
        difficulty_label,
        case_id,
        dict(selected_case.parameters),
        dict(zip(("LS", "LROM", "ROSE"), ws3_error_table[selected_index])),
    )
    axes[0, column].plot(
        selected_case.radius,
        np.real(selected_case.high_fidelity[0]),
        color="black",
        label="high fidelity",
    )
    axes[0, column].plot(
        selected_case.radius,
        np.real(ws3_ls_wf_test[selected_index]),
        color="blue",
        label="LS",
    )
    axes[0, column].plot(
        selected_case.radius,
        np.real(selected_case.lrom[0]),
        color="#E6AB02",
        linestyle=":",
        linewidth=2,
        label="LROM",
    )
    axes[0, column].plot(
        selected_case.radius,
        np.real(ws3_rose_wf_test[selected_index]),
        color="red",
        linestyle="--",
        label="ROSE",
    )
    axes[0, column].set_title(f"{difficulty_label}: {case_id}")
    axes[1, column].plot(
        selected_case.radius,
        np.maximum(
            np.abs(
                selected_case.high_fidelity[0]
                - ws3_ls_wf_test[selected_index]
            ),
            DISPLAY_ERROR_FLOOR,
        ),
        color="blue",
    )
    axes[1, column].plot(
        selected_case.radius,
        np.maximum(
            np.abs(selected_case.high_fidelity[0] - selected_case.lrom[0]),
            DISPLAY_ERROR_FLOOR,
        ),
        color="#E6AB02",
    )
    axes[1, column].plot(
        selected_case.radius,
        np.maximum(
            np.abs(
                selected_case.high_fidelity[0]
                - ws3_rose_wf_test[selected_index]
            ),
            DISPLAY_ERROR_FLOOR,
        ),
        color="red",
    )
    axes[1, column].set_yscale("log")
    axes[1, column].set_xlabel("r [fm]")
axes[0, 0].set_ylabel("Re(phi)")
axes[1, 0].set_ylabel("absolute difference")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Three aligned noncentral ws_3 testing cases")
fig.tight_layout()
plt.show()
```

- [ ] **Step 6: Run selector and full source-contract tests**

```bash
python -m pytest -q tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py tests/test_notebook01_rose_diagnostic.py
python -m ruff check tools/generate_notebook01.py tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
```

Expected: all tests PASS; Ruff reports `All checks passed!`.

- [ ] **Step 7: Commit and push the matched-case milestone**

```bash
git add tests/test_notebook01_lrom_flow.py tools/generate_notebook01.py
git commit -m "Select matched notebook 01 comparison cases"
git push origin main
```

---

### Task 5: Regenerate, Execute, Inspect, Document, and Hand Off

**Files:**
- Modify: `tests/test_notebook01_generation.py`
- Modify: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`
- Modify: `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify last: `../_memory/Daily Notes/2026-07-21.md`
- Modify last: `../_memory/Context/active-state.md`

**Interfaces:**
- Consumes: final `generate_notebook01.notebook_cells()` and the three-case selectors.
- Produces: source-synchronized executed notebook, recorded methodology, visual QA, and current memory handoff.

- [ ] **Step 1: Add a checked-in notebook/generator source-sync test**

Add to `tests/test_notebook01_generation.py`:

```python
def test_checked_in_notebook_source_matches_generator() -> None:
    import nbformat

    notebook = nbformat.read(
        ROOT / "notebooks" / "01_rbm_vs_lrom_single_wavefunction.ipynb",
        as_version=4,
    )
    generated = generate_notebook01.notebook_cells()
    assert [cell.cell_type for cell in notebook.cells] == [
        cell["cell_type"] for cell in generated
    ]
    assert [cell.source for cell in notebook.cells] == [
        cell["source"] for cell in generated
    ]
```

- [ ] **Step 2: Run the sync test to verify RED before regeneration**

```bash
python -m pytest -q tests/test_notebook01_generation.py::test_checked_in_notebook_source_matches_generator
```

Expected: FAIL because the user's current notebook and newly changed generator have not yet been regenerated together.

- [ ] **Step 3: Regenerate Notebook 01 from the authoritative generator**

```bash
python tools/generate_notebook01.py
```

Then verify the user's comment intent survived:

```bash
rg -n "FOM parameter rows|EIM interaction|free solution|reduced basis|optional LS floor" notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
```

Expected: every phrase appears in the regenerated notebook source.

- [ ] **Step 4: Run the sync and structural tests to verify GREEN**

```bash
python -m pytest -q tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py tests/test_notebook01_rose_diagnostic.py
```

Expected: PASS.

- [ ] **Step 5: Execute Notebook 01 end to end**

```bash
MPLCONFIGDIR=/private/tmp/lrom-notebook01-mpl python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
```

Expected: exit code zero and the notebook is written in place.

- [ ] **Step 6: Audit execution, selected cases, and array alignment**

```bash
python - <<'PY'
from pathlib import Path
import nbformat

path = Path("notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb")
notebook = nbformat.read(path, as_version=4)
code = [cell for cell in notebook.cells if cell.cell_type == "code"]
errors = [
    output
    for cell in code
    for output in cell.get("outputs", [])
    if output.output_type == "error"
]
assert all(cell.execution_count is not None for cell in code)
assert not errors
for cell in code:
    for output in cell.get("outputs", []):
        if output.output_type == "stream" and any(
            word in output.text for word in ("lower", "median", "upper")
        ):
            print(output.text)
print(f"executed={len(code)}, errors={len(errors)}")
PY
```

Expected: every code cell executed, zero errors, and six selected-case records print with noncentral parameters and all three method errors.

- [ ] **Step 7: Render and visually inspect every retained figure**

```bash
mkdir -p /private/tmp/lrom-notebook01-qa
python -m jupyter nbconvert --to markdown --output-dir=/private/tmp/lrom-notebook01-qa notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
find /private/tmp/lrom-notebook01-qa -maxdepth 2 -type f -name '*.png' -print
```

Inspect each PNG. Confirm:

- the two coordinate figures have three method rows and four coordinate columns;
- `a_j` labels appear only for LS/LROM and `c_j` only for ROSE;
- the `ws_3` coordinate figure has one readable `Rv` colorbar;
- both matched-case figures show three distinct noncentral cases;
- method colors are LS blue, LROM yellow, ROSE red;
- no singular spectrum, coordinate-difference, or standalone joint-variation figure remains;
- all spatial axes use `r [fm]`.

- [ ] **Step 8: Document the methodology in the architecture guide**

Add this section to `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`:

```markdown
## Notebook comparison-case selection

Notebook 01 aligns high-fidelity, LS, LROM, and ROSE outputs by the shared testing-row index. It excludes central, near-central, and exact training-overlap rows. Each method's relative L2 errors are ranked independently, the three ranks are averaged, and the notebook displays distinct cases nearest the lower-quartile, median, and upper-quartile combined difficulty. This prevents one method's error scale or one almost-central case from controlling the visual comparison.
```

- [ ] **Step 9: Run final verification**

```bash
python -m pytest -q
python -m ruff check lrom lrom_legacy tests tools
python -m ruff check notebooks --ignore E402
git diff --check
```

Expected: full pytest PASS, both Ruff commands report `All checks passed!`, and `git diff --check` prints nothing.

- [ ] **Step 10: Commit and push the executed notebook milestone**

```bash
git add notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb tests/test_notebook01_generation.py docs/LROM_ARCHITECTURE_UNDERSTANDING.md
git commit -m "Execute the focused notebook 01 comparison"
git push origin main
git rev-parse HEAD
git rev-parse origin/main
```

Expected: local and remote hashes match.

- [ ] **Step 11: Update workspace memory last**

Append `[LROM_Project]` entries to `../_memory/Daily Notes/2026-07-21.md` and update the LROM checklist in `../_memory/Context/active-state.md` with:

- preserved user edits;
- removed figures;
- selected six case IDs and parameter/error tables;
- notebook execution count and visual QA result;
- test/lint results;
- final local/remote commit hash;
- reminder that prose ownership remains with the user.

Do not store runtime arrays or generated images in `_memory/`.
