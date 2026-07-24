# Notebook 02 Cross-Section Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Correct Notebook 02 and benchmark 03 so ROSE, LROM, LS, and the high-fidelity reference use the same authoritative snapshots and all retained (l=0,1,2,3) channels.

**Architecture:** Keep the parked `lrom_legacy.v2_0` package unchanged. In each notebook, construct notebook-local ROSE `CustomBasis` objects from the exact LROM training wavefunctions while using ROSE's free reference, then build the ROSE reduced emulator over the notebook-owned EIM interaction. Replace ROSE's truncating observable methods with explicit all-channel S-matrix helpers and retain the existing LROM observable path.

**Tech Stack:** Python 3.12, Jupyter Notebook JSON, NumPy, SciPy, Numba, ROSE, parked `lrom_legacy.v2_0`, pytest.

## Global Constraints

- Retain (^{40}\mathrm{Ca}(n,n)), 14.1 MeV laboratory energy, (l=0,1,2,3), 20% parameter ranges, 600-point mesh, and angles 1--179 degrees.
- Keep Notebook 02 sparse: do not add explanatory Markdown beyond the existing headings.
- Keep `lrom_legacy.v2_0`, public `lrom`, Notebook 01, and scientific archives unchanged.
- Use exact shared ordered training/testing rows; held-out rows never train ROSE bases or EIM.
- Use ROSE free-reference bases and LROM central-reference bases; never compare their coefficient values as if they had the same origin.
- Do not call ROSE `exact_dsdo()` or `emulate_dsdo()` for the four-channel observable because those helpers omit the final constructed channel at `l_max=3`.
- Replace representative labels with `alpha selection A`, `alpha selection B`, and `alpha selection C`; include their exact alpha rows.
- Use `apply_patch` for source/test edits and commit each coherent task.

---

### Task 1: Lock the corrected notebook contract with failing tests

**Files:**
- Modify: `tests/test_notebook02_cross_sections.py`
- Test: `tests/test_notebook02_cross_sections.py`

**Interfaces:**
- Tests inspect code-source strings from both notebooks; no production API changes are introduced.

- [ ] **Step 1: Write the failing assertions**

Add assertions requiring:

```python
assert "rose.basis.CustomBasis(" in text
assert "solutions=np.asarray(" in text
assert "phi_0=np.asarray(" in text
assert "def exact_smatrix_all_channels" in text
assert "def emulated_smatrix_all_channels" in text
assert "exact_dsdo" not in text
assert "emulate_dsdo" not in text
assert "alpha selection A" in text
assert "alpha selection B" in text
assert "alpha selection C" in text
assert "percentile" not in text.lower()
```

Apply the corresponding assertions to benchmark 03. Add a structural check that each observable helper creates `len(sae.rbes)` S-matrix entries and that the validation table checks `len(partial_waves)` against `L_MAX + 1`.

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run:

```bash
pytest -q tests/test_notebook02_cross_sections.py
```

Expected: FAIL because the existing notebooks still call `from_train()`, `exact_dsdo()`, `emulate_dsdo()`, and use percentile labels.

- [ ] **Step 3: Commit the red test contract**

```bash
git add tests/test_notebook02_cross_sections.py
git commit -m "test: lock Notebook 02 all-channel ROSE correction"
```

### Task 2: Add shared-snapshot ROSE construction and all-channel observables

**Files:**
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb` code cells 9 and 11
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb` matching code cells

**Interfaces:**
- `build_rose_emulator(interaction, n_phi, solver, training_rows)` returns a `rose.ScatteringAmplitudeEmulator` with one `CustomBasis` row for every retained channel.
- `exact_smatrix_all_channels(sae, row)` returns `(splus, sminus)` arrays with shape `(L_MAX + 1,)`.
- `emulated_smatrix_all_channels(sae, row)` returns the same shape using each RBE's reduced `S_matrix_element`.
- `cross_section_from_smatrix(sae, row, splus, sminus)` returns the `.dsdo` array on `ANGLES_RAD`.

- [ ] **Step 1: Implement the notebook-local basis builder**

For every `ell` and spin row, construct:

```python
basis = rose.basis.CustomBasis(
    solutions=np.asarray(emulator.samples.training_wavefunctions[key], dtype=np.complex128).T.copy(),
    phi_0=np.asarray(
        [rose.free_solutions.phi_free(float(s), ell, emulator.kinematics.eta) for s in rho_mesh],
        dtype=np.complex128,
    ),
    rho_mesh=rho_mesh,
    n_basis=n_phi,
    solver=emulator.samples.full_order_models[key].solver,
    subtract_phi0=True,
    use_svd=True,
    center=False,
    scale=False,
)
```

Pass the resulting nested basis list to `rose.ScatteringAmplitudeEmulator`, preserving the existing interaction, angle, `s_0`, and tolerance settings.

- [ ] **Step 2: Implement explicit S-matrix and cross-section helpers**

Use the solver attached to each basis for exact values and each RBE for emulated values:

```python
def exact_smatrix_all_channels(sae, row):
    splus = np.empty(len(sae.rbes), dtype=np.complex128)
    sminus = np.empty_like(splus)
    for ell, rbe_row in enumerate(sae.rbes):
        splus[ell] = rbe_row[0].basis.solver.smatrix(row)
        sminus[ell] = splus[ell] if ell == 0 else rbe_row[1].basis.solver.smatrix(row)
    return splus, sminus


def emulated_smatrix_all_channels(sae, row):
    splus = np.empty(len(sae.rbes), dtype=np.complex128)
    sminus = np.empty_like(splus)
    for ell, rbe_row in enumerate(sae.rbes):
        splus[ell] = rbe_row[0].S_matrix_element(row)
        sminus[ell] = splus[ell] if ell == 0 else rbe_row[1].S_matrix_element(row)
    return splus, sminus


def cross_section_from_smatrix(sae, row, splus, sminus):
    return sae.calculate_xs(splus, sminus, row, angles=ANGLES_RAD).dsdo
```

Keep the channel ordering explicit: row 0 is (j=l+1/2), row 1 is (j=l-1/2) for (l>0).

- [ ] **Step 3: Replace the reference and ROSE result paths**

Construct ROSE bases from shared LROM snapshots for every `(n_phi, n_u)` pair. Generate `fom_train_xs` and `fom_test_xs` with `exact_smatrix_all_channels` and `cross_section_from_smatrix`. Generate ROSE results with `emulated_smatrix_all_channels` and the same cross-section helper. Replace `minimum_rose_times()` so it times that same explicit path.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
pytest -q tests/test_notebook02_cross_sections.py
```

Expected: the scientific-core string contract passes; execution tests remain pending until the notebook outputs are regenerated.

- [ ] **Step 5: Commit the notebook observable correction**

```bash
git add notebooks/02_rose_vs_lrom_cross_sections.ipynb notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
git commit -m "fix: align Notebook 02 ROSE cross-section channels"
```

### Task 3: Replace representative selection language with alpha selections

**Files:**
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb` code cell 11
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb` matching code cell

**Interfaces:**
- `selected_indices` remains an ordered list of three testing-row indices.
- `selected_labels` becomes `("alpha selection A", "alpha selection B", "alpha selection C")`.
- `alpha_selection_table` contains `selection`, `case_id`, and every parameter column.

- [ ] **Step 1: Replace the selection algorithm and labels**

Use the approved interior anchor rule:

```python
selection_positions = np.linspace(0, len(ordered_combined) - 1, 5, dtype=int)[1:4]
selected_indices = [int(ordered_combined[position]) for position in selection_positions]
assert len(set(selected_indices)) == 3
selected_labels = ("alpha selection A", "alpha selection B", "alpha selection C")
alpha_selection_table = pd.DataFrame(
    [
        {"selection": label, "case_id": test_ids[index], **{
            name: float(test_rows[index, column])
            for column, name in enumerate(parameter_names)
        }}
        for label, index in zip(selected_labels, selected_indices)
    ]
)
```

Display the table before the representative plots. Do not use the banned terminology in source, titles, or labels.

- [ ] **Step 2: Run the focused structural tests**

```bash
pytest -q tests/test_notebook02_cross_sections.py
```

Expected: PASS for all source-level alpha-selection and terminology assertions.

- [ ] **Step 3: Commit the selection-language change**

```bash
git add notebooks/02_rose_vs_lrom_cross_sections.ipynb notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb tests/test_notebook02_cross_sections.py
git commit -m "fix: label Notebook 02 cases as alpha selections"
```

### Task 4: Regenerate and validate Notebook 02 outputs

**Files:**
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb` outputs and execution counts
- Modify: `tests/test_notebook02_cross_sections.py` only if a verified output contract is missing

**Interfaces:**
- Notebook 02 must execute all code cells with no error outputs.
- `fom_train_xs`, `fom_test_xs`, `rose_results`, `ls_results`, and `lrom_results` retain their existing shapes and grid keys.

- [ ] **Step 1: Execute Notebook 02 at the full 200/100 profile**

Run:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_rose_vs_lrom_cross_sections.ipynb
```

Preserve the notebook's stored outputs and confirm no cell error output.

- [ ] **Step 2: Validate the numerical invariants**

Check exact shapes, finite/nonnegative cross sections, no train/test row overlap, complete four-channel S-matrix arrays, shared alpha-selection indices, and equality of `train_rows`/`test_rows` between LROM and ROSE.

- [ ] **Step 3: Inspect all corrected figures**

Inspect potential/predictor rainbows, representative curves/errors, alpha-selection table, violin distributions, grid summary, CAT plot, and validation table. Confirm physical (r) labels, no spin-orbit origin singularity, readable legends, and no truncation at (l=3).

- [ ] **Step 4: Commit regenerated Notebook 02 outputs**

```bash
git add notebooks/02_rose_vs_lrom_cross_sections.ipynb
git commit -m "test: regenerate corrected Notebook 02 results"
```

### Task 5: Validate benchmark 03 in reduced and full profiles

**Files:**
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb` outputs and execution counts

**Interfaces:**
- Default reduced profile remains 120/30.
- `LROM_BENCHMARK_PROFILE=full` remains 200/100.
- Corrected benchmark summary keys and shapes match Notebook 02 outside timing noise.

- [ ] **Step 1: Execute reduced benchmark mode**

Run:

```bash
LROM_BENCHMARK_PROFILE=reduced jupyter nbconvert --to notebook --execute --inplace notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
```

Confirm no error outputs and the reduced profile remains 120/30.

- [ ] **Step 2: Execute full benchmark mode**

Run:

```bash
LROM_BENCHMARK_PROFILE=full jupyter nbconvert --to notebook --execute --inplace notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
```

Confirm no error outputs and the full profile remains 200/100.

- [ ] **Step 3: Compare scientific summaries**

Compare the corrected benchmark and Notebook 02 for shared rows, grid keys, alpha-selection construction, four-channel reference arrays, and error summaries. Timing differences are expected; scientific values should agree within numerical tolerance.

- [ ] **Step 4: Commit regenerated benchmark outputs**

```bash
git add notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
git commit -m "test: regenerate corrected Notebook 02 benchmark"
```

### Task 6: Final verification and handoff

**Files:**
- Test: `tests/test_notebook02_cross_sections.py`
- Verify: both corrected notebooks and repository status

- [ ] **Step 1: Run focused tests**

```bash
pytest -q tests/test_notebook02_cross_sections.py
```

- [ ] **Step 2: Run the full available test suite**

```bash
pytest -q
```

Record any pre-existing parent-path failures separately from corrected-notebook failures.

- [ ] **Step 3: Verify protected files and git status**

Confirm no changes to `lrom/__init__.py`, `lrom_legacy/v1_2`, `lrom_legacy/v2_0`, Notebook 01, or scientific archives. Run `git diff --check` and `git status --short`.

- [ ] **Step 4: Commit final tests if needed and report evidence**

Report notebook execution counts, figure inspection, focused/full test results, corrected representative alpha IDs, and the local commit list. Do not push without explicit authorization.
