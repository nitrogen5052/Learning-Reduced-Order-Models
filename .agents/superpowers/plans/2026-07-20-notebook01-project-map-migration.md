# Notebook 01 Project-Map Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Notebook 01 to the simplified v1.2 interfaces while preserving and restoring the Paper Results Map scientific narrative.

**Architecture:** `tools/generate_notebook01.py` remains the only source. The notebook calls public `lrom`, calculates LS explicitly, and constructs ROSE EIM explicitly. Scientific plots remain inline and visible.

**Tech Stack:** Python, nbformat, Jupyter nbconvert, NumPy, pandas, Matplotlib, nuclear-rose, pytest.

## Global Constraints

- Do not redesign the two-act Notebook 01 narrative.
- Do not remove project-map sections or figures to reduce code.
- Edit the generator, never notebook JSON directly.
- Keep LROM orange/yellow, LS blue, and ROSE red.
- Show singular-value decay beside every basis figure.
- Use physical radius in fm and mesh 800 in both sections.
- Keep raw arrays and methodology visible; no plotting wrappers.
- Preserve existing numerical outputs unless the task names a scientific correction.
- Commit and push every completed task to `main`.

---

### Task 1: Strengthen the Notebook 01 project-map contract

**Files:**
- Modify: `tests/test_notebook01_generation.py`
- Modify: `tests/test_notebook01_lrom_flow.py`

**Interfaces:**
- Consumes: generator cell source.
- Produces: structural requirements that prevent accidental notebook redesign.

- [ ] **Step 1: Add preservation assertions for the current approved structure**

Add assertions for:

```python
assert "## Section 1. Parameter Varying Vv" in source
assert "## Section 2. Three-Parameter" in source
assert "## Section 3. Three-Parameter Wavefunction" in source
assert "Vv training potentials" in source
assert "High-fidelity training solutions" in source
assert source.count("normalized singular value") >= 4
assert "LROM central-reference basis" in source
assert "ROSE free-reference basis" in source
assert "potential predictors" in source
assert '("ls", "blue")' in source
assert '("lrom", "orange")' in source
assert '("rose", "red")' in source
assert "def plot_" not in source
```

Count required narrative sections and ensure basis/spectrum figures remain paired.

- [ ] **Step 2: Run the focused tests to verify the preservation lock**

Run: `python -m pytest -q tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py`

Expected: PASS against the currently approved notebook structure.

- [ ] **Step 3: Commit the preservation contract**

```bash
git add tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
git commit -m "Lock notebook 01 project-map structure"
git push origin main
```

### Task 2: Migrate package calls without changing the notebook story

**Files:**
- Modify: `tools/generate_notebook01.py`
- Modify: `tests/test_notebook01_generation.py`
- Modify: `tests/test_notebook01_lrom_flow.py`

**Interfaces:**
- Consumes: public v1.2 `LROM`, explicit `least_squares_baseline`, stored LROM-only results.
- Produces: the same Vv and ws_3 samples and LROM predictions through the simplified interface.

- [ ] **Step 1: Add failing interface-migration assertions**

```python
assert "import lrom" in setup
assert "lrom_legacy.v1_2" not in setup
assert 'high_fidelity_solver="runge_kutta"' in source
assert "eim_basis_size" not in source
assert "least_squares_baseline" in source
```

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `python -m pytest -q tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py`

Expected: FAIL on the old import, EIM argument, and implicit LS access.

- [ ] **Step 3: Switch to the public active package**

Use:

```python
import lrom
```

Keep the version print. Both sampling calls use:

```python
high_fidelity_solver="runge_kutta"
```

and no EIM argument. Set both meshes to 800.

- [ ] **Step 4: Calculate LS visibly after training**

For each study:

```python
ls_coordinates, ls_wavefunctions = lrom.least_squares_baseline(
    basis=emulator.basis[0],
    wavefunctions=emulator.samples.testing_wavefunctions[0],
)
ls_relative_l2 = lrom.relative_l2(
    prediction=ls_wavefunctions,
    reference=emulator.samples.testing_wavefunctions[0],
)
```

Use stored LROM results or explicit `predict()` output for LROM arrays. Do not reconstruct an LS baseline inside the package object.

- [ ] **Step 5: Preserve the save/load cell**

Keep `save`, `load`, and the portable prediction call unchanged except for the public import.

- [ ] **Step 6: Run structural tests**

Run: `python -m pytest -q tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py`

Expected: PASS. Tests for later restored details are added only in their own tasks.

- [ ] **Step 7: Commit and push**

```bash
git add tools/generate_notebook01.py tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
git commit -m "Migrate notebook 01 to explicit v1.2 analysis"
git push origin main
```

### Task 3: Move ROSE EIM construction into the notebook

**Files:**
- Modify: `tools/generate_notebook01.py`
- Modify: `tests/test_notebook01_lrom_flow.py`
- Modify: `.agents/validation/notebook01_rose_reference_diagnostic.py`

**Interfaces:**
- Consumes: package FOM kinematics, parameter bounds, exact potential callback, public ROSE API.
- Produces: notebook-owned `InteractionEIMSpace` and free-reference ROSE RBE.

- [ ] **Step 1: Add a failing ownership assertion**

```python
assert "rose.InteractionEIMSpace(" in source
assert ".full_order_model[0].interaction" not in source
assert "training_info=vv_rose_bounds" in source
assert "training_info=ws3_rose_bounds" in source
```

- [ ] **Step 2: Build Vv and ws_3 ROSE interactions explicitly**

For each study, form bounds from the central, training, and testing parameter rows to preserve current ROSE results:

```python
rose_values = np.vstack([central_row, training_rows, testing_rows])
rose_bounds = np.column_stack([rose_values.min(axis=0), rose_values.max(axis=0)])
rose_interactions = rose.InteractionEIMSpace(
    l_max=0,
    n_theta=len(parameter_names),
    mu=emulator.kinematics.mu,
    energy=emulator.kinematics.e_com,
    training_info=rose_bounds,
    n_basis=8,
    rho_mesh=emulator.samples.mesh.rho,
    coordinate_space_potential=notebook_potential,
    is_complex=False,
)
rose_interaction = rose_interactions.interactions[0][0]
```

Use this interaction only for the notebook-owned ROSE reduced emulator. Keep the corrected free-reference basis.

- [ ] **Step 3: Update the controlled diagnostic the same way**

The diagnostic must no longer depend on the package FOM carrying EIM state. Preserve its three case IDs and measured reference comparison.

- [ ] **Step 4: Run ROSE-focused tests**

Run: `python -m pytest -q tests/test_notebook01_lrom_flow.py tests/test_notebook01_rose_diagnostic.py`

Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add tools/generate_notebook01.py tests/test_notebook01_lrom_flow.py .agents/validation/notebook01_rose_reference_diagnostic.py
git commit -m "Make notebook 01 own its ROSE EIM"
git push origin main
```

### Task 4: Restore the Vv-only teaching details

**Files:**
- Modify: `tools/generate_notebook01.py`
- Modify: `tests/test_notebook01_lrom_flow.py`

**Interfaces:**
- Consumes: existing Vv data and map requirements.
- Produces: explicit fixed-parameter provenance, unchanged rainbows/basis evidence, and noncentral representative case.

- [ ] **Step 1: Add failing Vv-detail assertions**

```python
assert "Koning-Daelroche" not in source
assert "Koning-Delaroche" in source
assert "fixed Rv" in source
assert "fixed av" in source
assert "noncentral" in source
```

- [ ] **Step 2: Run the Vv-detail test to verify RED**

Run: `python -m pytest -q tests/test_notebook01_lrom_flow.py`

Expected: FAIL on missing provenance and the central representative selector.

- [ ] **Step 3: Add the KD provenance cell**

Print and explain:

```python
print("Vv varies over the requested ranges")
print("fixed Rv [fm]:", vv_center["Rv"])
print("fixed av [fm]:", vv_center["av"])
```

The preceding prose states these fixed values come from Koning-Delaroche global systematics for the selected target/projectile and 14.1 MeV laboratory energy.

- [ ] **Step 4: Choose a noncentral representative deterministically**

```python
candidate_indices = np.flatnonzero(vv_plot_mask)
vv_representative_index = candidate_indices[len(candidate_indices) // 2]
```

Label the case as noncentral and print its distance from `Vv0`. Do not reintroduce the central overlap into aggregate plots.

- [ ] **Step 5: Keep the project-map Vv figures**

Retain potential rainbow, wavefunction rainbow, LROM basis/spectrum, ROSE basis/spectrum, coordinate behavior, representative reproduction, and testing-error summary. Add plain one- or two-sentence narrative before each figure.

- [ ] **Step 6: Run focused tests and commit**

Run: `python -m pytest -q tests/test_notebook01_lrom_flow.py`

Expected: Vv project-map assertions PASS.

```bash
git add tools/generate_notebook01.py tests/test_notebook01_lrom_flow.py
git commit -m "Restore notebook 01 Vv teaching details"
git push origin main
```

### Task 5: Restore the three-parameter predictor narrative

**Files:**
- Modify: `tools/generate_notebook01.py`
- Modify: `tests/test_notebook01_lrom_flow.py`

**Interfaces:**
- Consumes: one ws_3 sampling state; parameter and potential predictor training paths.
- Produces: visible raw-parameter comparison, av variation, selected potential radii, and parameter-colored coefficient figures.

- [ ] **Step 1: Add failing three-parameter narrative assertions**

```python
assert "av variation" in source
assert "raw parameter predictors" in source
assert "potential predictors" in source
assert "colored by" in source
assert "case_number" not in source
```

- [ ] **Step 2: Run the three-parameter test to verify RED**

Run: `python -m pytest -q tests/test_notebook01_lrom_flow.py`

Expected: FAIL on the missing raw-parameter/av narrative and case-index plot.

- [ ] **Step 3: Show Vv, Rv, and av sample variation**

Add a three-panel parameter-distribution/rainbow figure that explicitly includes `av`. Keep potential and high-fidelity plots separate where required by the map.

- [ ] **Step 4: Train the raw-parameter diagnostic without rerunning FOM sampling**

```python
ws3_emulator.train(
    basis_size=BASIS_SIZE,
    predictor="parameters",
    predictor_count=3,
)
parameter_lrom = np.asarray(ws3_emulator.testing_results.lrom[0]).copy()
parameter_relative_l2 = np.asarray(
    ws3_emulator.testing_results.metrics["relative_l2"][0]["lrom"]
).copy()
```

Then retrain the same sampled object with six potential predictors and preserve those arrays separately. This changes learned models, not FOM samples.

- [ ] **Step 5: Replace case-index coefficient plots**

Create visible inline panels such as:

```python
scatter = ax.scatter(
    ws3_test_rows[:, x_index],
    np.real(coordinates[:, coefficient_index]),
    c=ws3_test_rows[:, color_index],
    cmap="viridis",
)
fig.colorbar(scatter, ax=ax, label=f"{names[color_index]}")
```

Include one panel per useful parameter pair, with legends/titles explaining which coordinate and color parameter are shown. ROSE-native coordinates remain separate from LROM/LS coordinates.

- [ ] **Step 6: Preserve performance summaries**

Keep the representative wavefunction, split violin, and interpolation/extrapolation table. Add raw-parameter LROM only where it explains the motivation; do not replace the main potential-predictor comparison.

- [ ] **Step 7: Run focused tests and commit**

Run: `python -m pytest -q tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py`

Expected: PASS.

```bash
git add tools/generate_notebook01.py tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
git commit -m "Restore notebook 01 predictor narrative"
git push origin main
```

### Task 6: Generate, execute, compare, and visually inspect Notebook 01

**Files:**
- Modify: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`
- Modify: `.agents/validation/2026-07-20-notebook01-rose-reference-results.md` only if controlled results change because of explicit notebook-owned EIM construction.

**Interfaces:**
- Consumes: completed generator.
- Produces: fully executed notebook satisfying the map and parity contracts.

- [ ] **Step 1: Generate the notebook**

Run: `python tools/generate_notebook01.py`

Expected: generator completes and writes the target notebook.

- [ ] **Step 2: Execute end to end**

Run: `MPLCONFIGDIR=/private/tmp/lrom-notebook01-mpl python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`

Expected: exit 0.

- [ ] **Step 3: Validate execution metadata**

Read the notebook with `json` or `nbformat`; assert every code cell has an execution count and no output has `output_type == "error"`.

- [ ] **Step 4: Compare numerical invariants**

Record existing potential-predictor LROM and corrected ROSE interpolation/extrapolation medians. Pure interface migration must match the pre-migration values within `rtol=1e-12`, `atol=1e-14`. Document any new raw-parameter diagnostic separately.

- [ ] **Step 5: Render and inspect every figure**

Extract PNG outputs to a temporary directory. Check physical radius labels, color laws, singular spectra, noncentral representative selection, av visibility, coefficient axes, violin medians, and readable legends.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest -q tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py tests/test_notebook01_rose_diagnostic.py`

Run: `python -m pytest -q`

Expected: PASS.

```bash
git add notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb .agents/validation/2026-07-20-notebook01-rose-reference-results.md
git commit -m "Execute notebook 01 on the simplified v1.2 package"
git push origin main
```

### Task 7: Update the user architecture guide and ownership handoff

**Files:**
- Modify: `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify: `tests/test_project_metadata.py`

**Interfaces:**
- Consumes: final notebook and package boundaries.
- Produces: exact references from notebook cells to package sections and pending prose checklist.

- [ ] **Step 1: Add a notebook-to-package map**

Map each public notebook call to the numbered package section, input arrays, output arrays, scientific assumption, and falsifying test.

- [ ] **Step 2: Add the change explanation**

Record Methodology, Before, After, Execution, and What did not change for LS separation, EIM ownership, and restored map content.

- [ ] **Step 3: Preserve the user ownership gate**

List the exact markdown cells and figure narratives added or changed. Mark final prose ownership pending the user's sentence-by-sentence review.

- [ ] **Step 4: Verify, commit, and push**

Run: `python -m pytest -q tests/test_project_metadata.py`

Expected: PASS.

```bash
git add docs/LROM_ARCHITECTURE_UNDERSTANDING.md tests/test_project_metadata.py
git commit -m "Map notebook 01 to the simplified package"
git push origin main
```
