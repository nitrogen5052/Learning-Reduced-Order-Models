# Benchmark 02 ROSE Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace only the ROSE construction and benchmarking blocks in `benchmark_02.ipynb` with independent, per-study held-out validation and EIM tuning modeled on the public ROSE tutorials.

**Architecture:** The existing LROM objects, parameter rows, predictors, scan diagnostics, and 16 figures remain unchanged. Each of the three physical studies builds three notebook-owned ROSE candidates at fixed `n_phi=4`, validates them on a deterministic ROSE-only Latin-hypercube set using exact ROSE wavefunctions, times their online evaluation, selects the most accurate candidate, and feeds only that selected emulator into the existing ROSE coefficient and wavefunction variables.

**Tech Stack:** Python 3.12, Jupyter/nbformat, NumPy, SciPy QMC, Matplotlib, public `nuclear-rose`, pytest

## Global Constraints

- Work only inside `/Users/Kitkat/Documents/Documents-Agent/LROM_Project`.
- Read and obey both workspace and project `CLAUDE.md` files.
- Do not read, write, or otherwise touch `.git/` or `.obsidian/`; therefore this plan contains no commit or push steps.
- Do not modify `lrom/`, `lrom_legacy/`, or `scientific_archive/`.
- Do not change any LROM sampling rows, training calls, predictors, metrics, scan windows, figure layouts, or figure markers.
- ROSE remains notebook-owned and uses its native free-reference basis.
- Keep coefficients within convention: ROSE versus ROSE-native LS; LROM versus central-basis LS.
- Keep physical radius `r` in fm in every researcher-facing spatial plot.
- Keep `DIFFERENCE_FLOOR = 1e-5` as display-only clipping.
- Keep EIM `training_info` bounds inclusive of original testing rows.
- Do not add plotting wrappers, ROSE workflow wrappers, notebook generators, caches, or serialized results.

---

### Task 1: Lock The Tutorial-Style ROSE Contract

**Files:**
- Modify: `tests/test_benchmark_02_notebook.py`
- Test: `tests/test_benchmark_02_notebook.py`

**Interfaces:**
- Consumes: notebook source returned by existing `notebook_text()`.
- Produces: a structural test that requires three independent held-out ROSE validation/tuning blocks without changing the existing 16-figure contract.

- [ ] **Step 1: Add the failing ROSE methodology test**

Append this test after `test_benchmark_02_notebook_contract`:

```python
def test_rose_benchmark_follows_the_heldout_tutorial_workflow() -> None:
    text = notebook_text()
    assert "from scipy.stats import qmc" in text
    assert "ROSE_EIM_CANDIDATES = (4, 8, 12)" in text
    assert "ROSE_VALIDATION_SIZE = 20" in text
    assert "ROSE_TIMING_REPEATS = 3" in text
    assert text.count("qmc.LatinHypercube") == 3
    assert text.count("for n_u in ROSE_EIM_CANDIDATES:") == 3
    assert text.count(".exact_wave_functions(") == 3
    assert text.count(".emulate_wave_functions(") >= 6
    assert text.count("time.perf_counter()") >= 6
    assert text.count("n_basis=BASIS_SIZE") >= 3
    assert text.count("selected ROSE configuration") == 3
    assert "def build_rose" not in text
    assert "def validate_rose" not in text
    assert "CATPerformance" not in text
```

The final assertion prevents use of ROSE's pointwise-relative `CATPerformance`,
which divides by the exact wavefunction at its nodes. The notebook instead
uses its existing complex-safe relative-L2 metric.

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run:

```bash
python -m pytest -q tests/test_benchmark_02_notebook.py
```

Expected: one failure in
`test_rose_benchmark_follows_the_heldout_tutorial_workflow` because the shared
ROSE constants and held-out loops are absent; the three pre-existing tests
remain green.

---

### Task 2: Add Shared ROSE Validation Configuration

**Files:**
- Modify: `notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb`
- Test: `tests/test_benchmark_02_notebook.py`

**Interfaces:**
- Consumes: existing imports and shared benchmark constants.
- Produces: `qmc`, `time`, `ROSE_EIM_CANDIDATES`, `ROSE_VALIDATION_SIZE`, and `ROSE_TIMING_REPEATS` for the three independent study blocks.

- [ ] **Step 1: Extend only the imports needed by ROSE benchmarking**

In code cell 1, add:

```python
import time
from scipy.stats import qmc
```

Do not alter the SciPy `sph_harm` compatibility alias, `rose`, `lrom`, or
Matplotlib configuration.

- [ ] **Step 2: Add shared ROSE-only knobs**

In the shared configuration cell, immediately after `DIFFERENCE_FLOOR`, add:

```python
ROSE_EIM_CANDIDATES = (4, 8, 12)
ROSE_VALIDATION_SIZE = 20
ROSE_TIMING_REPEATS = 3
```

Keep `BASIS_SIZE = 4`, `EIM_BASIS_SIZE = 8`, the global scan limits, and all
LROM constants unchanged. `EIM_BASIS_SIZE` remains part of the LROM sampling
request; the new tuple belongs only to the notebook-owned ROSE benchmark.

- [ ] **Step 3: Clarify provenance without restructuring sections**

Extend the introductory markdown with this paragraph:

```markdown
Within each physical study, ROSE is benchmarked following its public tutorial
workflow: the same training rows used by LROM train several independent ROSE
EIM configurations, a deterministic held-out Latin-hypercube design is solved
with both ROSE's exact and reduced paths, and held-out wavefunction accuracy
and online time select the ROSE configuration used in the existing figures.
The held-out rows affect only ROSE model selection; the legacy LROM designs and
diagnostics remain unchanged.
```

- [ ] **Step 4: Run the notebook contract**

Run:

```bash
python -m pytest -q tests/test_benchmark_02_notebook.py
```

Expected: the new methodology test still fails only on the missing three
per-study loops; JSON parsing and code-cell compilation pass.

---

### Task 3: Replace The Vv-Only ROSE Block

**Files:**
- Modify: `notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb` (existing Vv setup cell)
- Test: `tests/test_benchmark_02_notebook.py`

**Interfaces:**
- Consumes: `vv_emulator`, `vv_rose_rows`, `vv_rose_test_rows`, `vv_rose_rho`, shared ROSE constants, and `rose_real_woods_saxon`.
- Produces: the existing variables `vv_rose_emulator`, `vv_rose_rbe`, `vv_rose_coeff_train`, `vv_rose_coeff_test`, `vv_rose_wf_train`, `vv_rose_wf_test`, `vv_rose_ls_train`, `vv_rose_ls_test`, `vv_rose_error_train`, and `vv_rose_error_test`, now derived from a held-out-selected ROSE candidate.

- [ ] **Step 1: Generate and validate the one-dimensional held-out LHS**

Keep the existing LROM construction and the assignments through
`vv_rose_rho`. Replace the single-ROSE construction with:

```python
vv_rose_validation_unit = qmc.LatinHypercube(d=1, seed=12041).random(
    ROSE_VALIDATION_SIZE
)
vv_rose_validation_rows = np.repeat(
    vv_rose_rows[:1], ROSE_VALIDATION_SIZE, axis=0
)
vv_rose_validation_rows[:, 0] = qmc.scale(
    vv_rose_validation_unit,
    [vv_rose_rows[:, 0].min()],
    [vv_rose_rows[:, 0].max()],
)[:, 0]
assert vv_rose_validation_rows.shape == (ROSE_VALIDATION_SIZE, 3)
assert np.isfinite(vv_rose_validation_rows).all()
assert np.all(vv_rose_validation_rows >= vv_rose_rows.min(axis=0) - 1e-12)
assert np.all(vv_rose_validation_rows <= vv_rose_rows.max(axis=0) + 1e-12)
vv_rose_duplicates = np.isclose(
    vv_rose_validation_rows[:, None, :],
    vv_rose_rows[None, :, :],
    rtol=0.0,
    atol=1e-12,
).all(axis=2)
assert not vv_rose_duplicates.any()

vv_rose_bounds_rows = np.vstack(
    [vv_rose_rows, vv_rose_validation_rows, vv_rose_test_rows]
)
vv_rose_bounds = np.column_stack(
    [vv_rose_bounds_rows.min(axis=0), vv_rose_bounds_rows.max(axis=0)]
)
vv_rose_solver = rose.SchroedingerEquation.make_base_solver(
    s_0=6 * np.pi,
    rk_tols=[1e-9, 1e-9],
    domain=np.array([vv_rose_rho[0], vv_rose_rho[-1]]),
)
```

- [ ] **Step 2: Train, validate, and time the three Vv candidates**

Continue in the same cell with:

```python
vv_rose_candidates = {}
vv_rose_validation_summary = []
vv_rose_exact_validation = None
for n_u in ROSE_EIM_CANDIDATES:
    vv_candidate_interactions = rose.InteractionEIMSpace(
        l_max=0,
        coordinate_space_potential=rose_real_woods_saxon,
        n_theta=3,
        mu=vv_emulator.kinematics.mu,
        energy=vv_emulator.kinematics.e_com,
        is_complex=False,
        training_info=vv_rose_bounds,
        n_basis=n_u,
        rho_mesh=vv_rose_rho,
    )
    vv_candidate_emulator = rose.ScatteringAmplitudeEmulator.from_train(
        vv_candidate_interactions,
        vv_rose_rows,
        base_solver=vv_rose_solver,
        l_max=0,
        angles=np.linspace(1, 179, 120) * np.pi / 180,
        n_basis=BASIS_SIZE,
        use_svd=True,
        scale=False,
        s_mesh=vv_rose_rho,
        Smatrix_abs_tol=1e-8,
    )
    if vv_rose_exact_validation is None:
        vv_rose_exact_validation = np.asarray([
            vv_candidate_emulator.exact_wave_functions(row)[0][0]
            for row in vv_rose_validation_rows
        ])
    vv_candidate_validation = np.asarray([
        vv_candidate_emulator.emulate_wave_functions(row)[0][0]
        for row in vv_rose_validation_rows
    ])
    assert vv_rose_exact_validation.shape == (
        ROSE_VALIDATION_SIZE, MESH_SIZE
    )
    assert vv_candidate_validation.shape == vv_rose_exact_validation.shape
    assert np.isfinite(vv_rose_exact_validation).all()
    assert np.isfinite(vv_candidate_validation).all()
    vv_candidate_errors = np.linalg.norm(
        vv_candidate_validation - vv_rose_exact_validation, axis=1
    ) / np.maximum(
        np.linalg.norm(vv_rose_exact_validation, axis=1), 1e-300
    )
    vv_candidate_emulator.emulate_wave_functions(vv_rose_validation_rows[0])
    vv_candidate_timings = []
    for _ in range(ROSE_TIMING_REPEATS):
        vv_start = time.perf_counter()
        for row in vv_rose_validation_rows:
            vv_candidate_emulator.emulate_wave_functions(row)
        vv_candidate_timings.append(
            (time.perf_counter() - vv_start) / ROSE_VALIDATION_SIZE
        )
    vv_rose_candidates[n_u] = vv_candidate_emulator
    vv_rose_validation_summary.append({
        "n_phi": BASIS_SIZE,
        "n_U": n_u,
        "median_relative_l2": float(np.median(vv_candidate_errors)),
        "max_relative_l2": float(np.max(vv_candidate_errors)),
        "seconds_per_sample": float(min(vv_candidate_timings)),
    })

vv_rose_selected = min(
    vv_rose_validation_summary,
    key=lambda row: (
        row["median_relative_l2"],
        row["seconds_per_sample"],
        row["n_U"],
    ),
)
print("Vv ROSE held-out validation")
for row in vv_rose_validation_summary:
    print(
        f"  (n_phi={row['n_phi']}, n_U={row['n_U']:2d})  "
        f"median={row['median_relative_l2']:.3e}  "
        f"max={row['max_relative_l2']:.3e}  "
        f"online={row['seconds_per_sample'] * 1e3:.3f} ms/sample"
    )
print(
    "Vv selected ROSE configuration: "
    f"(n_phi={vv_rose_selected['n_phi']}, n_U={vv_rose_selected['n_U']})"
)
vv_rose_emulator = vv_rose_candidates[vv_rose_selected["n_U"]]
vv_rose_rbe = vv_rose_emulator.rbes[0][0]
```

- [ ] **Step 3: Preserve the existing Vv diagnostic assignments**

Retain the current code beginning with:

```python
vv_rose_coeff_train = np.asarray([
    vv_rose_rbe.coefficients(row) for row in vv_rose_rows
])
```

through the definitions of `vv_rose_error_train` and `vv_rose_error_test`.
Do not modify the three Vv figure cells.

- [ ] **Step 4: Run focused structural and compilation tests**

Run:

```bash
python -m pytest -q tests/test_benchmark_02_notebook.py
```

Expected: the methodology test still fails because only one of three study
loops exists; all code cells compile.

---

### Task 4: Replace The Rv-Only ROSE Block

**Files:**
- Modify: `notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb` (existing Rv setup cell)
- Test: `tests/test_benchmark_02_notebook.py`

**Interfaces:**
- Consumes: `rv_emulator`, `rv_rose_rows`, `rv_rose_test_rows`, `rv_rose_rho`, shared ROSE constants, and `rose_real_woods_saxon`.
- Produces: all existing `rv_rose_*` coefficient, wavefunction, LS, and error variables from the held-out-selected Rv emulator.

- [ ] **Step 1: Add the Rv held-out design and inclusive EIM bounds**

Keep the LROM portion of the Rv setup cell unchanged through `rv_rose_rho`,
then add these exact Rv definitions:

```python
rv_rose_validation_unit = qmc.LatinHypercube(d=1, seed=12042).random(
    ROSE_VALIDATION_SIZE
)
rv_rose_validation_rows = np.repeat(
    rv_rose_rows[:1], ROSE_VALIDATION_SIZE, axis=0
)
rv_rose_validation_rows[:, 1] = qmc.scale(
    rv_rose_validation_unit,
    [rv_rose_rows[:, 1].min()],
    [rv_rose_rows[:, 1].max()],
)[:, 0]
assert rv_rose_validation_rows.shape == (ROSE_VALIDATION_SIZE, 3)
assert np.isfinite(rv_rose_validation_rows).all()
assert np.all(rv_rose_validation_rows >= rv_rose_rows.min(axis=0) - 1e-12)
assert np.all(rv_rose_validation_rows <= rv_rose_rows.max(axis=0) + 1e-12)
rv_rose_duplicates = np.isclose(
    rv_rose_validation_rows[:, None, :],
    rv_rose_rows[None, :, :],
    rtol=0.0,
    atol=1e-12,
).all(axis=2)
assert not rv_rose_duplicates.any()
rv_rose_bounds_rows = np.vstack(
    [rv_rose_rows, rv_rose_validation_rows, rv_rose_test_rows]
)
rv_rose_bounds = np.column_stack(
    [rv_rose_bounds_rows.min(axis=0), rv_rose_bounds_rows.max(axis=0)]
)
rv_rose_solver = rose.SchroedingerEquation.make_base_solver(
    s_0=6 * np.pi,
    rk_tols=[1e-9, 1e-9],
    domain=np.array([rv_rose_rho[0], rv_rose_rho[-1]]),
)
```

- [ ] **Step 2: Add the independent Rv candidate loop**

Continue in the Rv cell with:

```python
rv_rose_candidates = {}
rv_rose_validation_summary = []
rv_rose_exact_validation = None
for n_u in ROSE_EIM_CANDIDATES:
    rv_candidate_interactions = rose.InteractionEIMSpace(
        l_max=0,
        coordinate_space_potential=rose_real_woods_saxon,
        n_theta=3,
        mu=rv_emulator.kinematics.mu,
        energy=rv_emulator.kinematics.e_com,
        is_complex=False,
        training_info=rv_rose_bounds,
        n_basis=n_u,
        rho_mesh=rv_rose_rho,
    )
    rv_candidate_emulator = rose.ScatteringAmplitudeEmulator.from_train(
        rv_candidate_interactions,
        rv_rose_rows,
        base_solver=rv_rose_solver,
        l_max=0,
        angles=np.linspace(1, 179, 120) * np.pi / 180,
        n_basis=BASIS_SIZE,
        use_svd=True,
        scale=False,
        s_mesh=rv_rose_rho,
        Smatrix_abs_tol=1e-8,
    )
    if rv_rose_exact_validation is None:
        rv_rose_exact_validation = np.asarray([
            rv_candidate_emulator.exact_wave_functions(row)[0][0]
            for row in rv_rose_validation_rows
        ])
    rv_candidate_validation = np.asarray([
        rv_candidate_emulator.emulate_wave_functions(row)[0][0]
        for row in rv_rose_validation_rows
    ])
    assert rv_rose_exact_validation.shape == (
        ROSE_VALIDATION_SIZE, MESH_SIZE
    )
    assert rv_candidate_validation.shape == rv_rose_exact_validation.shape
    assert np.isfinite(rv_rose_exact_validation).all()
    assert np.isfinite(rv_candidate_validation).all()
    rv_candidate_errors = np.linalg.norm(
        rv_candidate_validation - rv_rose_exact_validation, axis=1
    ) / np.maximum(
        np.linalg.norm(rv_rose_exact_validation, axis=1), 1e-300
    )
    rv_candidate_emulator.emulate_wave_functions(rv_rose_validation_rows[0])
    rv_candidate_timings = []
    for _ in range(ROSE_TIMING_REPEATS):
        rv_start = time.perf_counter()
        for row in rv_rose_validation_rows:
            rv_candidate_emulator.emulate_wave_functions(row)
        rv_candidate_timings.append(
            (time.perf_counter() - rv_start) / ROSE_VALIDATION_SIZE
        )
    rv_rose_candidates[n_u] = rv_candidate_emulator
    rv_rose_validation_summary.append({
        "n_phi": BASIS_SIZE,
        "n_U": n_u,
        "median_relative_l2": float(np.median(rv_candidate_errors)),
        "max_relative_l2": float(np.max(rv_candidate_errors)),
        "seconds_per_sample": float(min(rv_candidate_timings)),
    })

rv_rose_selected = min(
    rv_rose_validation_summary,
    key=lambda row: (
        row["median_relative_l2"],
        row["seconds_per_sample"],
        row["n_U"],
    ),
)
print("Rv ROSE held-out validation")
for row in rv_rose_validation_summary:
    print(
        f"  (n_phi={row['n_phi']}, n_U={row['n_U']:2d})  "
        f"median={row['median_relative_l2']:.3e}  "
        f"max={row['max_relative_l2']:.3e}  "
        f"online={row['seconds_per_sample'] * 1e3:.3f} ms/sample"
    )
print(
    "Rv selected ROSE configuration: "
    f"(n_phi={rv_rose_selected['n_phi']}, n_U={rv_rose_selected['n_U']})"
)
rv_rose_emulator = rv_rose_candidates[rv_rose_selected["n_U"]]
rv_rose_rbe = rv_rose_emulator.rbes[0][0]
```

Then retain every existing `rv_rose_coeff_*`, `rv_rose_wf_*`,
`rv_rose_ls_*`, and `rv_rose_error_*` calculation. Do not modify the Rv
figures.

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m pytest -q tests/test_benchmark_02_notebook.py
```

Expected: the methodology test still fails because the broad loop is absent;
all cells compile.

---

### Task 5: Replace The Broad Vv/Rv ROSE Block

**Files:**
- Modify: `notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb` (existing broad setup cell)
- Test: `tests/test_benchmark_02_notebook.py`

**Interfaces:**
- Consumes: unchanged `broad_emulator`, `broad_rose_rows`, `broad_rose_test_rows`, `broad_rose_rho`, and scan rows.
- Produces: the existing broad ROSE diagnostic arrays and aliases reused by the potential-predictor study.

- [ ] **Step 1: Add the two-dimensional broad held-out LHS**

Keep the broad LROM construction and rows unchanged. Replace only the current
single-ROSE setup with:

```python
broad_rose_validation_unit = qmc.LatinHypercube(d=2, seed=12043).random(
    ROSE_VALIDATION_SIZE
)
broad_rose_validation_rows = np.repeat(
    broad_rose_rows[:1], ROSE_VALIDATION_SIZE, axis=0
)
broad_rose_validation_rows[:, :2] = qmc.scale(
    broad_rose_validation_unit,
    broad_rose_rows[:, :2].min(axis=0),
    broad_rose_rows[:, :2].max(axis=0),
)
assert broad_rose_validation_rows.shape == (ROSE_VALIDATION_SIZE, 3)
assert np.isfinite(broad_rose_validation_rows).all()
assert np.all(
    broad_rose_validation_rows >= broad_rose_rows.min(axis=0) - 1e-12
)
assert np.all(
    broad_rose_validation_rows <= broad_rose_rows.max(axis=0) + 1e-12
)
broad_rose_duplicates = np.isclose(
    broad_rose_validation_rows[:, None, :],
    broad_rose_rows[None, :, :],
    rtol=0.0,
    atol=1e-12,
).all(axis=2)
assert not broad_rose_duplicates.any()
broad_rose_bounds_rows = np.vstack(
    [broad_rose_rows, broad_rose_validation_rows, broad_rose_test_rows]
)
broad_rose_bounds = np.column_stack(
    [broad_rose_bounds_rows.min(axis=0), broad_rose_bounds_rows.max(axis=0)]
)
broad_rose_solver = rose.SchroedingerEquation.make_base_solver(
    s_0=6 * np.pi,
    rk_tols=[1e-9, 1e-9],
    domain=np.array([broad_rose_rho[0], broad_rose_rho[-1]]),
)
```

- [ ] **Step 2: Add the independent broad candidate loop**

Continue in the broad cell with:

```python
broad_rose_candidates = {}
broad_rose_validation_summary = []
broad_rose_exact_validation = None
for n_u in ROSE_EIM_CANDIDATES:
    broad_candidate_interactions = rose.InteractionEIMSpace(
        l_max=0,
        coordinate_space_potential=rose_real_woods_saxon,
        n_theta=3,
        mu=broad_emulator.kinematics.mu,
        energy=broad_emulator.kinematics.e_com,
        is_complex=False,
        training_info=broad_rose_bounds,
        n_basis=n_u,
        rho_mesh=broad_rose_rho,
    )
    broad_candidate_emulator = rose.ScatteringAmplitudeEmulator.from_train(
        broad_candidate_interactions,
        broad_rose_rows,
        base_solver=broad_rose_solver,
        l_max=0,
        angles=np.linspace(1, 179, 120) * np.pi / 180,
        n_basis=BASIS_SIZE,
        use_svd=True,
        scale=False,
        s_mesh=broad_rose_rho,
        Smatrix_abs_tol=1e-8,
    )
    if broad_rose_exact_validation is None:
        broad_rose_exact_validation = np.asarray([
            broad_candidate_emulator.exact_wave_functions(row)[0][0]
            for row in broad_rose_validation_rows
        ])
    broad_candidate_validation = np.asarray([
        broad_candidate_emulator.emulate_wave_functions(row)[0][0]
        for row in broad_rose_validation_rows
    ])
    assert broad_rose_exact_validation.shape == (
        ROSE_VALIDATION_SIZE, MESH_SIZE
    )
    assert broad_candidate_validation.shape == broad_rose_exact_validation.shape
    assert np.isfinite(broad_rose_exact_validation).all()
    assert np.isfinite(broad_candidate_validation).all()
    broad_candidate_errors = np.linalg.norm(
        broad_candidate_validation - broad_rose_exact_validation, axis=1
    ) / np.maximum(
        np.linalg.norm(broad_rose_exact_validation, axis=1), 1e-300
    )
    broad_candidate_emulator.emulate_wave_functions(
        broad_rose_validation_rows[0]
    )
    broad_candidate_timings = []
    for _ in range(ROSE_TIMING_REPEATS):
        broad_start = time.perf_counter()
        for row in broad_rose_validation_rows:
            broad_candidate_emulator.emulate_wave_functions(row)
        broad_candidate_timings.append(
            (time.perf_counter() - broad_start) / ROSE_VALIDATION_SIZE
        )
    broad_rose_candidates[n_u] = broad_candidate_emulator
    broad_rose_validation_summary.append({
        "n_phi": BASIS_SIZE,
        "n_U": n_u,
        "median_relative_l2": float(np.median(broad_candidate_errors)),
        "max_relative_l2": float(np.max(broad_candidate_errors)),
        "seconds_per_sample": float(min(broad_candidate_timings)),
    })

broad_rose_selected = min(
    broad_rose_validation_summary,
    key=lambda row: (
        row["median_relative_l2"],
        row["seconds_per_sample"],
        row["n_U"],
    ),
)
print("broad Vv/Rv ROSE held-out validation")
for row in broad_rose_validation_summary:
    print(
        f"  (n_phi={row['n_phi']}, n_U={row['n_U']:2d})  "
        f"median={row['median_relative_l2']:.3e}  "
        f"max={row['max_relative_l2']:.3e}  "
        f"online={row['seconds_per_sample'] * 1e3:.3f} ms/sample"
    )
print(
    "broad selected ROSE configuration: "
    f"(n_phi={broad_rose_selected['n_phi']}, "
    f"n_U={broad_rose_selected['n_U']})"
)
broad_rose_emulator = broad_rose_candidates[broad_rose_selected["n_U"]]
broad_rose_rbe = broad_rose_emulator.rbes[0][0]
```

Retain every existing broad coefficient, wavefunction, weighted-LS, and
relative-L2 assignment after this point.

- [ ] **Step 3: Preserve predictor-study reuse**

Keep the current potential-predictor aliases unchanged:

```python
predictor_rose_emulator = broad_rose_emulator
predictor_rose_rbe = broad_rose_rbe
predictor_rose_coeff_train = broad_rose_coeff_train
predictor_rose_coeff_test = broad_rose_coeff_test
predictor_rose_ls_train = broad_rose_ls_train
predictor_rose_ls_test = broad_rose_ls_test
predictor_rose_wf_train = broad_rose_wf_train
predictor_rose_wf_test = broad_rose_wf_test
predictor_rose_error_train = broad_rose_error_train
predictor_rose_error_test = broad_rose_error_test
```

Do not change any broad, predictor, multiparameter, or violin figure cells.

- [ ] **Step 4: Run the complete notebook contract**

Run:

```bash
python -m pytest -q tests/test_benchmark_02_notebook.py
```

Expected: `4 passed`.

---

### Task 6: Execute And Inspect Benchmark 02

**Files:**
- Modify outputs in place: `notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb`
- Verify: all 16 existing embedded figures and three ROSE validation summaries

**Interfaces:**
- Consumes: completed notebook source and local `nuclear-rose` installation.
- Produces: a clean, fully executed notebook with deterministic scientific arrays and refreshed outputs.

- [ ] **Step 1: Execute the notebook end-to-end**

Run from `LROM_Project` with a writable Matplotlib cache:

```bash
MPLCONFIGDIR=/private/tmp/lrom-benchmark-02-mpl \
python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb
```

Expected: exit code 0, all code cells receive sequential execution counts,
and no error output is embedded.

- [ ] **Step 2: Check the three ROSE summaries**

Run:

```bash
jq -r '
  .cells[].outputs[]?
  | select(.output_type == "stream")
  | .text[]
' notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb \
| rg 'held-out validation|selected ROSE configuration|median=|online='
```

Expected: three held-out-validation headings, nine candidate rows, and three
selected-configuration lines. Errors and times must be finite and nonnegative.

- [ ] **Step 3: Check notebook execution integrity**

Run:

```bash
jq -r '[.cells[] | select(.cell_type == "code" and .execution_count == null)] | length' \
  notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb
jq -r '[.cells[].outputs[]? | select(.output_type == "error")] | length' \
  notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb
```

Expected: `0` and `0`.

- [ ] **Step 4: Visually inspect representative existing figures**

Extract and inspect at minimum:

- `vv-coefficients`;
- `rv-wavefunction-errors`;
- `broad-wavefunction-errors`;
- `multiparameter-violins`.

Confirm that axes, labels, legends, physical units, log floors, and LROM curves
are unchanged in structure and that the ROSE curves are finite and legible.

---

### Task 7: Final Regression And Memory Handoff

**Files:**
- Verify: entire repository test suite
- Modify: `../_memory/Daily Notes/2026-07-20.md`
- Modify: `../_memory/Context/active-state.md`

**Interfaces:**
- Consumes: verified executed notebook and test results.
- Produces: durable project handoff recording the ROSE-only redesign and any measured selected configurations.

- [ ] **Step 1: Run the complete repository test suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass; with the added methodology test the expected count is
`77 passed`.

- [ ] **Step 2: Confirm protected and out-of-scope source files were not edited**

Run:

```bash
shasum -a 256 \
  lrom/__init__.py \
  scientific_archive/ROSE_Guide/'ROSEPaper[7945].pdf' \
  scientific_archive/ROSE_Guide/ROSE_tutorial_0_quickstart.ipynb \
  scientific_archive/ROSE_Guide/ROSE_tutorial_1_building_an_emulator.ipynb \
  scientific_archive/ROSE_Guide/ROSE_tutorial_2_optical_potential_surmise_UQ.ipynb \
  scientific_archive/legacy_code/Legacy_benchmark/notebooks/02_lrom_method_walkthrough.ipynb
```

Expected hashes, recorded before implementation:

```text
7b9f815b1e210f46836bf5ee70201af463bf7c7059179f8dca08be53bec937c3  lrom/__init__.py
bd223834ae7564290d8eea33e6a77762fe9d808c2049a946f0ba49f5096ff299  scientific_archive/ROSE_Guide/ROSEPaper[7945].pdf
dfebacf7c087e7dcd3a8d7ca7fb4aaa9f53b4362a99b76e49ffdf0f78a988efb  scientific_archive/ROSE_Guide/ROSE_tutorial_0_quickstart.ipynb
cbd9c1d1b91ebe93ef8248776536ace631b0ad3a0f8f45ef357f6b370ca1c274  scientific_archive/ROSE_Guide/ROSE_tutorial_1_building_an_emulator.ipynb
8d3b7e9446bdd39b731cb55a597a9b40aed68d709979f0fd5769c5d0350a9134  scientific_archive/ROSE_Guide/ROSE_tutorial_2_optical_potential_surmise_UQ.ipynb
9ae3ad016b2112684cb49f629eb99ab8a0c7ee2eb8108762deac6696949eadcb  scientific_archive/legacy_code/Legacy_benchmark/notebooks/02_lrom_method_walkthrough.ipynb
```

The only project implementation files modified are the benchmark notebook and
its focused test, plus the approved spec and plan in `.agents/`.

- [ ] **Step 3: Append the daily log**

Append a bullet with the mandated prefix:

```markdown
- [LROM_Project] Rebuilt only the ROSE benchmarking path in `benchmark_notebooks/1.0/benchmark_02.ipynb`: each Vv, Rv, and broad Vv/Rv study now tunes `(n_phi=4, n_U in {4,8,12})` on deterministic held-out Latin-hypercube wavefunctions using public ROSE exact/emulated paths and min-of-three online timing; preserved all LROM state and 16 legacy figures; focused and full tests pass and the notebook executes end-to-end.
```

- [ ] **Step 4: Update active state**

Replace the pending Benchmark 02 physics-verification item with a completed
item naming the notebook and selected per-study `n_U` values observed during
execution. Record any remaining scientific blocker separately; do not mix it
with trading or other project state.

- [ ] **Step 5: Report completion with evidence**

Report the focused test result, full-suite result, notebook execution result,
the three selected ROSE configurations and their held-out errors/times, and
links to the notebook, test, design, and plan. State explicitly that `lrom/`
and `scientific_archive/` were not modified.
