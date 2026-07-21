# v2.0 Shell and Benchmark Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the validated exact-FOM boundary to the parked 2.0 shell and route every benchmark to its explicit package milestone without changing scientific results.

**Architecture:** `lrom_legacy.v2_0` remains the future cross-section shell and is not promoted. Version-2 benchmarks import it explicitly; wavefunction benchmarks import public v1.2. Notebook simplification removes obsolete arguments and duplicate setup only.

**Tech Stack:** Python, NumPy, nuclear-rose, Jupyter, pytest, Ruff.

## Global Constraints

- Do not add new 2.0 features.
- Preserve 2.0 cross-section and spin-orbit-aware behavior.
- Keep the public v1.2 object stable.
- Remove EIM only from high-fidelity FOM sampling; notebook-owned ROSE comparisons retain explicit EIM.
- Pure benchmark migrations preserve numerical summaries.
- Do not remove benchmark scientific sections or plots.
- Commit and push each task directly to `main`.

---

### Task 1: Characterize the parked 2.0 shell

**Files:**
- Create: `tests/test_v2_0_shell_characterization.py`

**Interfaces:**
- Consumes: parked `lrom_legacy.v2_0` single-file implementation.
- Produces: public signature, wavefunction parity, and cross-section API locks.

- [ ] **Step 1: Write the characterization tests**

```python
import inspect
import lrom_legacy.v2_0 as v2_0


def test_v2_shell_identity_and_api() -> None:
    assert v2_0.__version__ == "2.0.0"
    for name in ("sampling", "train", "predict", "save"):
        assert callable(getattr(v2_0.LROM, name))
    training_parameters = inspect.signature(v2_0.LROM.train).parameters
    assert "observable" in training_parameters
    assert "angles_degrees" in training_parameters
    assert "eim_basis_size" in inspect.signature(v2_0.LROM.sampling).parameters
```

Reuse the v1.2 small ws_1 characterization settings and assert parked v2.0 central, training, testing, basis, RF-operator, and prediction arrays match v1.2 before the EIM change.

- [ ] **Step 2: Run and commit the characterization**

Run: `python -m pytest -q tests/test_v2_0_shell_characterization.py`

Expected: PASS.

```bash
git add tests/test_v2_0_shell_characterization.py
git commit -m "Lock the parked 2.0 shell behavior"
git push origin main
```

### Task 2: Remove EIM from the parked 2.0 high-fidelity boundary

**Files:**
- Modify: `lrom_legacy/v2_0/__init__.py`
- Modify: `tests/test_v2_0_shell_characterization.py`

**Interfaces:**
- Consumes: v1.2 exact-boundary pattern.
- Produces: `sampling(..., high_fidelity_solver="runge_kutta")` without EIM.

- [ ] **Step 1: Change the test to the target signature**

```python
parameters = inspect.signature(v2_0.LROM.sampling).parameters
assert "eim_basis_size" not in parameters
assert parameters["high_fidelity_solver"].default == "runge_kutta"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_v2_0_shell_characterization.py`

Expected: FAIL on the old signature.

- [ ] **Step 3: Apply the exact interaction change**

Mirror the approved v1.2 boundary while retaining complex KD and spin-orbit configuration. Use `rose.InteractionSpace`, retain the existing Runge-Kutta base solver, and remove combined test-bound EIM construction.

- [ ] **Step 4: Verify parity and lint**

Run: `ruff check lrom_legacy/v2_0/__init__.py`

Run: `python -m pytest -q tests/test_v2_0_shell_characterization.py tests/test_lrom_fom.py`

Expected: PASS; characterized wavefunction arrays remain equal to v1.2 for shared wavefunction configurations.

- [ ] **Step 5: Commit and push**

```bash
git add lrom_legacy/v2_0/__init__.py tests/test_v2_0_shell_characterization.py
git commit -m "Use exact FOM interactions in the 2.0 shell"
git push origin main
```

### Task 3: Route the 2.0 benchmarks explicitly

**Files:**
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_01.ipynb`
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
- Modify: `tests/test_benchmark_2_0_notebooks.py`

**Interfaces:**
- Consumes: explicit `lrom_legacy.v2_0` and public `lrom` v1.2.
- Produces: unambiguous benchmark milestone imports.

- [ ] **Step 1: Write failing import assertions**

```python
assert "import lrom_legacy.v2_0 as v2_0" in benchmark_01_text
assert "import lrom as v1_2" in benchmark_01_text
assert "import lrom_legacy.v2_0 as lrom" in benchmark_03_text
assert "eim_basis_size" not in benchmark_01_text
assert "eim_basis_size" not in benchmark_03_text
```

- [ ] **Step 2: Update imports and sampling arguments mechanically**

Do not alter parameter rows, basis sizes, predictor counts, tolerances, timing loops, or figures. Add `high_fidelity_solver="runge_kutta"` to visible sampling calls.

- [ ] **Step 3: Run structural tests**

Run: `python -m pytest -q tests/test_benchmark_2_0_notebooks.py`

Expected: PASS.

- [ ] **Step 4: Commit and push**

```bash
git add notebooks/benchmark_notebooks/2.0 tests/test_benchmark_2_0_notebooks.py
git commit -m "Route 2.0 benchmarks to the parked shell"
git push origin main
```

### Task 4: Route and simplify the wavefunction benchmark

**Files:**
- Modify: `notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb`
- Modify: `tests/test_benchmark_02_notebook.py`

**Interfaces:**
- Consumes: public v1.2 and notebook-owned ROSE EIM methodology.
- Produces: same held-out ROSE validation and selected EIM configurations without package EIM arguments.

- [ ] **Step 1: Strengthen the routing/parity assertions**

```python
assert "import lrom" in text
assert "lrom_legacy" not in text
assert "eim_basis_size" not in text
assert 'high_fidelity_solver="runge_kutta"' in text
assert "rose.InteractionEIMSpace" in text
```

- [ ] **Step 2: Remove only obsolete plumbing**

Delete package `eim_basis_size` arguments and add the visible solver option. Consolidate duplicate imports and repeated array aliases only when the resulting cell still shows the Vv, Rv, and broad-study methodology.

- [ ] **Step 3: Preserve selected ROSE configurations**

After execution, selections must remain Vv `(n_phi=4, n_U=8)`, Rv `(4, 12)`, and broad Vv/Rv `(4, 12)` unless the exact same data produces a measured deterministic difference. Any difference stops the task for investigation.

- [ ] **Step 4: Run structural tests and commit**

Run: `python -m pytest -q tests/test_benchmark_02_notebook.py`

Expected: PASS.

```bash
git add notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb tests/test_benchmark_02_notebook.py
git commit -m "Route benchmark 02 to public v1.2"
git push origin main
```

### Task 5: Execute and verify all benchmark notebooks

**Files:**
- Modify execution outputs in all three benchmark notebooks.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: executed, version-explicit benchmark suite.

- [ ] **Step 1: Execute benchmark 01**

Run: `MPLCONFIGDIR=/private/tmp/lrom-benchmark-mpl python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/benchmark_notebooks/2.0/benchmark_01.ipynb`

Expected: zero error outputs and v2.0-vs-v1.2 parity remains at the characterized floor.

- [ ] **Step 2: Execute benchmark 02**

Run the same command for `notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb`.

Expected: zero errors and the three selected ROSE configurations remain unchanged.

- [ ] **Step 3: Execute benchmark 03**

Run the same command for `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`.

Expected: zero errors; CAT methodology and cross-section summaries remain unchanged within stored tolerances.

- [ ] **Step 4: Validate all notebook metadata**

Programmatically assert no unexecuted code cells and no error outputs. Visually inspect any figure changed by regenerated output.

- [ ] **Step 5: Run full verification**

Run: `python -m pytest -q`

Run: `ruff check lrom/__init__.py lrom_legacy/v1_2/__init__.py lrom_legacy/v2_0/__init__.py`

Run: `git diff --check`

Expected: PASS/clean.

- [ ] **Step 6: Commit and push executed notebooks**

```bash
git add notebooks/benchmark_notebooks
git commit -m "Execute benchmarks with explicit package routing"
git push origin main
```

### Task 6: Final documentation, memory, and ownership report

**Files:**
- Modify: `docs/VERSIONING.md`
- Modify: `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify: `CLAUDE.md`
- Modify: workspace `_memory/Context/active-state.md`
- Modify: workspace daily note

**Interfaces:**
- Consumes: fully verified package and notebooks.
- Produces: final reproducible architecture handoff.

- [ ] **Step 1: Record exact final routing and public contracts**

Document public v1.2, parked v2.0, exact FOM interactions, notebook-owned EIM, explicit LS analysis, and major-restructure approval rule.

- [ ] **Step 2: Record measured performance and parity**

Include EIM construction timing, training-stage evaluation reduction, test counts, notebook execution counts, and numerical parity summaries.

- [ ] **Step 3: Produce the user's change explanation**

For each functional change, report Methodology, Before, After, Execution, and What did not change. List notebook prose awaiting the user's ownership pass.

- [ ] **Step 4: Verify synchronization**

Run: `git rev-parse HEAD origin/main`

Expected: identical hashes after the final documentation commit and push.
