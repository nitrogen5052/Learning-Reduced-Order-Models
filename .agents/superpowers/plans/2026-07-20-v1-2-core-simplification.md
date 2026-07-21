# v1.2 Core Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the validated v1.2 wavefunction implementation the public LROM package, remove unused EIM and automatic LS-baseline work, and preserve the rigid public object lifecycle.

**Architecture:** Keep one authoritative implementation in `lrom_legacy/v1_2/__init__.py`; make `lrom/__init__.py` a thin public entry point. Preserve the verified current 2.0 single-file implementation at `lrom_legacy/v2_0/__init__.py` before changing the public route. Keep the exact ROSE high-fidelity boundary because RF-LROM requires authoritative snapshots.

**Tech Stack:** Python 3.12, NumPy, SciPy, Numba, nuclear-rose, pytest, Ruff.

## Global Constraints

- Keep `LROM(...)`, `sampling()`, `train()`, `predict()`, `save()`, and `load()` lifecycle meanings stable.
- Approved signature changes: remove `eim_basis_size`; add `high_fidelity_solver="runge_kutta"`.
- Do not remove public result properties or `testing_case()` in this pass.
- Keep the active implementation in one file.
- No package plotting functions.
- All spatial arrays exposed to users use physical radius in fm.
- Pure refactors preserve characterized arrays; scientific changes require measured before/after evidence.
- Commit and push every completed task directly to `main`.

---

### Task 1: Lock v1.2 characterization and public-lifecycle contracts

**Files:**
- Create: `tests/test_v1_2_characterization.py`
- Modify: none

**Interfaces:**
- Consumes: current `lrom_legacy.v1_2.LROM` workflow.
- Produces: deterministic array hashes and method-signature guards used by later tasks.

- [ ] **Step 1: Write the characterization test**

Create a test helper and one small ws_1 workflow:

```python
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import lrom_legacy.v1_2 as v1_2


def array_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def build_characterization_emulator() -> v1_2.LROM:
    emulator = v1_2.LROM(
        target=(40, 20), projectile=(1, 0), lab_energy=14.1,
        l=0, fom="nucl-scatter-eq", potential="ws_1",
    )
    center = dict(emulator.central_parameters)
    vv = center["Vv"]
    emulator.sampling(
        training_ranges={"Vv": (0.9 * vv, 1.1 * vv)},
        testing_ranges={"Vv": (0.8 * vv, 1.2 * vv)},
        training_size=5, testing_size=5, mesh_size=96,
        strategy="linspace", seed=1204, eim_basis_size=4,
    )
    emulator.train(basis_size=3, predictor="parameters", predictor_count=1)
    emulator.predict(parameters={"Vv": 0.95 * vv})
    return emulator


def test_v1_2_characterization_hashes() -> None:
    emulator = build_characterization_emulator()
    expected = {
        "central": "edf364ef6f5c4d13af1570ed5ea7d279ec0b3d246a090b06fccffe56305e2eff",
        "training": "b835d1011a0e887dc3c70279fadd5436821b896a7f3ddc375ce5cfc055fca27a",
        "testing": "6c1dc1711c24df55ea7e8287b923151cec2c558e309dc49850513766079b15b9",
        "basis": "ddac951fe5ba271a32c70b25477df6e860a2d3d9d3493ac32234b2db34b0c43a",
        "matrices": "8ef5cc64e00d55abf32463c6b5fdf1cad4ad50ebc92bdbfa83056f6f4ee3445d",
        "vectors": "c003fe245264ac00df36bd846966f4e5ac387e2a0e0501f53ba639f15628a281",
        "prediction": "28d898140842a85bb8d7d2f0eccc69d10d7ba251e97968cb4bdc0b8b93e9b874",
    }
    actual = {
        "central": array_hash(emulator.samples.central_wavefunctions[0]),
        "training": array_hash(emulator.samples.training_wavefunctions[0]),
        "testing": array_hash(emulator.samples.testing_wavefunctions[0]),
        "basis": array_hash(emulator.basis[0].vectors),
        "matrices": array_hash(emulator.rf_lrom[0].matrices),
        "vectors": array_hash(emulator.rf_lrom[0].vectors),
        "prediction": array_hash(emulator.predictions.wavefunctions[0]),
    }
    assert actual == expected


def test_public_lifecycle_methods_exist() -> None:
    for name in ("sampling", "train", "predict", "save"):
        assert callable(getattr(v1_2.LROM, name))
    assert callable(v1_2.load)
    assert "eim_basis_size" in inspect.signature(v1_2.LROM.sampling).parameters
```

- [ ] **Step 2: Run the characterization test**

Run: `MPLCONFIGDIR=/private/tmp/lrom-v1-plan python -m pytest -q tests/test_v1_2_characterization.py`

Expected: PASS against the pre-refactor implementation.

- [ ] **Step 3: Commit and push the characterization lock**

```bash
git add tests/test_v1_2_characterization.py
git commit -m "Lock v1.2 package characterization"
git push origin main
```

### Task 2: Preserve verified 2.0 and remove its obsolete modular duplicate

**Files:**
- Replace: `lrom_legacy/v2_0/__init__.py`
- Delete: `lrom_legacy/v2_0/artifacts.py`
- Delete: `lrom_legacy/v2_0/basis.py`
- Delete: `lrom_legacy/v2_0/config.py`
- Delete: `lrom_legacy/v2_0/diagnostics.py`
- Delete: `lrom_legacy/v2_0/emulator.py`
- Delete: `lrom_legacy/v2_0/errors.py`
- Delete: `lrom_legacy/v2_0/fom.py`
- Delete: `lrom_legacy/v2_0/potentials.py`
- Delete: `lrom_legacy/v2_0/predictors.py`
- Delete: `lrom_legacy/v2_0/rf.py`
- Delete: `lrom_legacy/v2_0/sampling.py`
- Delete: `lrom_legacy/v2_0/state.py`
- Delete: `lrom_legacy/v2_0/training.py`
- Modify: `tests/test_lrom_public_api.py`

**Interfaces:**
- Consumes: verified current `lrom/__init__.py` version 2.0.0.
- Produces: one-file `lrom_legacy.v2_0` future shell with the same public API and cross-section implementation.

- [ ] **Step 1: Add a preservation test before moving code**

Add:

```python
import inspect


def test_parked_v2_is_verified_single_file_shell() -> None:
    import lrom
    assert lrom.__version__ == "2.0.0"
    assert hasattr(lrom.LROM, "predict")
    parameters = inspect.signature(lrom.LROM.train).parameters
    assert "observable" in parameters
    assert "angles_degrees" in parameters
```

- [ ] **Step 2: Copy the verified single-file implementation and remove the modular donor**

Mechanically replace `lrom_legacy/v2_0/__init__.py` with the exact current contents of `lrom/__init__.py`. Delete the thirteen now-unused sibling modules with versioned patches. Do not edit the copied physics in this task.

- [ ] **Step 3: Point the preservation test at the parked module**

```python
def test_parked_v2_is_verified_single_file_shell() -> None:
    import lrom_legacy.v2_0 as v2_0
    assert v2_0.__version__ == "2.0.0"
    assert hasattr(v2_0.LROM, "predict")
    parameters = inspect.signature(v2_0.LROM.train).parameters
    assert "observable" in parameters
    assert "angles_degrees" in parameters
```

- [ ] **Step 4: Verify import and full baseline**

Run: `python -m pytest -q tests/test_lrom_public_api.py tests/test_benchmark_2_0_notebooks.py`

Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add lrom_legacy/v2_0 tests/test_lrom_public_api.py
git commit -m "Park verified 2.0 as the future shell"
git push origin main
```

### Task 3: Replace v1.2 sampling EIM with the exact ROSE interaction

**Files:**
- Modify: `lrom_legacy/v1_2/__init__.py`
- Create: `tests/test_v1_2_exact_fom.py`
- Modify: `tests/test_v1_2_characterization.py`

**Interfaces:**
- Consumes: `rose.InteractionSpace`, `_real_ws_interaction`, existing FOM configuration.
- Produces: `sampling(..., high_fidelity_solver="runge_kutta")` without `eim_basis_size`.

- [ ] **Step 1: Write failing signature and interaction tests**

```python
import inspect
import numpy as np
import pytest
import lrom_legacy.v1_2 as v1_2


def test_sampling_exposes_only_the_exact_runge_kutta_solver() -> None:
    parameters = inspect.signature(v1_2.LROM.sampling).parameters
    assert "eim_basis_size" not in parameters
    assert parameters["high_fidelity_solver"].default == "runge_kutta"


def test_sampling_rejects_unknown_high_fidelity_solver() -> None:
    emulator = v1_2.LROM(
        target=(40, 20), projectile=(1, 0), lab_energy=14.1,
        l=0, potential="ws_1",
    )
    with pytest.raises(v1_2.LROMSamplingError, match="runge_kutta"):
        emulator.sampling(
            training_ranges={"Vv": (45.0, 50.0)},
            testing_ranges={"Vv": (44.0, 51.0)},
            training_size=3, testing_size=3,
            high_fidelity_solver="unknown",
        )
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest -q tests/test_v1_2_exact_fom.py`

Expected: FAIL because the old signature still contains `eim_basis_size`.

- [ ] **Step 3: Implement the exact boundary**

Change the provider signature to accept `high_fidelity_solver: str`. Validate it with:

```python
if high_fidelity_solver != "runge_kutta":
    raise LROMSamplingError(
        "high_fidelity_solver must be 'runge_kutta'"
    )
```

Delete the combined train/test bounds and EIM keyword block. Preserve the existing potential selection, but pass it to ordinary `InteractionSpace`:

```python
interaction_options: dict[str, object] = {
    "l_max": max(config.channels),
    "n_theta": len(config.parameter_names),
    "mu": kinematics.mu,
    "energy": kinematics.e_com,
}
if config.potential.name == "woods-saxon":
    interaction_options.update(
        coordinate_space_potential=rose.koning_delaroche.KD_simple,
        spin_orbit_term=rose.koning_delaroche.KD_simple_so,
        is_complex=True,
    )
else:
    interaction_options.update(
        coordinate_space_potential=(
            _real_ws_interaction
            if config.potential.name in {"ws_1", "ws_3"}
            else config.potential.function
        ),
        is_complex=False,
    )
interactions = rose.InteractionSpace(**interaction_options)
```

Keep the existing `SchroedingerEquation.make_base_solver` and `ChannelFOM.solve` path unchanged.

- [ ] **Step 4: Update the characterization call and prove parity**

Remove `eim_basis_size=4` from the characterization setup and add `high_fidelity_solver="runge_kutta"`. Keep every expected hash unchanged.

Run: `MPLCONFIGDIR=/private/tmp/lrom-v1-plan python -m pytest -q tests/test_v1_2_exact_fom.py tests/test_v1_2_characterization.py`

Expected: PASS with the original hashes.

- [ ] **Step 5: Commit and push**

```bash
git add lrom_legacy/v1_2/__init__.py tests/test_v1_2_exact_fom.py tests/test_v1_2_characterization.py
git commit -m "Use exact ROSE interactions for v1.2 sampling"
git push origin main
```

### Task 4: Separate optional LS analysis from RF-LROM training

**Files:**
- Modify: `lrom_legacy/v1_2/__init__.py`
- Create: `tests/test_v1_2_ls_separation.py`
- Modify: `tests/test_v1_2_characterization.py`

**Interfaces:**
- Consumes: `BasisState`, `project_coordinates`, `reconstruct`, high-fidelity wavefunction arrays.
- Produces: `least_squares_baseline(...) -> tuple[np.ndarray, np.ndarray]`; automatic results retain LROM/HF but do not calculate LS.

- [ ] **Step 1: Write the failing explicit-baseline test**

```python
import numpy as np
import pytest
import lrom_legacy.v1_2 as v1_2


@pytest.fixture
def trained_v1_2_emulator() -> v1_2.LROM:
    emulator = v1_2.LROM(
        target=(40, 20), projectile=(1, 0), lab_energy=14.1,
        l=0, potential="ws_1",
    )
    center = dict(emulator.central_parameters)["Vv"]
    emulator.sampling(
        training_ranges={"Vv": (0.9 * center, 1.1 * center)},
        testing_ranges={"Vv": (0.8 * center, 1.2 * center)},
        training_size=5, testing_size=5, mesh_size=96,
        strategy="linspace", high_fidelity_solver="runge_kutta",
    )
    emulator.train(basis_size=3, predictor="parameters", predictor_count=1)
    return emulator


def test_least_squares_baseline_is_explicit_and_optimal(
    trained_v1_2_emulator: v1_2.LROM,
) -> None:
    basis = trained_v1_2_emulator.basis[0]
    high_fidelity = trained_v1_2_emulator.samples.testing_wavefunctions[0]
    coordinates, wavefunctions = v1_2.least_squares_baseline(
        basis=basis, wavefunctions=high_fidelity,
    )
    weights = v1_2._sqrt_trapezoid_weights(basis.radius)
    ls_error = np.linalg.norm((wavefunctions - high_fidelity) * weights, axis=1)
    lrom_error = np.linalg.norm(
        (trained_v1_2_emulator.testing_results.lrom[0] - high_fidelity) * weights,
        axis=1,
    )
    assert coordinates.shape == (len(high_fidelity), basis.basis_size)
    assert np.all(ls_error <= lrom_error + 1e-12)


def test_train_does_not_store_an_ls_baseline(
    trained_v1_2_emulator: v1_2.LROM,
) -> None:
    assert trained_v1_2_emulator.testing_results.ls is None
    assert trained_v1_2_emulator.training_results.ls is None
    assert "ls" not in trained_v1_2_emulator.testing_results.coefficients
    assert "ls" not in trained_v1_2_emulator.testing_errors[0]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_v1_2_ls_separation.py`

Expected: FAIL because the baseline function does not exist and `train()` stores LS.

- [ ] **Step 3: Add the explicit raw-array baseline**

```python
def least_squares_baseline(
    *, basis: BasisState, wavefunctions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return oracle coordinates and reconstructions for analysis only."""
    coordinates = project_coordinates(basis=basis, wavefunctions=wavefunctions)
    return coordinates, reconstruct(basis=basis, coordinates=coordinates)
```

Change automatic evaluation so it computes only RF-LROM coordinates, reconstructed wavefunctions, and LROM metrics. Preserve the result objects and set their compatibility `ls` field to `None`. `testing_case()` remains callable and returns `ls=None`.

- [ ] **Step 4: Keep RF training coordinates unchanged**

Retain this required training step:

```python
train_coordinates = project_coordinates(
    basis=basis,
    wavefunctions=samples.training_wavefunctions[channel],
)
model = fit_rf_lrom(
    predictors=predictor_state.training_features,
    coordinates=train_coordinates,
)
```

- [ ] **Step 5: Run focused and characterization tests**

Run: `python -m pytest -q tests/test_v1_2_ls_separation.py tests/test_v1_2_characterization.py`

Expected: PASS; basis, operator, and prediction hashes remain unchanged.

- [ ] **Step 6: Commit and push**

```bash
git add lrom_legacy/v1_2/__init__.py tests/test_v1_2_ls_separation.py tests/test_v1_2_characterization.py
git commit -m "Separate LS analysis from v1.2 training"
git push origin main
```

### Task 5: Explain and simplify the RF-LROM numerical core

**Files:**
- Modify: `lrom_legacy/v1_2/__init__.py`
- Modify: `tests/test_v1_2_characterization.py`
- Modify: `tests/test_project_metadata.py`

**Interfaces:**
- Consumes: `fit_rf_lrom(predictors, coordinates)` and `solve_rf_lrom(model, predictors)` behavior.
- Produces: numbered package boundaries and methodology comments without changed arrays.

- [ ] **Step 1: Add documentation contract tests**

```python
from pathlib import Path


def test_v1_2_explains_rf_lrom_least_squares() -> None:
    source = Path("lrom_legacy/v1_2/__init__.py").read_text()
    for phrase in (
        "weighted basis projection",
        "stacked RF-LROM system",
        "normal equations",
        "condition number",
        "online reduced solve",
    ):
        assert phrase in source
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_project_metadata.py::test_v1_2_explains_rf_lrom_least_squares`

Expected: FAIL on missing explanatory phrases.

- [ ] **Step 3: Consolidate imports and add numbered boundaries**

Use one standard-library import block, then NumPy/SciPy/Numba imports. Replace generic section banners with the eleven approved numbered boundaries. Each boundary receives a two-sentence ownership comment.

- [ ] **Step 4: Explain the two least-squares systems beside their code**

In `project_coordinates`, explain `A = W**1/2 Phi` and `b = W**1/2(phi-phi0)`. In `fit_rf_lrom`, explain the design shape `(sample_count * basis_size, predictor_count * (basis_size**2 + basis_size))`, the single stacked solve, unpacking, and why normal equations are avoided.

Do not vectorize the clear reduced-dimension loops unless a benchmark proves the replacement shorter and faster.

- [ ] **Step 5: Flatten only the private training wrapper**

Replace the one-instance `TrainingEngine` class and `LROM._trainer()` cache with private functions that receive `emulator` explicitly. Keep `LROM.train()` and `LROM.reduced_basis()` signatures and state transitions unchanged.

- [ ] **Step 6: Verify exact characterization and lint**

Run: `ruff check lrom_legacy/v1_2/__init__.py`

Expected: PASS.

Run: `MPLCONFIGDIR=/private/tmp/lrom-v1-plan python -m pytest -q tests/test_v1_2_characterization.py tests/test_v1_2_ls_separation.py tests/test_project_metadata.py`

Expected: PASS with unchanged core hashes.

- [ ] **Step 7: Commit and push**

```bash
git add lrom_legacy/v1_2/__init__.py tests/test_v1_2_characterization.py tests/test_project_metadata.py
git commit -m "Clarify the v1.2 RF-LROM core"
git push origin main
```

### Task 6: Route public `lrom` to the active v1.2 implementation

**Files:**
- Replace: `lrom/__init__.py`
- Modify: `lrom_legacy/v1_2/__init__.py`
- Modify: `tests/test_lrom_public_api.py`
- Modify: `tests/test_project_metadata.py`

**Interfaces:**
- Consumes: intentional `v1_2.__all__`.
- Produces: `import lrom` version 1.2.0; explicit `lrom_legacy.v2_0` version 2.0.0.

- [ ] **Step 1: Write failing routing tests**

```python
def test_public_lrom_routes_to_v1_2() -> None:
    import lrom
    import lrom_legacy.v1_2 as v1_2
    import lrom_legacy.v2_0 as v2_0
    assert lrom.__version__ == "1.2.0"
    assert lrom.LROM is v1_2.LROM
    assert v2_0.__version__ == "2.0.0"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_lrom_public_api.py::test_public_lrom_routes_to_v1_2`

Expected: FAIL because public `lrom` is still 2.0.0.

- [ ] **Step 3: Define the intentional v1.2 public surface**

Add required state types and analysis functions to `v1_2.__all__`, including `LROM`, public errors, `BasisState`, `PredictionState`, `least_squares_baseline`, `relative_l2`, `reconstruct`, and `load`.

Replace `lrom/__init__.py` with:

```python
"""Public entry point for the active v1.2 wavefunction LROM."""

from lrom_legacy.v1_2 import *
from lrom_legacy.v1_2 import __all__, __version__
```

- [ ] **Step 4: Update public tests to the v1.2 milestone**

Remove cross-section expectations from public `lrom` tests; retain them under explicit `lrom_legacy.v2_0` tests.

- [ ] **Step 5: Run the core suite**

Run: `python -m pytest -q tests/test_lrom_public_api.py tests/test_lrom_config.py tests/test_lrom_sampling.py tests/test_lrom_basis.py tests/test_lrom_lifecycle.py tests/test_lrom_training.py tests/test_lrom_artifacts.py tests/test_lrom_fom.py tests/test_v1_2_characterization.py tests/test_v1_2_exact_fom.py tests/test_v1_2_ls_separation.py`

Expected: PASS after tests are aligned with the approved v1.2 public milestone and LS separation.

- [ ] **Step 6: Commit and push**

```bash
git add lrom/__init__.py lrom_legacy/v1_2/__init__.py tests
git commit -m "Make v1.2 the public LROM package"
git push origin main
```

### Task 7: Document the stable active-package architecture

**Files:**
- Modify: `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify: `docs/VERSIONING.md`
- Modify: `CLAUDE.md`
- Modify: `tests/test_project_metadata.py`

**Interfaces:**
- Consumes: completed public routing and lifecycle.
- Produces: user-facing code map with stable function names and major-restructure rule.

- [ ] **Step 1: Write failing metadata assertions**

Assert the documents contain `v1.2 active package`, `Exact ROSE high-fidelity boundary`, `LS baseline is opt-in`, and `major restructure requires user approval`.

- [ ] **Step 2: Update the architecture guide**

Add a numbered map matching the package banners, a table mapping every public object call to its private numerical function, the two LS meanings, and the exact/EIM boundary.

- [ ] **Step 3: Update versioning and CLAUDE rules**

State that public `lrom` is v1.2, `lrom_legacy.v2_0` is the parked future shell, and public object methods are rigid. Replace the obsolete EIM-testing-bounds decision with the new notebook-owned EIM rule.

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q tests/test_project_metadata.py`

Expected: PASS.

```bash
git add docs/LROM_ARCHITECTURE_UNDERSTANDING.md docs/VERSIONING.md CLAUDE.md tests/test_project_metadata.py
git commit -m "Document the v1.2 active package boundary"
git push origin main
```

### Task 8: Core milestone verification

**Files:**
- Modify only if verification exposes a defect in an approved task.

**Interfaces:**
- Consumes: Tasks 1-7.
- Produces: verified core ready for notebook migration.

- [ ] **Step 1: Run lint and focused tests**

Run: `ruff check lrom/__init__.py lrom_legacy/v1_2/__init__.py lrom_legacy/v2_0/__init__.py`

Run: `python -m pytest -q tests/test_v1_2_characterization.py tests/test_v1_2_exact_fom.py tests/test_v1_2_ls_separation.py tests/test_lrom_public_api.py`

Expected: PASS.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`

Expected: PASS; notebook-contract tests may remain intentionally RED only if their implementation belongs to the next written plan and they are not merged early.

- [ ] **Step 3: Verify repository scope**

Run: `git diff --check`

Run: `git status --short --branch`

Expected: only the pre-existing untracked advisor backlog and restored ROSE guide remain.

- [ ] **Step 4: Record and push the milestone**

Update workspace daily notes and active state with exact test counts, hashes, and public API changes. Commit any verification correction separately, then verify `HEAD` equals `origin/main`.
