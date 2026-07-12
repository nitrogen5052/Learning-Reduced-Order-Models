# Object-Oriented LROM Notebook 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the new keyword-only `lrom.LROM` package and reproduce the approved Notebook 1 high-fidelity/ROSE/LROM wavefunction comparisons without modifying Notebook 2.

**Architecture:** Add `lrom` alongside the temporarily retained `lrom_bench` compatibility package. `LROM` owns authoritative state and delegates to focused private modules for potential schemas, sampling, ROSE FOM/RBM, basis construction, predictors, RF-LROM, diagnostics, and portable artifacts. Migrate only Notebook 1; retain `lrom_bench` until later notebooks no longer depend on it.

**Tech Stack:** Python 3.11-3.13, NumPy, SciPy, nuclear-rose, pytest, nbformat, Matplotlib

---

### Task 1: Package Metadata and Public Surface

**Files:**
- Modify: `pyproject.toml`
- Create: `lrom/__init__.py`
- Create: `lrom/errors.py`
- Create: `lrom/emulator.py`
- Create: `tests/test_lrom_public_api.py`
- Modify: `tests/test_project_metadata.py`

- [ ] **Step 1: Write failing public-package tests**

```python
import inspect
from pathlib import Path

import lrom


def test_public_package_exports() -> None:
    assert lrom.__version__ == "0.1.0"
    assert lrom.LROM.__name__ == "LROM"
    assert callable(lrom.load)


def test_constructor_is_keyword_only() -> None:
    parameters = inspect.signature(lrom.LROM).parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )


def test_supported_python_window_and_package_name() -> None:
    text = Path("pyproject.toml").read_text()
    assert 'name = "lrom"' in text
    assert 'requires-python = ">=3.11,<3.14"' in text
```

- [ ] **Step 2: Run the test and confirm import failure**

Run: `pytest tests/test_lrom_public_api.py tests/test_project_metadata.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'lrom'`.

- [ ] **Step 3: Add package metadata and error types**

Set `requires-python = ">=3.11,<3.14"`, change the project name to `lrom`, and include `lrom*` packages. Retain `lrom_bench*` in package discovery during migration so untouched Notebook 2 remains usable.

Create these errors:

```python
class LROMError(Exception):
    """Base error for the public LROM workflow."""


class LROMConfigurationError(LROMError, ValueError):
    pass


class LROMSamplingError(LROMError, ValueError):
    pass


class LROMStateError(LROMError, RuntimeError):
    pass


class LROMArtifactError(LROMError, ValueError):
    pass
```

Create the exact approved keyword-only constructor shell in `lrom/emulator.py`; store its raw inputs temporarily without numerical work so the public signature is real from the first commit. Export `LROM`, the four specific errors, and `__version__` from `lrom/__init__.py`. Define a keyword-only `load(*, path)` that lazily imports `lrom.artifacts.load` when called, allowing the public surface to exist before artifact implementation without creating a fake artifact.

- [ ] **Step 4: Run the focused test**

Run: `pytest tests/test_lrom_public_api.py tests/test_project_metadata.py -q`

Expected: all public-surface tests pass.

- [ ] **Step 5: Commit the package shell**

```bash
git add pyproject.toml lrom/__init__.py lrom/errors.py lrom/emulator.py tests/test_lrom_public_api.py tests/test_project_metadata.py
git commit -m "Create lrom package surface"
```

### Task 2: Immutable Configuration and Potential Schemas

**Files:**
- Create: `lrom/config.py`
- Create: `lrom/potentials.py`
- Create: `tests/test_lrom_config.py`

- [ ] **Step 1: Write configuration tests**

Test these exact behaviors:

```python
def test_l_is_normalized_without_range_expansion() -> None:
    one = LROMConfig.create(target=(40, 20), projectile=(1, 0), lab_energy=14.1, l=3)
    many = LROMConfig.create(
        target=(40, 20), projectile=(1, 0), lab_energy=14.1, l=(0, 1, 3)
    )
    assert one.channels == (3,)
    assert many.channels == (0, 1, 3)


def test_ws_schemas_have_named_parameters() -> None:
    assert resolve_potential("ws_1").sampleable_names == ("Vv",)
    assert resolve_potential("ws_3").sampleable_names == ("Vv", "Rv", "av")
    assert len(resolve_potential("woods-saxon").sampleable_names) == 15


def test_custom_potential_requires_named_central_parameters() -> None:
    def custom(r, alpha):
        return r * 0 + alpha[0]

    with pytest.raises(LROMConfigurationError, match="central_parameters"):
        LROMConfig.create(
            target=(40, 20), projectile=(1, 0), lab_energy=14.1, potential=custom
        )
```

- [ ] **Step 2: Run the tests and confirm missing-module failure**

Run: `pytest tests/test_lrom_config.py -q`

Expected: collection fails because `lrom.config` and `lrom.potentials` do not exist.

- [ ] **Step 3: Implement potential schemas**

Create an immutable `PotentialSpec` with `name`, `function`, `parameter_names`, and `sampleable_names`. Use canonical full-KD names:

```python
KD_NAMES = (
    "Vv", "Rv", "av", "Wv", "Rwv", "awv", "Wd", "Rd", "ad",
    "Vso", "Rso", "aso", "Wso", "Rwso", "awso",
)
```

`ws_1` and `ws_3` both evaluate the real three-parameter Woods-Saxon function; `ws_1.sampleable_names` contains only `Vv`. A custom callable keeps the signature `potential(r, alpha)` and obtains `parameter_names` from the insertion order of the required central mapping, which is persisted explicitly in configuration and artifacts.

- [ ] **Step 4: Implement immutable configuration**

Create `LROMConfig.create(...)` as a keyword-only factory that validates `(A, Z)` tuples, positive finite energy, non-negative integer channels, `fom="nucl-scatter-eq"`, potential selection, and named overrides. Store `channels`, `parameter_names`, and a human-readable `description` mapping including exact `l` semantics.

- [ ] **Step 5: Run the configuration tests**

Run: `pytest tests/test_lrom_config.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit configuration**

```bash
git add lrom/config.py lrom/potentials.py tests/test_lrom_config.py
git commit -m "Add named LROM configuration"
```

### Task 3: Named Sampling Designs

**Files:**
- Create: `lrom/sampling.py`
- Create: `lrom/state.py`
- Create: `tests/test_lrom_sampling.py`

- [ ] **Step 1: Write failing sampling tests**

```python
def test_linspace_uses_separate_training_and_testing_ranges() -> None:
    design = create_sampling_design(
        parameter_names=("Vv", "Rv", "av"),
        central={"Vv": 50.0, "Rv": 4.0, "av": 0.65},
        training_ranges={"Vv": (45.0, 55.0)},
        testing_ranges={"Vv": (32.5, 67.5)},
        training_size=35,
        testing_size=41,
        strategy="linspace",
        seed=1204,
    )
    assert design.training.values.shape == (35, 3)
    assert design.testing.values.shape == (41, 3)
    assert design.training.case_ids[0] == "train-0000"
    assert np.all(design.training.values[:, 1] == 4.0)


def test_lhs_is_deterministic_and_named() -> None:
    kwargs = dict(
        parameter_names=("Vv", "Rv", "av"),
        central={"Vv": 50.0, "Rv": 4.0, "av": 0.65},
        training_ranges={"Vv": (45, 55), "Rv": (3.6, 4.4), "av": (.585, .715)},
        testing_ranges={"Vv": (39, 61), "Rv": (3.2, 4.8), "av": (.52, .78)},
        training_size=70,
        testing_size=81,
        strategy="latin_hypercube",
        seed=1204,
    )
    assert np.allclose(create_sampling_design(**kwargs).training.values,
                       create_sampling_design(**kwargs).training.values)
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest tests/test_lrom_sampling.py -q`

Expected: collection fails because the sampling API does not exist.

- [ ] **Step 3: Implement sampling state and designs**

Create immutable `ParameterCases` and `SamplingDesign` containers. `ParameterCases` stores `case_ids`, ordered `parameter_names`, numeric `values`, and a `named(index=...)` accessor. Implement `linspace` only when one parameter varies and LHS otherwise. Derive independent deterministic training/testing seeds from one `numpy.random.SeedSequence`.

- [ ] **Step 4: Validate errors**

Reject unknown parameter names, non-finite/reversed bounds, non-positive sizes, mismatched training/testing range keys, and multidimensional `linspace` using `LROMSamplingError` with the offending names in the message.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_lrom_sampling.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit sampling**

```bash
git add lrom/sampling.py lrom/state.py tests/test_lrom_sampling.py
git commit -m "Add named LROM sampling designs"
```

### Task 4: Stateful LROM Lifecycle

**Files:**
- Modify: `lrom/emulator.py`
- Modify: `lrom/__init__.py`
- Modify: `lrom/state.py`
- Modify: `tests/test_lrom_public_api.py`
- Create: `tests/test_lrom_lifecycle.py`

- [ ] **Step 1: Write lifecycle tests with an injected fake FOM**

```python
def test_train_before_sampling_is_rejected() -> None:
    emulator = LROM(target=(40, 20), projectile=(1, 0), lab_energy=14.1)
    with pytest.raises(LROMStateError, match="sampling"):
        emulator.train()


def test_resampling_invalidates_downstream_state(fake_fom) -> None:
    emulator = make_test_emulator(fake_fom=fake_fom)
    emulator.sampling(**small_sampling_kwargs())
    emulator.train()
    assert emulator.is_trained
    emulator.sampling(**small_sampling_kwargs(seed=2))
    assert emulator.is_sampled
    assert not emulator.is_trained
    assert emulator.predictions is None
```

- [ ] **Step 2: Run tests and confirm missing lifecycle behavior**

Run: `pytest tests/test_lrom_public_api.py tests/test_lrom_lifecycle.py -q`

Expected: tests fail because `LROM` has not been implemented.

- [ ] **Step 3: Implement the public class shell**

Implement the approved keyword-only constructor and properties `config`, `kinematics`, `central_parameters`, `parameter_names`, `sampleable_parameters`, `partial_waves`, `description`, `samples`, `mesh`, `full_order_model`, `basis`, `predictors`, `rf_lrom`, `rose_rbm`, `testing_results`, `testing_errors`, `predictions`, `provenance`, `is_sampled`, `is_trained`, and `can_predict`.

Methods mutate internal state and return `None`. Add private `_clear_training_state()` and `_clear_prediction_state()` methods. Support private dependency injection only through an underscore-prefixed testing hook, not a public constructor option.

- [ ] **Step 4: Implement sampling orchestration against the FOM protocol**

`LROM.sampling(...)` creates the named design, calls a FOM provider with keyword arguments, stores returned `SamplingState`, and clears downstream state. Keep the provider protocol small: resolve central physical state, create meshes, solve central/training/testing wavefunctions, evaluate potentials, and expose per-channel runtime objects.

- [ ] **Step 5: Run lifecycle tests**

Run: `pytest tests/test_lrom_public_api.py tests/test_lrom_lifecycle.py -q`

Expected: all fake-FOM lifecycle tests pass without ROSE.

- [ ] **Step 6: Commit lifecycle**

```bash
git add lrom/emulator.py lrom/__init__.py lrom/state.py tests/test_lrom_public_api.py tests/test_lrom_lifecycle.py
git commit -m "Add stateful LROM lifecycle"
```

### Task 5: `nucl-scatter-eq` FOM and Shared ROSE Basis

**Files:**
- Create: `lrom/fom.py`
- Create: `lrom/basis.py`
- Modify: `lrom/state.py`
- Create: `tests/test_lrom_fom.py`
- Create: `tests/test_lrom_basis.py`

- [ ] **Step 1: Write pure basis tests**

Verify that `build_basis(phi0=..., snapshots=..., radius=..., basis_size=...)` stores a central state, produces vectors with shape `(mesh_size, basis_size)`, and projects/reconstructs complex wavefunctions with the existing trapezoid-weighted LS convention.

- [ ] **Step 2: Write the smallest ROSE integration test**

Construct `LROM(target=(40, 20), projectile=(1, 0), lab_energy=14.1, l=0, potential="ws_1")`, sample three Vv training and three testing points on a 64-point mesh, and assert central/train/test shapes and physical-radius monotonicity.

- [ ] **Step 3: Run tests and confirm failure**

Run: `pytest tests/test_lrom_basis.py tests/test_lrom_fom.py -q`

Expected: tests fail because the basis and ROSE provider are absent.

- [ ] **Step 4: Port numerical basis behavior**

Adapt the already-tested weighted trapezoid and centered-SVD behavior from `lrom_bench.numerics` and `lrom_bench.reduced_basis` into `lrom/basis.py`. Do not import `lrom_bench` from the new package.

- [ ] **Step 5: Implement the ROSE provider**

Adapt compatibility import handling, KD kinematics, the three-parameter real Woods-Saxon interaction, EIM construction, base solver, physical-radius mesh, and per-channel solve behavior from `lrom_bench.rose_fom`. Resolve full KD central values once, merge named overrides, and use exact selected channels. Return structured `SamplingState` rather than filesystem artifacts.

- [ ] **Step 6: Build the shared ROSE custom basis**

Construct one central `CustomBasis` per channel from the LROM basis arrays and pass that exact object to ROSE `ReducedBasisEmulator`. Store ROSE runtime objects in live training state, not portable state.

- [ ] **Step 7: Run focused integration tests**

Run: `pytest tests/test_lrom_basis.py tests/test_lrom_fom.py -q`

Expected: all tests pass with the installed `nuclear-rose` package.

- [ ] **Step 8: Commit FOM and basis**

```bash
git add lrom/fom.py lrom/basis.py lrom/state.py tests/test_lrom_fom.py tests/test_lrom_basis.py
git commit -m "Add nuclear scattering FOM and shared basis"
```

### Task 6: Predictors, RF-LROM, and Testing Diagnostics

**Files:**
- Create: `lrom/predictors.py`
- Create: `lrom/rf.py`
- Create: `lrom/diagnostics.py`
- Modify: `lrom/emulator.py`
- Modify: `lrom/state.py`
- Create: `tests/test_lrom_training.py`

- [ ] **Step 1: Write predictor and training tests**

Cover `predictor="parameters"` for `ws_1`, `predictor="potential"` for `ws_3`, six selected physical radii, shared ROSE/LROM basis identity, one RF model per exact channel, and test diagnostics containing `rose`, `lrom`, and `ls` arrays of shape `(testing_size, mesh_size)`.

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/test_lrom_training.py -q`

Expected: tests fail because training internals are absent.

- [ ] **Step 3: Port predictor algorithms**

Adapt centered named-parameter scaling and potential SVD/maxvol selection from `lrom_bench.predictors`. Store predictor kind, scales, selected grid indices, selected physical radii, central potential values, and singular values. Potential predictors are the default.

- [ ] **Step 4: Port RF-LROM and inference algorithms**

Adapt residual-fit training and the online reduced solve from `lrom_bench.rf_lrom` and `lrom_bench.prediction`. Keep public parameter mappings named; convert to ordered numeric arrays only inside the predictor component.

- [ ] **Step 5: Implement `reduced_basis()`, `train()`, and `predict()`**

`train()` builds bases, ROSE RBMs, the selected predictor, per-channel RF models, and automatic testing predictions. `predict()` accepts one mapping or a sequence, fills omitted parameters with central values, rejects unknown names, solves each selected channel, reconstructs wavefunctions, stores `PredictionState`, and returns `None`.

- [ ] **Step 6: Implement testing diagnostics and accessors**

Store high-fidelity, ROSE, LROM, and LS wavefunctions and coefficients. Compute complex-safe pointwise absolute errors plus relative/absolute L2 summaries. Add `testing_case(case_id=...)` and convenience `testing_errors` access without plotting code.

- [ ] **Step 7: Run training tests and regression suite**

Run: `pytest tests/test_lrom_training.py tests/test_predictors.py tests/test_rf_lrom.py tests/test_reduced_basis.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit training**

```bash
git add lrom/predictors.py lrom/rf.py lrom/diagnostics.py lrom/emulator.py lrom/state.py tests/test_lrom_training.py
git commit -m "Add LROM training and diagnostics"
```

### Task 7: Portable Prediction Artifact

**Files:**
- Create: `lrom/artifacts.py`
- Modify: `lrom/emulator.py`
- Modify: `lrom/__init__.py`
- Create: `tests/test_lrom_artifacts.py`

- [ ] **Step 1: Write artifact round-trip tests**

Train with a fake FOM, save to `tmp_path / "model.lrom"`, load through `lrom.load(path=...)`, assert `samples is None`, `full_order_model is None`, `is_trained`, and predictions numerically match the pre-save object.

- [ ] **Step 2: Run and confirm failure**

Run: `pytest tests/test_lrom_artifacts.py -q`

Expected: tests fail because save/load are absent.

- [ ] **Step 3: Implement versioned JSON-plus-NPZ storage**

Use a ZIP container with `metadata.json` and `arrays.npz`. Store only JSON primitives and `allow_pickle=False` arrays. Include artifact schema, package version, configuration hash, Python/NumPy training metadata, physical configuration, ordered parameter schema, meshes, basis state, predictor state, and RF matrices/vectors. Reject corrupt, missing, object-dtype, or unsupported-schema data with `LROMArtifactError`.

- [ ] **Step 4: Implement inference-only load semantics**

`lrom.load(path=...)` creates a trained `LROM` with no FOM samples or ROSE runtime, permits `predict()`, and rejects `sampling()` and `train()` with an explanatory `LROMStateError`.

- [ ] **Step 5: Run artifact tests**

Run: `pytest tests/test_lrom_artifacts.py -q`

Expected: all tests pass without invoking ROSE.

- [ ] **Step 6: Commit artifacts**

```bash
git add lrom/artifacts.py lrom/emulator.py lrom/__init__.py tests/test_lrom_artifacts.py
git commit -m "Add portable LROM inference artifacts"
```

### Task 8: Notebook 1 Migration and Explicit Figures

**Files:**
- Modify: `scripts/generate_notebook01.py`
- Modify: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`
- Modify: `tests/test_notebook01_generation.py`
- Create: `tests/test_notebook01_lrom_flow.py`

- [ ] **Step 1: Update structural notebook tests first**

Assert the generated notebook imports `lrom`, constructs `vv_emulator` and `ws3_emulator`, uses separate training/testing ranges, includes `predictor="parameters"` and `predictor="potential"`, does not import `lrom_bench`, contains no cross-section calls, and includes explicit Matplotlib code for selected predictor points, representative `Re(phi)` curves, and log-scale testing errors.

- [ ] **Step 2: Run and confirm failure**

Run: `pytest tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py -q`

Expected: tests fail against the current generator/notebook.

- [ ] **Step 3: Rewrite the generator around the two stateful objects**

Keep plotting code in notebook cells. Use 35/41 linspace Vv cases at +/-10% and +/-35%; use 70/81 LHS ws3 cases at training +/-10% and testing `Vv` +/-22%, `Rv`/`av` +/-20%; use `n_phi=4`, `n_U=8`, and six potential predictors.

- [ ] **Step 4: Add approved figures and summaries**

Generate potential and `Re(phi(r))` rainbows, basis vectors, shared-coordinate coefficients, wavefunction error curves, ws3 selected-radius overlay, representative high-fidelity/ROSE/LROM wavefunctions, all-testing-case logarithmic pointwise errors for ROSE/LROM/LS with transparency, and interpolation/extrapolation summaries. Use physical radius in fm everywhere and omit cross sections and the older linear-LROM baseline.

- [ ] **Step 5: Regenerate twice and verify determinism**

Run:

```bash
python scripts/generate_notebook01.py
cp notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb /tmp/notebook01-first.ipynb
python scripts/generate_notebook01.py
cmp /tmp/notebook01-first.ipynb notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
```

Expected: `cmp` exits successfully.

- [ ] **Step 6: Run notebook tests**

Run: `pytest tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit Notebook 1 migration**

```bash
git add scripts/generate_notebook01.py notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb tests/test_notebook01_generation.py tests/test_notebook01_lrom_flow.py
git commit -m "Migrate Notebook 1 to lrom object workflow"
```

### Task 9: Compatibility Matrix and Final Verification

**Files:**
- Create: `.github/workflows/test.yml`
- Modify: `README.md`

- [ ] **Step 1: Add metadata and CI tests**

Extend metadata coverage to assert package discovery includes `lrom*`, and assert README shows the canonical construct/sample/train/save/load workflow.

- [ ] **Step 2: Add the Python matrix**

Configure GitHub Actions for Python `3.11`, `3.12`, and `3.13`, install the package with test dependencies, run pure tests on all versions, and run the smallest ROSE integration test where the dependency installation succeeds. Do not mark failed training dependencies as successful.

- [ ] **Step 3: Run the complete local suite**

Run: `pytest -q`

Expected: all tests pass; record exact count.

- [ ] **Step 4: Run direct Notebook 1 code cells if Jupyter execution is unavailable**

Execute code cells sequentially in one Python namespace, skipping only notebook display-only cells if necessary. Expected: both emulators sample and train; all approved arrays and figures are produced without traceback.

- [ ] **Step 5: Verify repository boundaries**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Existing unrelated modifications to Notebook 2, `tests/test_metrics_artifacts.py`, and its supporting untracked scripts remain outside implementation commits.

- [ ] **Step 6: Commit docs and CI**

```bash
git add .github/workflows/test.yml README.md tests/test_project_metadata.py
git commit -m "Document and test supported LROM workflow"
```
