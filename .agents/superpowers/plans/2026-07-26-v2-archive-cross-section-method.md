# V2 Archive Cross-Section Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the scientific archive's channel-effective-interaction RF-LROM method and packed cross-section path to parked v2, then use it in Notebook 02 and benchmark 03.

**Architecture:** Preserve the existing flat `lrom_legacy.v2_0` module and add one explicit `effective-interaction` predictor strategy. Each channel owns its predictor state and intercept-bearing RF model; cross-section-only prediction batches the reduced solves and maps coefficients directly to \(S_l^\pm\). ROSE reduced-emulator construction, tuning, timing, and comparison remain notebook-local.

**Tech Stack:** Python 3.12, NumPy, SciPy, nuclear-rose, Matplotlib, pandas, Jupyter `nbconvert`, pytest, Ruff.

## Global Constraints

- Public `lrom` v1.2 and Notebook 01 remain byte-for-byte unchanged.
- The scientific archive remains read-only.
- Main comparison conditions remain \(l=0,1,2,3\), ±20%, 600 radial points, angles \(1^\circ,\ldots,179^\circ\), and seed 1204.
- Notebook 02 uses 200/100 training/testing rows; benchmark 03 defaults to 120/30 and retains its 200/100 full profile.
- Every user-facing spatial value uses physical radius \(r\) in fm.
- ROSE reduced-basis/EIM construction and comparison remain outside the LROM package.
- Existing `parameters` and `potential` predictor behavior remains available.
- New notebook Markdown is limited to short headings; the user owns the narrative.
- Use named alpha selections, never rank labels expressed as distribution cutoffs.
- Do not assert runtime ordering in unit tests.
- Keep the package flat and functional; do not create wrapper hierarchies.

---

## File Map

- Modify `lrom_legacy/v2_0/__init__.py`
  - predictor selection, channel feature construction, RF intercept, packed
    solve/\(S\)-matrix path, prediction option, and schema-2 persistence.
- Create `tests/test_v2_archive_method.py`
  - focused, inexpensive unit and small integration tests for the new v2
    method.
- Modify `tests/test_notebook02_cross_sections.py`
  - notebook/benchmark structure, metric, separation, and output contracts.
- Modify `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
  - full-profile scientific artifact.
- Modify `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
  - reduced/full reproducibility artifact.
- Modify `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
  - final package/notebook boundary and measured results.
- Create `outputs/notebook02_archive_method_summary.json`
  - machine-readable fixed-profile accuracy/timing evidence produced from the
    executed notebook results.

### Stable interfaces introduced by this plan

```python
def _refined_maxvol_indices(
    basis: np.ndarray,
    *,
    max_swaps: int = 50,
    tolerance: float = 1e-10,
) -> np.ndarray: ...

def build_effective_interaction_predictors(
    *,
    full_order_models: Mapping[Any, ChannelFOM],
    rho: np.ndarray,
    radius: np.ndarray,
    central_values: np.ndarray,
    training_values: np.ndarray,
    testing_values: np.ndarray,
    predictor_count: int,
    minimum_radius: float = 0.5,
) -> dict[Any, PredictorState]: ...

def effective_interaction_features(
    *,
    emulator: LROM,
    predictors: Mapping[Any, PredictorState],
    values: np.ndarray,
) -> dict[Any, np.ndarray]: ...

def fit(
    *,
    predictors: np.ndarray,
    coordinates: np.ndarray,
    include_intercept: bool = False,
) -> RFLROMModel: ...

def predict(
    *,
    emulator: LROM,
    parameters: Mapping[str, float] | Sequence[Mapping[str, float]],
    reconstruct_wavefunctions: bool = True,
) -> PredictionState: ...
```

`LROM.predict()` receives the same optional
`reconstruct_wavefunctions: bool = True` keyword and continues returning
state through `emulator.predictions`.

---

### Task 1: Refined max-volume selection and channel predictors

**Files:**
- Create: `tests/test_v2_archive_method.py`
- Modify: `lrom_legacy/v2_0/__init__.py`

**Interfaces:**
- Consumes: existing `PredictorState`, `ChannelFOM`, `SamplingState`.
- Produces: `_refined_maxvol_indices()` and
  `build_effective_interaction_predictors()` with the signatures in the file
  map.

- [ ] **Step 1: Write failing selector tests**

Add deterministic selector tests using a complex candidate basis:

```python
def test_refined_maxvol_is_unique_deterministic_and_improves_volume():
    basis = np.array(
        [
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1],
            [0.1, 0.1, 1.0],
            [0.9, 0.8, 0.0],
            [0.2j, 0.1, 0.8],
        ],
        dtype=np.complex128,
    )
    selected_a = v2._refined_maxvol_indices(basis)
    selected_b = v2._refined_maxvol_indices(basis)
    assert np.array_equal(selected_a, selected_b)
    assert selected_a.shape == (basis.shape[1],)
    assert np.unique(selected_a).size == selected_a.size
    assert abs(np.linalg.det(basis[selected_a])) > 0.0
```

Also test invalid shapes and `max_swaps < 0`.

- [ ] **Step 2: Run the selector test and verify RED**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k refined_maxvol
```

Expected: failure because `_refined_maxvol_indices` does not exist.

- [ ] **Step 3: Implement QR initialization and max-volume swaps**

Implement the archive algorithm without changing
`_greedy_maxvol_indices()`:

```python
def _refined_maxvol_indices(basis, *, max_swaps=50, tolerance=1e-10):
    basis = np.asarray(basis)
    if basis.ndim != 2 or basis.shape[1] > basis.shape[0]:
        raise ValueError("maxvol basis must have shape (points, modes)")
    if max_swaps < 0:
        raise ValueError("max_swaps must be non-negative")
    selected = [int(np.argmax(np.linalg.norm(basis, axis=1)))]
    for _ in range(1, basis.shape[1]):
        q, *_ = np.linalg.qr(basis[selected].T, mode="reduced")
        residual = basis - (basis @ q) @ q.T.conj()
        scores = np.linalg.norm(residual, axis=1)
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
    for _ in range(max_swaps):
        try:
            coefficients = basis @ np.linalg.inv(basis[selected])
        except np.linalg.LinAlgError:
            break
        magnitudes = np.abs(coefficients)
        magnitudes[selected, :] = 0.0
        row, column = np.unravel_index(np.argmax(magnitudes), magnitudes.shape)
        if magnitudes[row, column] <= 1.0 + tolerance:
            break
        selected[column] = int(row)
    return np.asarray(sorted(selected), dtype=int)
```

- [ ] **Step 4: Run selector tests and verify GREEN**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k refined_maxvol
```

Expected: all selector tests pass.

- [ ] **Step 5: Write failing effective-interaction predictor tests**

Use inexpensive fake channels whose `tilde` values depend on both channel and
alpha:

```python
class FakeInteraction:
    def __init__(self, channel_scale):
        self.channel_scale = channel_scale

    def tilde(self, rho, alpha):
        rho = np.asarray(rho)
        a0, a1 = np.asarray(alpha)
        return self.channel_scale * (
            a0 * np.exp(-rho) + 1j * a1 * rho * np.exp(-0.5 * rho)
        )


def test_effective_interaction_predictors_are_channel_specific_and_physical():
    rho = np.linspace(0.01, 8.0, 96)
    radius = rho / 1.4
    models = {
        0: SimpleNamespace(interaction=FakeInteraction(1.0)),
        (1, 0): SimpleNamespace(interaction=FakeInteraction(2.0)),
        (1, 1): SimpleNamespace(interaction=FakeInteraction(-3.0)),
    }
    center = np.array([1.0, 2.0])
    train = np.array([[0.8, 1.7], [1.2, 2.3], [1.1, 1.8], [0.9, 2.2]])
    test = np.array([[1.05, 2.1]])
    states = v2.build_effective_interaction_predictors(
        full_order_models=models,
        rho=rho,
        radius=radius,
        central_values=center,
        training_values=train,
        testing_values=test,
        predictor_count=2,
    )
    assert tuple(states) == (0, (1, 0), (1, 1))
    assert all(np.all(state.selected_radii >= 0.5) for state in states.values())
    assert not np.array_equal(
        states[(1, 0)].training_features,
        states[(1, 1)].training_features,
    )
    assert all(
        np.max(np.abs(state.training_features), axis=0).max()
        <= 1.0 + 1e-12
        for state in states.values()
    )
```

Add tests for insufficient allowed radii, infeasible predictor count, and a
zero-variation channel.

- [ ] **Step 6: Run predictor tests and verify RED**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k effective_interaction_predictors
```

Expected: failure because the builder does not exist.

- [ ] **Step 7: Implement channel predictor construction**

For each sorted channel:

```python
center_profile = model.interaction.tilde(rho, central_values)
training_profiles = np.asarray(
    [model.interaction.tilde(rho, row) for row in training_values]
)
testing_profiles = np.asarray(
    [model.interaction.tilde(rho, row) for row in testing_values]
)
allowed = np.flatnonzero(radius >= minimum_radius)
delta = (training_profiles - center_profile[np.newaxis, :]).T
u, singular_values, _ = np.linalg.svd(delta[allowed], full_matrices=False)
local = _refined_maxvol_indices(u[:, :predictor_count])
selected = allowed[local]
raw_training = training_profiles[:, selected] - center_profile[selected]
scales = np.max(np.abs(raw_training), axis=0)
```

Reject any scale at or below `1e-14`; do not replace it silently. Store
`selected_radii=radius[selected]` and use `selected_indices` only for
internal \(\rho\)-mesh lookup.

- [ ] **Step 8: Run Task 1 tests and commit**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k "refined_maxvol or effective_interaction_predictors"
ruff check lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git add lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git commit -m "feat: add channel interaction predictors"
```

Expected: tests and Ruff pass.

---

### Task 2: Intercept-bearing RF-LROM fit

**Files:**
- Modify: `tests/test_v2_archive_method.py`
- Modify: `lrom_legacy/v2_0/__init__.py`

**Interfaces:**
- Consumes: existing `fit_rf_lrom`, `solve_rf_lrom`.
- Produces: `RFLROMModel.constant_vector` and
  `fit(..., include_intercept=False)`.

- [ ] **Step 1: Write a failing exact synthetic-system test**

Generate training coordinates from known matrices and vectors:

```python
def test_intercept_fit_recovers_held_out_coordinates():
    rng = np.random.default_rng(18)
    features = rng.uniform(-0.8, 0.8, size=(80, 2))
    matrices = np.array(
        [
            [[0.05, 0.01], [-0.02, 0.03]],
            [[-0.01, 0.02], [0.01, 0.04]],
        ],
        dtype=np.complex128,
    )
    vectors = np.array([[0.2, -0.1], [0.04, 0.06]], dtype=np.complex128)
    constant = np.array([0.35 + 0.1j, -0.2 + 0.05j])

    def exact(rows):
        result = []
        for row in rows:
            matrix = np.eye(2) + np.einsum("k,kij->ij", row, matrices)
            rhs = constant + np.einsum("k,kj->j", row, vectors)
            result.append(np.linalg.solve(matrix, rhs))
        return np.asarray(result)

    model = v2.fit_rf_lrom(
        predictors=features,
        coordinates=exact(features),
        include_intercept=True,
    )
    held_out = rng.uniform(-0.7, 0.7, size=(12, 2))
    assert np.allclose(
        v2.solve_rf_lrom(model=model, predictors=held_out),
        exact(held_out),
        rtol=1e-9,
        atol=1e-10,
    )
    assert np.linalg.norm(model.constant_vector) > 0.0
```

Add a compatibility test asserting `include_intercept=False` returns a zero
constant vector and reproduces the pre-change fit/solve arrays.

- [ ] **Step 2: Run RF tests and verify RED**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k intercept
```

Expected: `fit()` rejects `include_intercept` or the model has no
`constant_vector`.

- [ ] **Step 3: Add the constant vector to model state**

Extend `RFLROMModel`:

```python
@dataclass(frozen=True)
class RFLROMModel:
    matrices: np.ndarray
    vectors: np.ndarray
    residual_mse: float
    rank: int
    singular_values: np.ndarray
    constant_vector: np.ndarray | None = None
```

Use a property or a small internal helper to interpret `None` as a zero
vector so manually constructed older test models remain valid.

- [ ] **Step 4: Extend the fit design**

When `include_intercept=True`, allocate `basis_size` extra vector unknowns.
For every equation row, place `-1` in that row's constant-vector column.
Keep the matrix and predictor-vector columns identical to the current fit.
Extract the constant vector separately after `numpy.linalg.lstsq`.

When `include_intercept=False`, execute the current column layout exactly and
store an explicit zero vector.

- [ ] **Step 5: Update online solve**

Change:

```python
rhs = np.einsum("k,kj->j", row, model.vectors)
```

to:

```python
rhs = _constant_vector(model) + np.einsum(
    "k,kj->j", row, model.vectors
)
```

- [ ] **Step 6: Run Task 2 tests and compatibility characterization**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k "intercept or zero_intercept"
pytest -q ../tests/test_v2_0_shell_characterization.py
ruff check lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
```

Expected: new tests pass and parked v2 still matches v1.2 for the shared
zero-intercept workflow.

- [ ] **Step 7: Commit**

```bash
git add lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git commit -m "feat: fit RF-LROM source intercept"
```

---

### Task 3: Integrate channel predictors into training and inference

**Files:**
- Modify: `tests/test_v2_archive_method.py`
- Modify: `lrom_legacy/v2_0/__init__.py`

**Interfaces:**
- Consumes: Task 1 predictor mapping and Task 2 intercept fit.
- Produces: `effective_interaction_features()` and a training/evaluation path
  accepting either one shared feature array or a channel feature mapping.

- [ ] **Step 1: Write a failing small nuclear integration test**

Use two partial waves and a small mesh/profile:

```python
def small_cross_section_emulator():
    emulator = v2.LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=(0, 1),
        potential="full_woods-saxon",
    )
    center = dict(emulator.central_parameters)
    ranges = {
        name: tuple(sorted((0.95 * value, 1.05 * value)))
        for name, value in center.items()
    }
    emulator.sampling(
        training_ranges=ranges,
        testing_ranges=ranges,
        training_size=24,
        testing_size=6,
        mesh_size=96,
        strategy="latin_hypercube",
        seed=1204,
    )
    return emulator


def test_effective_interaction_training_uses_one_feature_set_per_channel():
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
    )
    assert set(emulator.predictors) == set(emulator.rf_lrom)
    assert all(state.kind == "effective-interaction" for state in emulator.predictors.values())
    assert all(
        np.linalg.norm(model.constant_vector) > 0.0
        for model in emulator.rf_lrom.values()
    )
```

Add a test evaluating two new alpha rows and comparing every returned feature
to direct `interaction.tilde` values at
`samples.mesh.rho[state.selected_indices]`.

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k "training_uses_one or direct_tilde"
```

Expected: `predictor` validation rejects `"effective-interaction"`.

- [ ] **Step 3: Route `_predictor()`**

Add:

```python
if kind == "effective-interaction":
    return build_effective_interaction_predictors(
        full_order_models=samples.full_order_models,
        rho=samples.mesh.rho,
        radius=samples.mesh.radius,
        central_values=central,
        training_values=samples.design.training.values,
        testing_values=samples.design.testing.values,
        predictor_count=count,
        minimum_radius=0.5,
    )
```

Update the validation message to list all three predictor choices.

- [ ] **Step 4: Implement feature dispatch**

`effective_interaction_features()` must:

1. use each live `ChannelFOM.interaction.tilde` when sampled state exists;
2. otherwise evaluate a registered portable full Woods-Saxon interaction
   from the stored physical radius, channel key, kinematics, and potential
   functions;
3. center and scale with the channel's `PredictorState`;
4. return `{channel: features}` with shape `(cases, K)` per channel.

For live state, the core calculation is:

```python
rho_points = emulator.mesh.rho[state.selected_indices]
raw = np.asarray(
    [model.interaction.tilde(rho_points, row) for row in values]
)
features[channel] = (
    raw - state.central_values[np.newaxis, :]
) / state.scales[np.newaxis, :]
```

- [ ] **Step 5: Make `_evaluate()` feature-source agnostic**

Add a helper:

```python
def _channel_features(predictor_features, channel):
    if isinstance(predictor_features, Mapping):
        return predictor_features[channel]
    return predictor_features
```

Use it for every `solve_rf_lrom()` call. During training, call
`fit_rf_lrom(..., include_intercept=True)` only for the new strategy.

- [ ] **Step 6: Update normal `predict()` coefficient inference**

Dispatch shared predictors through `features_for_values()` and channel
predictors through `effective_interaction_features()`. Preserve the returned
coefficient mapping and default wavefunction reconstruction.

- [ ] **Step 7: Run focused and compatibility tests**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k "effective_interaction"
pytest -q ../tests/test_v2_0_shell_characterization.py
pytest -q tests/test_notebook02_cross_sections.py
ruff check lrom_legacy/v2_0/__init__.py tests
```

Expected: focused tests and old workflow characterization pass. Notebook
tests still describe the old Notebook 02 method and may remain unchanged at
this task.

- [ ] **Step 8: Commit**

```bash
git add lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git commit -m "feat: train channel-specific RF-LROM models"
```

---

### Task 4: Packed coefficient-to-cross-section prediction

**Files:**
- Modify: `tests/test_v2_archive_method.py`
- Modify: `lrom_legacy/v2_0/__init__.py`

**Interfaces:**
- Consumes: channel feature mappings and intercept-bearing RF models.
- Produces: internal packed solve/\(S\)-matrix helpers and
  `reconstruct_wavefunctions` on functional and stateful prediction APIs.

- [ ] **Step 1: Write failing batched-solve equivalence tests**

After training the small emulator from Task 3, obtain a multi-row feature
mapping and assert:

```python
packed = v2._packed_coefficients(
    channels=tuple(emulator.rf_lrom),
    models=emulator.rf_lrom,
    features=features,
)
for offset, channel in enumerate(emulator.rf_lrom):
    scalar = v2.solve_rf_lrom(
        model=emulator.rf_lrom[channel],
        predictors=features[channel],
    )
    assert np.allclose(packed[:, offset], scalar)
```

Add tests showing `_packed_smatrix_from_coefficients()` matches
`_s_matrix_from_coefficients()` channel by channel.

- [ ] **Step 2: Run packed tests and verify RED**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k packed
```

Expected: packed helpers do not exist.

- [ ] **Step 3: Implement packed model arrays**

Because Notebook 02 uses equal `basis_size` and `predictor_count` for all
channels, stack:

```python
matrices = np.asarray([models[channel].matrices for channel in channels])
vectors = np.asarray([models[channel].vectors for channel in channels])
constants = np.asarray([_constant_vector(models[channel]) for channel in channels])
feature_array = np.stack([features[channel] for channel in channels], axis=1)
systems = (
    np.eye(basis_size)[None, None, :, :]
    + np.einsum("sck,ckij->scij", feature_array, matrices, optimize=True)
)
rhs = constants[None, :, :] + np.einsum(
    "sck,ckj->scj", feature_array, vectors, optimize=True
)
coordinates = np.linalg.solve(systems, rhs[..., np.newaxis])[..., 0]
```

Validate identical basis and predictor widths before stacking. Raise a clear
error identifying the inconsistent channel.

- [ ] **Step 4: Pack asymptotic conversion data**

Build and cache arrays from the existing LROM-owned central-reference ROSE
RBE objects:

- `asymptotic_vals`;
- `asymptotic_ders`;
- `Hm`, `Hp`, `Hmp`, `Hpp`;
- `s_0`;
- channel keys and their \(l,\pm\) slots.

Compute all channel \(S\)-matrix values with array contractions and place them
into `(cases, l_max + 1)` `splus` and `sminus` arrays. For \(l=0\), copy the
plus value into minus.

- [ ] **Step 5: Add observable-only prediction**

Change both prediction signatures:

```python
def predict(*, emulator, parameters, reconstruct_wavefunctions=True):
    ...

class LROM:
    def predict(self, *, parameters, reconstruct_wavefunctions=True):
        self._prediction_state = predict(
            emulator=self,
            parameters=parameters,
            reconstruct_wavefunctions=reconstruct_wavefunctions,
        )
```

If false, return `wavefunctions={}`. If the trained observable is not
`"cross_section"`, reject false with a clear `LROMStateError`.

For effective-interaction cross-section models, use packed coefficients and
direct \(S\)-matrix conversion. Preserve the existing scalar path for older
predictor strategies.

- [ ] **Step 6: Write and run end-to-end equivalence tests**

Test the same alpha rows twice:

```python
emulator.predict(parameters=rows, reconstruct_wavefunctions=True)
full = emulator.predictions
emulator.predict(parameters=rows, reconstruct_wavefunctions=False)
packed = emulator.predictions
assert packed.wavefunctions == {}
for channel in full.coefficients:
    assert np.allclose(full.coefficients[channel], packed.coefficients[channel])
assert np.allclose(full.smatrix.splus, packed.smatrix.splus)
assert np.allclose(full.smatrix.sminus, packed.smatrix.sminus)
assert np.allclose(full.cross_sections.values, packed.cross_sections.values)
```

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k "packed or reconstruct_wavefunctions"
ruff check lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
```

Expected: all packed equivalence tests pass.

- [ ] **Step 7: Commit**

```bash
git add lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git commit -m "feat: add packed LROM cross-section inference"
```

---

### Task 5: Persist new v2 state safely

**Files:**
- Modify: `tests/test_v2_archive_method.py`
- Modify: `lrom_legacy/v2_0/__init__.py`

**Interfaces:**
- Consumes: channel predictor mappings, constant vectors, packed asymptotic
  arrays.
- Produces: artifact schema 2 with schema-1 compatibility.

- [ ] **Step 1: Write failing schema-2 round-trip tests**

Train the smallest valid \(l=(0,1)\) cross-section model, save it, reload it,
and compare channel predictor arrays, constant vectors, coefficients,
\(S\)-matrices, and differential cross sections for two alpha rows.

Inspect the zip and assert:

```python
with zipfile.ZipFile(path) as archive:
    metadata = json.loads(archive.read("metadata.json"))
    arrays = np.load(io.BytesIO(archive.read("arrays.npz")), allow_pickle=False)
assert metadata["artifact_schema"] == 2
assert all(value.dtype != object for value in arrays.values())
assert metadata["predictor"]["layout"] == "by_channel"
```

Add a schema-1 compatibility fixture by saving a current zero-intercept
single-channel model before changing the writer, or construct the two-file
zip from fixed numeric arrays in the test. Loading it must produce a zero
constant vector.

- [ ] **Step 2: Run artifact tests and verify RED**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k artifact
```

Expected: current writer rejects `full_woods-saxon` and schema 2.

- [ ] **Step 3: Add stable channel-key encoding**

Use JSON-safe names:

```python
def _channel_token(channel):
    return f"l{channel}" if isinstance(channel, int) else f"l{channel[0]}_s{channel[1]}"

def _channel_from_meta(value):
    return int(value) if isinstance(value, int) else tuple(int(item) for item in value)
```

Metadata stores each key as either an integer or a two-integer list. Array
names use `_channel_token`.

- [ ] **Step 4: Write schema-2 predictor and RF arrays**

Set `ARTIFACT_SCHEMA = 2`. For shared predictors, keep one predictor block
and set `layout="shared"`. For channel predictors, write every
`PredictorState` under its channel token and set `layout="by_channel"`.

For every RF model, write `rf_constant_vector` in addition to existing
matrices/vectors/singular values. Write packed asymptotic arrays for trained
cross-section models.

- [ ] **Step 5: Load schema 1 and schema 2**

Accept `{1, 2}`:

- schema 1: existing shared predictor arrays, integer channels, zero RF
  constant vectors;
- schema 2 shared: same logical state with explicit constants;
- schema 2 by-channel: rebuild the predictor mapping and tuple channel keys.

For portable effective-interaction features, evaluate the registered
full-Woods-Saxon central and radial spin-orbit forms at stored physical
radii, apply the channel \(l\cdot s\) factor, and divide by `e_com`. Compare
this path to live `interaction.tilde` in the round-trip test.

For the approved neutron observable, assemble the differential cross section
from packed \(S_l^\pm\), stored kinematics, and the ROSE neutral-scattering
utility. Charged portable observable assembly raises a specific unsupported
artifact error until its Coulomb state is serialized.

- [ ] **Step 6: Run artifact and broader compatibility tests**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k artifact
pytest -q ../tests/test_lrom_artifacts.py
pytest -q ../tests/test_v2_0_shell_characterization.py
ruff check lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
```

Expected: v2 schema tests pass; public v1.2 schema remains unchanged at 1.

- [ ] **Step 7: Commit**

```bash
git add lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git commit -m "feat: persist archive-method v2 models"
```

---

### Task 6: Update Notebook 02 contracts and full-profile workflow

**Files:**
- Modify: `tests/test_notebook02_cross_sections.py`
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`

**Interfaces:**
- Consumes: Tasks 1-5 v2 APIs.
- Produces: executed full-profile Notebook 02 and its fixed result summary.

- [ ] **Step 1: Record protected-file hashes**

Run:

```bash
shasum -a 256 lrom/__init__.py lrom_legacy/v1_2/__init__.py notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
```

Copy the three hashes into the task notes and compare them again in Task 8.

- [ ] **Step 2: Change notebook contract tests first**

Replace old shared-potential expectations with:

```python
assert 'predictor="effective-interaction"' in text
assert "reconstruct_wavefunctions=False" in text
assert "minimum_radius=0.5" in text or "selected_radii >= 0.5" in text
assert "median_pointwise_relative_error" in text
assert "maximum_over_angle_relative_error" in text
assert "np.median(pointwise_relative_error" in text
assert "np.max(pointwise_relative_error" in text
assert "old_v2_results" in text
assert "archive_lrom_results" in text
```

Keep the existing assertions for explicit parked-v2 import, shared alpha rows,
notebook-owned `InteractionEIMSpace`, free-reference `CustomBasis`,
all-channel helpers, named alpha selections, sparse headings, and figure
markers.

Add a separation assertion that the v2 source contains neither
`InteractionEIMSpace` nor a ROSE comparison result table.

- [ ] **Step 3: Run notebook tests and verify RED**

Run:

```bash
pytest -q tests/test_notebook02_cross_sections.py
```

Expected: failures for the new predictor, metrics, packed prediction, and
ablation names.

- [ ] **Step 4: Update the LROM construction cells**

Keep one old-v2 default model with `predictor="potential"` solely for the
controlled ablation. Change the primary grid to:

```python
emulator.train(
    basis_size=n_phi,
    predictor="effective-interaction",
    predictor_count=predictor_count,
    observable="cross_section",
    angles_degrees=ANGLES_DEGREES,
)
```

Generate training/testing results from package-owned testing state. For CAT
timing, call:

```python
emulator.predict(
    parameters=timing_rows,
    reconstruct_wavefunctions=False,
)
```

Do not call this path from the ROSE timing helper.

- [ ] **Step 5: Update predictor visualization**

Display one row per \((l,j)\) channel or a compact channel-colored overlay.
Use `state.selected_radii` on the x-axis and label it `r [fm]`. Do not expose
\(\rho\). Retain the central/spin-orbit potential rainbow as physical context
and add a small table listing channel keys and selected radii.

- [ ] **Step 6: Add the controlled ablation**

On identical default basis size 6, predictor count 8, and testing rows,
report:

- current shared central/spin-orbit potential predictor v2;
- archive effective-interaction predictor v2;
- each method's median pointwise test error;
- each method's maximum-over-angle diagnostic;
- complete online cross-section time.

Assert the archive method's primary error is lower before accepting the
notebook output.

- [ ] **Step 7: Change the CAT metric**

Define:

```python
def median_pointwise_relative_error(prediction, reference):
    pointwise = pointwise_relative_error(prediction, reference)
    return np.median(pointwise, axis=1)


def maximum_over_angle_relative_error(prediction, reference):
    pointwise = pointwise_relative_error(prediction, reference)
    return np.max(pointwise, axis=1)
```

Use the median of the first array as CAT y. Keep the second array in the
summary/robustness table and error figure. Label both explicitly.

- [ ] **Step 8: Preserve separate ROSE workflow and figures**

Retain notebook-local:

- ROSE interaction EIM construction;
- free-reference `CustomBasis`;
- `exact_smatrix_all_channels`;
- `emulated_smatrix_all_channels`;
- `cross_section_from_smatrix`;
- ROSE timing helper.

Update representative alpha selections A/B/C to show exact, LS floor, ROSE,
and archive-method LROM curves plus matching pointwise-error panels.

- [ ] **Step 9: Execute the full notebook**

Use the environment already validated for Notebook 02:

```bash
MPLCONFIGDIR=/tmp/lrom-mpl jupyter nbconvert \
  --to notebook \
  --execute notebooks/02_rose_vs_lrom_cross_sections.ipynb \
  --output 02_rose_vs_lrom_cross_sections.ipynb \
  --ExecutePreprocessor.timeout=3600
```

Expected:

- all code cells execute;
- zero error outputs;
- archive-method default primary error is below old-v2 default;
- at least one representative grid configuration beats or closely matches
  ROSE;
- all required figures and tables are stored.

- [ ] **Step 10: Inspect every stored figure**

Extract PNG outputs to `tmp/notebook02-archive-method-figures/` and inspect
them with the image viewer. Confirm:

- predictor x-axes are physical fm;
- exact/LS/ROSE/LROM legends are readable;
- diffraction-minimum error panels are not clipped;
- the CAT legend is outside dense data;
- no title or legend overlaps;
- alpha selections A/B/C match their displayed parameter rows.

Revise and re-execute if any figure fails.

- [ ] **Step 11: Write the machine-readable summary**

Create `outputs/notebook02_archive_method_summary.json` directly from the
displayed notebook variables:

```python
summary = {
    "profile": {
        "l_max": L_MAX,
        "half_width": HALF_WIDTH,
        "train": N_TRAIN,
        "test": N_TEST,
    },
    "default": {
        "basis_size": DEFAULT_BASIS_SIZE,
        "compression": DEFAULT_COMPRESSION_SIZE,
        "old_v2_median_pointwise_error": float(
            np.median(old_v2_results["median_pointwise_error"])
        ),
        "archive_lrom_median_pointwise_error": float(
            np.median(default_lrom_result["median_pointwise_error"])
        ),
        "rose_median_pointwise_error": float(
            np.median(default_rose_result["median_pointwise_error"])
        ),
    },
    "cat_rows": summary_table.to_dict(orient="records"),
    "packed_equivalence_max_abs": packed_equivalence_max_abs,
}
Path("outputs/notebook02_archive_method_summary.json").write_text(
    json.dumps(summary, indent=2)
)
```

This file is execution evidence, not a notebook generator.

- [ ] **Step 12: Run tests and commit**

Run:

```bash
pytest -q tests/test_notebook02_cross_sections.py
pytest -q tests/test_v2_archive_method.py
ruff check lrom_legacy/v2_0/__init__.py tests
git diff --check
git add notebooks/02_rose_vs_lrom_cross_sections.ipynb tests/test_notebook02_cross_sections.py outputs/notebook02_archive_method_summary.json
git commit -m "feat: apply archive LROM method in notebook 02"
```

---

### Task 7: Mirror the method in benchmark 03

**Files:**
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
- Modify: `tests/test_notebook02_cross_sections.py`

**Interfaces:**
- Consumes: Notebook 02 method and fixed full-profile summary.
- Produces: independently executed reduced benchmark and optional full-profile
  parity evidence.

- [ ] **Step 1: Add benchmark-specific failing assertions**

Require:

```python
assert 'predictor="effective-interaction"' in text
assert "reconstruct_wavefunctions=False" in text
assert "median_pointwise_relative_error" in text
assert "maximum_over_angle_relative_error" in text
assert "archive_lrom_results" in text
assert "old_v2_results" in text
assert '"reduced": (120, 30)' in text
assert '"full": (200, 100)' in text
```

Also require a reduced-profile numerical assertion that archive-method LROM
beats the old-v2 default on identical rows.

- [ ] **Step 2: Run benchmark contract and verify RED**

Run:

```bash
pytest -q tests/test_notebook02_cross_sections.py -k benchmark03
```

Expected: failures for the archive-method strings.

- [ ] **Step 3: Apply the Notebook 02 computation path**

Mirror the same:

- channel predictor construction;
- intercept-bearing LROM training;
- old-v2 ablation;
- packed CAT timing;
- primary and robustness metrics;
- separate notebook-owned ROSE workflow;
- representative alpha selections and figures.

Only profile sizes differ. Do not copy Notebook 02 stored numerical results
into benchmark output.

- [ ] **Step 4: Execute reduced benchmark**

Run:

```bash
MPLCONFIGDIR=/tmp/lrom-mpl LROM_BENCHMARK_PROFILE=reduced \
  jupyter nbconvert \
  --to notebook \
  --execute notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb \
  --output benchmark_03.ipynb \
  --ExecutePreprocessor.timeout=3600
```

Expected: zero error outputs and all numerical assertions pass.

- [ ] **Step 5: Validate full-profile parity without overwriting reduced output**

Execute a temporary copy under `/tmp` with
`LROM_BENCHMARK_PROFILE=full`. Compare its deterministic alpha table,
accuracy rows, and packed-equivalence values to
`outputs/notebook02_archive_method_summary.json`. Timing may differ and is
not required to be identical.

- [ ] **Step 6: Inspect benchmark figures**

Extract all stored benchmark PNGs and apply the same visual checks as
Notebook 02. Confirm the output explicitly reports the active profile.

- [ ] **Step 7: Run contracts and commit**

Run:

```bash
pytest -q tests/test_notebook02_cross_sections.py
ruff check tests
git diff --check
git add notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb tests/test_notebook02_cross_sections.py
git commit -m "test: benchmark archive-method LROM cross sections"
```

---

### Task 8: Final scientific validation and change ledger

**Files:**
- Modify: `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify: `_memory/Daily Notes/2026-07-26.md` outside the repository
- Modify: `_memory/Context/active-state.md` outside the repository

**Interfaces:**
- Consumes: all executed package/notebook/benchmark evidence.
- Produces: final verified documentation and handoff.

- [ ] **Step 1: Run focused tests**

```bash
pytest -q tests/test_v2_archive_method.py tests/test_notebook02_cross_sections.py
```

Expected: all pass.

- [ ] **Step 2: Run broader parent suite**

```bash
pytest -q ../tests tests
```

Expected: all relevant tests pass. If parent-path layout tests fail, classify
each failure and show that it is unrelated before proceeding.

- [ ] **Step 3: Run static checks**

```bash
ruff check lrom lrom_legacy tests ../tests
git diff --check
```

Expected: pass, with only already documented notebook path-setup exceptions
if the repository configuration excludes them.

- [ ] **Step 4: Verify executed notebook health**

Inspect both notebook JSON documents and assert:

- zero outputs whose `output_type` is `"error"`;
- expected code-cell execution counts;
- all required figure markers have stored image output;
- no banned distribution-rank wording;
- all displayed alpha tables use named selections.

- [ ] **Step 5: Verify protected hashes and repository scope**

Re-run:

```bash
shasum -a 256 lrom/__init__.py lrom_legacy/v1_2/__init__.py notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
git status --short
git diff HEAD~7 --name-only
```

Confirm public v1.2, Notebook 01, and scientific archive files were not
modified.

- [ ] **Step 6: Document the measured result**

Add to `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`:

- old-v2 versus archive-method default accuracy;
- archive-method versus ROSE representative accuracy;
- packed versus full inference equivalence;
- complete online timing;
- explanation of per-channel interaction predictors and \(b_0\);
- explicit ROSE comparison boundary;
- current \(l=3\), ±20% scope versus archive \(l_{\max}=10\) caveat.

- [ ] **Step 7: Update memory handoff**

Append `[LROM_Project]` entries to the daily note and update the physics
checklist/last-session handoff with modified files, commits, numerical
results, validation commands, and the unpushed branch state.

- [ ] **Step 8: Commit documentation**

```bash
git add docs/LROM_ARCHITECTURE_UNDERSTANDING.md
git commit -m "docs: record archive-method LROM validation"
```

- [ ] **Step 9: Final verification after the last commit**

```bash
pytest -q tests/test_v2_archive_method.py tests/test_notebook02_cross_sections.py
ruff check lrom_legacy/v2_0/__init__.py tests
git status --short --branch
git log -8 --oneline
```

Expected: focused tests and Ruff pass; worktree is clean; branch remains
ahead of origin and is not pushed without explicit user authorization.

---

## Self-Review Record

- Spec coverage: all approved package, ROSE-boundary, notebook, CAT,
  artifact, benchmark, and reporting requirements map to Tasks 1-8.
- Completeness scan: every implementation and error path names the concrete
  function, assertion, or command required.
- Type consistency: the predictor mapping, `constant_vector`,
  `include_intercept`, and `reconstruct_wavefunctions` names are consistent
  across package, tests, notebooks, and persistence.
- Scope: Notebook 01, public v1.2, and the scientific archive are validation
  targets only and are never implementation targets.
