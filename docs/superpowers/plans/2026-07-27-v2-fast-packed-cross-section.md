# Parked v2 Fast-Packed Cross-Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce parked-v2 cross-section online latency by compiling the invariant ROSE/LROM arrays once, evaluating Woods-Saxon predictors in one flattened operation, solving all reduced systems as a batch, and reusing ROSE angular tables without changing scientific results.

**Architecture:** Keep the implementation private, flat, and functional inside `lrom_legacy/v2_0/__init__.py`. Add one derived dictionary cache on `LROM`, populate it eagerly after cross-section training and lazily after schema-2 loading, and invalidate it on sampling or retraining. The optimized path is available only for the registered `full_woods-saxon` model and observable-only cross-section prediction; custom potentials and wavefunction reconstruction continue through the existing reference path.

**Tech Stack:** Python 3.12, NumPy, ROSE, pytest, Jupyter/nbconvert, Ruff.

## Global Constraints

- Work only in `/Users/Kitkat/Documents/Documents-Agent/LROM_Project/lrom_git`, except for the final architecture note at `../docs/LROM_ARCHITECTURE_UNDERSTANDING.md` and required memory logs.
- Do not edit public v1.2, Notebook 01, the scientific archive, `.git/`, or `.obsidian/`.
- Preserve Ca40(n,n), 14.1 MeV, `l=0..3`, the 600-point mesh, seed 1204, the `±20%` alpha ranges, all sampled alpha rows, and all basis/predictor grids.
- Do not use “percentile” in notebook prose or output. Use “alpha selection” and the existing A/B/C labels.
- Keep ROSE comparison construction, training, prediction, and timing notebook-owned. The package may use ROSE only for its existing physical cross-section assembly.
- Preserve the current ROSE-interaction evaluation radii. Do not correct the deferred `k=0.8073091` versus `k=0.8102365 fm^-1` mismatch in this work.
- Do not add a dependency, public class, artifact schema revision, or persisted cache arrays.
- Use `numpy.ndarray` values and simple dictionaries; do not introduce defensive wrapper hierarchies.
- Create tests before implementation in each task. Run the new focused test and confirm that it fails for the intended reason before writing production code.
- Use `apply_patch` for file edits.
- Leave the existing untracked `tmp/` directory untouched and out of every commit.

---

## Task 1: Compile and invalidate immutable cross-section state

**Files:**

- Modify: `tests/test_v2_archive_method.py`
- Modify: `lrom_legacy/v2_0/__init__.py` near `LROM.__init__`, `_clear_training_state`, `sampling`, `train`, and the existing packed helpers

- [ ] **Step 1: Add a focused trained-cache lifecycle test**

Add a test using `small_cross_section_emulator()` that asserts:

```python
def test_cross_section_cache_is_compiled_and_invalidated() -> None:
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_sizes=2,
        predictor_count=3,
        observable="cross_section",
        validate=False,
    )

    cache = emulator._packed_cross_section_cache
    assert cache is not None
    channel_count = len(emulator.rf_lrom)
    assert cache["channel_keys"] == tuple(emulator.rf_lrom)
    assert cache["centers"].shape == (channel_count, 3)
    assert cache["matrices"].shape == (channel_count, 3, 2, 2)
    assert cache["vectors"].shape == (channel_count, 3, 2)
    assert cache["constants"].shape == (channel_count, 2)
    assert cache["identity"].shape == (2, 2)
    assert cache["evaluation_radii"].shape == (channel_count, 3)

    original_cache = cache
    emulator._clear_training_state()
    assert emulator._packed_cross_section_cache is None
    assert emulator._sae_cache is None
    assert original_cache is not emulator._packed_cross_section_cache
```

The existing helper returns a sampled, untrained emulator, so no helper signature change is needed.

- [ ] **Step 2: Run the lifecycle test and verify the expected failure**

Run:

```bash
pytest -q tests/test_v2_archive_method.py::test_cross_section_cache_is_compiled_and_invalidated
```

Expected: failure because `LROM` has no `_packed_cross_section_cache`.

- [ ] **Step 3: Add the private cache attribute and invalidation**

In `LROM.__init__`, initialize:

```python
self._packed_cross_section_cache: dict[str, Any] | None = None
```

In `_clear_training_state`, clear both derived caches:

```python
self._sae_cache = None
self._packed_cross_section_cache = None
```

Ensure both `sampling()` and `train()` reach `_clear_training_state()` before replacing state. Do not add an independent invalidation policy that can diverge from the existing lifecycle.

- [ ] **Step 4: Add a single interaction-space resolver**

Add:

```python
def _prediction_interaction_space(*, emulator: LROM) -> Mapping[PartialWave, Any]:
    if emulator.samples is not None:
        return emulator.samples.interaction_space
    interactions = emulator._portable_interaction_cache
    if interactions is None:
        rose = _import_rose()
        options = {
            "l_max": max(emulator.partial_waves),
            "n_theta": len(emulator.parameter_names),
            "mu": emulator.kinematics.mu,
            "energy": emulator.kinematics.e_com,
        }
        # Move the existing potential-name options branch here unchanged.
        interactions = rose.InteractionSpace(**options)
        emulator._portable_interaction_cache = interactions
    return interactions
```

Initialize `_portable_interaction_cache` to `None` in `LROM.__init__`. Move the exact existing `full_woods-saxon`, `woods-saxon`, and real-Woods-Saxon options branch from `effective_interaction_features()` into this helper. Replace duplicated live/portable interaction-space selection in `effective_interaction_features()` and `_scattering_amplitude_emulator()` with the helper without changing either branch’s coordinate formula.

- [ ] **Step 5: Implement `_compile_cross_section_cache`**

Add `_compile_cross_section_cache(*, emulator: LROM) -> dict[str, Any]` and `_cross_section_cache(*, emulator: LROM) -> dict[str, Any]`. The latter contains only the lazy guard: if `_packed_cross_section_cache` is `None`, assign the compiler result, then return the stored dictionary.

The compiler must:

1. Require trained cross-section state and raise the existing `LROMStateError` style if it is absent.
2. Preserve `tuple(emulator.rf_lrom)` channel ordering exactly; this existing mapping already groups the scalar and spin-coupled channels in partial-wave order.
3. Require one common basis size and one common predictor count across channels.
4. Stack these exact keys:

```text
channel_keys
partial_waves
potential_name
evaluation_radii
ell
spin
ldots
centers
scales
matrices
vectors
constants
identity
asymptotic_values
asymptotic_derivatives
hminus
hplus
hminus_derivative
hplus_derivative
s0
plus_channel_indices
plus_ell_indices
minus_channel_indices
minus_ell_indices
sae
```

5. Derive live evaluation coordinates exactly as the reference path does:

```python
rho_points = emulator.samples.mesh.rho[predictor.selected_indices]
evaluation_radii[channel_index] = rho_points / interaction.k
```

6. Derive portable evaluation coordinates exactly as the loaded reference path does:

```python
rho_points = predictor.selected_radii * emulator.kinematics.k
evaluation_radii[channel_index] = rho_points / interaction.k
```

7. Read each channel’s `l_dot_s` from the interaction’s spin-orbit term rather than recomputing it.
8. Compute `H^-`, `H^+`, and derivatives once from the initialized ROSE basis/solver values.
9. Encode the existing integer-`j` versus half-integer-`j` `splus`/`sminus` placement rules as integer index arrays.
10. Make arrays contiguous with explicit `float`, `complex128`, or integer dtypes as appropriate; do not create object-dtype arrays.

- [ ] **Step 6: Compile eagerly after cross-section training**

At the end of successful `train(..., observable="cross_section")`, after all fitted predictor and basis states have been assigned:

```python
_cross_section_cache(emulator=self)
```

Do not compile for wavefunction-only training. A loaded artifact remains lazy because `load()` must not perform hidden ROSE initialization work.

- [ ] **Step 7: Run focused and neighboring tests**

Run:

```bash
pytest -q \
  tests/test_v2_archive_method.py::test_cross_section_cache_is_compiled_and_invalidated \
  tests/test_v2_archive_method.py::test_effective_interaction_training_uses_one_feature_set_per_channel
```

Expected: both pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git commit -m "perf(v2): compile cross-section prediction state"
```

---

## Task 2: Flatten full Woods-Saxon predictor evaluation

**Files:**

- Modify: `tests/test_v2_archive_method.py`
- Modify: `lrom_legacy/v2_0/__init__.py` near `effective_interaction_features`

- [ ] **Step 1: Add live multi-sample feature-parity coverage**

Add:

```python
def test_flattened_full_woods_saxon_features_match_reference() -> None:
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )
    cache = v2._cross_section_cache(emulator=emulator)
    rows = np.asarray(emulator.samples.design.testing.values[:2], dtype=float)
    reference = v2.effective_interaction_features(
        emulator=emulator,
        predictors=emulator.predictors,
        values=rows,
    )

    actual = v2._packed_effective_interaction_features(
        emulator=emulator,
        values=rows,
        cache=cache,
    )
    expected = np.stack(
        [reference[channel] for channel in cache["channel_keys"]],
        axis=1,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
```

Expected shape: `(2, channels, predictor_count)`.

- [ ] **Step 2: Add fallback coverage for nonregistered potential behavior**

Copy the cache dictionary, set its private fast-potential marker to `None`, call `_packed_effective_interaction_features`, and assert the result still equals the per-channel reference stack. This verifies that unsupported/custom potential registrations continue to use the generic dispatch path.

- [ ] **Step 3: Run the new feature tests and verify failure**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k "flattened_full_woods_saxon_features or packed_features_fall_back"
```

Expected: failure because `_packed_effective_interaction_features` does not exist.

- [ ] **Step 4: Implement the flattened registered-potential kernel**

Add:

```python
def _packed_effective_interaction_features(
    *,
    emulator: LROM,
    values: NDArray[np.float64],
    cache: Mapping[str, Any],
) -> NDArray[np.complex128]:
    rows = np.asarray(values, dtype=float)
    if rows.ndim == 1:
        rows = rows[np.newaxis, :]
    if cache["potential_name"] != "full_woods-saxon":
        return _generic_packed_effective_interaction_features(
            emulator=emulator,
            rows=rows,
        )
    shape = cache["evaluation_radii"].shape
    radii = cache["evaluation_radii"].reshape(-1)
    ldots = np.repeat(cache["ldots"], shape[1])
    raw = np.asarray(
        [
            full_woods_saxon(radii, row)
            + ldots * full_woods_saxon_spin_orbit(radii, row)
            for row in rows
        ],
        dtype=np.complex128,
    ).reshape(rows.shape[0], *shape)
    return (
        raw - cache["centers"][None, :, :]
    ) / cache["scales"][None, :, :]
```

For the registered fast branch:

1. Flatten `cache["evaluation_radii"]` and repeat `cache["ldots"]` over the predictor axis.
2. Evaluate the existing central and spin-orbit Woods-Saxon NumPy functions once per parameter row on that flattened radius array.
3. Compute `central + ldots * spin_orbit`, reshape to `(samples, channels, predictor_count)`, and normalize with:

```python
(raw_features - cache["centers"][None, :, :]) / cache["scales"][None, :, :]
```

4. Preserve row order and channel order.
5. Do not round or expose the evaluation radii.
6. Do not route this operation through new ROSE interaction objects.

The parameter rows already follow `emulator.parameter_names`, which is the exact positional contract used by `full_woods_saxon()` and `full_woods_saxon_spin_orbit()`. Do not introduce a second column mapping.

- [ ] **Step 5: Keep the generic branch as the reference implementation**

Implement `_generic_packed_effective_interaction_features()` only as a thin stack over the current `effective_interaction_features()` function. It is a compatibility fallback, not a second optimized framework.

- [ ] **Step 6: Run feature parity and existing predictor tests**

Run:

```bash
pytest -q \
  tests/test_v2_archive_method.py::test_flattened_full_woods_saxon_features_match_reference \
  tests/test_v2_archive_method.py::test_packed_features_fall_back_to_reference \
  tests/test_v2_archive_method.py::test_effective_interaction_predictors_are_channel_specific_and_physical
```

Expected: all pass with the `2e-13` feature tolerance.

- [ ] **Step 7: Commit Task 2**

```bash
git add lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git commit -m "perf(v2): flatten Woods-Saxon predictor evaluation"
```

---

## Task 3: Batch reduced solves and pack coefficient-to-S conversion

**Files:**

- Modify: `tests/test_v2_archive_method.py`
- Modify: `lrom_legacy/v2_0/__init__.py` near `_packed_coefficients` and `_packed_smatrix_from_coefficients`

- [ ] **Step 1: Add multi-sample reduced-coordinate parity**

Extend the existing packed-coordinate test to compare two parameter rows:

```python
features = v2._packed_effective_interaction_features(
    emulator=emulator,
    values=rows,
    cache=cache,
)
actual = v2._solve_packed_coordinates(features=features, cache=cache)

expected = np.stack(
    [
        v2.solve_rf_lrom(
            model=emulator.rf_lrom[channel],
            predictors=features[:, channel_index, :],
        )
        for channel_index, channel in enumerate(cache["channel_keys"])
    ],
    axis=1,
)
np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
```

Assert the result shape is `(samples, channels, basis_size)`.

- [ ] **Step 2: Add direct packed S-matrix parity**

Extend `test_packed_smatrix_matches_scalar_channel_conversion` to call:

```python
actual = v2._smatrix_from_packed_coordinates(
    coordinates=coordinates,
    cache=cache,
)
```

Compare `actual.splus` and `actual.sminus` with the current scalar/reference `_packed_smatrix_from_coefficients()` output at `rtol=2e-12, atol=2e-12`. Include two samples so index placement is exercised across the leading dimension.

- [ ] **Step 3: Run the tests and verify failure**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k "packed_coefficients_match or packed_smatrix_matches"
```

Expected: failure because the new batched helpers do not exist.

- [ ] **Step 4: Implement `_solve_packed_coordinates`**

Add:

```python
def _solve_packed_coordinates(
    *,
    features: NDArray[np.complex128],
    cache: Mapping[str, Any],
) -> NDArray[np.complex128]:
    systems = cache["identity"][None, None, :, :] + np.einsum(
        "sck,ckij->scij",
        features,
        cache["matrices"],
        optimize=True,
    )
    rhs = cache["constants"][None, :, :] + np.einsum(
        "sck,ckj->scj",
        features,
        cache["vectors"],
        optimize=True,
    )
    return np.linalg.solve(systems, rhs[..., None])[..., 0]
```

The final function must make exactly one `np.linalg.solve` call for the full `(samples, channels)` batch.

- [ ] **Step 5: Implement `_smatrix_from_packed_coordinates`**

Add:

```python
def _smatrix_from_packed_coordinates(
    *,
    coordinates: NDArray[np.complex128],
    cache: Mapping[str, Any],
) -> SmatrixState:
    coefficients = np.concatenate(
        (
            np.ones((*coordinates.shape[:-1], 1), dtype=np.complex128),
            coordinates,
        ),
        axis=-1,
    )
    phi = np.einsum(
        "scb,cb->sc",
        coefficients,
        cache["asymptotic_values"],
        optimize=True,
    )
    phi_prime = np.einsum(
        "scb,cb->sc",
        coefficients,
        cache["asymptotic_derivatives"],
        optimize=True,
    )
    r_matrix = phi / (cache["s0"][None, :] * phi_prime)
    s_flat = (
        cache["hminus"][None, :]
        - cache["s0"][None, :]
        * r_matrix
        * cache["hminus_derivative"][None, :]
    ) / (
        cache["hplus"][None, :]
        - cache["s0"][None, :]
        * r_matrix
        * cache["hplus_derivative"][None, :]
    )
    sample_count = coordinates.shape[0]
    ell_count = len(cache["partial_waves"])
    splus = np.zeros((sample_count, ell_count), dtype=np.complex128)
    sminus = np.zeros_like(splus)
    splus[:, cache["plus_ell_indices"]] = (
        s_flat[:, cache["plus_channel_indices"]]
    )
    sminus[:, cache["minus_ell_indices"]] = (
        s_flat[:, cache["minus_channel_indices"]]
    )
    return SmatrixState(
        partial_waves=cache["partial_waves"],
        splus=splus,
        sminus=sminus,
    )
```

Contract coefficients with cached asymptotic values and derivatives, then reproduce the existing Wronskian/Hankel equation using `hminus`, `hplus`, their derivatives, and `s0`. Allocate `splus` and `sminus` once and assign channel results with the four integer index arrays.

Return the existing `SmatrixState`; do not add a new result type.

- [ ] **Step 6: Preserve the scalar helpers as testable reference paths**

Do not delete `_packed_coefficients()` or `_packed_smatrix_from_coefficients()` in this task. Mark neither as deprecated. They remain private numerical references used by parity tests and by reconstruction/custom fallbacks.

- [ ] **Step 7: Run focused numerical tests**

Run:

```bash
pytest -q \
  tests/test_v2_archive_method.py::test_packed_coefficients_match_individual_channel_solves \
  tests/test_v2_archive_method.py::test_packed_smatrix_matches_scalar_channel_conversion
```

Expected: pass with no tolerance relaxation.

- [ ] **Step 8: Commit Task 3**

```bash
git add lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git commit -m "perf(v2): batch reduced solves and S matrices"
```

---

## Task 4: Integrate the fast observable path and schema-2 loading

**Files:**

- Modify: `tests/test_v2_archive_method.py`
- Modify: `lrom_legacy/v2_0/__init__.py` near `_scattering_amplitude_emulator`, `_cross_section_prediction`, `predict`, `save`, and `load`

- [ ] **Step 1: Add end-to-end trained prediction parity**

Add:

```python
def test_fast_observable_prediction_matches_reference_path() -> None:
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )
    rows = [
        dict(zip(emulator.parameter_names, row))
        for row in (
            emulator.samples.design.testing.values[1],
            emulator.samples.design.testing.values[-2],
        )
    ]

    emulator.predict(parameters=rows, reconstruct_wavefunctions=False)
    fast = emulator.predictions
    emulator.predict(parameters=rows, reconstruct_wavefunctions=True)
    reference = emulator.predictions
    cache = v2._cross_section_cache(emulator=emulator)

    for channel in cache["channel_keys"]:
        np.testing.assert_allclose(
            fast.coefficients[channel],
            reference.coefficients[channel],
            rtol=2e-12,
            atol=2e-12,
        )
    np.testing.assert_allclose(
        fast.smatrix.splus,
        reference.smatrix.splus,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        fast.smatrix.sminus,
        reference.smatrix.sminus,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        fast.cross_sections.values,
        reference.cross_sections.values,
        rtol=2e-11,
        atol=1e-10,
    )
```

Use the `v2.` prefix for private helpers in the test module. The reconstruction flag leaves wavefunctions populated only on the reference result; compare only shared numerical outputs.

- [ ] **Step 2: Add cached-angle regression coverage**

Monkeypatch or wrap the cached SAE’s `calculate_xs` and capture the `angles` keyword:

```python
assert calls
assert all(call.get("angles") is None for call in calls)
```

This test must fail while `_cross_section_prediction()` still passes the unchanged angle grid explicitly.

- [ ] **Step 3: Add schema-2 cross-section artifact round-trip coverage**

Extend the existing artifact test or add:

```python
def test_cross_section_artifact_lazily_rebuilds_fast_cache(tmp_path: Path) -> None:
    trained = small_cross_section_emulator()
    trained.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )
    row = dict(
        zip(
            trained.parameter_names,
            trained.samples.design.testing.values[2],
        )
    )
    trained.predict(parameters=row, reconstruct_wavefunctions=False)
    expected = trained.predictions
    artifact = tmp_path / "cross-section.lrom"
    trained.save(path=artifact)

    loaded = v2.load(path=artifact)
    assert loaded._packed_cross_section_cache is None

    loaded.predict(parameters=row, reconstruct_wavefunctions=False)
    actual = loaded.predictions
    assert loaded._packed_cross_section_cache is not None
    np.testing.assert_allclose(
        actual.cross_sections.values,
        expected.cross_sections.values,
        rtol=2e-11,
        atol=1e-10,
    )
```

Inspect `metadata.json` in the ZIP and assert `metadata["artifact_schema"] == 2`. Also compare loaded/trained `splus` and `sminus` at `2e-12`.

- [ ] **Step 4: Run the new integration tests and verify intended failures**

Run:

```bash
pytest -q tests/test_v2_archive_method.py -k "fast_observable or cached_angles or artifact_lazily"
```

Expected failures:

- observable-only prediction still uses the general path;
- `angles` is explicitly passed;
- loaded cross-section artifact cannot yet rebuild all derived ROSE state or cache.

- [ ] **Step 5: Route eligible predictions through the packed path**

In the module-level `predict()` function, compute:

```python
options = emulator.training_options or {}
use_fast_observable_path = (
    options.get("observable") == "cross_section"
    and not reconstruct_wavefunctions
    and emulator.config.potential.name == "full_woods-saxon"
    and isinstance(emulator.predictors, Mapping)
)
```

When eligible:

1. get `_cross_section_cache(emulator=emulator)`;
2. compute `_packed_effective_interaction_features`;
3. compute `_solve_packed_coordinates`;
4. create the existing coefficient mapping from `cache["channel_keys"]`;
5. compute `_smatrix_from_packed_coordinates`;
6. compute cross sections with `_cross_sections_from_smatrix`; and
7. populate the existing `PredictionState` fields.

Use the current path unchanged when reconstructing wavefunctions or when the registered potential does not support the flattened kernel.

- [ ] **Step 6: Reuse cached ROSE angular tables**

Add:

```python
def _cross_sections_from_smatrix(
    *,
    emulator,
    values: np.ndarray,
    smatrix: SmatrixState,
) -> CrossSectionState:
    options = emulator.training_options or {}
    angles_degrees = np.asarray(options.get("angles_degrees"), dtype=float)
    angles = np.deg2rad(angles_degrees)
    sae = _scattering_amplitude_emulator(emulator=emulator)
    if not np.array_equal(np.asarray(sae.angles), angles):
        raise ValueError("ROSE angle cache does not match the trained grid")
    cross_sections = [
        sae.calculate_xs(splus, sminus, row).dsdo
        for row, splus, sminus in zip(values, smatrix.splus, smatrix.sminus)
    ]
    return CrossSectionState(
        angles_degrees=angles_degrees,
        values=np.asarray(cross_sections, dtype=float),
    )
```

Change `_cross_section_prediction()` to obtain its reference-path S matrix and delegate cross-section assembly to this helper. Do not pass `angles=` when the grid is unchanged.

- [ ] **Step 7: Make loaded schema-2 cross-section state reconstructible**

Refactor `_scattering_amplitude_emulator()` to build channel bases from either:

- live training wavefunctions, as it does now; or
- loaded `BasisState.phi0` and `BasisState.vectors`.

For the loaded branch, initialize the existing ROSE `CustomBasis` with:

```python
solutions = np.column_stack((basis_state.phi0, basis_state.vectors))
```

then overwrite its learned basis arrays with the artifact arrays exactly as the live branch already does. Build the channel solver from `_prediction_interaction_space()` and:

```python
base_solver = rose.SchroedingerEquation.make_base_solver(
    s_0=6.0 * np.pi,
    rk_tols=[1e-9, 1e-9],
    domain=np.asarray(
        [emulator.mesh.rho[0], emulator.mesh.rho[-1]],
        dtype=float,
    ),
)
solver = base_solver.clone_for_new_interaction(interaction)
```

These values reproduce the current default sampling solver used by the schema-2 study. This is a reconstruction of existing schema-2 state, not a schema change. The test must document that artifacts trained with nondefault `solver_options` cannot reconstruct cross-section asymptotics exactly because schema 2 does not persist those options; do not broaden this performance change into schema 3.

Do not persist `sae`, packed arrays, or angular tables. After `load()`, all derived caches must be `None`.

- [ ] **Step 8: Run all parked-v2 archive-method tests**

Run:

```bash
pytest -q tests/test_v2_archive_method.py
```

Expected: all pass. The maximum cross-section absolute difference in the explicit diagnostic assertion is `<= 1e-10`.

- [ ] **Step 9: Commit Task 4**

```bash
git add lrom_legacy/v2_0/__init__.py tests/test_v2_archive_method.py
git commit -m "perf(v2): use fast packed cross-section prediction"
```

---

## Task 5: Make Notebook 02 and Benchmark 03 timing fair and stable

**Files:**

- Modify: `tests/test_notebook02_cross_sections.py`
- Modify: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
- Modify: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`

- [ ] **Step 1: Strengthen notebook source-contract tests**

In `tests/test_notebook02_cross_sections.py`, assert both notebooks contain:

```text
TIMING_REPEATS = 3
TIMING_INNER_LOOPS = 20
time.perf_counter_ns()
/ TIMING_INNER_LOOPS
sae.calculate_xs(splus, sminus, parameters)
```

Assert neither notebook contains:

```text
calculate_xs(splus, sminus, parameters, angles=ANGLES_RAD)
```

Keep the existing “no percentile” assertion.

For Benchmark 03 only, require:

```text
LROM_BENCHMARK_L_MAX
L_MAX = int(os.environ.get("LROM_BENCHMARK_L_MAX", "3"))
if L_MAX not in (3, 10):
```

Notebook 02 must continue to contain literal `L_MAX = 3` and must not read this environment variable.

- [ ] **Step 2: Run contract tests and verify failure**

Run:

```bash
pytest -q tests/test_notebook02_cross_sections.py
```

Expected: failure because timing uses one call per repeat, passes `angles=`, and Benchmark 03 has no separate `l=10` diagnostic switch.

- [ ] **Step 3: Update Notebook 02 timing cells without adding prose**

Make targeted JSON source edits only. Add:

```python
TIMING_INNER_LOOPS = 20
```

In `cross_section_from_smatrix`, first assert:

```python
if not np.array_equal(sae.angles, ANGLES_RAD):
    raise ValueError("ROSE angle cache does not match the study grid.")
```

Then call `sae.calculate_xs(...)` without `angles=`.

For each timed LROM or ROSE case:

1. perform one untimed warm-up;
2. use `time.perf_counter_ns()`;
3. run the identical callable `TIMING_INNER_LOOPS` times inside each repeat;
4. divide elapsed nanoseconds by `1e9 * TIMING_INNER_LOOPS`; and
5. keep the minimum of `TIMING_REPEATS`.

Do not change markdown cells, alpha rows, case ordering, figures, or plot labels.

- [ ] **Step 4: Apply the same timing kernel to Benchmark 03**

Use the same warm-up, inner-loop, and cached-angle logic.

Add the separate diagnostic control:

```python
L_MAX = int(os.environ.get("LROM_BENCHMARK_L_MAX", "3"))
if L_MAX not in (3, 10):
    raise ValueError("LROM_BENCHMARK_L_MAX must be 3 or 10.")
```

The saved notebook must be executed and committed with the default `l=3`. The `l=10` mode is diagnostic only and must be executed to a temporary output path so it cannot overwrite the selected study’s saved results.

- [ ] **Step 5: Run notebook contract and compile tests**

Run:

```bash
pytest -q tests/test_notebook02_cross_sections.py
```

Expected: all pass.

- [ ] **Step 6: Commit Task 5 source changes**

```bash
git add \
  tests/test_notebook02_cross_sections.py \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
git commit -m "perf(notebooks): stabilize LROM and ROSE timing"
```

---

## Task 6: Execute scientific regressions and record performance evidence

**Files:**

- Modify generated outputs in: `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
- Modify generated outputs in: `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
- Modify: `../docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
- Modify: `_memory/Daily Notes/2026-07-27.md`
- Modify: `_memory/Context/active-state.md`

- [ ] **Step 1: Record protected-file hashes before notebook execution**

Run:

```bash
shasum -a 256 \
  notebooks/01_*.ipynb \
  ../scientific_archive/**/* 2>/dev/null
```

If shell globbing does not cover the archive safely, use `find` read-only:

```bash
find ../scientific_archive -type f -print0 | sort -z | xargs -0 shasum -a 256
```

Save the output under `/tmp/lrom-protected-before.sha256`, not in the repository or memory layer.

- [ ] **Step 2: Run static and unit verification before expensive notebooks**

Run:

```bash
ruff check lrom_legacy/v2_0/__init__.py tests
pytest -q tests/test_v2_archive_method.py tests/test_notebook02_cross_sections.py
pytest -q
```

Expected: Ruff clean and full pytest suite passing. Stop and diagnose any failure before notebook execution.

- [ ] **Step 3: Execute Notebook 02 in place**

Run:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb
```

Expected:

- 10 of 10 code cells execute;
- no error output;
- five figures remain;
- alpha rows and scientific accuracy tables match the previous committed output within floating-point tolerance;
- only online timing values and timing-derived CAT x-coordinates materially change.

- [ ] **Step 4: Execute Benchmark 03 default reduced profile in place**

Run:

```bash
LROM_BENCHMARK_PROFILE=reduced \
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
```

Expected:

- stored `L_MAX` is 3;
- `N_TRAIN=120`, `N_TEST=30`;
- no error output;
- five figures remain;
- the representative `n_phi=6`, `K=8` LROM timing is near 60–75 microseconds per sample on the current machine;
- LROM is faster than comparable notebook-owned ROSE timing rows.

The last two performance bullets are evidence targets, not CI assertions. If missed, profile the four known stages again before accepting the task.

- [ ] **Step 5: Execute the separate archive-condition diagnostic**

Execute Benchmark 03 to a temporary copy:

```bash
cp \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb \
  /tmp/lrom-benchmark-03-l10.ipynb
LROM_BENCHMARK_PROFILE=reduced \
LROM_BENCHMARK_L_MAX=10 \
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=7200 \
  /tmp/lrom-benchmark-03-l10.ipynb
```

Extract and record:

- LROM time for `n_phi=6`, `K=8`;
- ROSE time for the comparable `n_phi=6`, `n_U=8` case;
- the observed speed ratio;
- the fact that 21 channels were assembled.

Do not copy the temporary `l=10` output over the committed default notebook. Treat the archive’s historical 3–5x range as context, not a pass/fail threshold.

- [ ] **Step 6: Verify notebook scientific invariants programmatically**

Use a read-only Python/Jupyter inspection command to assert:

- both stored notebooks have 10 executed code cells and no error outputs;
- `ALPHA_LOW == -0.2`, `ALPHA_HIGH == 0.2`, `SEED == 1204`, `L_MAX == 3`;
- Notebook 02 stores `N_TRAIN == 200`, `N_TEST == 100`;
- Benchmark 03 stores `N_TRAIN == 120`, `N_TEST == 30`;
- every results table contains the existing alpha-selection labels and no “percentile”;
- figure count is five in each notebook.

Compare the pre-optimization notebook outputs at commit `cdea30d` with the new outputs using:

```bash
git show cdea30d:notebooks/02_rose_vs_lrom_cross_sections.ipynb
git show cdea30d:notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
```

Permit differences only in timing arrays, timing-derived speedups, CAT x-coordinates, and execution metadata. Cross sections must satisfy `max_abs_difference <= 1e-10`.

- [ ] **Step 7: Visually inspect all regenerated figures**

Render or extract the notebook PNG outputs to a temporary directory and inspect:

- Notebook 02 CAT plot;
- Notebook 02 three A/B/C cross-section panels;
- Notebook 02 error comparison;
- Benchmark 03 CAT plot;
- Benchmark 03 three A/B/C cross-section panels;
- Benchmark 03 error comparison.

Confirm physical radius labels remain in fm wherever radius appears, legends are readable, curves are not clipped, and CAT timing points reflect the optimized measurements.

- [ ] **Step 8: Verify protected files are unchanged**

Repeat the Step 1 hashes into `/tmp/lrom-protected-after.sha256` and compare:

```bash
diff -u /tmp/lrom-protected-before.sha256 /tmp/lrom-protected-after.sha256
```

Expected: no differences.

- [ ] **Step 9: Update the architecture understanding document**

In `../docs/LROM_ARCHITECTURE_UNDERSTANDING.md`, record:

- the four removed online overhead sources;
- the private cache lifecycle;
- the fast-path eligibility and generic/reconstruction fallbacks;
- trained versus loaded schema-2 behavior;
- the measured `l=3` and separate `l=10` timing evidence;
- the unchanged scientific outputs and `<=1e-10` cross-section parity;
- ROSE’s continued notebook-owned separation; and
- the still-deferred momentum/radius discrepancy.

Do not describe the timing as an unconditional order-of-magnitude improvement.

- [ ] **Step 10: Update memory logs**

Append `[LROM_Project]` entries to `_memory/Daily Notes/2026-07-27.md` with implemented files, verification, notebook executions, and measured timings. Update the physics checklist in `_memory/Context/active-state.md` with the completed milestone and any remaining physics-validation blocker.

- [ ] **Step 11: Commit generated outputs and documentation**

From `lrom_git`:

```bash
git add \
  notebooks/02_rose_vs_lrom_cross_sections.ipynb \
  notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb
git commit -m "results(v2): record fast packed benchmark outputs"
```

If `../docs/LROM_ARCHITECTURE_UNDERSTANDING.md` belongs to a different repository, commit it only in that repository after checking its local status; otherwise report it as an adjacent workspace documentation change. Never stage `tmp/`.

- [ ] **Step 12: Final verification**

Run:

```bash
git status --short
git log --oneline -7
```

Expected: only the pre-existing untracked `tmp/` remains in `lrom_git`; implementation, tests, plan, and notebook outputs are committed in scoped commits.

Report:

- exact changed files;
- exact test counts and commands;
- max cross-section difference;
- default `l=3` LROM/ROSE times and ratio;
- diagnostic `l=10` LROM/ROSE times and ratio;
- confirmation that public v1.2, Notebook 01, and scientific archive files were untouched; and
- the deferred momentum/radius physics-validation issue.

---

## Final Plan Review Checklist

- [ ] Search this plan for unresolved placeholders:

```bash
rg -n "TO[D]O|TB[D]|FIXM[E]" \
  docs/superpowers/plans/2026-07-27-v2-fast-packed-cross-section.md
```

Expected: no matches.

- [ ] Confirm every approved design requirement maps to a task:

```text
compiled immutable arrays -> Task 1
flattened Woods-Saxon features -> Task 2
one batched solve -> Task 3
direct packed S conversion -> Task 3
cached ROSE angular tables -> Tasks 4 and 5
schema-2 lazy rebuild -> Task 4
generic and reconstruction fallbacks -> Tasks 2 and 4
Notebook 02 and Benchmark 03 -> Tasks 5 and 6
default l=3 plus separate l=10 diagnostic -> Tasks 5 and 6
scientific parity and protected files -> Task 6
deferred momentum/radius correction -> Global Constraints and Task 6
```

- [ ] Confirm every private interface uses consistent names in tests and implementation:

```text
_prediction_interaction_space
_compile_cross_section_cache
_cross_section_cache
_packed_effective_interaction_features
_generic_packed_effective_interaction_features
_solve_packed_coordinates
_smatrix_from_packed_coordinates
_packed_cross_section_cache
```

- [ ] Confirm no task changes public v1.2, Notebook 01, the scientific archive, artifact schema, alpha rows, or the selected `l=3` study.
