import io
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

import lrom_legacy.v2_0 as v2


class FakeInteraction:
    def __init__(self, channel_scale):
        self.channel_scale = channel_scale

    def tilde(self, rho, alpha):
        rho = np.asarray(rho)
        a0, a1 = np.asarray(alpha)
        return self.channel_scale * (
            a0 * np.exp(-rho) + 1j * a1 * rho * np.exp(-0.5 * rho)
        )


def test_refined_maxvol_is_unique_and_deterministic():
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
    assert np.linalg.matrix_rank(basis[selected_a]) == basis.shape[1]


def test_refined_maxvol_rejects_invalid_shape():
    with pytest.raises(ValueError, match="points, modes"):
        v2._refined_maxvol_indices(np.ones((2, 3)))


def test_effective_interaction_predictors_are_channel_specific_and_physical():
    rho = np.linspace(0.01, 8.0, 96)
    radius = rho / 1.4
    models = {
        0: SimpleNamespace(interaction=FakeInteraction(1.0)),
        (1, 0): SimpleNamespace(interaction=FakeInteraction(2.0)),
        (1, 1): SimpleNamespace(interaction=FakeInteraction(-3.0)),
    }
    center = np.array([1.0, 2.0])
    train = np.array(
        [[0.8, 1.7], [1.2, 2.3], [1.1, 1.8], [0.9, 2.2]]
    )
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
    assert all(
        np.all(state.selected_radii >= 0.5) for state in states.values()
    )
    assert not np.array_equal(
        states[(1, 0)].training_features,
        states[(1, 1)].training_features,
    )
    assert all(
        np.max(np.abs(state.training_features), axis=0).max()
        <= 1.0 + 1e-12
        for state in states.values()
    )


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
    vectors = np.array(
        [[0.2, -0.1], [0.04, 0.06]], dtype=np.complex128
    )
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


def test_zero_intercept_fit_preserves_zero_constant_vector():
    predictors = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    coordinates = np.column_stack(
        [0.2 * predictors[:, 0], -0.1 * predictors[:, 0]]
    )

    model = v2.fit_rf_lrom(
        predictors=predictors,
        coordinates=coordinates,
    )

    assert np.array_equal(model.constant_vector, np.zeros(2))


def small_cross_section_emulator():
    emulator = v2.LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=(0, 1),
        potential="full_woods-saxon",
    )
    ranges = {
        name: tuple(sorted((0.95 * value, 1.05 * value)))
        for name, value in emulator.central_parameters.items()
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


def test_cross_section_cache_is_compiled_and_invalidated():
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=3,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
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


def test_shared_potential_cross_section_predictor_skips_packed_cache():
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="potential",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )

    assert emulator._packed_cross_section_cache is None
    row = dict(
        zip(
            emulator.parameter_names,
            emulator.samples.design.testing.values[0],
        )
    )
    emulator.predict(parameters=row, reconstruct_wavefunctions=False)
    assert np.all(np.isfinite(emulator.predictions.cross_sections.values))


def test_effective_interaction_training_uses_one_feature_set_per_channel():
    emulator = small_cross_section_emulator()

    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
    )

    assert set(emulator.predictors) == set(emulator.rf_lrom)
    assert all(
        state.kind == "effective-interaction"
        for state in emulator.predictors.values()
    )
    assert all(
        np.linalg.norm(model.constant_vector) > 0.0
        for model in emulator.rf_lrom.values()
    )


def test_packed_coefficients_match_individual_channel_solves():
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )
    values = emulator.samples.design.testing.values[:3]
    features = v2.effective_interaction_features(
        emulator=emulator,
        predictors=emulator.predictors,
        values=values,
    )
    channels = tuple(emulator.rf_lrom)

    packed = v2._packed_coefficients(
        channels=channels,
        models=emulator.rf_lrom,
        features=features,
    )

    for offset, channel in enumerate(channels):
        scalar = v2.solve_rf_lrom(
            model=emulator.rf_lrom[channel],
            predictors=features[channel],
        )
        assert np.allclose(packed[:, offset], scalar)

    cache = v2._cross_section_cache(emulator=emulator)
    packed_features = v2._packed_effective_interaction_features(
        emulator=emulator,
        values=values,
        cache=cache,
    )
    actual = v2._solve_packed_coordinates(
        features=packed_features,
        cache=cache,
    )
    expected = np.stack(
        [
            v2.solve_rf_lrom(
                model=emulator.rf_lrom[channel],
                predictors=packed_features[:, channel_index, :],
            )
            for channel_index, channel in enumerate(cache["channel_keys"])
        ],
        axis=1,
    )
    assert actual.shape == (3, len(channels), 2)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_flattened_full_woods_saxon_features_match_reference():
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )
    cache = v2._cross_section_cache(emulator=emulator)
    rows = np.asarray(
        emulator.samples.design.testing.values[:2], dtype=float
    )
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

    assert actual.shape == (2, len(emulator.rf_lrom), 2)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_packed_features_fall_back_to_reference():
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )
    cache = v2._cross_section_cache(emulator=emulator).copy()
    cache["potential_name"] = None
    rows = np.asarray(
        emulator.samples.design.testing.values[:2], dtype=float
    )
    reference = v2.effective_interaction_features(
        emulator=emulator,
        predictors=emulator.predictors,
        values=rows,
    )
    expected = np.stack(
        [reference[channel] for channel in cache["channel_keys"]],
        axis=1,
    )

    actual = v2._packed_effective_interaction_features(
        emulator=emulator,
        values=rows,
        cache=cache,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_observable_only_prediction_matches_full_prediction():
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
        for row in emulator.samples.design.testing.values[:2]
    ]

    emulator.predict(parameters=rows, reconstruct_wavefunctions=True)
    full = emulator.predictions
    emulator.predict(parameters=rows, reconstruct_wavefunctions=False)
    packed = emulator.predictions

    assert packed.wavefunctions == {}
    for channel in full.coefficients:
        assert np.allclose(
            full.coefficients[channel], packed.coefficients[channel]
        )
    assert np.allclose(full.smatrix.splus, packed.smatrix.splus)
    assert np.allclose(full.smatrix.sminus, packed.smatrix.sminus)
    assert np.allclose(
        full.cross_sections.values, packed.cross_sections.values
    )


def test_observable_only_full_woods_saxon_bypasses_channel_dispatch(
    monkeypatch,
):
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )
    row = dict(
        zip(
            emulator.parameter_names,
            emulator.samples.design.testing.values[0],
        )
    )

    def fail_channel_dispatch(**_kwargs):
        raise AssertionError("observable-only fast path used channel dispatch")

    monkeypatch.setattr(
        v2,
        "effective_interaction_features",
        fail_channel_dispatch,
    )

    emulator.predict(
        parameters=row,
        reconstruct_wavefunctions=False,
    )

    assert np.all(np.isfinite(emulator.predictions.cross_sections.values))


def test_cross_section_prediction_reuses_cached_angles(monkeypatch):
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )
    row = dict(
        zip(
            emulator.parameter_names,
            emulator.samples.design.testing.values[0],
        )
    )
    sae = v2._scattering_amplitude_emulator(emulator=emulator)
    original = sae.calculate_xs
    calls = []

    def calculate_xs(splus, sminus, parameters, *args, **kwargs):
        calls.append(dict(kwargs))
        return original(splus, sminus, parameters, *args, **kwargs)

    monkeypatch.setattr(sae, "calculate_xs", calculate_xs)

    emulator.predict(
        parameters=row,
        reconstruct_wavefunctions=False,
    )

    assert calls
    assert all("angles" not in kwargs for kwargs in calls)


def test_packed_smatrix_matches_scalar_channel_conversion():
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
        observable="cross_section",
        angles_degrees=np.linspace(10.0, 170.0, 9),
    )
    values = emulator.samples.design.testing.values[:2]
    features = v2.effective_interaction_features(
        emulator=emulator,
        predictors=emulator.predictors,
        values=values,
    )
    channels = tuple(emulator.rf_lrom)
    packed_coefficients = v2._packed_coefficients(
        channels=channels,
        models=emulator.rf_lrom,
        features=features,
    )
    coefficients = {
        channel: packed_coefficients[:, offset]
        for offset, channel in enumerate(channels)
    }
    sae = v2._scattering_amplitude_emulator(emulator=emulator)

    smatrix = v2._packed_smatrix_from_coefficients(
        emulator=emulator,
        sae=sae,
        coefficients=coefficients,
    )
    cache = v2._cross_section_cache(emulator=emulator)
    packed_smatrix = v2._smatrix_from_packed_coordinates(
        coordinates=packed_coefficients,
        cache=cache,
    )

    np.testing.assert_allclose(
        packed_smatrix.splus,
        smatrix.splus,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        packed_smatrix.sminus,
        smatrix.sminus,
        rtol=2e-12,
        atol=2e-12,
    )

    for case_index in range(values.shape[0]):
        for ell, rbe_row in enumerate(sae.rbes):
            assert np.allclose(
                smatrix.splus[case_index, ell],
                v2._s_matrix_from_coefficients(
                    rbe_row[0], coefficients[0 if ell == 0 else (ell, 0)][case_index]
                ),
            )
            if ell == 0:
                assert smatrix.sminus[case_index, ell] == smatrix.splus[case_index, ell]
            else:
                assert np.allclose(
                    smatrix.sminus[case_index, ell],
                    v2._s_matrix_from_coefficients(
                        rbe_row[1], coefficients[(ell, 1)][case_index]
                    ),
                )


def test_effective_interaction_artifact_round_trips_wavefunction_prediction(
    tmp_path,
):
    emulator = small_cross_section_emulator()
    emulator.train(
        basis_size=2,
        predictor="effective-interaction",
        predictor_count=2,
    )
    row = dict(
        zip(
            emulator.parameter_names,
            emulator.samples.design.testing.values[0],
        )
    )
    emulator.predict(parameters=row)
    expected = {
        channel: values.copy()
        for channel, values in emulator.predictions.wavefunctions.items()
    }
    path = tmp_path / "archive-method.lrom"

    emulator.save(path=path)
    loaded = v2.load(path=path)
    loaded.predict(parameters=row)

    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        arrays = np.load(
            io.BytesIO(archive.read("arrays.npz")), allow_pickle=False
        )
    assert metadata["artifact_schema"] == 2
    assert metadata["predictor"]["layout"] == "by_channel"
    assert all(value.dtype != object for value in arrays.values())
    assert set(loaded.predictors) == set(emulator.predictors)
    for channel, values in expected.items():
        assert np.allclose(loaded.predictions.wavefunctions[channel], values)


def test_cross_section_artifact_lazily_rebuilds_fast_cache(tmp_path):
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
    path = tmp_path / "cross-section.lrom"
    trained.save(path=path)

    loaded = v2.load(path=path)
    assert loaded._packed_cross_section_cache is None

    loaded.predict(parameters=row, reconstruct_wavefunctions=False)
    actual = loaded.predictions

    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata["artifact_schema"] == 2
    assert loaded._packed_cross_section_cache is not None
    np.testing.assert_allclose(
        actual.smatrix.splus,
        expected.smatrix.splus,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual.smatrix.sminus,
        expected.smatrix.sminus,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual.cross_sections.values,
        expected.cross_sections.values,
        rtol=2e-11,
        atol=1e-10,
    )
