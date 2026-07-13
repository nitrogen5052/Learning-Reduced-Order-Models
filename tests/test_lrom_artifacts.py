from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

import lrom
from lrom import LROMArtifactError, LROMStateError


def trained_emulator() -> lrom.LROM:
    emulator = lrom.LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=0,
        potential="ws_1",
    )
    emulator.sampling(
        training_ranges={"Vv": (47.0, 53.0)},
        testing_ranges={"Vv": (45.0, 55.0)},
        training_size=5,
        testing_size=3,
        mesh_size=64,
        strategy="linspace",
        seed=5,
        eim_basis_size=2,
    )
    emulator.train(basis_size=2, predictor="parameters")
    return emulator


def test_portable_artifact_round_trips_prediction_state(tmp_path) -> None:
    emulator = trained_emulator()
    emulator.predict(parameters={"Vv": 51.25})
    expected = emulator.predictions.wavefunctions[0].copy()
    path = tmp_path / "ca40.lrom"

    emulator.save(path=path)
    loaded = lrom.load(path=path)
    loaded.predict(parameters={"Vv": 51.25})

    assert loaded.samples is None
    assert loaded.full_order_model is None
    assert loaded.training_results is None
    assert loaded.testing_results is None
    assert loaded.is_trained
    assert loaded.can_predict
    assert loaded.mesh.radius.shape == (64,)
    assert np.allclose(loaded.predictions.wavefunctions[0], expected)
    assert loaded.provenance["artifact_schema"] == 1

    with pytest.raises(LROMStateError, match="cannot run sampling"):
        loaded.sampling(
            training_ranges={"Vv": (47.0, 53.0)},
            testing_ranges={"Vv": (45.0, 55.0)},
            training_size=3,
            testing_size=3,
        )
    with pytest.raises(LROMStateError, match="cannot be retrained"):
        loaded.train()


def test_artifact_is_json_and_pickle_free_numpy(tmp_path) -> None:
    emulator = trained_emulator()
    path = tmp_path / "safe.lrom"

    emulator.save(path=path)

    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"metadata.json", "arrays.npz"}
        metadata = json.loads(archive.read("metadata.json"))
        arrays = np.load(io.BytesIO(archive.read("arrays.npz")), allow_pickle=False)
        assert metadata["artifact_schema"] == 1
        assert all(array.dtype != object for array in arrays.values())


def test_invalid_artifact_has_clear_error(tmp_path) -> None:
    path = tmp_path / "broken.lrom"
    path.write_bytes(b"not a zip archive")

    with pytest.raises(LROMArtifactError, match="artifact"):
        lrom.load(path=path)
