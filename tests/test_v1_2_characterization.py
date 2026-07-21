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
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=0,
        fom="nucl-scatter-eq",
        potential="ws_1",
    )
    vv = dict(emulator.central_parameters)["Vv"]
    emulator.sampling(
        training_ranges={"Vv": (0.9 * vv, 1.1 * vv)},
        testing_ranges={"Vv": (0.8 * vv, 1.2 * vv)},
        training_size=5,
        testing_size=5,
        mesh_size=96,
        strategy="linspace",
        seed=1204,
        eim_basis_size=4,
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
