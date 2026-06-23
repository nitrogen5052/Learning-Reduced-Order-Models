from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from lrom import LROM
from lrom.errors import LROMStateError
from lrom.state import Kinematics, MeshState, SamplingState, TrainingState


class FakeFOMProvider:
    def resolve(self, *, config):
        central = {"Vv": 50.0, "Rv": 4.0, "av": 0.65}
        central.update(config.central_overrides)
        kinematics = Kinematics(
            mu=1.0,
            e_com=13.8,
            k=0.8,
            eta=0.0,
            coulomb_radius=4.0,
        )
        return MappingProxyType(central), kinematics

    def sample(
        self,
        *,
        config,
        design,
        mesh_size,
        radial_domain,
        eim_basis_size,
        solver_options,
    ) -> SamplingState:
        radius = np.linspace(0.0, 8.0, mesh_size)
        central, kinematics = self.resolve(config=config)
        central_wavefunctions = {
            channel: np.full(mesh_size, channel + 1, dtype=np.complex128)
            for channel in config.channels
        }
        training_wavefunctions = {
            channel: np.tile(central_wavefunctions[channel], (len(design.training.case_ids), 1))
            for channel in config.channels
        }
        testing_wavefunctions = {
            channel: np.tile(central_wavefunctions[channel], (len(design.testing.case_ids), 1))
            for channel in config.channels
        }
        return SamplingState(
            design=design,
            central_parameters=central,
            kinematics=kinematics,
            mesh=MeshState(rho=radius * 0.8, radius=radius),
            central_wavefunctions=central_wavefunctions,
            training_wavefunctions=training_wavefunctions,
            testing_wavefunctions=testing_wavefunctions,
            training_potentials=np.zeros((len(design.training.case_ids), mesh_size)),
            full_order_models={channel: f"fom-{channel}" for channel in config.channels},
        )


class FakeTrainingEngine:
    def train(self, *, emulator, basis_size, predictor, predictor_count) -> TrainingState:
        return TrainingState(
            basis={channel: f"basis-{channel}" for channel in emulator.partial_waves},
            predictors={"kind": predictor, "count": predictor_count},
            rf_lrom={channel: f"model-{channel}" for channel in emulator.partial_waves},
            rose_rbm={channel: f"rose-{channel}" for channel in emulator.partial_waves},
            testing_results={"basis_size": basis_size},
            testing_errors={channel: {} for channel in emulator.partial_waves},
        )


def make_emulator(*, l: int | tuple[int, ...] = 0) -> LROM:
    emulator = LROM(
        target=(40, 20),
        projectile=(1, 0),
        lab_energy=14.1,
        l=l,
        potential="ws_3",
    )
    emulator._fom_provider = FakeFOMProvider()
    emulator._training_engine = FakeTrainingEngine()
    return emulator


def sampling_kwargs(*, seed: int = 1) -> dict[str, object]:
    return {
        "training_ranges": {"Vv": (45.0, 55.0)},
        "testing_ranges": {"Vv": (40.0, 60.0)},
        "training_size": 3,
        "testing_size": 4,
        "mesh_size": 16,
        "strategy": "linspace",
        "seed": seed,
    }


def test_constructor_exposes_normalized_configuration() -> None:
    emulator = make_emulator(l=(3, 0, 1))

    assert emulator.config.target == (40, 20)
    assert emulator.partial_waves == (0, 1, 3)
    assert emulator.parameter_names == ("Vv", "Rv", "av")
    assert emulator.sampleable_parameters == ("Vv", "Rv", "av")
    assert not emulator.is_sampled
    assert not emulator.is_trained
    assert not emulator.can_predict


def test_train_before_sampling_is_rejected() -> None:
    emulator = make_emulator()

    with pytest.raises(LROMStateError, match="sampling"):
        emulator.train()


def test_sampling_updates_authoritative_state() -> None:
    emulator = make_emulator(l=(0, 2))

    result = emulator.sampling(**sampling_kwargs())

    assert result is None
    assert emulator.is_sampled
    assert emulator.central_parameters == {"Vv": 50.0, "Rv": 4.0, "av": 0.65}
    assert emulator.mesh.radius.shape == (16,)
    assert emulator.samples.training_wavefunctions[2].shape == (3, 16)
    assert emulator.full_order_model == {0: "fom-0", 2: "fom-2"}


def test_resampling_invalidates_downstream_state() -> None:
    emulator = make_emulator()
    emulator.sampling(**sampling_kwargs())
    emulator.train()
    emulator._prediction_state = {"old": True}
    assert emulator.is_trained

    emulator.sampling(**sampling_kwargs(seed=2))

    assert emulator.is_sampled
    assert not emulator.is_trained
    assert emulator.basis is None
    assert emulator.predictors is None
    assert emulator.predictions is None


def test_train_updates_state_and_returns_none() -> None:
    emulator = make_emulator(l=(0, 1))
    emulator.sampling(**sampling_kwargs())

    result = emulator.train(basis_size=4, predictor="potential", predictor_count=6)

    assert result is None
    assert emulator.is_trained
    assert emulator.can_predict
    assert emulator.basis == {0: "basis-0", 1: "basis-1"}
    assert emulator.predictors == {"kind": "potential", "count": 6}
    assert emulator.rf_lrom == {0: "model-0", 1: "model-1"}


def test_predict_before_training_is_rejected() -> None:
    emulator = make_emulator()
    emulator.sampling(**sampling_kwargs())

    with pytest.raises(LROMStateError, match="train"):
        emulator.predict(parameters={"Vv": 50.0})
