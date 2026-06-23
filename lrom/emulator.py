"""Central public LROM object and lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import LROMConfig
from .errors import LROMStateError
from .potentials import PotentialFunction
from .sampling import create_sampling_design
from .state import Kinematics, MeshState, SamplingState, TestingCase, TrainingState


class LROM:
    """Stateful learned reduced-operator model workflow."""

    def __init__(
        self,
        *,
        target: tuple[int, int],
        projectile: tuple[int, int],
        lab_energy: float,
        l: int | tuple[int, ...] = 0,
        fom: str = "nucl-scatter-eq",
        potential: str | PotentialFunction = "ws_3",
        central_parameters: Mapping[str, float] | None = None,
    ) -> None:
        self._config = LROMConfig.create(
            target=target,
            projectile=projectile,
            lab_energy=lab_energy,
            l=l,
            fom=fom,
            potential=potential,
            central_parameters=central_parameters,
        )
        self._central_parameters: Mapping[str, float] = self._config.central_overrides
        self._kinematics: Kinematics | None = None
        self._sampling_state: SamplingState | None = None
        self._portable_mesh: MeshState | None = None
        self._training_state: TrainingState | None = None
        self._prediction_state: Any = None
        self._fom_provider: Any = None
        self._training_engine: Any = None
        self._inference_only = False
        self._provenance: dict[str, Any] = {}

    @property
    def config(self) -> LROMConfig:
        return self._config

    @property
    def kinematics(self) -> Kinematics | None:
        return self._kinematics

    @property
    def central_parameters(self) -> Mapping[str, float]:
        return self._central_parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._config.parameter_names

    @property
    def sampleable_parameters(self) -> tuple[str, ...]:
        return self._config.sampleable_names

    @property
    def partial_waves(self) -> tuple[int, ...]:
        return self._config.channels

    @property
    def description(self) -> Mapping[str, str]:
        return self._config.description

    @property
    def samples(self) -> SamplingState | None:
        return self._sampling_state

    @property
    def mesh(self) -> MeshState | None:
        if self._sampling_state is not None:
            return self._sampling_state.mesh
        return self._portable_mesh

    @property
    def full_order_model(self) -> Mapping[int, Any] | None:
        return (
            None
            if self._sampling_state is None
            else self._sampling_state.full_order_models
        )

    @property
    def basis(self) -> Mapping[int, Any] | None:
        return None if self._training_state is None else self._training_state.basis

    @property
    def predictors(self) -> Any:
        return None if self._training_state is None else self._training_state.predictors

    @property
    def rf_lrom(self) -> Mapping[int, Any] | None:
        return None if self._training_state is None else self._training_state.rf_lrom

    @property
    def rose_rbm(self) -> Mapping[int, Any] | None:
        return None if self._training_state is None else self._training_state.rose_rbm

    @property
    def testing_results(self) -> Any:
        return None if self._training_state is None else self._training_state.testing_results

    @property
    def testing_errors(self) -> Mapping[int, Any] | None:
        return None if self._training_state is None else self._training_state.testing_errors

    @property
    def predictions(self) -> Any:
        return self._prediction_state

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    @property
    def is_sampled(self) -> bool:
        return self._sampling_state is not None

    @property
    def is_trained(self) -> bool:
        return self._training_state is not None

    @property
    def can_predict(self) -> bool:
        return self.is_trained

    def _provider(self) -> Any:
        if self._fom_provider is None:
            from .fom import NuclearScatteringFOM

            self._fom_provider = NuclearScatteringFOM()
        return self._fom_provider

    def _trainer(self) -> Any:
        if self._training_engine is None:
            from .training import TrainingEngine

            self._training_engine = TrainingEngine()
        return self._training_engine

    def _clear_training_state(self) -> None:
        self._training_state = None
        self._clear_prediction_state()

    def _clear_prediction_state(self) -> None:
        self._prediction_state = None

    def sampling(
        self,
        *,
        training_ranges: Mapping[str, tuple[float, float]],
        testing_ranges: Mapping[str, tuple[float, float]],
        training_size: int,
        testing_size: int,
        mesh_size: int = 900,
        radial_domain: tuple[float, float] | None = None,
        strategy: str = "latin_hypercube",
        seed: int | None = None,
        eim_basis_size: int = 8,
        solver_options: Mapping[str, object] | None = None,
    ) -> None:
        if self._inference_only:
            raise LROMStateError("a portable inference artifact cannot run sampling")
        provider = self._provider()
        central, kinematics = provider.resolve(config=self._config)
        design = create_sampling_design(
            parameter_names=self.parameter_names,
            sampleable_names=self.sampleable_parameters,
            central=central,
            training_ranges=training_ranges,
            testing_ranges=testing_ranges,
            training_size=training_size,
            testing_size=testing_size,
            strategy=strategy,
            seed=seed,
        )
        state = provider.sample(
            config=self._config,
            design=design,
            mesh_size=mesh_size,
            radial_domain=radial_domain,
            eim_basis_size=eim_basis_size,
            solver_options=solver_options,
        )
        self._sampling_state = state
        self._central_parameters = state.central_parameters
        self._kinematics = state.kinematics
        self._clear_training_state()

    def reduced_basis(self, *, basis_size: int) -> None:
        if not self.is_sampled:
            raise LROMStateError("call sampling() before reduced_basis()")
        trainer = self._trainer()
        self._training_state = trainer.reduced_basis(
            emulator=self, basis_size=basis_size
        )
        self._clear_prediction_state()

    def train(
        self,
        *,
        basis_size: int = 4,
        predictor: str = "potential",
        predictor_count: int = 6,
    ) -> None:
        if self._inference_only:
            raise LROMStateError("a portable inference artifact cannot be retrained")
        if not self.is_sampled:
            raise LROMStateError("call sampling() before train()")
        self._training_state = self._trainer().train(
            emulator=self,
            basis_size=basis_size,
            predictor=predictor,
            predictor_count=predictor_count,
        )
        self._clear_prediction_state()

    def predict(
        self,
        *,
        parameters: Mapping[str, float] | Sequence[Mapping[str, float]],
    ) -> None:
        if not self.can_predict:
            raise LROMStateError("call train() before predict()")
        from .training import predict

        self._prediction_state = predict(emulator=self, parameters=parameters)

    def testing_case(self, *, case_id: str) -> TestingCase:
        if self._training_state is None or self._sampling_state is None:
            raise LROMStateError("call train() before requesting a testing case")
        try:
            index = self._sampling_state.design.testing.case_ids.index(case_id)
        except ValueError as exc:
            raise LROMStateError(f"unknown testing case_id {case_id!r}") from exc
        results = self._training_state.testing_results
        return TestingCase(
            case_id=case_id,
            parameters=self._sampling_state.design.testing.named(index=index),
            radius=self._sampling_state.mesh.radius,
            high_fidelity={
                channel: values[index]
                for channel, values in results.high_fidelity.items()
            },
            rose={channel: values[index] for channel, values in results.rose.items()},
            lrom={channel: values[index] for channel, values in results.lrom.items()},
            ls={channel: values[index] for channel, values in results.ls.items()},
        )

    def save(self, *, path: str | Path) -> None:
        if not self.can_predict:
            raise LROMStateError("call train() before save()")
        from .artifacts import save

        save(path=path, emulator=self)
