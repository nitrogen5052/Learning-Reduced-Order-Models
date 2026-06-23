"""ROSE-backed nuclear scattering full-order model provider."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from numba import njit
import numpy as np

from .config import LROMConfig
from .errors import LROMConfigurationError, LROMSamplingError
from .potentials import real_woods_saxon
from .state import Kinematics, MeshState, SamplingDesign, SamplingState


def _import_rose() -> Any:
    import scipy.special

    if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
        def sph_harm(m: Any, n: Any, theta: Any, phi: Any) -> Any:
            return scipy.special.sph_harm_y(n, m, theta, phi)

        scipy.special.sph_harm = sph_harm
    try:
        import rose
    except ImportError as exc:
        raise ImportError(
            "LROM sampling with 'nucl-scatter-eq' requires nuclear-rose"
        ) from exc
    return rose


@njit
def _real_ws_interaction(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    vv = alpha[0]
    rv = alpha[1]
    av = alpha[2]
    return -vv / (1.0 + np.exp((r - rv) / av))


@dataclass(frozen=True)
class ChannelFOM:
    """Live ROSE solver state for one exact partial-wave channel."""

    channel: int
    interaction: Any
    solver: Any
    base_solver: Any
    rho_mesh: np.ndarray
    radius_mesh: np.ndarray

    def solve(self, *, parameters: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.solver.phi(np.asarray(parameters, dtype=float), self.rho_mesh),
            dtype=np.complex128,
        )


class NuclearScatteringFOM:
    """Construct and sample the built-in ROSE scattering equation."""

    def resolve(
        self, *, config: LROMConfig
    ) -> tuple[Mapping[str, float], Kinematics]:
        rose = _import_rose()
        projectile_lookup = {
            (1, 0): rose.Projectile.neutron,
            (1, 1): rose.Projectile.proton,
        }
        try:
            projectile = projectile_lookup[config.projectile]
        except KeyError as exc:
            raise LROMConfigurationError(
                "nucl-scatter-eq currently supports neutron (1, 0) or proton (1, 1)"
            ) from exc
        mu, e_com, k, eta = rose.kinematics(
            target=config.target,
            projectile=config.projectile,
            E_lab=config.lab_energy,
        )
        kd = rose.koning_delaroche.KDGlobal(projectile)
        coulomb_radius, kd_values = kd.get_params(
            config.target[0], config.target[1], mu, config.lab_energy, k
        )
        if config.potential.name in {"ws_1", "ws_3"}:
            values = np.asarray(kd_values[:3], dtype=float)
        elif config.potential.name == "woods-saxon":
            values = np.asarray(kd_values, dtype=float)
        else:
            values = np.asarray(
                [config.central_overrides[name] for name in config.parameter_names],
                dtype=float,
            )
        central = dict(zip(config.parameter_names, values))
        central.update(config.central_overrides)
        return MappingProxyType(central), Kinematics(
            mu=float(mu),
            e_com=float(e_com),
            k=float(k),
            eta=float(eta),
            coulomb_radius=float(coulomb_radius),
        )

    def sample(
        self,
        *,
        config: LROMConfig,
        design: SamplingDesign,
        mesh_size: int,
        radial_domain: tuple[float, float] | None,
        eim_basis_size: int,
        solver_options: Mapping[str, object] | None,
    ) -> SamplingState:
        if isinstance(mesh_size, bool) or not isinstance(mesh_size, int) or mesh_size < 16:
            raise LROMSamplingError("mesh_size must be an integer of at least 16")
        if (
            isinstance(eim_basis_size, bool)
            or not isinstance(eim_basis_size, int)
            or eim_basis_size < 1
        ):
            raise LROMSamplingError("eim_basis_size must be positive")
        central, kinematics = self.resolve(config=config)
        central_vector = np.asarray(
            [central[name] for name in config.parameter_names], dtype=float
        )
        rose = _import_rose()

        if radial_domain is None:
            rho_domain = (1e-8, float(8.0 * np.pi))
        else:
            if len(radial_domain) != 2 or radial_domain[0] < 0 or radial_domain[0] >= radial_domain[1]:
                raise LROMSamplingError(
                    "radial_domain must be an increasing non-negative (minimum, maximum) tuple"
                )
            rho_domain = (
                max(1e-8, float(radial_domain[0]) * kinematics.k),
                float(radial_domain[1]) * kinematics.k,
            )
        rho_mesh = np.linspace(*rho_domain, mesh_size)
        radius_mesh = rho_mesh / kinematics.k

        all_values = np.vstack(
            [central_vector, design.training.values, design.testing.values]
        )
        bounds = np.column_stack([all_values.min(axis=0), all_values.max(axis=0)])
        kwargs: dict[str, Any] = {
            "l_max": max(config.channels),
            "n_theta": len(config.parameter_names),
            "mu": kinematics.mu,
            "energy": kinematics.e_com,
            "training_info": bounds,
            "n_basis": eim_basis_size,
            "rho_mesh": rho_mesh,
        }
        if config.potential.name == "woods-saxon":
            kwargs.update(
                coordinate_space_potential=rose.koning_delaroche.KD_simple,
                spin_orbit_term=rose.koning_delaroche.KD_simple_so,
                is_complex=True,
            )
            potential_function = rose.koning_delaroche.KD_simple
        else:
            kwargs.update(
                coordinate_space_potential=(
                    _real_ws_interaction
                    if config.potential.name in {"ws_1", "ws_3"}
                    else config.potential.function
                ),
                is_complex=False,
            )
            potential_function = config.potential.function
        interactions = rose.InteractionEIMSpace(**kwargs)

        options = dict(solver_options or {})
        s_0 = float(options.pop("s_0", 6.0 * np.pi))
        rk_tols = options.pop("rk_tols", (1e-9, 1e-9))
        if options:
            raise LROMSamplingError(
                f"unknown solver_options: {sorted(options)}"
            )
        base_solver = rose.SchroedingerEquation.make_base_solver(
            s_0=s_0,
            rk_tols=list(rk_tols),
            domain=np.asarray(rho_domain, dtype=float),
        )
        full_order_models: dict[int, ChannelFOM] = {}
        for channel in config.channels:
            interaction = interactions.interactions[channel][0]
            solver = base_solver.clone_for_new_interaction(interaction)
            full_order_models[channel] = ChannelFOM(
                channel=channel,
                interaction=interaction,
                solver=solver,
                base_solver=base_solver,
                rho_mesh=rho_mesh,
                radius_mesh=radius_mesh,
            )

        central_wavefunctions = {
            channel: model.solve(parameters=central_vector)
            for channel, model in full_order_models.items()
        }
        training_wavefunctions = {
            channel: np.asarray(
                [model.solve(parameters=row) for row in design.training.values]
            )
            for channel, model in full_order_models.items()
        }
        testing_wavefunctions = {
            channel: np.asarray(
                [model.solve(parameters=row) for row in design.testing.values]
            )
            for channel, model in full_order_models.items()
        }
        if potential_function is None:
            raise LROMSamplingError("selected potential cannot be evaluated")
        training_potentials = np.asarray(
            [potential_function(radius_mesh, row) for row in design.training.values]
        )
        return SamplingState(
            design=design,
            central_parameters=central,
            kinematics=kinematics,
            mesh=MeshState(rho=rho_mesh, radius=radius_mesh),
            central_wavefunctions=central_wavefunctions,
            training_wavefunctions=training_wavefunctions,
            testing_wavefunctions=testing_wavefunctions,
            training_potentials=training_potentials,
            full_order_models=full_order_models,
        )
