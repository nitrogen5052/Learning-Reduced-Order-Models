"""Safe, versioned portable LROM inference artifacts."""

from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import platform
from types import MappingProxyType
import zipfile

import numpy as np

from .emulator import LROM
from .errors import LROMArtifactError
from .predictors import PredictorState
from .rf import RFLROMModel
from .state import BasisState, Kinematics, MeshState, TrainingState


ARTIFACT_SCHEMA = 1


def _json_config(emulator: LROM) -> dict[str, object]:
    potential_name = emulator.config.potential.name
    if potential_name not in {"ws_1", "ws_3", "woods-saxon"}:
        raise LROMArtifactError(
            "portable artifacts require a registered potential name"
        )
    return {
        "target": list(emulator.config.target),
        "projectile": list(emulator.config.projectile),
        "lab_energy": emulator.config.lab_energy,
        "channels": list(emulator.partial_waves),
        "fom": emulator.config.fom,
        "potential": potential_name,
        "central_parameters": dict(emulator.central_parameters),
        "parameter_names": list(emulator.parameter_names),
    }


def save(*, path: str | Path, emulator: LROM) -> None:
    """Write prediction-critical state without pickle or live ROSE objects."""
    if emulator.mesh is None or emulator.kinematics is None:
        raise LROMArtifactError("trained emulator is missing mesh or kinematics")
    config = _json_config(emulator)
    predictor = emulator.predictors
    arrays: dict[str, np.ndarray] = {
        "mesh_rho": np.asarray(emulator.mesh.rho),
        "mesh_radius": np.asarray(emulator.mesh.radius),
        "predictor_parameter_indices": np.asarray(predictor.parameter_indices),
        "predictor_center": np.asarray(predictor.center),
        "predictor_scales": np.asarray(predictor.scales),
        "predictor_selected_indices": np.asarray(predictor.selected_indices),
        "predictor_selected_radii": np.asarray(predictor.selected_radii),
        "predictor_central_values": np.asarray(predictor.central_values),
        "predictor_singular_values": np.asarray(predictor.singular_values),
    }
    channels: dict[str, dict[str, object]] = {}
    for channel in emulator.partial_waves:
        basis = emulator.basis[channel]
        model = emulator.rf_lrom[channel]
        prefix = f"l{channel}"
        arrays[f"{prefix}_phi0"] = np.asarray(basis.phi0)
        arrays[f"{prefix}_basis_vectors"] = np.asarray(basis.vectors)
        arrays[f"{prefix}_basis_singular_values"] = np.asarray(
            basis.singular_values
        )
        arrays[f"{prefix}_rf_matrices"] = np.asarray(model.matrices)
        arrays[f"{prefix}_rf_vectors"] = np.asarray(model.vectors)
        arrays[f"{prefix}_rf_singular_values"] = np.asarray(
            model.singular_values
        )
        channels[str(channel)] = {
            "residual_mse": model.residual_mse,
            "rank": model.rank,
        }
    config_hash = sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    metadata = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "package_version": "0.1.0",
        "config_hash": config_hash,
        "config": config,
        "kinematics": {
            "mu": emulator.kinematics.mu,
            "e_com": emulator.kinematics.e_com,
            "k": emulator.kinematics.k,
            "eta": emulator.kinematics.eta,
            "coulomb_radius": emulator.kinematics.coulomb_radius,
        },
        "predictor": {
            "kind": predictor.kind,
            "names": list(predictor.names),
            "parameter_names": list(predictor.parameter_names),
        },
        "channels": channels,
        "training_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    array_buffer = io.BytesIO()
    np.savez_compressed(array_buffer, **arrays)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            destination, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))
            archive.writestr("arrays.npz", array_buffer.getvalue())
    except OSError as exc:
        raise LROMArtifactError(f"could not write artifact {destination}") from exc


def _required_array(arrays, name: str) -> np.ndarray:
    try:
        value = np.asarray(arrays[name])
    except KeyError as exc:
        raise LROMArtifactError(f"artifact is missing array {name!r}") from exc
    if value.dtype == object:
        raise LROMArtifactError(f"artifact array {name!r} has unsafe object dtype")
    return value


def load(*, path: str | Path) -> LROM:
    """Load a portable prediction-only emulator."""
    source = Path(path)
    try:
        with zipfile.ZipFile(source) as archive:
            if set(archive.namelist()) != {"metadata.json", "arrays.npz"}:
                raise LROMArtifactError("artifact must contain metadata.json and arrays.npz")
            metadata = json.loads(archive.read("metadata.json"))
            array_bytes = archive.read("arrays.npz")
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise LROMArtifactError(f"invalid LROM artifact {source}") from exc
    if metadata.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise LROMArtifactError(
            f"unsupported artifact schema {metadata.get('artifact_schema')!r}"
        )
    try:
        arrays = np.load(io.BytesIO(array_bytes), allow_pickle=False)
        config = metadata["config"]
        emulator = LROM(
            target=tuple(config["target"]),
            projectile=tuple(config["projectile"]),
            lab_energy=float(config["lab_energy"]),
            l=tuple(config["channels"]),
            fom=config["fom"],
            potential=config["potential"],
            central_parameters=config["central_parameters"],
        )
        emulator._central_parameters = MappingProxyType(
            {name: float(value) for name, value in config["central_parameters"].items()}
        )
        kin = metadata["kinematics"]
        emulator._kinematics = Kinematics(
            mu=float(kin["mu"]),
            e_com=float(kin["e_com"]),
            k=float(kin["k"]),
            eta=float(kin["eta"]),
            coulomb_radius=float(kin["coulomb_radius"]),
        )
        emulator._portable_mesh = MeshState(
            rho=_required_array(arrays, "mesh_rho"),
            radius=_required_array(arrays, "mesh_radius"),
        )
        predictor_meta = metadata["predictor"]
        predictor = PredictorState(
            kind=predictor_meta["kind"],
            names=tuple(predictor_meta["names"]),
            parameter_names=tuple(predictor_meta["parameter_names"]),
            parameter_indices=_required_array(
                arrays, "predictor_parameter_indices"
            ).astype(int),
            center=_required_array(arrays, "predictor_center"),
            scales=_required_array(arrays, "predictor_scales"),
            training_features=np.empty((0, 0)),
            testing_features=np.empty((0, 0)),
            selected_indices=_required_array(
                arrays, "predictor_selected_indices"
            ).astype(int),
            selected_radii=_required_array(arrays, "predictor_selected_radii"),
            central_values=_required_array(arrays, "predictor_central_values"),
            singular_values=_required_array(
                arrays, "predictor_singular_values"
            ),
        )
        bases = {}
        models = {}
        for channel_text, model_meta in metadata["channels"].items():
            channel = int(channel_text)
            prefix = f"l{channel}"
            bases[channel] = BasisState(
                phi0=_required_array(arrays, f"{prefix}_phi0"),
                vectors=_required_array(arrays, f"{prefix}_basis_vectors"),
                radius=emulator._portable_mesh.radius,
                singular_values=_required_array(
                    arrays, f"{prefix}_basis_singular_values"
                ),
            )
            models[channel] = RFLROMModel(
                matrices=_required_array(arrays, f"{prefix}_rf_matrices"),
                vectors=_required_array(arrays, f"{prefix}_rf_vectors"),
                residual_mse=float(model_meta["residual_mse"]),
                rank=int(model_meta["rank"]),
                singular_values=_required_array(
                    arrays, f"{prefix}_rf_singular_values"
                ),
            )
        emulator._training_state = TrainingState(
            basis=bases,
            predictors=predictor,
            rf_lrom=models,
            testing_results=None,
            testing_errors={channel: {} for channel in bases},
        )
        emulator._inference_only = True
        emulator._provenance = {
            "artifact_schema": ARTIFACT_SCHEMA,
            "package_version": metadata["package_version"],
            "config_hash": metadata["config_hash"],
            "training_environment": metadata["training_environment"],
        }
        arrays.close()
        return emulator
    except LROMArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise LROMArtifactError(f"invalid LROM artifact {source}") from exc
