from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


METADATA_KEY = "__metadata_json__"


@dataclass(frozen=True)
class ArrayComparison:
    name: str
    passed: bool
    max_abs_error: float


def save_npz_artifact(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    if METADATA_KEY in arrays:
        raise ValueError(f"array key is reserved for metadata: {METADATA_KEY}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload[METADATA_KEY] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez(path, **payload)


def load_npz_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data[METADATA_KEY]))
        arrays = {key: data[key] for key in data.files if key != METADATA_KEY}
    return arrays, metadata


def compare_npz_artifacts(
    legacy_path: Path,
    new_path: Path,
    rtol: float,
    atol: float,
) -> list[ArrayComparison]:
    legacy_arrays, _legacy_metadata = load_npz_artifact(legacy_path)
    new_arrays, _new_metadata = load_npz_artifact(new_path)
    if set(legacy_arrays) != set(new_arrays):
        missing = sorted(set(legacy_arrays).symmetric_difference(new_arrays))
        raise ValueError(f"artifact keys differ: {missing}")

    comparisons = []
    for name in sorted(legacy_arrays):
        legacy = legacy_arrays[name]
        new = new_arrays[name]
        if legacy.shape != new.shape:
            raise ValueError(f"shape mismatch for {name}: {legacy.shape} != {new.shape}")
        diff = np.max(np.abs(legacy - new)) if legacy.size else 0.0
        comparisons.append(
            ArrayComparison(
                name=name,
                passed=bool(np.allclose(legacy, new, rtol=rtol, atol=atol)),
                max_abs_error=float(diff),
            )
        )
    return comparisons


def write_parity_report(
    path: Path,
    comparisons: list[ArrayComparison],
    metadata: dict[str, Any],
) -> None:
    if not comparisons:
        raise ValueError("parity report requires at least one comparison")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": all(item.passed for item in comparisons),
        "metadata": metadata,
        "comparisons": [asdict(item) for item in comparisons],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
