from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lrom_bench.artifacts import (
    ArrayComparison,
    compare_npz_artifacts,
    load_npz_artifact,
    save_npz_artifact,
    write_parity_report,
)


def test_save_and_load_npz_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.npz"
    arrays = {"coeff": np.array([1.0, 2.0])}
    metadata = {"config_hash": "abc123", "note": "unit test"}

    save_npz_artifact(path, arrays=arrays, metadata=metadata)
    loaded_arrays, loaded_metadata = load_npz_artifact(path)

    assert np.allclose(loaded_arrays["coeff"], arrays["coeff"])
    assert loaded_metadata == metadata


def test_compare_npz_artifacts_and_write_report(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.npz"
    new = tmp_path / "new.npz"
    report = tmp_path / "report.json"
    metadata = {"config_hash": "abc123"}

    save_npz_artifact(legacy, arrays={"x": np.array([1.0, 2.0])}, metadata=metadata)
    save_npz_artifact(new, arrays={"x": np.array([1.0, 2.0 + 1e-12])}, metadata=metadata)
    comparisons = compare_npz_artifacts(legacy, new, rtol=1e-9, atol=1e-9)
    write_parity_report(report, comparisons=comparisons, metadata=metadata)

    assert comparisons == [
        ArrayComparison(name="x", passed=True, max_abs_error=1.000088900582341e-12)
    ]
    payload = json.loads(report.read_text())
    assert payload["passed"] is True
    assert payload["metadata"] == metadata
