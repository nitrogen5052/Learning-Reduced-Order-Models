from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Notebook02Config:
    target_a: int = 40
    target_z: int = 20
    projectile_a: int = 1
    projectile_z: int = 0
    e_lab: float = 14.1
    n_mesh: int = 900
    n_phi: int = 4
    n_u: int = 8
    l_max: int = 1
    vv_width: float = 8.0
    rv_width: float = 0.35
    broad_vv_width: float = 10.0
    broad_rv_width: float = 0.45
    n_scan: int = 17
    n_box_train: int = 49
    n_box_test: int = 81
    n_predictors: int = 6
    seed_train: int = 20260616
    seed_test: int = 20260617
    rtol: float = 1e-8
    atol: float = 1e-10

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.as_jsonable(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class BenchmarkPaths:
    root: Path

    @property
    def benchmark_root(self) -> Path:
        return self.root / "outputs" / "benchmarks"

    def legacy_npz(self, stem: str) -> Path:
        return self.benchmark_root / "legacy" / f"{stem}.npz"

    def new_npz(self, stem: str) -> Path:
        return self.benchmark_root / "new" / f"{stem}.npz"

    def report_json(self, stem: str) -> Path:
        return self.benchmark_root / "reports" / f"{stem}.json"
