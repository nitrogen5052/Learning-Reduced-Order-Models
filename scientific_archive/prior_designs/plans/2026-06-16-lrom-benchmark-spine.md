# LROM Benchmark Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Python research benchmark package for Notebook 02 RF-LROM parity while keeping the scientific notebook narrative visible.

**Architecture:** Add a small `lrom_bench` package organized by the benchmark spine: config, sampling, reduced-basis utilities, predictors, RF-LROM fitting, prediction, metrics, and artifacts. Keep pure NumPy functionality unit-tested first, then add ROSE-backed integration code and a package-native Notebook 02 benchmark generator.

**Tech Stack:** Python 3.11-compatible code, NumPy, SciPy where already used, `nuclear-rose` for integration paths, nbformat for notebook generation, pytest for tests, JSON/NPZ artifacts for parity evidence.

---

## File Structure

Create these files:

- `pyproject.toml`: minimal project/test configuration for local editable imports and pytest.
- `lrom_bench/__init__.py`: public package version and top-level exports.
- `lrom_bench/config.py`: benchmark configs, tolerances, path helpers, and config hashing.
- `lrom_bench/sampling.py`: deterministic one-parameter and box sampling.
- `lrom_bench/numerics.py`: trapezoid integration and weighted least-squares helpers that avoid NumPy alias drift.
- `lrom_bench/reduced_basis.py`: central-reference basis data containers and LS coordinate projection.
- `lrom_bench/predictors.py`: centered raw-parameter predictors and delta-maxvol/potential predictor packs.
- `lrom_bench/rf_lrom.py`: residual-fit linear least-squares training and model container.
- `lrom_bench/prediction.py`: online reduced-system assembly, coefficient prediction, and reconstruction.
- `lrom_bench/metrics.py`: coefficient, wavefunction, and parity comparison metrics.
- `lrom_bench/artifacts.py`: save/load NPZ artifacts and JSON parity reports.
- `lrom_bench/rose_fom.py`: ROSE-specific setup and FOM helpers copied in small pieces from `Legacy_benchmark/lrom_demo/core.py`.
- `scripts/generate_benchmark_notebooks.py`: generate the new package-native Notebook 02.
- `scripts/run_notebook02_parity.py`: run or refresh Notebook 02 artifacts and parity report.
- `notebooks/02_lrom_method_walkthrough.ipynb`: generated package-native benchmark notebook.
- `docs/architecture/lrom-benchmark-spine.md`: Mermaid architecture figures at multiple abstraction levels.
- `tests/test_config.py`
- `tests/test_sampling.py`
- `tests/test_numerics.py`
- `tests/test_reduced_basis.py`
- `tests/test_predictors.py`
- `tests/test_rf_lrom.py`
- `tests/test_metrics_artifacts.py`

Modify these files:

- `Legacy_benchmark/README.md`: add a short migration note pointing to the new package-native benchmark once it exists.

Do not delete legacy notebooks during this plan.

---

### Task 1: Project And Test Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `lrom_bench/__init__.py`
- Verify: `docs/architecture/lrom-benchmark-spine.md`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create the minimal project configuration**

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "lrom-bench"
version = "0.1.0"
description = "Research benchmark package for LROM notebook parity"
requires-python = ">=3.11"
dependencies = [
  "numpy",
  "scipy",
  "matplotlib",
  "pandas",
  "nbformat",
  "nbclient",
  "nuclear-rose>=1.1.7",
]

[project.optional-dependencies]
test = ["pytest"]

[tool.setuptools.packages.find]
where = ["."]
include = ["lrom_bench*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create the package entrypoint**

Create `lrom_bench/__init__.py` with:

```python
"""Research benchmark package for learned reduced-operator model parity."""

from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 3: Write the first failing import test**

Create `tests/test_config.py` with:

```python
from __future__ import annotations


def test_package_imports() -> None:
    import lrom_bench

    assert lrom_bench.__version__ == "0.1.0"
```

- [ ] **Step 4: Run the scaffold test**

Run:

```bash
python -m pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Verify Mermaid architecture figures exist**

Run:

```bash
python - <<'PY'
from pathlib import Path
path = Path("docs/architecture/lrom-benchmark-spine.md")
text = path.read_text()
assert text.count("```mermaid") == 5
assert "Figure 1: Project Context" in text
assert "Figure 2: Package Spine" in text
assert "Figure 3: Notebook 02 Scientific Flow" in text
assert "Figure 4: RF-LROM Training And Prediction" in text
assert "Figure 5: Parity And Future Memory" in text
print(path)
PY
```

Expected: prints `docs/architecture/lrom-benchmark-spine.md`.

- [ ] **Step 6: Commit scaffold and architecture figures**

Run:

```bash
git add pyproject.toml lrom_bench/__init__.py tests/test_config.py docs/architecture/lrom-benchmark-spine.md
git commit -m "Add LROM benchmark package scaffold"
```

---

### Task 2: Config, Paths, And Artifact Metadata

**Files:**
- Modify: `lrom_bench/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add failing tests for config hashing and paths**

Replace `tests/test_config.py` with:

```python
from __future__ import annotations

from pathlib import Path

from lrom_bench.config import BenchmarkPaths, Notebook02Config


def test_package_imports() -> None:
    import lrom_bench

    assert lrom_bench.__version__ == "0.1.0"


def test_notebook02_config_hash_is_stable() -> None:
    cfg = Notebook02Config(n_mesh=64, n_phi=3, n_u=5)

    assert cfg.config_hash() == Notebook02Config(n_mesh=64, n_phi=3, n_u=5).config_hash()
    assert cfg.config_hash() != Notebook02Config(n_mesh=65, n_phi=3, n_u=5).config_hash()
    assert len(cfg.config_hash()) == 16


def test_benchmark_paths_are_stable(tmp_path: Path) -> None:
    paths = BenchmarkPaths(root=tmp_path)

    assert paths.legacy_npz("notebook02").as_posix().endswith(
        "outputs/benchmarks/legacy/notebook02.npz"
    )
    assert paths.new_npz("notebook02").as_posix().endswith(
        "outputs/benchmarks/new/notebook02.npz"
    )
    assert paths.report_json("notebook02").as_posix().endswith(
        "outputs/benchmarks/reports/notebook02.json"
    )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_config.py -v
```

Expected: FAIL because `lrom_bench.config` does not exist.

- [ ] **Step 3: Implement config objects**

Create `lrom_bench/config.py` with:

```python
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
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit config**

Run:

```bash
git add lrom_bench/config.py tests/test_config.py
git commit -m "Add benchmark configuration and paths"
```

---

### Task 3: Sampling And Portable Numerics

**Files:**
- Create: `lrom_bench/sampling.py`
- Create: `lrom_bench/numerics.py`
- Create: `tests/test_sampling.py`
- Create: `tests/test_numerics.py`

- [ ] **Step 1: Write failing sampling tests**

Create `tests/test_sampling.py` with:

```python
from __future__ import annotations

import numpy as np

from lrom_bench.sampling import centered_box_samples, one_at_a_time_scan_samples


def test_one_at_a_time_scan_samples_preserves_center_except_active_coordinate() -> None:
    center = np.array([10.0, 2.0])
    widths = np.array([1.0, 0.5])

    samples, info = one_at_a_time_scan_samples(center, widths, n_scan=3, names=("Vv", "Rv"))

    assert samples.shape == (6, 2)
    assert info[0].name == "Vv"
    assert info[0].values.tolist() == [9.0, 10.0, 11.0]
    assert np.allclose(samples[:3, 1], 2.0)
    assert info[1].name == "Rv"
    assert info[1].values.tolist() == [1.5, 2.0, 2.5]
    assert np.allclose(samples[3:, 0], 10.0)


def test_centered_box_samples_are_deterministic_and_include_center() -> None:
    center = np.array([10.0, 2.0])
    widths = np.array([1.0, 0.5])

    samples_a = centered_box_samples(center, widths, n_samples=5, seed=123, include_center=True)
    samples_b = centered_box_samples(center, widths, n_samples=5, seed=123, include_center=True)

    assert samples_a.shape == (6, 2)
    assert np.allclose(samples_a, samples_b)
    assert np.allclose(samples_a[0], center)
    assert np.all(samples_a[:, 0] >= 9.0)
    assert np.all(samples_a[:, 0] <= 11.0)
    assert np.all(samples_a[:, 1] >= 1.5)
    assert np.all(samples_a[:, 1] <= 2.5)
```

- [ ] **Step 2: Write failing numerics tests**

Create `tests/test_numerics.py` with:

```python
from __future__ import annotations

import numpy as np

from lrom_bench.numerics import least_squares_basis_coefficients, trapezoid_integral


def test_trapezoid_integral_matches_linear_function_area() -> None:
    x = np.array([0.0, 0.5, 1.0])
    y = 2.0 * x

    assert trapezoid_integral(y, x=x) == 1.0


def test_least_squares_basis_coefficients_recovers_known_coefficients() -> None:
    mesh = np.array([0.0, 1.0, 2.0])
    phi0 = np.array([1.0, 1.0, 1.0])
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    coeff_true = np.array([[2.0, -3.0], [0.5, 4.0]])
    wavefunctions = phi0[np.newaxis, :] + coeff_true @ vectors.T

    coeff = least_squares_basis_coefficients(vectors, phi0, wavefunctions, mesh)

    assert np.allclose(coeff, coeff_true)
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_sampling.py tests/test_numerics.py -v
```

Expected: FAIL because modules do not exist.

- [ ] **Step 4: Implement sampling**

Create `lrom_bench/sampling.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class ScanInfo:
    name: str
    values: np.ndarray
    slc: slice


def one_at_a_time_scan_samples(
    center: np.ndarray,
    widths: np.ndarray,
    n_scan: int,
    names: tuple[str, ...],
) -> tuple[np.ndarray, list[ScanInfo]]:
    center = np.asarray(center, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if center.shape != widths.shape:
        raise ValueError("center and widths must have the same shape")
    if len(names) != center.size:
        raise ValueError("names must have one entry per parameter")
    if n_scan < 2:
        raise ValueError("n_scan must be at least 2")

    samples = []
    info: list[ScanInfo] = []
    start = 0
    for j, name in enumerate(names):
        values = np.linspace(center[j] - widths[j], center[j] + widths[j], n_scan)
        block = np.tile(center, (n_scan, 1))
        block[:, j] = values
        stop = start + n_scan
        samples.append(block)
        info.append(ScanInfo(name=name, values=values, slc=slice(start, stop)))
        start = stop
    return np.vstack(samples), info


def centered_box_samples(
    center: np.ndarray,
    widths: np.ndarray,
    n_samples: int,
    seed: int,
    include_center: bool = False,
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if center.shape != widths.shape:
        raise ValueError("center and widths must have the same shape")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    lower = center - widths
    upper = center + widths
    sampler = qmc.LatinHypercube(d=center.size, seed=seed)
    samples = qmc.scale(sampler.random(n_samples), lower, upper)
    if include_center:
        samples = np.vstack([center, samples])
    return samples
```

- [ ] **Step 5: Implement portable numerics**

Create `lrom_bench/numerics.py` with:

```python
from __future__ import annotations

import numpy as np


def trapezoid_integral(
    y: np.ndarray,
    x: np.ndarray | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> np.ndarray:
    y = np.asarray(y)
    if y.shape[axis] < 2:
        return np.sum(y, axis=axis) * 0.0
    moved = np.moveaxis(y, axis, -1)
    if x is None:
        widths = np.asarray(dx)
    else:
        x = np.asarray(x)
        widths = np.diff(x, axis=0 if x.ndim > 1 else -1)
    area = (moved[..., 1:] + moved[..., :-1]) * 0.5
    return np.sum(area * widths, axis=-1)


def trapezoid_sqrt_weights(mesh: np.ndarray) -> np.ndarray:
    mesh = np.asarray(mesh, dtype=float)
    if mesh.ndim != 1 or mesh.size < 2:
        raise ValueError("mesh must be a one-dimensional array with at least two points")
    weights = np.empty_like(mesh, dtype=float)
    weights[1:-1] = 0.5 * (mesh[2:] - mesh[:-2])
    weights[0] = 0.5 * (mesh[1] - mesh[0])
    weights[-1] = 0.5 * (mesh[-1] - mesh[-2])
    return np.sqrt(weights)


def least_squares_basis_coefficients(
    vectors: np.ndarray,
    phi0: np.ndarray,
    wavefunctions: np.ndarray,
    mesh: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.complex128)
    phi0 = np.asarray(phi0, dtype=np.complex128)
    wavefunctions = np.asarray(wavefunctions, dtype=np.complex128)
    weights = trapezoid_sqrt_weights(mesh)
    weighted_vectors = weights[:, np.newaxis] * vectors
    coeffs = []
    for phi in wavefunctions:
        rhs = weights * (phi - phi0)
        coeff, *_ = np.linalg.lstsq(weighted_vectors, rhs, rcond=None)
        coeffs.append(coeff)
    return np.asarray(coeffs)
```

- [ ] **Step 6: Run tests**

Run:

```bash
python -m pytest tests/test_sampling.py tests/test_numerics.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit sampling and numerics**

Run:

```bash
git add lrom_bench/sampling.py lrom_bench/numerics.py tests/test_sampling.py tests/test_numerics.py
git commit -m "Add benchmark sampling and portable numerics"
```

---

### Task 4: Reduced-Basis Boundary

**Files:**
- Create: `lrom_bench/reduced_basis.py`
- Create: `tests/test_reduced_basis.py`

- [ ] **Step 1: Write failing reduced-basis tests**

Create `tests/test_reduced_basis.py` with:

```python
from __future__ import annotations

import numpy as np

from lrom_bench.reduced_basis import CentralBasisData, project_ls_coordinates


def test_project_ls_coordinates_wraps_central_basis_data() -> None:
    mesh = np.array([0.0, 1.0, 2.0])
    phi0 = np.array([1.0, 1.0, 1.0])
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    basis = CentralBasisData(phi0=phi0, vectors=vectors, mesh=mesh)
    coeff_true = np.array([[2.0, -3.0], [0.5, 4.0]])
    wavefunctions = phi0[np.newaxis, :] + coeff_true @ vectors.T

    coeff = project_ls_coordinates(basis, wavefunctions)

    assert basis.n_basis == 2
    assert basis.n_mesh == 3
    assert np.allclose(coeff, coeff_true)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_reduced_basis.py -v
```

Expected: FAIL because `lrom_bench.reduced_basis` does not exist.

- [ ] **Step 3: Implement reduced-basis data boundary**

Create `lrom_bench/reduced_basis.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lrom_bench.numerics import least_squares_basis_coefficients


@dataclass(frozen=True)
class CentralBasisData:
    phi0: np.ndarray
    vectors: np.ndarray
    mesh: np.ndarray

    def __post_init__(self) -> None:
        phi0 = np.asarray(self.phi0)
        vectors = np.asarray(self.vectors)
        mesh = np.asarray(self.mesh)
        if phi0.ndim != 1:
            raise ValueError("phi0 must be one-dimensional")
        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (n_mesh, n_basis)")
        if mesh.ndim != 1:
            raise ValueError("mesh must be one-dimensional")
        if vectors.shape[0] != phi0.size:
            raise ValueError("vectors and phi0 must share the mesh dimension")
        if mesh.size != phi0.size:
            raise ValueError("mesh and phi0 must have the same length")

    @property
    def n_mesh(self) -> int:
        return int(np.asarray(self.phi0).size)

    @property
    def n_basis(self) -> int:
        return int(np.asarray(self.vectors).shape[1])


def project_ls_coordinates(
    basis: CentralBasisData,
    wavefunctions: np.ndarray,
) -> np.ndarray:
    return least_squares_basis_coefficients(
        vectors=basis.vectors,
        phi0=basis.phi0,
        wavefunctions=wavefunctions,
        mesh=basis.mesh,
    )
```

- [ ] **Step 4: Run reduced-basis tests**

Run:

```bash
python -m pytest tests/test_reduced_basis.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit reduced-basis boundary**

Run:

```bash
git add lrom_bench/reduced_basis.py tests/test_reduced_basis.py
git commit -m "Add central reduced-basis boundary"
```

---

### Task 5: Predictors And Delta-Maxvol Selection

**Files:**
- Create: `lrom_bench/predictors.py`
- Create: `tests/test_predictors.py`

- [ ] **Step 1: Write failing predictor tests**

Create `tests/test_predictors.py` with:

```python
from __future__ import annotations

import numpy as np

from lrom_bench.predictors import (
    PredictorPack,
    centered_parameter_predictors,
    centered_potential_predictors,
    greedy_maxvol_indices,
    make_potential_predictor_pack,
)


def test_centered_parameter_predictors() -> None:
    samples = np.array([[2.0, 5.0], [4.0, 9.0]])
    center = np.array([3.0, 7.0])
    scales = np.array([2.0, 4.0])

    got = centered_parameter_predictors(samples, center, scales)

    assert np.allclose(got, [[-0.5, -0.5], [0.5, 0.5]])


def test_greedy_maxvol_indices_selects_independent_rows() -> None:
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.1, 0.1],
        ]
    )

    got = greedy_maxvol_indices(basis)

    assert set(got.tolist()) == {0, 1}


def test_potential_predictor_pack_centers_and_scales_values() -> None:
    mesh = np.linspace(0.0, 1.0, 5)
    alphas = np.array([[1.0], [2.0], [3.0]])
    center = np.array([2.0])

    def potential(points: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return alpha[0] * (1.0 + points)

    pack = make_potential_predictor_pack(
        potential=potential,
        train_alphas=alphas,
        central_alpha=center,
        mesh=mesh,
        n_predictors=2,
        min_mesh_value=0.0,
    )
    features = centered_potential_predictors(potential, np.array([[2.0], [3.0]]), pack)

    assert isinstance(pack, PredictorPack)
    assert pack.s_points.shape == (2,)
    assert np.allclose(features[0], 0.0)
    assert np.all(np.isfinite(features[1]))
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_predictors.py -v
```

Expected: FAIL because `lrom_bench.predictors` does not exist.

- [ ] **Step 3: Implement predictors**

Create `lrom_bench/predictors.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


PotentialFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PredictorPack:
    s_points: np.ndarray
    center_values: np.ndarray
    scales: np.ndarray
    singular_values: np.ndarray


def centered_parameter_predictors(
    samples: np.ndarray,
    center: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    center = np.asarray(center, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if np.any(scales == 0.0):
        raise ValueError("scales must be nonzero")
    return (samples - center) / scales


def greedy_maxvol_indices(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis)
    if basis.ndim != 2:
        raise ValueError("basis must be two-dimensional")
    n_rows, n_cols = basis.shape
    if n_cols > n_rows:
        raise ValueError("basis must have at least as many rows as columns")

    selected: list[int] = []
    residual_basis = basis.copy()
    for _ in range(n_cols):
        row_norms = np.linalg.norm(residual_basis, axis=1)
        row_norms[selected] = -np.inf
        idx = int(np.argmax(row_norms))
        selected.append(idx)
        selected_matrix = basis[selected]
        projection_coeffs = np.linalg.lstsq(selected_matrix.T, basis.T, rcond=None)[0]
        projected = (selected_matrix.T @ projection_coeffs).T
        residual_basis = basis - projected
    return np.array(selected, dtype=int)


def raw_potential_predictors(
    potential: PotentialFn,
    alphas: np.ndarray,
    s_points: np.ndarray,
) -> np.ndarray:
    return np.asarray([potential(s_points, alpha) for alpha in np.asarray(alphas)])


def make_potential_predictor_pack(
    potential: PotentialFn,
    train_alphas: np.ndarray,
    central_alpha: np.ndarray,
    mesh: np.ndarray,
    n_predictors: int,
    min_mesh_value: float = 0.0,
) -> PredictorPack:
    mesh = np.asarray(mesh, dtype=float)
    center = potential(mesh, central_alpha)
    delta = np.asarray([potential(mesh, alpha) - center for alpha in train_alphas]).T
    allowed = np.flatnonzero(mesh >= min_mesh_value)
    if allowed.size < n_predictors:
        raise ValueError("not enough allowed mesh points for predictor selection")
    u, singular_values, _ = np.linalg.svd(delta[allowed], full_matrices=False)
    local = greedy_maxvol_indices(u[:, :n_predictors])
    indices = allowed[local]
    s_points = mesh[indices]
    center_values = potential(s_points, central_alpha)
    raw_train = raw_potential_predictors(potential, train_alphas, s_points)
    scales = np.maximum(np.std(raw_train - center_values[np.newaxis, :], axis=0), 1e-12)
    return PredictorPack(
        s_points=s_points,
        center_values=center_values,
        scales=scales,
        singular_values=singular_values,
    )


def centered_potential_predictors(
    potential: PotentialFn,
    alphas: np.ndarray,
    pack: PredictorPack,
    normalize: bool = True,
) -> np.ndarray:
    values = raw_potential_predictors(potential, alphas, pack.s_points)
    centered = values - pack.center_values[np.newaxis, :]
    if normalize:
        centered = centered / pack.scales[np.newaxis, :]
    return centered
```

- [ ] **Step 4: Run predictor tests**

Run:

```bash
python -m pytest tests/test_predictors.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit predictors**

Run:

```bash
git add lrom_bench/predictors.py tests/test_predictors.py
git commit -m "Add LROM predictor utilities"
```

---

### Task 6: RF-LROM Fit, Prediction, And Metrics

**Files:**
- Create: `lrom_bench/rf_lrom.py`
- Create: `lrom_bench/prediction.py`
- Create: `lrom_bench/metrics.py`
- Create: `tests/test_rf_lrom.py`

- [ ] **Step 1: Write failing RF-LROM tests**

Create `tests/test_rf_lrom.py` with:

```python
from __future__ import annotations

import numpy as np

from lrom_bench.metrics import relative_l2_rows
from lrom_bench.prediction import predict_coefficients, reconstruct_from_basis
from lrom_bench.rf_lrom import fit_central_lrom


def test_fit_central_lrom_recovers_linear_coefficients() -> None:
    predictors = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=float)
    coeff_targets = np.array([[-2.0], [0.0], [2.0], [4.0]], dtype=float)

    model = fit_central_lrom("linear", predictors, coeff_targets)
    pred = predict_coefficients(model, predictors)

    assert model.n_basis == 1
    assert model.n_predictors == 1
    assert model.residual_mse < 1e-24
    assert np.allclose(pred, coeff_targets)


def test_reconstruct_from_basis_and_relative_l2_rows() -> None:
    phi0 = np.array([1.0, 1.0, 1.0])
    vectors = np.array([[1.0], [0.0], [-1.0]])
    coeffs = np.array([[2.0], [3.0]])

    recon = reconstruct_from_basis(phi0, vectors, coeffs)

    assert np.allclose(recon[0], [3.0, 1.0, -1.0])
    assert np.allclose(relative_l2_rows(recon, recon), 0.0)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_rf_lrom.py -v
```

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement RF-LROM fitting**

Create `lrom_bench/rf_lrom.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CentralLROM:
    name: str
    matrices: np.ndarray
    vectors: np.ndarray
    residual_mse: float
    rank: int
    singular_values: np.ndarray

    @property
    def n_basis(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def n_predictors(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def n_complex_parameters(self) -> int:
        n = self.n_basis
        return self.n_predictors * (n * n + n)


def fit_central_lrom(
    name: str,
    predictors: np.ndarray,
    coeff_targets: np.ndarray,
) -> CentralLROM:
    predictors = np.asarray(predictors, dtype=np.complex128)
    coeff_targets = np.asarray(coeff_targets, dtype=np.complex128)
    if predictors.ndim != 2:
        raise ValueError("predictors must have shape (n_samples, K)")
    if coeff_targets.ndim != 2:
        raise ValueError("coeff_targets must have shape (n_samples, n_basis)")
    if predictors.shape[0] != coeff_targets.shape[0]:
        raise ValueError("predictors and coeff_targets need the same sample count")

    n_samples, k_pred = predictors.shape
    n_basis = coeff_targets.shape[1]
    n_unknown = k_pred * (n_basis * n_basis + n_basis)
    design = np.zeros((n_samples * n_basis, n_unknown), dtype=np.complex128)
    target = -coeff_targets.reshape(-1)

    def matrix_offset(j: int) -> int:
        return j * (n_basis * n_basis + n_basis)

    def vector_offset(j: int) -> int:
        return matrix_offset(j) + n_basis * n_basis

    for sample_idx, (p_row, a_row) in enumerate(zip(predictors, coeff_targets)):
        for row in range(n_basis):
            eq = sample_idx * n_basis + row
            for j, p in enumerate(p_row):
                moff = matrix_offset(j)
                voff = vector_offset(j)
                design[eq, moff + row * n_basis : moff + (row + 1) * n_basis] = p * a_row
                design[eq, voff + row] = -p

    solution, _residuals, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)

    matrices = np.zeros((k_pred, n_basis, n_basis), dtype=np.complex128)
    vectors = np.zeros((k_pred, n_basis), dtype=np.complex128)
    for j in range(k_pred):
        moff = matrix_offset(j)
        voff = vector_offset(j)
        matrices[j] = solution[moff : moff + n_basis * n_basis].reshape(n_basis, n_basis)
        vectors[j] = solution[voff : voff + n_basis]

    residual = design @ solution - target
    return CentralLROM(
        name=name,
        matrices=matrices,
        vectors=vectors,
        residual_mse=float(np.mean(np.abs(residual) ** 2)),
        rank=int(rank),
        singular_values=singular_values,
    )
```

- [ ] **Step 4: Implement prediction helpers**

Create `lrom_bench/prediction.py` with:

```python
from __future__ import annotations

import numpy as np

from lrom_bench.rf_lrom import CentralLROM


def predict_coefficients(model: CentralLROM, predictors: np.ndarray) -> np.ndarray:
    predictors = np.asarray(predictors, dtype=np.complex128)
    if predictors.ndim == 1:
        predictors = predictors[np.newaxis, :]
    if predictors.shape[1] != model.n_predictors:
        raise ValueError("predictor width does not match model")

    n_basis = model.n_basis
    identity = np.eye(n_basis, dtype=np.complex128)
    coeffs = np.empty((predictors.shape[0], n_basis), dtype=np.complex128)
    for i, p_row in enumerate(predictors):
        matrix = identity + np.einsum("k,kij->ij", p_row, model.matrices)
        rhs = np.einsum("k,kj->j", p_row, model.vectors)
        coeffs[i] = np.linalg.solve(matrix, rhs)
    return coeffs


def reconstruct_from_basis(
    phi0: np.ndarray,
    vectors: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    phi0 = np.asarray(phi0, dtype=np.complex128)
    vectors = np.asarray(vectors, dtype=np.complex128)
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    return phi0[np.newaxis, :] + coeffs @ vectors.T
```

- [ ] **Step 5: Implement metrics**

Create `lrom_bench/metrics.py` with:

```python
from __future__ import annotations

import numpy as np


def relative_l2_rows(pred: np.ndarray, ref: np.ndarray, floor: float = 1e-300) -> np.ndarray:
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError("pred and ref must have the same shape")
    return np.linalg.norm(pred - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), floor)


def absolute_l2_rows(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError("pred and ref must have the same shape")
    return np.linalg.norm(pred - ref, axis=1)
```

- [ ] **Step 6: Run RF-LROM tests**

Run:

```bash
python -m pytest tests/test_rf_lrom.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit RF-LROM core**

Run:

```bash
git add lrom_bench/rf_lrom.py lrom_bench/prediction.py lrom_bench/metrics.py tests/test_rf_lrom.py
git commit -m "Add residual-fit LROM core"
```

---

### Task 7: Artifact Save/Load And Parity Reports

**Files:**
- Create: `lrom_bench/artifacts.py`
- Create: `tests/test_metrics_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

Create `tests/test_metrics_artifacts.py` with:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_metrics_artifacts.py -v
```

Expected: FAIL because `lrom_bench.artifacts` does not exist.

- [ ] **Step 3: Implement artifact helpers**

Create `lrom_bench/artifacts.py` with:

```python
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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": all(item.passed for item in comparisons),
        "metadata": metadata,
        "comparisons": [asdict(item) for item in comparisons],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run artifact tests**

Run:

```bash
python -m pytest tests/test_metrics_artifacts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit artifacts**

Run:

```bash
git add lrom_bench/artifacts.py tests/test_metrics_artifacts.py
git commit -m "Add benchmark artifact parity helpers"
```

---

### Task 8: ROSE Integration Boundary

**Files:**
- Create: `lrom_bench/rose_fom.py`
- Create: `tests/test_rose_fom_import.py`

- [ ] **Step 1: Write import-boundary test**

Create `tests/test_rose_fom_import.py` with:

```python
from __future__ import annotations


def test_rose_fom_module_imports_without_importing_rose_immediately() -> None:
    import lrom_bench.rose_fom as rose_fom

    assert rose_fom.TARGET == (40, 20)
    assert rose_fom.PROJECTILE == (1, 0)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/test_rose_fom_import.py -v
```

Expected: FAIL because `lrom_bench.rose_fom` does not exist.

- [ ] **Step 3: Implement lazy ROSE helpers**

Create `lrom_bench/rose_fom.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

TARGET = (40, 20)
PROJECTILE = (1, 0)
E_LAB = 14.1


def import_rose() -> Any:
    try:
        import rose
    except ImportError as exc:
        raise ImportError(
            "The LROM benchmark requires nuclear-rose, imported as rose. "
            "Install it with `python -m pip install nuclear-rose`."
        ) from exc
    return rose


@dataclass(frozen=True)
class KDParameters:
    mu: float
    e_com: float
    k: float
    eta: float
    r_c: float
    alpha: np.ndarray


def central_kd_parameters(
    target: tuple[int, int] = TARGET,
    projectile: tuple[int, int] = PROJECTILE,
    e_lab: float = E_LAB,
) -> KDParameters:
    rose = import_rose()
    mu, e_com, k, eta = rose.kinematics(target=target, projectile=projectile, E_lab=e_lab)
    kd = rose.koning_delaroche.KDGlobal(rose.Projectile.neutron)
    r_c, alpha = kd.get_params(target[0], target[1], mu, e_lab, k)
    return KDParameters(
        mu=float(mu),
        e_com=float(e_com),
        k=float(k),
        eta=float(eta),
        r_c=float(r_c),
        alpha=np.asarray(alpha, dtype=float),
    )


def make_alphas(alpha0: np.ndarray, param_index: int, values: np.ndarray) -> np.ndarray:
    alphas = np.tile(np.asarray(alpha0, dtype=float), (len(values), 1))
    alphas[:, param_index] = values
    return alphas
```

- [ ] **Step 4: Run import-boundary test**

Run:

```bash
python -m pytest tests/test_rose_fom_import.py -v
```

Expected: PASS.

- [ ] **Step 5: Add an optional local smoke command**

Run this only in an environment with `nuclear-rose` installed:

```bash
python - <<'PY'
from lrom_bench.rose_fom import central_kd_parameters
params = central_kd_parameters()
print(params.alpha.shape)
PY
```

Expected: prints `(15,)`.

- [ ] **Step 6: Commit ROSE boundary**

Run:

```bash
git add lrom_bench/rose_fom.py tests/test_rose_fom_import.py
git commit -m "Add lazy ROSE integration boundary"
```

---

### Task 9: Package-Native Notebook 02 Generator

**Files:**
- Create: `scripts/generate_benchmark_notebooks.py`
- Create: `notebooks/02_lrom_method_walkthrough.ipynb`
- Modify: `Legacy_benchmark/README.md`

- [ ] **Step 1: Create notebook generator script**

Create `scripts/generate_benchmark_notebooks.py` with:

```python
from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def write_notebook(path: Path, cells: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, path)


def notebook02_cells() -> list:
    setup = r"""
    from pathlib import Path
    import sys

    import matplotlib.pyplot as plt
    import numpy as np

    ROOT = Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from lrom_bench.config import BenchmarkPaths, Notebook02Config
    from lrom_bench import predictors
    from lrom_bench import rf_lrom
    from lrom_bench import prediction
    from lrom_bench import metrics

    cfg = Notebook02Config()
    paths = BenchmarkPaths(ROOT)
    print("config hash:", cfg.config_hash())
    """
    return [
        md(
            """
            # 02. LROM Method Walkthrough

            This package-native benchmark keeps the legacy scientific narrative visible:
            `Vv`, `Rv`, broad `Vv`/`Rv`, and operator-informed potential predictors.

            The notebook is also a parity artifact. It should save new benchmark arrays
            and compare them against frozen legacy outputs before the legacy notebook is retired.
            """
        ),
        code(setup),
        md("## 1. Setup And Benchmark Configuration"),
        code("cfg"),
        md("## 2. Central Reference State And Reduced-Basis Convention"),
        code(
            """
            # ROSE-backed central state construction is added in the implementation tasks.
            # This cell stays visible because phi0 = phi(alpha_c) is the core convention.
            """
        ),
        md("## 3. Vv-Only Scan"),
        code(
            """
            # Build Vv samples, LS target coordinates, fit RF-LROM, and plot coefficient/wavefunction errors.
            """
        ),
        md("## 4. Rv-Only Scan"),
        code(
            """
            # Repeat the one-parameter RF-LROM check for the radius parameter.
            """
        ),
        md("## 5. Broad Vv/Rv Box"),
        code(
            """
            # Show where raw centered parameter predictors begin to struggle.
            """
        ),
        md("## 6. Operator-Informed Potential Predictors"),
        code(
            """
            # Select delta-maxvol potential points and fit the predictor RF-LROM.
            """
        ),
        md("## 7. Frozen Parity Comparison"),
        code(
            """
            # Save new outputs and compare against outputs/benchmarks/legacy/notebook02_lrom_walkthrough.npz.
            """
        ),
    ]


def main() -> None:
    write_notebook(NOTEBOOK_DIR / "02_lrom_method_walkthrough.ipynb", notebook02_cells())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the notebook**

Run:

```bash
python scripts/generate_benchmark_notebooks.py
```

Expected: creates `notebooks/02_lrom_method_walkthrough.ipynb`.

- [ ] **Step 3: Verify notebook headings**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path("notebooks/02_lrom_method_walkthrough.ipynb").read_text())
headings = [
    line.strip()
    for cell in nb["cells"]
    if cell["cell_type"] == "markdown"
    for line in "".join(cell["source"]).splitlines()
    if line.startswith("#")
]
print("\n".join(headings))
PY
```

Expected output includes:

```text
# 02. LROM Method Walkthrough
## 1. Setup And Benchmark Configuration
## 2. Central Reference State And Reduced-Basis Convention
## 3. Vv-Only Scan
## 4. Rv-Only Scan
## 5. Broad Vv/Rv Box
## 6. Operator-Informed Potential Predictors
## 7. Frozen Parity Comparison
```

- [ ] **Step 4: Add migration note**

Append this paragraph to `Legacy_benchmark/README.md`:

```markdown

## Migration Note

The package-native benchmark spine is being developed at the repository root under
`lrom_bench/`, with the first parity target in `notebooks/02_lrom_method_walkthrough.ipynb`.
Keep these legacy notebooks until the new Notebook 02 parity report passes against the
frozen legacy artifacts and the user approves deletion.
```

- [ ] **Step 5: Commit notebook generator**

Run:

```bash
git add scripts/generate_benchmark_notebooks.py notebooks/02_lrom_method_walkthrough.ipynb Legacy_benchmark/README.md
git commit -m "Add package-native Notebook 02 generator"
```

---

### Task 10: Notebook 02 Parity Runner

**Files:**
- Create: `scripts/run_notebook02_parity.py`
- Modify: `lrom_bench/artifacts.py`
- Test: `tests/test_metrics_artifacts.py`

- [ ] **Step 1: Add report metadata helper test**

Append to `tests/test_metrics_artifacts.py`:

```python

def test_report_metadata_can_include_scientific_notes(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    comparison = ArrayComparison(name="x", passed=True, max_abs_error=0.0)
    metadata = {
        "config_hash": "abc123",
        "scientific_notes": ["frozen artifact parity"],
    }

    write_parity_report(report, comparisons=[comparison], metadata=metadata)

    payload = json.loads(report.read_text())
    assert payload["metadata"]["scientific_notes"] == ["frozen artifact parity"]
```

- [ ] **Step 2: Run artifact tests**

Run:

```bash
python -m pytest tests/test_metrics_artifacts.py -v
```

Expected: PASS.

- [ ] **Step 3: Create parity runner skeleton**

Create `scripts/run_notebook02_parity.py` with:

```python
from __future__ import annotations

from pathlib import Path

from lrom_bench.artifacts import compare_npz_artifacts, write_parity_report
from lrom_bench.config import BenchmarkPaths, Notebook02Config


STEM = "notebook02_lrom_walkthrough"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = Notebook02Config()
    paths = BenchmarkPaths(root)
    legacy_path = paths.legacy_npz(STEM)
    new_path = paths.new_npz(STEM)
    report_path = paths.report_json("notebook02_parity")

    if not legacy_path.exists():
        raise FileNotFoundError(f"missing frozen legacy artifact: {legacy_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"missing new benchmark artifact: {new_path}")

    comparisons = compare_npz_artifacts(
        legacy_path,
        new_path,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    write_parity_report(
        report_path,
        comparisons=comparisons,
        metadata={
            "config_hash": cfg.config_hash(),
            "legacy_artifact": str(legacy_path),
            "new_artifact": str(new_path),
            "rtol": cfg.rtol,
            "atol": cfg.atol,
            "scientific_notes": ["frozen artifact parity"],
        },
    )
    print(report_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run pure unit tests**

Run:

```bash
python -m pytest tests/test_metrics_artifacts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit parity runner**

Run:

```bash
git add scripts/run_notebook02_parity.py tests/test_metrics_artifacts.py
git commit -m "Add Notebook 02 parity runner"
```

---

### Task 11: Full Verification Pass

**Files:**
- No new files.

- [ ] **Step 1: Run all unit tests**

Run:

```bash
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Regenerate package-native Notebook 02**

Run:

```bash
python scripts/generate_benchmark_notebooks.py
```

Expected: completes with no traceback and updates `notebooks/02_lrom_method_walkthrough.ipynb`.

- [ ] **Step 3: Check git diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only intentional files from completed tasks are modified. Existing unrelated workspace changes under moved legacy files may still be present; do not revert them.

- [ ] **Step 4: Commit verification-only notebook refresh if needed**

If Step 2 changed only `notebooks/02_lrom_method_walkthrough.ipynb`, run:

```bash
git add notebooks/02_lrom_method_walkthrough.ipynb
git commit -m "Refresh generated Notebook 02 benchmark"
```

If Step 2 produced no diff, skip this commit.

---

## Self-Review Checklist

- Spec coverage: Tasks 1-8 build the Python benchmark package, Task 1 includes Mermaid architecture figures, Task 9 builds the readable package-native Notebook 02, Task 10 adds frozen artifact parity reporting, and Task 11 verifies the full slice.
- Notebook narrative: Task 9 preserves the `Vv`, `Rv`, broad `Vv`/`Rv`, and potential-predictor section order.
- Two-layer parity: Task 10 implements frozen artifact parity; rerun-legacy parity remains a later implementation slice after frozen references exist.
- Future graph/vector layer: Task 7 and Task 10 create metadata-rich JSON/NPZ artifacts that can be indexed later without database infrastructure.
- Type consistency: package names, function names, and dataclass names are consistent across tests and implementation steps.
