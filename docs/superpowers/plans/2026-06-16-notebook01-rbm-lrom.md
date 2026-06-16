# Notebook 01 RBM/LROM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one package-native Notebook 1 that compares ROSE/RBM and RF-LROM for a single `l = 0` real Woods-Saxon scattering wavefunction.

**Architecture:** Keep notebook plotting visible in notebook cells. Add only small package helpers that produce scientific objects: configs, samples, real Woods-Saxon setup, wavefunctions, basis data, coefficients, predictors, and metrics. Update architecture documentation when package modules gain Notebook 1 responsibilities.

**Tech Stack:** Python 3.11+, NumPy, SciPy, nbformat, pytest, nuclear-rose imported as `rose`, Matplotlib used only inside notebook cells.

---

## File Structure

- Modify `lrom_bench/config.py`: add `Notebook01Config`.
- Modify `tests/test_config.py`: test the Notebook 1 config hash and defaults.
- Modify `lrom_bench/sampling.py`: add `centered_1d_values`.
- Modify `tests/test_sampling.py`: test deterministic centered one-dimensional scan values.
- Modify `lrom_bench/reduced_basis.py`: add `build_centered_svd_basis`.
- Modify `tests/test_reduced_basis.py`: test centered SVD basis construction.
- Modify `lrom_bench/rose_fom.py`: add real Woods-Saxon parameters, problem construction, wavefunction and ROSE/RBM coefficient helpers.
- Create `tests/test_rose_fom_real_ws.py`: test import safety and the small real Woods-Saxon helper surfaces.
- Create `scripts/generate_notebook01.py`: generate Notebook 1 with visible plotting code in cells.
- Create `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`: generated notebook artifact.
- Modify `docs/architecture/lrom-benchmark-spine.md`: record Notebook 1 helper additions if `rose_fom.py` and `reduced_basis.py` are expanded.

Do not create package plotting functions. Do not create a function that runs the whole Notebook 1 workflow.

---

### Task 1: Notebook 1 Config And One-Dimensional Sampling

**Files:**
- Modify: `lrom_bench/config.py`
- Modify: `lrom_bench/sampling.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_sampling.py`

- [ ] **Step 1: Add failing tests for Notebook 1 config and centered 1D values**

Append to `tests/test_config.py`:

```python
from lrom_bench.config import Notebook01Config


def test_notebook01_config_hash_is_stable() -> None:
    cfg = Notebook01Config(n_mesh=64, n_basis=3, n_u=5)

    assert cfg.config_hash() == Notebook01Config(n_mesh=64, n_basis=3, n_u=5).config_hash()
    assert cfg.config_hash() != Notebook01Config(n_mesh=65, n_basis=3, n_u=5).config_hash()
    assert cfg.parameter_names == ("Vv", "Rv", "av")
    assert len(cfg.config_hash()) == 16
```

Append to `tests/test_sampling.py`:

```python
from lrom_bench.sampling import centered_1d_values


def test_centered_1d_values_are_symmetric_and_include_center() -> None:
    values = centered_1d_values(center=10.0, width=2.0, n=5)

    assert values.tolist() == [8.0, 9.0, 10.0, 11.0, 12.0]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_config.py tests/test_sampling.py -v
```

Expected: FAIL because `Notebook01Config` and `centered_1d_values` do not exist.

- [ ] **Step 3: Implement `Notebook01Config`**

Add to `lrom_bench/config.py` after imports and before `Notebook02Config`:

```python
@dataclass(frozen=True)
class Notebook01Config:
    target_a: int = 40
    target_z: int = 20
    projectile_a: int = 1
    projectile_z: int = 0
    e_lab: float = 14.1
    l: int = 0
    n_mesh: int = 900
    n_basis: int = 4
    n_u: int = 8
    vv_width: float = 8.0
    rv_width: float = 0.35
    av_width: float = 0.12
    n_vv_train: int = 17
    n_vv_test: int = 33
    n_box_train: int = 49
    n_box_test: int = 81
    n_predictors: int = 6
    seed_train: int = 20260616
    seed_test: int = 20260617
    min_predictor_radius: float = 0.2
    parameter_names: tuple[str, str, str] = ("Vv", "Rv", "av")
    rtol: float = 1e-8
    atol: float = 1e-10

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.as_jsonable(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 4: Implement `centered_1d_values`**

Add to `lrom_bench/sampling.py` after `ScanInfo`:

```python
def centered_1d_values(center: float, width: float, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    if width <= 0.0:
        raise ValueError("width must be positive")
    return np.linspace(float(center) - float(width), float(center) + float(width), int(n))
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_config.py tests/test_sampling.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add lrom_bench/config.py lrom_bench/sampling.py tests/test_config.py tests/test_sampling.py
git commit -m "Add Notebook 1 config and sampling helpers"
```

---

### Task 2: Centered SVD Basis Helper

**Files:**
- Modify: `lrom_bench/reduced_basis.py`
- Modify: `tests/test_reduced_basis.py`

- [ ] **Step 1: Add failing centered-SVD basis test**

Append to `tests/test_reduced_basis.py`:

```python
from lrom_bench.reduced_basis import build_centered_svd_basis


def test_build_centered_svd_basis_uses_phi0_centered_snapshots() -> None:
    mesh = np.array([0.0, 1.0, 2.0])
    phi0 = np.array([1.0, 1.0, 1.0])
    snapshots = np.array(
        [
            [2.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    basis = build_centered_svd_basis(phi0=phi0, snapshots=snapshots, mesh=mesh, n_basis=2)

    assert basis.n_mesh == 3
    assert basis.n_basis == 2
    assert np.allclose(basis.phi0, phi0)
    assert np.allclose(basis.vectors.conj().T @ basis.vectors, np.eye(2))
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python -m pytest tests/test_reduced_basis.py -v
```

Expected: FAIL because `build_centered_svd_basis` does not exist.

- [ ] **Step 3: Implement centered-SVD basis helper**

Add to `lrom_bench/reduced_basis.py` before `project_ls_coordinates`:

```python
def build_centered_svd_basis(
    phi0: np.ndarray,
    snapshots: np.ndarray,
    mesh: np.ndarray,
    n_basis: int,
) -> CentralBasisData:
    phi0 = np.asarray(phi0, dtype=np.complex128)
    snapshots = np.asarray(snapshots, dtype=np.complex128)
    mesh = np.asarray(mesh, dtype=float)
    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape (n_samples, n_mesh)")
    if phi0.ndim != 1:
        raise ValueError("phi0 must be one-dimensional")
    if snapshots.shape[1] != phi0.size:
        raise ValueError("snapshots and phi0 must share the mesh dimension")
    if n_basis < 1:
        raise ValueError("n_basis must be positive")
    if n_basis > min(snapshots.shape):
        raise ValueError("n_basis cannot exceed the snapshot matrix rank limit")

    centered = snapshots - phi0[np.newaxis, :]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    vectors = vh[:n_basis].T
    return CentralBasisData(phi0=phi0, vectors=vectors, mesh=mesh)
```

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
python -m pytest tests/test_reduced_basis.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add lrom_bench/reduced_basis.py tests/test_reduced_basis.py
git commit -m "Add centered SVD basis helper"
```

---

### Task 3: Real Woods-Saxon ROSE Boundary Helpers

**Files:**
- Modify: `lrom_bench/rose_fom.py`
- Create: `tests/test_rose_fom_real_ws.py`

- [ ] **Step 1: Add failing tests for real Woods-Saxon helpers**

Create `tests/test_rose_fom_real_ws.py`:

```python
from __future__ import annotations

import numpy as np

from lrom_bench.rose_fom import (
    RealWSParameters,
    central_real_ws_parameters,
    make_alphas,
    real_woods_saxon_potential,
)


def test_real_woods_saxon_potential_is_attractive_and_depth_scaled() -> None:
    r = np.array([0.5, 1.0, 2.0])
    alpha = np.array([50.0, 1.2, 0.7])
    deeper = np.array([60.0, 1.2, 0.7])

    values = real_woods_saxon_potential(r, alpha)
    deeper_values = real_woods_saxon_potential(r, deeper)

    assert values.shape == r.shape
    assert np.all(values < 0.0)
    assert np.all(np.abs(deeper_values) > np.abs(values))


def test_central_real_ws_parameters_exposes_three_parameter_alpha() -> None:
    params = central_real_ws_parameters()

    assert isinstance(params, RealWSParameters)
    assert params.alpha.shape == (3,)
    assert np.all(np.isfinite(params.alpha))


def test_make_alphas_changes_only_requested_column() -> None:
    alpha0 = np.array([50.0, 1.2, 0.7])
    values = np.array([48.0, 50.0, 52.0])

    alphas = make_alphas(alpha0, param_index=0, values=values)

    assert np.allclose(alphas[:, 0], values)
    assert np.allclose(alphas[:, 1:], alpha0[1:])
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python -m pytest tests/test_rose_fom_import.py tests/test_rose_fom_real_ws.py -v
```

Expected: FAIL because the real Woods-Saxon helpers do not exist.

- [ ] **Step 3: Implement real Woods-Saxon parameters and potential**

Add to `lrom_bench/rose_fom.py` after `KDParameters`:

```python
@dataclass(frozen=True)
class RealWSParameters:
    mu: float
    e_com: float
    k: float
    eta: float
    r_c: float
    alpha: np.ndarray


def central_real_ws_parameters(
    target: tuple[int, int] = TARGET,
    projectile: tuple[int, int] = PROJECTILE,
    e_lab: float = E_LAB,
) -> RealWSParameters:
    params = central_kd_parameters(target=target, projectile=projectile, e_lab=e_lab)
    return RealWSParameters(
        mu=params.mu,
        e_com=params.e_com,
        k=params.k,
        eta=params.eta,
        r_c=params.r_c,
        alpha=params.alpha[:3].copy(),
    )


def real_woods_saxon_potential(r: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rose = import_rose()
    r = np.asarray(r, dtype=float)
    vv, rv, av = np.asarray(alpha, dtype=float)
    return -vv * rose.koning_delaroche.woods_saxon_safe(r, rv, av)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
python -m pytest tests/test_rose_fom_import.py tests/test_rose_fom_real_ws.py -v
```

Expected: PASS.

- [ ] **Step 5: Add a focused optional ROSE smoke for central real-WS setup**

Run:

```bash
python - <<'PY'
from lrom_bench.rose_fom import central_real_ws_parameters
params = central_real_ws_parameters()
print(params.alpha.shape)
PY
```

Expected: prints `(3,)`.

- [ ] **Step 6: Commit**

Run:

```bash
git add lrom_bench/rose_fom.py tests/test_rose_fom_real_ws.py
git commit -m "Add real Woods-Saxon ROSE helpers"
```

---

### Task 4: Architecture Note For Notebook-Driven Package Helpers

**Files:**
- Modify: `docs/architecture/lrom-benchmark-spine.md`

- [ ] **Step 1: Update architecture markdown**

Add a short section after the package-spine diagram:

```markdown
## Notebook-Driven Helper Rule

Notebook 1 adds only small reusable helpers needed by the notebook:

- `Notebook01Config` records the single-wavefunction benchmark settings.
- `sampling.centered_1d_values` creates the visible `Vv` scan.
- `reduced_basis.build_centered_svd_basis` creates the central-reference basis used by the notebook.
- `rose_fom.central_real_ws_parameters` and `rose_fom.real_woods_saxon_potential` expose the real Woods-Saxon teaching setup.

Plotting remains in notebook cells. The package should not grow plotting functions or one-call notebook workflow functions.
```

- [ ] **Step 2: Verify architecture doc still has Mermaid figures**

Run:

```bash
python - <<'PY'
from pathlib import Path
path = Path("docs/architecture/lrom-benchmark-spine.md")
text = path.read_text()
assert text.count("```mermaid") == 5
assert "Notebook-Driven Helper Rule" in text
print(path)
PY
```

Expected: prints `docs/architecture/lrom-benchmark-spine.md`.

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/architecture/lrom-benchmark-spine.md
git commit -m "Document notebook-driven helper rule"
```

---

### Task 5: Notebook 1 Generator With Visible Plotting Cells

**Files:**
- Create: `scripts/generate_notebook01.py`
- Create: `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`

- [ ] **Step 1: Create generator script**

Create `scripts/generate_notebook01.py`:

```python
from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_rbm_vs_lrom_single_wavefunction.ipynb"


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


def notebook_cells() -> list:
    setup = r"""
    from pathlib import Path
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ROOT = Path.cwd()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from lrom_bench.config import Notebook01Config
    from lrom_bench import metrics, prediction, predictors, reduced_basis, rf_lrom, rose_fom, sampling

    cfg = Notebook01Config()
    params = rose_fom.central_real_ws_parameters()
    alpha_c = params.alpha
    rho_mesh = np.linspace(1e-8, 8 * np.pi, cfg.n_mesh)
    r_mesh = rho_mesh / params.k

    print("config hash:", cfg.config_hash())
    print("central [Vv, Rv, av]:", alpha_c)
    """
    vv_samples = r"""
    vv_train = sampling.centered_1d_values(alpha_c[0], cfg.vv_width, cfg.n_vv_train)
    vv_test = sampling.centered_1d_values(alpha_c[0], cfg.vv_width, cfg.n_vv_test)

    train_alphas = rose_fom.make_alphas(alpha_c, param_index=0, values=vv_train)
    test_alphas = rose_fom.make_alphas(alpha_c, param_index=0, values=vv_test)

    potentials_train = np.array(
        [rose_fom.real_woods_saxon_potential(r_mesh, alpha) for alpha in train_alphas]
    )
    """
    vv_plot = r"""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    for value, potential in zip(vv_train, potentials_train):
        ax.plot(r_mesh, potential, alpha=0.55, label=f"{value:.1f}" if value in vv_train[[0, -1]] else None)
    ax.set_xlabel("r [fm]")
    ax.set_ylabel("V(r)")
    ax.set_title("Real Woods-Saxon potential rainbow: Vv-only scan")
    ax.grid(alpha=0.25)
    ax.legend(title="Vv edge values")
    fig.tight_layout()
    """
    basis_placeholder = r"""
    # The first executable implementation slice wires ROSE wavefunctions here.
    # Once phi_train is available:
    #
    # phi0 = problem.solve_phi(alpha_c)
    # basis = reduced_basis.build_centered_svd_basis(phi0, phi_train, rho_mesh, cfg.n_basis)
    # coeff_ls_train = reduced_basis.project_ls_coordinates(basis, phi_train)
    # phi_ls_train = prediction.reconstruct_from_basis(basis.phi0, basis.vectors, coeff_ls_train)
    # ls_error = metrics.relative_l2_rows(phi_ls_train, phi_train)
    """
    return [
        md(
            """
            # 01. RBM vs LROM for a Single Scattering Wavefunction

            This notebook compares a traditional reduced-basis view and the RF-LROM view for one `l = 0`
            real Woods-Saxon scattering wavefunction. The first act varies only `Vv`; the second act expands
            to `Vv`, `Rv`, and `av` to motivate operator-informed predictors.
            """
        ),
        md("## 1. Scientific Setup"),
        code(setup),
        md("## 2. Vv-Only Samples"),
        code(vv_samples),
        code(vv_plot),
        md("## 3. Reduced Basis And LS Floor"),
        code(basis_placeholder),
        md("## 4. RBM/ROSE vs LROM Coordinates"),
        code("# Fit RF-LROM after ROSE/RBM coefficients and LS coordinates are available."),
        md("## 5. Wavefunction Reproduction"),
        code("# Compare LS floor, ROSE/RBM, and LROM wavefunction errors here."),
        md("## 6. Three-Parameter Samples"),
        code("# Build centered-box samples for [Vv, Rv, av] and add Vv/Rv/av rainbow summaries."),
        md("## 7. Why Raw Parameters Are Not Enough"),
        code("# Fit and inspect a raw-parameter LROM diagnostic without hiding the flow."),
        md("## 8. Operator-Informed Potential Predictors"),
        code("# Build maxvol-selected potential predictors and visualize selected points inline."),
        md("## 9. Three-Parameter Performance"),
        code("# Compare LS floor, ROSE/RBM, raw-parameter LROM, and potential-predictor LROM."),
        md("## 10. Notebook 1 Takeaways"),
        code("pd.DataFrame({'next': ['Notebook 2 moves to cross-section-level comparisons.']})"),
    ]


def main() -> None:
    write_notebook(NOTEBOOK_PATH, notebook_cells())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate Notebook 1**

Run:

```bash
python scripts/generate_notebook01.py
```

Expected: creates `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`.

- [ ] **Step 3: Verify section headings and visible plotting**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb")
nb = json.loads(path.read_text())
headings = [
    line.strip()
    for cell in nb["cells"]
    if cell["cell_type"] == "markdown"
    for line in "".join(cell["source"]).splitlines()
    if line.startswith("#")
]
source = "\n".join("".join(cell["source"]) for cell in nb["cells"])
expected = [
    "# 01. RBM vs LROM for a Single Scattering Wavefunction",
    "## 1. Scientific Setup",
    "## 2. Vv-Only Samples",
    "## 3. Reduced Basis And LS Floor",
    "## 4. RBM/ROSE vs LROM Coordinates",
    "## 5. Wavefunction Reproduction",
    "## 6. Three-Parameter Samples",
    "## 7. Why Raw Parameters Are Not Enough",
    "## 8. Operator-Informed Potential Predictors",
    "## 9. Three-Parameter Performance",
    "## 10. Notebook 1 Takeaways",
]
assert headings == expected
assert "fig, ax = plt.subplots" in source
assert "plot_notebook" not in source
assert "run_notebook01" not in source
print(path)
PY
```

Expected: prints `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`.

- [ ] **Step 4: Commit**

Run:

```bash
git add scripts/generate_notebook01.py notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb
git commit -m "Add Notebook 1 RBM LROM generator"
```

---

### Task 6: Verification Pass

**Files:**
- No new files.

- [ ] **Step 1: Run all unit tests**

Run:

```bash
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Regenerate Notebook 1**

Run:

```bash
python scripts/generate_notebook01.py
```

Expected: completes with no traceback.

- [ ] **Step 3: Run the real Woods-Saxon smoke**

Run:

```bash
python - <<'PY'
import numpy as np
from lrom_bench.config import Notebook01Config
from lrom_bench import rose_fom

cfg = Notebook01Config(n_mesh=32)
params = rose_fom.central_real_ws_parameters()
rho_mesh = np.linspace(1e-8, 8 * np.pi, cfg.n_mesh)
r_mesh = rho_mesh / params.k
potential = rose_fom.real_woods_saxon_potential(r_mesh, params.alpha)
print(potential.shape, float(np.min(potential)))
PY
```

Expected: prints `(32,)` and a negative minimum potential value.

- [ ] **Step 4: Check worktree**

Run:

```bash
git status --short
git diff --stat
```

Expected: intentional Notebook 1 files plus existing unrelated workspace migration changes. Do not revert unrelated changes.
