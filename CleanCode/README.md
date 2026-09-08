# Scattering LROM collaborator release

This repository is a self-contained, executable presentation of the learned
reduced-order model (LROM) workflow for neutron elastic scattering.  It keeps
the detailed physics narrative of the original research notebooks while using
the current implementation choices: origin-shooting wavefunctions, ridge-
regularized learned equations, hard-routed energy windows, and an optional
compiled JAX/GPU online solver.

The repository includes the curated neutron-elastic measurements used by
Notebook 3 and the pretrained three-window global LROM.  None of the notebooks
requires Pablo's original `ScatteringData`, `Scratch`, or `CodeRestructure`
directories.

## The three notebooks

1. **`notebooks/01_single_wavefunction.ipynb`** is the detailed microscope.
   It constructs the full-order radial problem, builds training and test
   snapshots, studies the snapshot SVD and max-volume basis, learns implicit
   coordinate equations, reconstructs held-out wavefunctions, and compares
   OLS with normalized ridge regularization.  It also examines the offline
   design spectrum, online matrix conditioning, coefficient norms, and compiled
   batched CPU/GPU evaluation.

2. **`notebooks/02_elastic_cross_sections.ipynb`** assembles all partial waves
   into elastic cross sections.  It retains the original five-configuration
   convergence study and ROSE comparison, then adds controlled OLS-versus-
   ridge comparisons using identical bases and predictors, held-out condition
   diagnostics, error violins, representative cross sections, and throughput
   measurements.

3. **`notebooks/03_curated_data_and_global_lrom.ipynb`** audits the vendored
   KDUQ and held-out neutron-elastic datasets.  For every retained experimental
   record it independently evaluates the current KD full-order model (solid
   black), the pretrained hard-routed global LROM (dashed orange), and the
   measurements.  It reports errors by corpus and energy window, compares
   profile-normalized chi-square values, inspects the worst cases, writes a
   complete paginated plot atlas, and measures deployment throughput and
   conditioning.

All notebooks in the repository have already been executed successfully and
include their outputs.  The source cells derive the repository root from the
notebook location and import only from this repository.

## Installation

Python 3.11 or newer is required.  From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Linux or macOS, activate with `source .venv/bin/activate` instead.

The normal installation is CPU-only.  It is sufficient for every notebook and
uses NumPy for the reduced solves.

### Optional GPU backend

The fitted equations are identical on CPU and GPU; only the online batched
linear algebra backend changes.  Install a GPU-enabled JAX build that matches
the local CUDA driver by following the official JAX installation selector:
<https://docs.jax.dev/en/latest/installation.html>.  Then use:

```python
from lrom import get_runtime

runtime = get_runtime("gpu", precision="double")
# or fall back automatically when no GPU is visible
runtime = get_runtime("auto", precision="double")
```

The first GPU call includes JIT compilation and should not be used as the
steady-state timing.  GPU acceleration is intended for large batches of
likelihood/cross-section evaluations; a single small reduced solve can be
faster on CPU because of dispatch and transfer overhead.  `precision="single"`
is available for an explicit degraded-precision experiment, but double
precision remains the scientific default.

## Reproducing the notebooks

Run all three notebooks in place:

```bash
python tools/run_notebooks.py
```

Or open `notebooks/` in Jupyter and run them individually.  Notebook 2 uses the
vendored cached ROSE reference arrays in `benchmarks/cache/`; rebuilding those
reference calculations is optional.  Notebook 3 uses the supplied models in
`models/three_window/` and writes the complete data atlas to
`artifacts/real_data_kd_overlays/`.

Run the test suite with:

```bash
pytest -q
```

For a final packaging audit (including hidden-path checks and notebook stored-
output checks), run:

```bash
python tools/release_check.py
```

## Repository layout

```text
benchmarks/                 ROSE adapter and cached reference arrays
data/curated/               vendored KDUQ and held-out neutron-elastic JSON
lrom/                       reusable FOM, training, deployment, and backend code
models/three_window/        pretrained ridge-14 hard-routed global LROM
notebooks/                  the three complete executable studies
tests/                      numerical and release-integrity tests
tools/                      notebook runner and release-building utilities
artifacts/                  regenerated plots and tables (not source data)
```

## Current deployment choice

The included global emulator uses three hard energy windows,
`[5, 70)`, `[70, 135)`, and `[135, 200]` MeV.  Each window owns its reduced
bases and learned equations, with basis size 14, 20 predictors, and normalized
ridge strength `1e-14`.  Training support extends beyond the hard routing
boundaries where possible; see `models/three_window/training_manifest.json`.

Hard routing is deliberate: energy is treated as an experimental/model-input
coordinate rather than a calibrated optical-potential coefficient, so routine
evaluation does not pay for two LROMs and blending.  Boundary behavior remains
a validation responsibility and is explicitly inspected in Notebook 3.

## Data and model provenance

The curated measurements are a compact vendored subset of
<https://github.com/beykyle/nucleon-nucleus-data>, pinned to commit
`adc8558fe9fcf629af41bc909514f8da63d4f590`.  Details and the upstream-license
warning are in `data/curated/PROVENANCE.md`.

The `.pkl` files are trusted repository artifacts produced by this project.
Python pickle is not a safe interchange format for untrusted files: do not
replace them with downloads from an unknown source and load them.

## Scientific scope

This is research software.  Notebook 3 separates two questions that should not
be conflated:

- **emulator fidelity:** whether the LROM reproduces the KD full-order solver;
- **model adequacy:** whether KD reproduces experimental measurements.

A low LROM/FOM discrepancy does not imply a statistically adequate optical
model, and an experimental mismatch is not automatically an emulator failure.
The reported chi-square is therefore accompanied by direct FOM-versus-LROM
metrics and record-level plots.
