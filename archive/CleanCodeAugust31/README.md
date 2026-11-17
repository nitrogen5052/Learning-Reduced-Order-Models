# Scattering LROM

A compact, collaborator-facing implementation of learned reduced-order models
(LROMs) for neutron elastic scattering. The repository keeps the numerical
ideas that survived the exploratory study and leaves the large diagnostic
archives, cached training databases, and one-off ablations behind.

The scientific path is intentionally short:

1. solve radial scattering equations with an independent Numerov full-order
   model (FOM) using the regular-origin shooting convention;
2. construct centered wavefunction bases from FOM snapshots;
3. learn implicit reduced-coordinate equations from selected potential values;
4. assemble partial-wave S matrices and elastic differential cross sections;
5. optionally use a weak, scale-independent ridge regularization and a GPU for
   large batches of the small online linear systems.

## Repository map

```text
.
├── lrom/                    reusable package
│   ├── physics.py           optical potentials and kinematics
│   ├── fom.py               regular-origin Numerov solver and S matrices
│   ├── reduced.py           bases, predictors, OLS/ridge implicit fits
│   ├── emulator.py          training and online LROM interface
│   ├── observables.py       elastic cross sections
│   ├── backends.py          CPU / optional JAX-GPU batched solves
│   ├── global_kd.py         explicit 37-coefficient KD map
│   └── curated_data.py      curated JSON reader with provenance
├── notebooks/
│   ├── 01_single_wavefunction.ipynb
│   ├── 02_elastic_cross_sections.ipynb
│   └── 03_real_data_pilot.ipynb
├── data/README.md           how to locate the external curated corpus
├── tests/                   unit and end-to-end smoke tests
└── artifacts/              intentionally empty, ignored output directory
```

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the package and notebook dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name scattering-lrom --display-name "Scattering LROM"
```

Fetch the external curated data required by notebook 03:

```bash
python tools/fetch_data.py
```

The command shallow-fetches the validated commit from
[beykyle/nucleon-nucleus-data](https://github.com/beykyle/nucleon-nucleus-data)
into the ignored `external/nucleon-nucleus-data/` directory and prints the
upstream commit. A different revision must be requested explicitly. Notebooks
01 and 02 do not require this download.

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The notebooks default to a quick CPU configuration. Set `LROM_FULL_RUN=1`
before launching Jupyter to use the documented research-size settings.

## CPU and GPU modes

CPU is the default and requires no accelerator software:

```bash
set LROM_BACKEND=cpu
jupyter lab
```

On PowerShell use `$env:LROM_BACKEND = "cpu"`; on bash use
`export LROM_BACKEND=cpu`.

GPU acceleration is optional. Install a GPU-enabled JAX build appropriate for
the local CUDA platform, verify that `jax.devices("gpu")` returns a device, and
set:

```bash
export LROM_BACKEND=gpu
export LROM_PRECISION=double   # or single for an explicit accuracy experiment
```

`LROM_BACKEND=auto` chooses a visible JAX GPU and otherwise falls back to
NumPy/CPU. Explicit `gpu` mode fails loudly if no GPU is available; it never
silently reports a GPU benchmark while running on the CPU.

Only the batched online reduced solves are sent to JAX. FOM generation and
training remain on the CPU. This preserves one fitted model and one observable
implementation across devices. Small jobs may be faster on the CPU because GPU
transfer and compilation overhead dominate; the GPU option is intended for the
large likelihood batches used in calibration.

## Notebook sequence

### 01 — Single wavefunction

Varies the real Woods--Saxon depth for one s wave. It separates basis
projection error from learned-coordinate error and shows interpolation plus
mild extrapolation. This is the smallest place to inspect the equation

```text
(I + sum_j p_j M_j) a = sum_j p_j b_j.
```

### 02 — Elastic cross sections

Varies all ten local optical-potential parameters for a fixed neutron--40Ca
problem, fits every partial-wave channel, and compares OLS with the weak ridge
currently retained as the practical default. The training snapshots, reduced
bases, predictor layout, and observable are shared so the comparison isolates
the regression choice.

### 03 — Curated real-data pilot

Loads the separately curated neutron-elastic JSON corpus, audits the retained
records, and performs a deliberately narrow 40Ca exercise. It profiles the
correlated normalization for each angular distribution, adjusts two global KD
coefficients with the FOM, and trains a local LROM around that point. It is a
workflow demonstration, not the final 37-dimensional global calibration.

Run `python tools/fetch_data.py` first. An advanced user may instead set
`SCATTERING_DATA_ROOT` to another checkout containing the upstream `data/`
folder. See [data/README.md](data/README.md).

## Core API

```python
from lrom import BasisConfig, MaxVol, ScatteringLROM, ScatteringProblem

problem = ScatteringProblem(
    potential="kd_neutron", target_a=40, target_z=20,
    lab_energy=14.1, l_max=10,
)
training = problem.generate_training_data(samples)
emulator = ScatteringLROM(
    problem,
    BasisConfig(size=6),
    MaxVol(count=12, ridge=1e-14),
    packing_strategy="padded",
).train(training)

cross_section = emulator.cross_section(sample, angles)
```

For a batch:

```python
from lrom import get_runtime

runtime = get_runtime("auto")
cross_sections = emulator.cross_sections(
    samples, angles, linear_solver=runtime.solve
)
```

The ridge parameter is dimensionless. Internally
`alpha = ridge * sigma_max(D)**2`, where `D` is the offline design matrix.
Therefore `ridge=1e-14` is a weak perturbation on the scale of that regression,
not an absolute penalty with units inherited from the inputs. Use `ridge=0.0`
for ordinary least squares.

## Reproducibility and scope

- Random designs use fixed seeds in the notebooks.
- Experimental records keep corpus, EXFOR accession, record identifier,
  normalization uncertainty, and source-file provenance.
- The curated data are not copied into this repository.
- `artifacts/` is ignored so trained models and plots do not inflate Git history.
- The notebooks expose quick and full settings rather than hiding reduced test
  sizes inside helper functions.
- The FOM and LROM use the same regular-origin normalization. No per-snapshot
  normalization is applied.

Before a production calibration, validate the LROM on held-out isotopes,
energies, optical-parameter perturbations, matrix-condition scans, and the
small-cross-section tails relevant to the likelihood. The third notebook is a
safe starting point for that expansion, not a substitute for it.

## Regenerating notebooks

The checked-in notebooks are generated from one readable script:

```bash
python tools/build_notebooks.py
```

This makes reviewable changes to prose and code possible without editing raw
notebook JSON.
