# LROM

LROM is a Python package for learning reduced-operator models of nuclear
scattering wavefunctions. The public `lrom` entry point exposes the **one-file
v1.2 implementation** in `lrom_legacy/v1_2/__init__.py`. One object,
`lrom.LROM`, coordinates the physical
configuration, sampling design, full-order solutions, reduced basis,
predictors, fitted model, diagnostics, and predictions through three calls —
`.sampling()`, `.train()`, `.predict()`.

## Canonical workflow

```python
import lrom

emulator = lrom.LROM(
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
    l=0,
    potential="ws_3",
)

central = emulator.central_parameters
emulator.sampling(
    training_ranges={name: (0.90 * central[name], 1.10 * central[name])
                     for name in ("Vv", "Rv", "av")},
    testing_ranges={name: (0.80 * central[name], 1.20 * central[name])
                    for name in ("Vv", "Rv", "av")},
    training_size=70,
    testing_size=81,
    strategy="latin_hypercube",
)
emulator.train(basis_size=4, predictor="potential", predictor_count=6)

emulator.save(path="calcium40.lrom")
portable = lrom.load(path="calcium40.lrom")
portable.predict(parameters={"Vv": central["Vv"]})
wavefunction = portable.predictions.wavefunctions[0][0]
```

The `.lrom` artifact contains only versioned JSON metadata and NumPy arrays —
no pickle data or live solver objects — so a trained model can be moved to
another notebook or machine for prediction.

## Repository layout

- `lrom/`: public entry point for the current v1.2 package
- `lrom_legacy/v1_2/`: authoritative one-file wavefunction implementation
- `lrom_legacy/v2_0/`: parked future cross-section shell
- `notebooks/`: `01_rbm_vs_lrom_single_wavefunction.ipynb` (paper Notebook 1)
  and `benchmark_02.ipynb` (validates the package against the archived legacy
  notebook 02 — benchmarks are named after the legacy notebook they recreate)
- `tests/`: unit, integration, artifact, and notebook-contract tests
- `docs/`: user-facing documentation (`VERSIONING.md`)
- `.agents/`: AI-agent working area (plans, specs, backlog)
- `scientific_archive/`: read-only legacy code and references

Supported Python versions are 3.11, 3.12, and 3.13.
