# LROM

LROM is an object-oriented Python package for learning reduced-operator models
of nuclear scattering wavefunctions. A single `LROM` object owns the physical
configuration, named sampling design, full-order solutions, reduced basis,
predictors, fitted model, diagnostics, and latest predictions.

The public API is keyword-only. `l=3` means the exact `l=3` channel; use a tuple
such as `l=(0, 1, 3)` to request several explicit channels.

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
    training_ranges={
        "Vv": (0.90 * central["Vv"], 1.10 * central["Vv"]),
        "Rv": (0.90 * central["Rv"], 1.10 * central["Rv"]),
        "av": (0.90 * central["av"], 1.10 * central["av"]),
    },
    testing_ranges={
        "Vv": (0.78 * central["Vv"], 1.22 * central["Vv"]),
        "Rv": (0.80 * central["Rv"], 1.20 * central["Rv"]),
        "av": (0.80 * central["av"], 1.20 * central["av"]),
    },
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

The `.lrom` artifact contains only versioned JSON metadata and NumPy arrays. It
does not contain pickle data or live ROSE solver objects, so a trained model can
be moved to another notebook or compatible device for prediction.

## Repository layout

- `lrom/`: new stateful public package
- `notebooks/`: research workflows; Notebook 1 uses the new API
- `tests/`: unit, integration, artifact, and notebook-structure tests
- `docs/`: approved designs, plans, and deferred work
- `lrom_bench/`: temporary compatibility implementation for later notebooks
- `scientific_archive/`: retained legacy and scientifically valuable references

Supported Python versions are 3.11, 3.12, and 3.13.
