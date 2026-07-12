# Deferred Design: External FOM Sampling

Status: Backburner until the core in-process `lrom.LROM` sampling and training workflow is designed.

## Motivation

Some full-order models are too expensive, proprietary, or environment-specific to run inside the `LROM` object. The package should eventually let users generate a reproducible sampling request locally, solve it on another computer or cluster, and attach the returned solutions before calling `train()`.

## Proposed Workflow

```python
emulator = lrom.LROM(
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
    l=1,
    fom="external",
    potential="ws_3",
)

emulator.sampling(
    ranges={
        "Vv": (40.0, 60.0),
        "Rv": (3.8, 4.6),
        "av": (0.50, 0.80),
    },
    training_size=70,
    testing_size=81,
    mesh_size=900,
)

print(emulator.sampling_request)
emulator.export_sampling_request(path="ca40_sampling_request.npz")

# Solve externally, then return the fulfilled response.
emulator.attach_solutions(
    solutions=lrom.load_solutions(path="ca40_fom_solutions.npz"),
)

emulator.train()
```

## Sampling Request Contract

The inspectable request should report:

- A stable request ID and status.
- FOM and potential identifiers.
- Named parameter schema and central values.
- Unique case IDs with training/testing designations.
- Named parameter values for every case.
- Requested partial waves `0..l`.
- Grid name, size, domain, and expected solution type.
- Expected solution shape per partial wave.

`print(emulator.sampling_request)` should show a compact summary, while `to_dataframe()` should expose the individual cases.

## Returned Solution Contract

- Results match cases by stable case ID, never row position alone.
- Partial-wave results use a mapping keyed by `l`, avoiding a hidden channel axis.
- Each result includes its grid and complex wavefunction array.
- Import validation detects missing, duplicated, unexpected, or mismatched cases and reports exact shape errors.
- The first version may use one shared one-dimensional grid; arbitrary multidimensional FOM output remains an open future decision.

## Architectural Boundary

`predict()` remains an LROM approximation of the selected FOM solution. Exact FOM execution is used only for sampling and validation. Portable request/response files are transport helpers; the `LROM` object itself must not depend on a directory layout.

## Revisit Trigger

Revisit this design after the in-process `nucl-scatter-eq` sampling contract, training state, and prediction interface are stable enough to define a common FOM-provider boundary.
