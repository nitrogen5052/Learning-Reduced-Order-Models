# Object-Oriented LROM and Notebook 1 Design

Date: 2026-06-23

## Goal

Replace the benchmark-oriented `lrom_bench` public workflow with a stateful Python package named `lrom`. Users work through one keyword-configured `lrom.LROM` object that owns physical configuration, FOM samples, ROSE and LROM training results, predictions, diagnostics, and portable inference state.

The first implementation target is Notebook 1 from `Paper Results Map.pdf`: compare one `l=0` radial scattering wavefunction across high-fidelity ROSE solves, ROSE EIM/Galerkin reduced-basis emulation, and predictor RF-LROM. Cross sections and global isotope/energy emulation are deferred.

## Public Object

```python
import lrom

emulator = lrom.LROM(
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
    l=0,
    fom="nucl-scatter-eq",
    potential="ws_3",
)
```

The constructor is keyword-only:

```python
class LROM:
    def __init__(
        self,
        *,
        target: tuple[int, int],
        projectile: tuple[int, int],
        lab_energy: float,
        l: int | tuple[int, ...] = 0,
        fom: str = "nucl-scatter-eq",
        potential: str | PotentialFunction = "ws_3",
        central_parameters: Mapping[str, float] | None = None,
    ) -> None:
        ...
```

`l=3` selects only channel 3. `l=(0, 1, 3)` selects exactly those channels; no implicit range expansion occurs. Targets and projectiles use `(A, Z)` tuples only.

Construction validates and stores immutable physical configuration, resolves central kinematics and KD values, normalizes the selected potential schema, and initializes empty lifecycle state. It does not create meshes, solve wavefunctions, or train models.

Built-in potential schemas are:

- `"ws_1"`: `Vv` is sampleable; KD-derived `Rv` and `av` remain fixed but may be overridden.
- `"ws_3"`: `Vv`, `Rv`, and `av` are sampleable.
- `"woods-saxon"`: the full KD schema `Vv`, `Rv`, `av`, `Wv`, `Rwv`, `awv`, `Wd`, `Rd`, `ad`, `Vso`, `Rso`, `aso`, `Wso`, `Rwso`, `awso`.
- Custom `potential(r, alpha)` callables require a complete named `central_parameters` mapping. The mapping defines the public parameter schema; only the low-level numerical callable uses the ordered `alpha` vector.

Unknown parameter names are rejected. Built-in central overrides are merged by name onto KD defaults.

## Architecture and State Ownership

`LROM` is the sole normal public workflow. It delegates calculations to focused private helpers for parameter sampling, `nucl-scatter-eq`, ROSE RBM, basis construction, predictor construction, RF-LROM fitting, inference, diagnostics, and artifact I/O. Helpers return results and remain stateless where practical; authoritative scientific state is always stored on `emulator`.

State is grouped conceptually while remaining accessible through convenient properties:

```text
emulator
├── config and description
├── kinematics and central_parameters
├── samples, mesh, and full_order_model
├── basis, predictors, and rf_lrom
├── rose_rbm and testing_results
├── testing_errors and predictions
└── provenance
```

Channel-specific objects use mappings keyed by the physical integer `l`, never a hidden channel position. Named parameters and stable case IDs prevent public sample matching from depending on row position. Direct properties point into grouped state and do not duplicate large arrays.

The lifecycle is `initialized -> sampled -> trained -> predicted`. Resampling clears basis, training, diagnostics, and prediction state. Rebuilding a basis clears dependent models and predictions. Retraining clears predictions. Invalid ordering raises `LROMStateError`.

## Sampling Contract

```python
emulator.sampling(
    training_ranges={...},
    testing_ranges={...},
    training_size=70,
    testing_size=81,
    mesh_size=900,
    radial_domain=None,
    strategy="latin_hypercube",
    seed=1204,
    eim_basis_size=8,
    solver_options=None,
)
```

All arguments are keyword-only. Ranges are mappings from potential parameter names to absolute `(minimum, maximum)` bounds. Parameters omitted from a range remain at their central value. Separate training and testing domains make interpolation and extrapolation explicit.

Supported initial strategies are:

- `"linspace"` for exactly one varied parameter, using `training_size` and `testing_size` evenly spaced points.
- `"latin_hypercube"` for one or more varied parameters, using deterministic independent training and testing designs derived from `seed`.

The central point is solved separately and does not count toward either requested sample size. Sampling constructs the built-in `nucl-scatter-eq` FOM and ROSE EIM pieces for every selected channel, solves central/training/testing wavefunctions, evaluates potentials, stores physical-radius and internal solver meshes, and records case metadata.

For each `l`, central wavefunctions have shape `(mesh_size,)`; training and testing wavefunctions have shapes `(training_size, mesh_size)` and `(testing_size, mesh_size)`. The notebook presents physical radius `r` in fm and does not use the dimensionless `s` coordinate in figures.

## Training and Prediction

The public workflow mutates the emulator and returns no scientific result object:

```python
emulator.reduced_basis(basis_size=4)

emulator.train(
    basis_size=4,
    predictor="potential",
    predictor_count=6,
)

emulator.predict(
    parameters={"Vv": 52.0, "Rv": 4.1, "av": 0.65},
)
```

`train()` automatically:

1. Builds one central-reference basis per channel from training FOM solutions.
2. Stores `phi0`, basis vectors, singular values, and LS target coordinates.
3. Gives the exact same `phi0` and basis vectors to the ROSE Galerkin emulator and LROM so their coefficients share one coordinate system.
4. Constructs the requested predictor representation.
5. Fits one RF-LROM model per channel.
6. Evaluates the stored testing set with the LS floor, ROSE RBM, and LROM.
7. Stores coefficients, reconstructed wavefunctions, and pointwise and aggregate errors.

Predictor names are:

- `"potential"` (default): normalized potential differences at SVD/maxvol-selected physical radii. `predictor_count` controls the number of points.
- `"parameters"`: normalized named-parameter deviations from the central point.

At the central point both predictor forms are zero, so the reduced coordinates are zero and the reconstructed solution is `phi0`. Prediction accepts one named mapping or a sequence of named mappings; omitted parameters use central values and unknown names are rejected. Predictions are LROM approximations of the selected FOM, not fresh high-fidelity solves.

## Notebook 1 Specification

Notebook 1 contains two independent objects so resampling one experiment cannot destroy the other:

### Experiment A: `Vv` Only

```python
vv_emulator = lrom.LROM(
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
    l=0,
    potential="ws_1",
)
```

- Training range: central `Vv` +/-10%, 35 evenly spaced cases.
- Testing range: central `Vv` +/-35%, 41 evenly spaced cases.
- `strategy="linspace"`, `eim_basis_size=8`, and `basis_size=4`.
- `predictor="parameters"` to match the direct one-parameter LROM equation.
- Show potential and `Re(phi(r))` wavefunction rainbows, the shared reduced basis, ROSE/LS/LROM coefficients, and reconstruction errors across `Vv`.

### Experiment B: `Vv`, `Rv`, and `av`

```python
ws3_emulator = lrom.LROM(
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
    l=0,
    potential="ws_3",
)
```

- Training domain: central `Vv`, `Rv`, and `av` +/-10%, 70 Latin-hypercube cases.
- Testing domain: `Vv` +/-22%, `Rv` +/-20%, and `av` +/-20%, 81 Latin-hypercube cases.
- `eim_basis_size=8`, `basis_size=4`, `predictor="potential"`, and `predictor_count=6`.
- Overlay the six maxvol-selected physical radii on the training-potential rainbow plot.
- Compare high fidelity, ROSE, and LROM on a representative testing wavefunction using `Re(phi(r))`.
- Plot all testing-case pointwise errors on a logarithmic vertical scale with transparency:
  - `abs(phi_high_fidelity - phi_rose)`
  - `abs(phi_high_fidelity - phi_lrom)`
  - `abs(phi_high_fidelity - phi_ls)` when reusing existing LS state
- Present interpolation and extrapolation summaries using the narrower training and wider testing domains.
- Do not calculate cross sections and do not show the older linear-LROM baseline.

The package exposes arrays and concise testing-case accessors; plotting remains explicit in notebook cells. No package plotting subsystem is introduced.

## Portable Inference

```python
emulator.save(path="ca40_l0_ws3.lrom")
loaded = lrom.load(path="ca40_l0_ws3.lrom")
loaded.predict(parameters={...})
```

The versioned artifact uses JSON metadata and NumPy arrays rather than pickle. It contains configuration, parameter schema, meshes, `phi0`, basis vectors, predictor transformations and selected points, RF-LROM matrices/vectors, provenance, and schema versions. It excludes live ROSE objects and raw samples by default, requires no ROSE installation for prediction, and targets Python 3.11, 3.12, and 3.13.

A loaded inference artifact is trained and prediction-ready but cannot resample or retrain without FOM samples. A larger research-checkpoint format is deferred.

## Validation and Compatibility

The implementation must test:

- Keyword-only public configuration and workflow APIs.
- `ws_1`, `ws_3`, full `"woods-saxon"`, and custom-potential schemas.
- Named overrides, subset ranges, exact `l` selection, and explicit multi-channel tuples.
- Deterministic linspace and Latin-hypercube designs.
- Central point exclusion from training/testing counts.
- Shared-basis ROSE/LS/LROM coefficient comparisons.
- Potential maxvol points and test-set reconstruction errors.
- Lifecycle invalidation and clear configuration, sampling, state, and artifact errors.
- Portable save/load and prediction without ROSE.
- Full workflow compatibility on Python 3.11, 3.12, and 3.13 where upstream training dependencies permit it; dependency blockers must be documented precisely rather than silently reducing inference support.
- Numerical agreement with the current validated Notebook 1 implementation.

## Deferred Scope and Change Control

Notebook 2 cross sections, CAT plots, and interactive HTML; Notebook 3 isotope/energy ranges and black-box export; external FOM request/response sampling; arbitrary multidimensional external solutions; and full research checkpoints are deferred in `docs/backlog/`.

Minor additions that reuse approved state and component boundaries may proceed. Any proposal requiring a major structural change to this architecture must be presented to and approved by the user before implementation.
