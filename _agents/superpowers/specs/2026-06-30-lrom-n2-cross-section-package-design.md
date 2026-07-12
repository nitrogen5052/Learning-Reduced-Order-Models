# LROM N2 Cross-Section Package Design

## Goal

Extend the active `lrom` package for Notebook 2 so it can produce LROM cross
sections from the same user-facing workflow as Notebook 1:

```python
emulator = lrom.LROM(
    target=(40, 20),
    projectile=(1, 0),
    lab_energy=14.1,
    l=tuple(range(0, 11)),
    potential="full_woods-saxon",
)
emulator.sampling(...)
emulator.train(...)
emulator.predict(...)
```

The current Notebook 1 package snapshot is already frozen as
`lrom_legacy.N1`. When the Notebook 2 package reaches the same stable review
point, it will be frozen as `lrom_legacy.N2`.

## Package Boundary

The package owns physics data production and state. It does not own the
Notebook 2 plots. Validation notebooks in `ntbk_validation/` use explicit
Matplotlib cells and visible tables so the user can judge the physics directly.

Automated tests check mechanics: imports, state transitions, array shapes,
parameter names, deterministic sample bookkeeping, and serialization. They do
not decide scientific validity.

## User Workflow

`sampling(...)` chooses named training and testing parameter rows and solves or
stores full-order wavefunction data. It should not expose `eim_basis_size`.
Operator/EIM choices belong to training.

```python
emulator.sampling(
    training_ranges=training_ranges,
    testing_ranges=testing_ranges,
    training_size=100,
    testing_size=80,
    strategy="latin_hypercube",
    seed=20260524,
    mesh_size=700,
)
```

`train(...)` owns all reduced-model choices:

```python
emulator.train(
    basis_size=4,
    predictor="potential",
    predictor_count=10,
    operator_basis_size=10,
)
```

Meanings:

- `basis_size`: wavefunction/reduced-solution basis size `n`.
- `predictor_count`: selected potential predictor count `K`.
- `operator_basis_size`: operator/EIM basis size `n_U`, used for the
  full Woods-Saxon interaction/operator representation needed downstream.

Cross sections are produced by `predict(...)` when `train(...)` configures the
emulator for a cross-section observable:

```python
emulator.train(
    basis_size=4,
    predictor="potential",
    predictor_count=10,
    operator_basis_size=10,
    observable="cross_section",
    angles_degrees=np.linspace(1.0, 179.0, 120),
)
```

Then:

```python
emulator.predict(parameters={...})
```

saves wavefunctions, S-matrix arrays, and LROM cross sections on
`emulator.predictions`. Optional FOM/reference cross sections may be produced for
review through an explicit reference option later, but references are references,
not methods. There is no `methods=("fom", "ls", "rose", "lrom")` selector in the
Notebook 2 API.

## Potential Schema

Add a named potential schema:

```python
potential="full_woods-saxon"
```

It owns the 10 optical-potential parameters:

```text
Vv, Wv, Wd, Vso, Rv, Rd, Rso, av, ad, aso
```

Notebook 2 varies all of them. The package must preserve parameter names and
row order so notebooks can show the exact sample tables.

## Saved State

The package should expose the following state after the relevant lifecycle
steps.

### `emulator.samples`

- `design.training.case_ids`
- `design.testing.case_ids`
- `design.training.parameter_names`
- `design.testing.parameter_names`
- `design.training.values`
- `design.testing.values`
- `central_parameters`
- physical mesh and stored FOM wavefunctions needed to train channel models

### `emulator.training_state`

- channel bases for each selected partial wave/spin channel
- LS coefficients used as coefficient targets
- selected potential predictor locations and scales
- RF-LROM/operator matrices and vectors
- `basis_size`, `predictor_count`, and `operator_basis_size` provenance

### `emulator.predictions`

LROM is the default result, so fields do not repeat `.lrom`.

- `parameter_names`
- `parameters`
- `coefficients`
- `wavefunctions`
- `smatrix.splus`
- `smatrix.sminus`
- `cross_sections.angles_degrees`
- `cross_sections.values`

If reference outputs are explicitly requested later:

- `cross_sections.reference`

The notebook access pattern should be:

```python
pred = emulator.predictions
theta = pred.cross_sections.angles_degrees

plt.plot(theta, pred.cross_sections.values[0], label="LROM")

splus = pred.smatrix.splus[0]
sminus = pred.smatrix.sminus[0]
```

### Error and CAT Data

If requested, error and CAT helpers return arrays, not plots:

- median pointwise relative error
- max pointwise relative error
- relative L2 cross-section error
- online time per sample
- method/config label data if needed for Notebook 2 CAT comparisons

The notebook decides how to visualize these arrays.

## Math Pipeline

The package maps the legacy method into explicit state:

1. Build `full_woods-saxon` parameter samples.
2. Solve/store channel wavefunctions for all requested partial waves.
3. Build central-reference LROM bases.
4. Fit potential-predictor RF-LROM channel models.
5. Predict channel coefficients for requested parameter rows.
6. Convert coefficients to `S+` and `S-` channel arrays.
7. Use ROSE scattering-amplitude assembly to compute
   `d sigma / d Omega`.
8. Save cross-section and S-matrix arrays on `emulator.predictions` for
   notebook inspection.

Current implementation checkpoint:

- `predict()` produces inspectable cross-section state for the package's current
  trained state.
- Full `full_woods-saxon` spin-orbit cross sections preserve ROSE's plus/minus
  channel pair for `l > 0` internally while keeping the public partial-wave API
  as `l=(0, 1, ...)`.

## Validation Notebooks

`ntbk_validation/N2_physics_validation.ipynb` is the human review artifact. It
should show:

- optical-potential parameter table
- sampled parameter rows
- potential rainbows
- selected predictor locations
- S-matrix summaries
- representative cross-section curves
- optional reference overlays
- error distributions and CAT-style scatter plots when requested

These plots should be ordinary notebook Matplotlib code, not package plotting
helpers.

## Deferred Work

- Freezing the finished package as `lrom_legacy.N2`.
- Interactive HTML export.
- ROSE reduced cross-section models as a package-owned comparison path. Notebook
  2 currently requires LROM-only cross-section production, with optional FOM
  reference data.
