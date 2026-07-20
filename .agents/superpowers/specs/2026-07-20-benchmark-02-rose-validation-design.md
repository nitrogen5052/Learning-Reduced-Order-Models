# Benchmark 02 ROSE Validation Redesign

## Purpose

Rebuild only the ROSE benchmarking workflow inside
`notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb` so that it follows the
method demonstrated in `scientific_archive/ROSE_Guide`: construct ROSE
explicitly, reserve held-out parameter samples, compare emulated solutions
against ROSE's high-fidelity solver, measure online accuracy and time across
several emulator configurations, and select a validated configuration before
using it in scientific comparison plots.

The notebook remains a reconstruction of
`scientific_archive/legacy_code/Legacy_benchmark/notebooks/02_lrom_method_walkthrough.ipynb`.
Its LROM methodology, study order, sampling designs, predictors, diagnostics,
and 16-figure contract do not change.

## Scope Boundaries

- Modify `notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb` only where it
  constructs, validates, selects, evaluates, labels, or summarizes ROSE.
- Update `tests/test_benchmark_02_notebook.py` only to lock in the revised ROSE
  workflow and preserve the existing notebook contract.
- Do not modify `lrom/`, `lrom_legacy/`, package APIs, package physics, or
  package-generated arrays.
- Do not modify anything under `scientific_archive/`.
- Keep every existing LROM training row, testing row, basis size, predictor,
  coefficient array, wavefunction metric, scan window, and figure marker.
- Keep all Matplotlib code inline. Do not add plotting wrappers or a ROSE
  workflow wrapper.
- Continue showing spatial quantities against physical radius `r` in fm.
- Keep `DIFFERENCE_FLOOR = 1e-5` as display-only clipping.
- Keep EIM `training_info` bounds inclusive of the original testing rows; this
  is an intentional project decision.

## Per-Study ROSE Workflow

Apply the following workflow independently to the existing `Vv`-only,
`Rv`-only, and broad `Vv/Rv` studies. The potential-predictor study continues
to reuse the broad study's ROSE comparison because its physical samples and
ROSE baseline are unchanged.

### 1. Preserve the shared training rows

Each ROSE emulator is trained on the exact parameter rows already produced by
the corresponding LROM study:

- `vv_rose_rows` for the `Vv` study;
- `rv_rose_rows` for the `Rv` study;
- `broad_rose_rows` for the broad and potential-predictor studies.

This keeps the direct ROSE/LROM comparison fair and leaves LROM sampling
untouched.

### 2. Add a ROSE-only held-out design

For each study, generate 20 deterministic Latin-hypercube validation rows
inside the coordinate-wise minimum and maximum of that study's training rows.
Parameters that are fixed in the study remain fixed at their existing central
values. Use distinct, fixed seeds for the three studies so notebook execution
is reproducible.

The held-out rows are used only to validate ROSE configurations. They do not
enter LROM training, LROM testing, the one-at-a-time scan definitions, or the
legacy figures.

Assert that validation arrays have the expected `(20, 3)` shape, contain only
finite values, lie inside the training box, and do not duplicate a training
row to floating-point tolerance.

### 3. Tune the EIM size at a fixed wavefunction basis size

Use a shared candidate set:

```python
ROSE_EIM_CANDIDATES = (4, 8, 12)
ROSE_VALIDATION_SIZE = 20
ROSE_TIMING_REPEATS = 3
```

For every candidate `n_U`, construct a fresh public-ROSE
`InteractionEIMSpace` and
`ScatteringAmplitudeEmulator.from_train(...)`. Keep
`n_phi = BASIS_SIZE = 4` for every candidate. Fixing the wavefunction basis
size preserves the equal-basis scientific comparison with LROM while varying
the ROSE hyperparameter responsible for approximating the non-affine
interaction.

Use the existing radial mesh, kinematics, base-solver tolerances, free-reference
basis convention, `scale=False`, and `use_svd=True`. The EIM bounds include
the training rows, held-out validation rows, and original testing/scan rows.

### 4. Establish high-fidelity validation truth

Use the public ROSE exact-solver path exposed by the candidate scattering
amplitude emulator to calculate the held-out `l=0` wavefunctions. Calculate
this ground truth once per study and reuse it across candidate configurations.
The exact and emulated arrays must share the study's ROSE mesh and have shape
`(ROSE_VALIDATION_SIZE, MESH_SIZE)`.

Use the existing per-wavefunction relative-L2 definition:

```python
np.linalg.norm(phi_emu - phi_exact, axis=1) /
np.maximum(np.linalg.norm(phi_exact, axis=1), 1e-300)
```

This adapts the ROSE tutorial's accuracy-versus-time procedure to Notebook
02's wavefunction observable without introducing cross sections into the
version-1 benchmark.

### 5. Measure online time and select the configuration

Warm each candidate once. Then time three complete passes over the held-out
rows using `time.perf_counter()`. Store the minimum batch time divided by the
number of held-out rows as seconds per sample.

For each candidate, retain:

- `n_phi` and `n_U`;
- median and maximum held-out relative-L2 error;
- minimum seconds per held-out sample;
- the trained emulator and its `l=0` reduced-basis emulator.

Select the candidate with the smallest median held-out relative-L2 error.
Break an exact error tie with the lower online time, then the smaller `n_U`.
Print a compact, deterministic per-study table and identify the selected
configuration. The table replaces the current progress-bar-only ROSE output;
it does not add a new notebook section or figure.

### 6. Feed only the selected ROSE model into existing diagnostics

Evaluate the selected emulator on the study's existing training and testing
rows. Preserve the current notebook variable roles:

- ROSE Galerkin coefficients;
- ROSE-native trapezoid-weighted LS coefficients in the free-reference basis;
- ROSE wavefunctions;
- ROSE relative-L2 wavefunction errors.

All existing figures continue consuming those roles without changing their
layout, axes, LROM curves, or scientific interpretation. Coefficients are
still compared only within a common convention: ROSE versus ROSE-native LS,
and LROM versus central-basis LS.

## Notebook Structure And Presentation

The existing sections remain in their current order:

1. shared physical setup;
2. `Vv`-only study and three figures;
3. `Rv`-only study and three figures;
4. broad `Vv/Rv` study and three figures;
5. potential-predictor diagnostics and three figures;
6. linear-versus-predictor comparison and four figures;
7. final interpretation.

No new figure markers or plot layouts are introduced. Add concise markdown to
the introduction and shared-setup section explaining that ROSE is tuned
independently in each physical study using held-out Latin-hypercube rows,
while the LROM benchmark design remains the legacy design.

The final violin continues to use the existing training and structured testing
rows. Its purpose remains the legacy train/scan comparison; ROSE's held-out
validation errors are reported separately in the configuration tables and do
not silently replace the legacy distributions.

## Failure Handling

- Fail visibly if ROSE returns non-finite exact or emulated wavefunctions.
- Fail visibly if a candidate produces a wavefunction array with the wrong
  shape.
- Do not silently drop a failed candidate or substitute cached results.
- Do not write pickle files, cached emulators, or numerical artifacts.
- Notebook execution must be deterministic apart from measured wall-clock
  time.

## Verification

Extend `tests/test_benchmark_02_notebook.py` to require:

- deterministic Latin-hypercube ROSE validation rows;
- `ROSE_EIM_CANDIDATES = (4, 8, 12)`;
- fixed `n_phi = BASIS_SIZE` across candidates;
- public ROSE exact-wavefunction and emulated-wavefunction evaluation;
- `time.perf_counter()` timing with three repeats;
- held-out relative-L2 accuracy calculation and explicit selected
  configuration;
- preservation of all 16 existing figure markers;
- absence of plotting wrappers, ROSE workflow wrappers, archived helpers, and
  package-owned ROSE benchmarking.

Run the focused notebook contract tests, the complete test suite, and the
notebook end-to-end. Inspect the rendered figures and the three ROSE summary
tables before reporting completion.

## Non-Goals

- Changing LROM training or prediction behavior.
- Changing any package version or architecture.
- Adding cross-section, phase-shift, Bayesian-calibration, or uncertainty-
  quantification studies to Benchmark 02.
- Replacing the structured interpolation/extrapolation scans.
- Comparing ROSE coefficients directly with LROM coefficients.
- Adding notebook generators, plotting helpers, serialization, or caches.
