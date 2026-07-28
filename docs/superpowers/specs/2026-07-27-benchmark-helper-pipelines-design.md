# Benchmark Helper Pipeline Design

## Goal

Replace notebook-local ROSE and least-squares machinery with a small,
notebook-owned `benchmark_helper.py` API that substantially shortens the main
teaching notebooks while preserving the validated scientific workflows.

## Scope

This change applies to:

- `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`
- `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
- `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
- `notebooks/rose_helper.py`, renamed to `notebooks/benchmark_helper.py`
- focused tests for the helper and affected notebooks

This change does not modify:

- public `lrom`
- `lrom_legacy.v1_2`
- `lrom_legacy.v2_0`
- `notebooks/benchmark_notebooks/1.0/benchmark_02.ipynb`
- `notebooks/benchmark_notebooks/2.0/benchmark_01.ipynb`
- the scientific archive
- the physical inputs, deterministic rows, retained partial waves, basis
  sizes, compression grids, timing definition, alpha selections, or figures

Benchmark 01 and Benchmark 02 remain explicit because their purpose is to show
legacy/version-specific validation methodology rather than provide the main
teaching workflow.

## Design Principles

1. ROSE remains separate from the LROM package.
2. LS remains an explicit oracle and is not part of LROM prediction.
3. ROSE and LS can run independently.
4. A combined call is a convenience wrapper around the independent paths.
5. Notebook cells show scientific choices and results, not mechanical solver
   construction.
6. Results use typed dataclasses containing raw NumPy arrays and plain result
   dictionaries.
7. No factory, plugin system, nested configuration hierarchy, or new
   dependency is introduced.
8. All radial arrays and exposed predictor locations remain physical radius
   \(r\) in femtometers.

## Public Notebook API

The helper exposes two benchmark objects.

### Wavefunction benchmark

```python
benchmark = benchmark_helper.WavefunctionBenchmark(
    emulator=emulator,
    center=center,
)

rose_result = benchmark.run_rose(
    basis_size=BASIS_SIZE,
    eim_basis_size=8,
)
ls_result = benchmark.run_ls()
```

The common combined form is:

```python
result = benchmark.run(
    basis_size=BASIS_SIZE,
    eim_basis_size=8,
)
```

`run()` returns a `WavefunctionBenchmarkResult` containing the separately typed
ROSE and LS results.

### Cross-section benchmark

```python
benchmark = benchmark_helper.CrossSectionBenchmark(
    emulator=emulator,
    training_rows=train_rows,
    testing_rows=test_rows,
    angles_degrees=ANGLES_DEG,
)

rose_result = benchmark.run_rose(
    basis_sizes=BASIS_SIZES,
    eim_sizes=ROSE_EIM_SIZES,
)
ls_result = benchmark.run_ls(
    basis_sizes=BASIS_SIZES,
)
```

The common combined form is:

```python
result = benchmark.run(
    basis_sizes=BASIS_SIZES,
    eim_sizes=ROSE_EIM_SIZES,
)
```

`run()` returns a `CrossSectionBenchmarkResult` containing the independently
computed ROSE and LS results.

The combined method must call the same public `run_rose()` and `run_ls()`
methods. It may not maintain a second combined implementation.

## Result Types

The helper uses a few flat dataclasses.

```python
@dataclass(frozen=True)
class RoseWavefunctionResult:
    emulator: Any
    basis: Any
    interaction: Any
    training_rows: np.ndarray
    testing_rows: np.ndarray
    training_coefficients: np.ndarray
    testing_coefficients: np.ndarray
    training_wavefunctions: np.ndarray | None
    testing_wavefunctions: np.ndarray


@dataclass(frozen=True)
class LSWavefunctionResult:
    training_coefficients: np.ndarray
    testing_coefficients: np.ndarray
    training_wavefunctions: np.ndarray
    testing_wavefunctions: np.ndarray


@dataclass(frozen=True)
class WavefunctionBenchmarkResult:
    rose: RoseWavefunctionResult
    ls: LSWavefunctionResult
```

Cross-section results retain the already validated schema:

```python
{
    "train_xs": np.ndarray,
    "test_xs": np.ndarray,
    "train_median_over_angle_error": np.ndarray,
    "test_median_over_angle_error": np.ndarray,
    "train_maximum_over_angle_error": np.ndarray,
    "test_maximum_over_angle_error": np.ndarray,
    "test_seconds": np.ndarray,
}
```

The LS result omits `test_seconds` because it is an oracle accuracy floor, not
an online emulator benchmark.

```python
@dataclass(frozen=True)
class RoseCrossSectionResult:
    emulators: dict[tuple[int, int], Any]
    fom_training_cross_sections: np.ndarray
    fom_testing_cross_sections: np.ndarray
    results: dict[tuple[int, int], dict[str, np.ndarray]]


@dataclass(frozen=True)
class LSCrossSectionResult:
    results: dict[int, dict[str, np.ndarray]]


@dataclass(frozen=True)
class CrossSectionBenchmarkResult:
    rose: RoseCrossSectionResult
    ls: LSCrossSectionResult
```

The helper does not own LROM result dataclasses. LROM predictions remain
package results and the notebook's LROM configuration grid remains visibly
separate.

## ROSE Guide Methodology

The helper follows the construction order used in
`scientific_archive/ROSE_Guide`:

1. Construct `rose.InteractionEIMSpace`.
2. Construct the ROSE base solver and its matching domain/tolerances.
3. Establish the native free reference with
   `rose.free_solutions.phi_free`.
4. Construct the reduced bases.
5. Construct `rose.ScatteringAmplitudeEmulator`.
6. Evaluate exact or emulated wavefunctions and \(S\)-matrix elements.
7. Assemble cross sections on the cached angular grid.

Private helper names and source ordering must make this sequence readable in
`benchmark_helper.py`.

The archive tutorials commonly call
`ScatteringAmplitudeEmulator.from_train()`, which lets ROSE generate its own
training snapshots. The corrected main notebooks instead construct
`CustomBasis` objects from the exact shared LROM training snapshots and
ROSE's free reference. This difference is deliberate and must remain:

- it aligns ROSE, LS, LROM, and FOM to the same ordered physical samples;
- it preserves the validated correction in Notebook 01 and Notebook 02;
- it keeps coefficient conventions separate while comparing reconstructed
  physical quantities.

The helper follows the guide's stages and ROSE object boundaries without
reverting to independent snapshots in these comparisons.

## Wavefunction Pipeline

`WavefunctionBenchmark` owns only references to the supplied emulator and
central parameter mapping.

`run_rose()`:

1. Reads the emulator's physical rho/radius mesh and deterministic sample rows.
2. Builds the same single-channel Woods-Saxon EIM interaction as the current
   `rose_helper.build_rose()`.
3. Builds the native ROSE free reference.
4. Builds the `CustomBasis` from shared exact training wavefunctions.
5. Builds the reduced ROSE emulator.
6. Evaluates testing coefficients and wavefunctions.
7. Optionally evaluates training wavefunctions when requested.
8. Returns `RoseWavefunctionResult`.

`run_ls()`:

1. Calls the active v1.2 `least_squares_baseline()` on the supplied LROM basis
   and exact training wavefunctions.
2. Calls it independently on exact testing wavefunctions.
3. Returns `LSWavefunctionResult`.

Notebook 01 may run ROSE and LS independently or use `run()` where both are
needed. Existing plotting and case-selection data names are adapted at the
short call site without changing values.

## Cross-Section Pipeline

`CrossSectionBenchmark` owns the supplied parked-v2 emulator, ordered training
and testing rows, and angular grid. Potential callbacks, spin-orbit callback,
partial-wave maximum, EIM sizes, timing controls, and error floor remain
explicit method inputs when they vary by notebook profile.

`run_rose()`:

1. Builds the Cartesian ROSE configuration grid.
2. Uses explicit ordered training rows for deterministic EIM construction.
3. Builds one free-reference `CustomBasis` per \((l,j)\) channel from shared
   exact training snapshots.
4. Builds the scattering-amplitude emulators.
5. Uses one emulator as the exact all-channel FOM boundary.
6. Evaluates exact training and testing cross sections.
7. Evaluates and times each ROSE configuration.
8. Returns cross sections, named error arrays, times, and emulator mappings in
   `RoseCrossSectionResult`.

`run_ls()`:

1. Trains or reuses the requested LROM basis for each basis size using the
   same explicit training options as the notebook.
2. Projects exact training and testing wavefunctions into that basis.
3. Calls the parked-v2 cross-section assembly with those oracle coordinates.
4. Returns the same accuracy arrays without timing in
   `LSCrossSectionResult`.

The LS method accepts the predictor/training settings needed to reproduce the
current basis state. It does not call `predict()` and does not report an online
time.

Notebook 02 and Benchmark 03 retain their separate LROM grid evaluation. They
consume the helper's FOM, ROSE, and LS results for tables and figures.

## Notebook Structure After Migration

### Notebook 01

The notebook retains:

- physical problem definitions and sampling;
- LROM training and prediction;
- benchmark object construction;
- one `run_rose()`, `run_ls()`, or combined `run()` call per study;
- case selection, figures, and user-owned explanatory Markdown.

It no longer contains repeated LS calls or direct ROSE construction details.

### Notebook 02

The notebook retains:

- imports and physical constants;
- shared sampling;
- predictor-radius visualization;
- old-v2 and archive-method LROM grid evaluation;
- one cross-section benchmark object and short ROSE/LS calls;
- alpha selection, figures, tables, and sparse explanatory Markdown.

It no longer defines ROSE potential callbacks, basis/emulator constructors,
all-channel \(S\)-matrix helpers, FOM evaluation, ROSE timing/evaluation, or LS
projection/cross-section functions.

### Benchmark 03

Benchmark 03 uses the same `CrossSectionBenchmark` API with its reduced/full
profile inputs. It must not carry a copied ROSE/LS implementation.

## Validation and Error Handling

Validation is kept at meaningful method boundaries:

- requested angle grids must match the ROSE cached grid;
- every requested partial wave and spin channel must be present;
- training/testing rows must be two-dimensional arrays with matching parameter
  counts;
- ROSE configuration grids must contain every requested basis/EIM pair;
- cross sections and error arrays must be finite;
- the free-reference bases must use the supplied physical mesh;
- predictor radii remain at or above the existing 0.5 fm study limit.

No redundant defensive assertions are added inside small numerical helpers.
The notebooks retain scientific assertions that explain their study contract.

ROSE-only execution must not construct LS results. LS-only execution must not
construct ROSE interactions or emulators. Failure in one independent path does
not prevent a caller from using the other path.

## Testing Strategy

### Helper tests

Focused tests cover:

- typed result structure;
- separate ROSE-only execution;
- separate LS-only execution;
- combined execution delegating to both independent paths;
- exact shared training-row use;
- native free-reference `CustomBasis` construction;
- all-channel \(l=0,\ldots,l_{\max}\) assembly;
- cross-section error schema;
- angle-grid mismatch failure;
- absence of package mutation.

Small fakes or monkeypatches cover orchestration contracts. Existing numerical
tests and full notebook executions provide the scientific integration gate.

### Notebook contracts

Notebook tests require:

- import of `benchmark_helper`;
- absence of direct ROSE EIM, `CustomBasis`, all-channel \(S\)-matrix, and LS
  projection implementations in Notebook 02 and Benchmark 03;
- short high-level pipeline calls;
- unchanged physical constants and grids;
- unchanged figure/table markers;
- compilable code cells.

Notebook 01 tests require the new helper import and no direct repeated LS
pipeline.

### Full execution parity

Re-execute Notebook 01, Notebook 02, and Benchmark 03. Confirm:

- every code cell executes with zero error outputs;
- all stored figures remain present and visually legible;
- alpha IDs remain unchanged;
- scientific arrays and summary values match the pre-migration baseline;
- timing uses the same warmup, repeat, and inner-loop definition;
- protected package and scientific-archive digests remain unchanged.

Timing values may vary naturally. Timing procedure and ordering must not.

## Migration and Compatibility

`notebooks/rose_helper.py` is renamed to
`notebooks/benchmark_helper.py`. A compatibility shim is not retained because
only repository notebooks import the helper and all in-scope imports are
updated in the same change.

No helper API is exported from `lrom`. The helper is notebook support code and
is imported by inserting the notebook directory into `sys.path` where needed.

The existing untracked `tmp/` directory is not touched.

## Acceptance Criteria

The design is complete when:

1. ROSE and LS can each run independently.
2. Combined calls reuse the independent implementations.
3. The helper visibly follows the ROSE Guide construction sequence.
4. Notebook 01, Notebook 02, and Benchmark 03 contain substantially less
   ROSE/LS machinery.
5. ROSE remains notebook-owned and LS remains an explicit oracle.
6. All scientific outputs and figures remain unchanged outside timing noise.
7. No LROM package, legacy validation notebook, or scientific archive file is
   modified.
