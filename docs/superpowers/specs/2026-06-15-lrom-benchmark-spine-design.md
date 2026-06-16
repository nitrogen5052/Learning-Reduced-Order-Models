# LROM Benchmark Spine Design

Date: 2026-06-15

## Goal

Rewrite the legacy LROM benchmark into a simpler, organized Python research benchmark package that can reproduce the scientific results from the legacy notebooks. The first implementation target is Notebook 02, because it contains the core residual-fit learned reduced-operator model (RF-LROM) method story.

The near-term goal is not to publish a general library. The near-term goal is to build a trustworthy benchmark package, generated/reviewable notebooks, and parity artifacts that make it possible to retire the legacy notebook once the new package reproduces its results.

## Scope

In scope for the first architecture pass:

- Create a Python-first benchmark package for the RF-LROM workflow.
- Keep Notebook 02 as the canonical first parity gate.
- Preserve the visible scientific narrative in the new Notebook 02.
- Compare new package outputs against frozen legacy reference artifacts by default.
- Support optional rerun-legacy comparisons when dependencies or benchmark definitions intentionally change.
- Write metadata-rich benchmark reports that can later feed a vector database or graph memory layer.

Out of scope for the first architecture pass:

- A publishable package release workflow.
- A web UI or webview-based design companion.
- Building the vector database or graph memory layer immediately.
- Deleting legacy notebooks before parity gates pass.
- Refactoring all four notebooks at once.

## Architecture

The new package should follow a benchmark spine organized by scientific workflow stage:

```text
lrom_bench/
  config.py          # typed benchmark configs, meshes, seeds, tolerances
  sampling.py        # one-parameter scans, Latin hypercube samples, train/test splits
  rose_fom.py        # ROSE setup, KD parameters, high-fidelity wavefunction generation
  reduced_basis.py   # central state, centered basis, LS coefficient targets, LS floor
  predictors.py      # raw parameter predictors, potential predictors, delta-maxvol
  rf_lrom.py         # residual-fit linear solve and learned operator containers
  prediction.py      # online reduced solve and wavefunction reconstruction
  metrics.py         # coefficient, wavefunction, cross-section, and parity metrics
  artifacts.py       # save/load frozen legacy refs, new outputs, reports
```

This package is intentionally shaped around the benchmark workflow rather than abstract framework interfaces. The code should remain small, explicit, and scientifically auditable.

## Notebook 02 Benchmark Flow

Notebook 02 should stay narrative and package-native. It should call the package at meaningful stage boundaries rather than hiding the whole experiment behind one large runner.

The notebook flow should be:

1. Setup and benchmark configuration.
2. Central reference state and reduced-basis convention.
3. Real volume depth (`Vv`) one-parameter scan.
4. Radius (`Rv`) one-parameter scan.
5. Broad `Vv`/`Rv` box where raw linear predictors struggle.
6. Operator-informed potential predictors using delta-maxvol selected points.
7. Frozen parity comparison and scientific summary.

The notebook should show enough code and commentary for a physicist or machine-learning researcher to follow the method. Repeated mechanics belong in the package; the scientific sequence, assumptions, and acceptance evidence belong in the notebook.

## Parity Strategy

Notebook 02 gets two parity layers.

The default layer is frozen artifact parity. Legacy reference outputs are preserved and compared against new package outputs. This is the daily regression gate because it is stable and does not require rerunning the legacy notebook.

The optional layer is rerun-legacy parity. Use it when ROSE behavior, numerical dependencies, mesh choices, benchmark definitions, or tolerances intentionally change. In that mode, rerun the legacy extraction path and compare old and new outputs under documented tolerances.

Initial artifact paths:

```text
outputs/benchmarks/legacy/notebook02_lrom_walkthrough.npz
outputs/benchmarks/new/notebook02_lrom_walkthrough.npz
outputs/benchmarks/reports/notebook02_parity.json
```

The parity report should record:

- benchmark config hash
- git commit when available
- Python, NumPy, SciPy, and ROSE import/version metadata when available
- legacy artifact path and new artifact path
- tolerances
- metric names
- pass/fail status
- scientific notes for intentional changes

The first parity metrics should include:

- least-squares coefficient targets
- LS wavefunction floor
- RF-LROM residual MSE
- predicted coefficient errors
- reconstructed wavefunction relative L2 errors
- selected potential predictor points
- representative arrays needed by the notebook plots

## Scientific Model Boundaries

The package should preserve the current best RF-LROM workflow:

1. Choose a central parameter point.
2. Compute the central FOM wavefunction and set it as `phi0`.
3. Build a reduced basis from centered snapshots.
4. Compute least-squares target coordinates for training wavefunctions.
5. Construct predictors from raw parameters or operator-informed potential samples.
6. Fit the implicit reduced equation by one linear residual least-squares solve.
7. Predict new reduced coordinates by assembling and solving the small online system.
8. Reconstruct wavefunctions and evaluate scientific errors.

For operator-informed predictors, the package should keep the delta-maxvol selection explicit: form centered potential variations, compute a low-rank representation, choose informative spatial points, and evaluate normalized predictor values at those points.

## Error Handling And Reproducibility

The benchmark package should fail early with clear messages when:

- `nuclear-rose` is missing or a wrong `rose` package is imported.
- Array shapes do not match expected sample, basis, or predictor dimensions.
- A frozen legacy artifact is missing.
- A parity comparison is attempted with incompatible metadata.
- Required numerical aliases or compatibility shims are missing in an environment.

Notebook startup should remain robust to stale imports and mixed NumPy versions. Any package bootstrap used by notebooks should run before importing ROSE-dependent modules.

## Future Graph And Vector Memory Layer

The graph/vector memory layer is a second layer after the benchmark package is stable. The first implementation should write structured artifacts that are easy to index later.

The future graph should model entities such as:

```text
BenchmarkRun
Config
Dataset
ModelFit
PredictorSet
MetricResult
Artifact
NotebookSection
DecisionNote
```

The graph should capture relationships like:

- a run used a config
- a model fit used a dataset
- a predictor set selected specific potential points
- a parity report compared against a legacy artifact
- a notebook section explains a metric or method choice

The vector side can later index Markdown explanations, notebook section summaries, paper/report excerpts, and design decisions. The first architecture pass should not add database infrastructure.

## Acceptance Criteria

The first implementation is successful when:

- A new package-native Notebook 02 reproduces the legacy Notebook 02 scientific flow.
- Frozen legacy and new artifacts are written to stable benchmark paths.
- A parity report records metadata, metrics, tolerances, and pass/fail status.
- The notebook remains readable and preserves the visible `Vv`, `Rv`, broad `Vv/Rv`, and potential-predictor narrative.
- Legacy Notebook 02 is no longer needed to understand or verify the method, although it is not deleted until the user approves deletion after passing parity.

## Open Decisions For Implementation Planning

- Exact package name: provisional name is `lrom_bench`.
- Exact tolerance values for each metric.
- Whether benchmark artifacts should live at the repository root or inside `Legacy_benchmark/outputs/` during migration.
- Whether Notebook 02 should be generated from a Python source script, hand-maintained, or built with a notebook generation tool.
