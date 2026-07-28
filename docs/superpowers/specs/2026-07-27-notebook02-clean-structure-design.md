# Notebook 02 Clean-Structure Design

## Goal

Refactor `notebooks/02_rose_vs_lrom_cross_sections.ipynb` into a
self-contained teaching notebook whose imports, reusable functions, experiment
execution, and presentation outputs are visibly separated. Preserve the
existing scientific study, independent ROSE workflow, five figures, alpha
selections, result tables, and numerical results.

## Scope

This change modifies only:

- `notebooks/02_rose_vs_lrom_cross_sections.ipynb`;
- notebook-contract coverage in `tests/test_notebook02_cross_sections.py`;
- the project architecture note and required memory handoff after validation.

The extracted functions remain inside Notebook 02. No helper module, package
API, package implementation, Notebook 01, Benchmark 03, or scientific archive
file changes.

## Constraints

- Put every import in the first code cell. No later code cell may contain an
  `import` statement.
- Remove temporary inspection output, including `print(...)` and `.head()`.
  Tables and figures that are study results remain visible notebook outputs.
- Give reusable helpers explicit inputs, explicit return values, and standard
  Python or NumPy annotations.
- Do not introduce validation branches inside helpers merely to defend against
  impossible notebook states.
- Do not use global study constants from inside reusable helpers. Constants,
  models, arrays, angle grids, repeat counts, and floors required by a helper
  must be passed as arguments.
- Keep ROSE construction, training, prediction, and timing notebook-owned and
  separate from the LROM package.
- Preserve \(^{40}\mathrm{Ca}(n,n)\), 14.1 MeV, \(l=0,\ldots,3\), the
  ten-parameter full Woods-Saxon interaction, 600-point physical-radius mesh,
  200/100 sampling profile, seed 1204, closed ±20% alpha ranges, angle grid,
  and all basis/compression grids.
- Use “alpha selection” rather than “percentile” in prose and output.
- Keep notebook markdown sparse so the user can write the final explanatory
  prose.

## Notebook Structure

### Code cell 0: imports and environment

Group standard-library, plotting, NumPy, Pandas, Numba, SciPy compatibility,
ROSE, and parked-v2 imports here. Keep the project-root path setup because the
notebook must execute from its own directory. Remove the version `print`;
version provenance remains in the validation table.

### Code cell 1: study constants

Retain only physical, sampling, model-grid, timing, and plotting constants.
This cell performs no model construction.

### Code cell 2: reusable numerical utilities

Define small functional helpers:

- `parameter_dicts(rows, parameter_names) -> list[dict[str, float]]`;
- `pointwise_relative_error(predicted, reference, denominator_floor)
  -> np.ndarray`;
- `summarize_relative_error(predicted, reference, denominator_floor)
  -> dict[str, np.ndarray]`;
- `time_lrom_predictions(model, cases, repeats, inner_loops) -> np.ndarray`;
- `time_rose_predictions(model, rows, repeats, inner_loops,
  emulated_smatrix, cross_section) -> np.ndarray`.

The summary contains one median-over-angle metric and one maximum-over-angle
metric. Remove the current duplicated `median_error` and `cat_error` values.
LROM and ROSE timing remain separate functions because they time different
online workflows.

### Code cell 3: sampling function and study construction

Define a function that accepts the complete physical and sampling
configuration, constructs the parked-v2 emulator, performs deterministic
sampling, and returns the emulator, parameter names, parameter ranges, train
and test arrays, and case IDs.

Call the function in the same cell. Keep concise top-level scientific
invariants for expected row shapes, train/test non-overlap, and alpha bounds.
Remove `rose_train_rows` and `rose_test_rows`: ROSE receives `train_rows` and
`test_rows` directly.

### Code cell 4: predictor-selection presentation

Define a function that accepts the sampled emulator, central alpha,
parameter ranges, training/testing arrays, predictor count, and minimum
physical radius. It returns the predictor mapping, predictor-radius table, and
Matplotlib figure. The short execution block displays the figure and table.

The plotted predictor mapping must still be compared with the actual default
trained LROM predictor later in the study.

### Code cell 5: notebook-owned ROSE functions

Keep the Numba-compatible Woods-Saxon callables here. Define functions with
explicit arguments for:

- channel-key resolution;
- free-reference ROSE basis construction;
- scattering-amplitude emulator construction;
- exact all-channel \(S\)-matrix evaluation;
- emulated all-channel \(S\)-matrix evaluation;
- fixed-grid cross-section assembly;
- construction of the nine ROSE configurations.

These functions may call ROSE and therefore are not mathematically pure, but
their dependencies and outputs are explicit. They must not read notebook study
constants implicitly.

### Code cell 6: experiment functions and execution

Extract the long sequential experiment into functions:

- `evaluate_fom_cross_sections(...)`;
- `evaluate_old_lrom(...)`;
- `evaluate_lrom_grid(...)`;
- `evaluate_ls_oracle(...)`;
- `evaluate_rose_grid(...)`.

Each function receives all required models, rows, grids, timing controls,
angles, and error controls, then returns new result dictionaries or arrays.
The execution block constructs the ROSE grid and calls these functions in
scientific order:

1. exact FOM;
2. old shared-potential v2 reference;
3. archive-method LROM grid;
4. LS oracle;
5. independent ROSE grid.

No helper mutates a notebook-global result dictionary. Finite-result and
configuration-completeness checks remain concise top-level experiment checks.

### Code cell 7: alpha selection and representative figures

Define a selection function that accepts the default LROM and ROSE error
arrays and returns three distinct testing indices. Define figure functions for
representative cross sections and angle-resolved errors. Return figures rather
than calling `plt.show()` inside the helper.

The execution block displays the alpha-selection table and the two figures.
The selected IDs must remain `test-0002`, `test-0039`, and `test-0019`.

### Code cell 8: train/test distribution figure

Define and call one function that receives the default LS, ROSE, and LROM
results and returns the split violin figure.

### Code cell 9: numerical summary table

Define and call a function that converts the LROM and ROSE result mappings into
the existing configuration table. Use one unambiguous column name for the
median-over-angle held-out error.

### Code cell 10: CAT figure

Define and call a function that accepts the two result mappings, basis sizes,
compression sizes, plotting floor, error reference, and throughput reference.
It returns the accuracy-versus-time figure with the current marker semantics.

### Code cell 11: validation table

Define and call a function that receives every displayed validation value and
returns the final table. Include Python and platform provenance as explicit
arguments from the execution call rather than reading them inside the helper.

## Data and Method Boundaries

The data flow remains:

```text
shared alpha rows and exact snapshots
├── exact channel solvers -> FOM cross sections
├── LS projection of exact wavefunctions -> oracle cross sections
├── parked-v2 effective-interaction training -> LROM cross sections
└── notebook-owned free-reference basis + EIM -> ROSE cross sections
```

Sharing alpha rows and snapshots does not merge method implementations. LROM
continues to call the package `train()` and `predict()` lifecycle. ROSE
continues to build its own interaction EIM, free-reference bases, reduced
\(S\)-matrix elements, and cross sections inside the notebook.

## Output Preservation

The refactor must retain:

- five figures in the existing order and with the same physical labels;
- the channel predictor-radius table;
- the exact alpha-selection table and A/B/C IDs;
- the scientific values in the configuration and validation tables;
- the full 200/100 executed profile;
- observable-only LROM timing with `reconstruct_wavefunctions=False`;
- warmed, repeated, complete alpha-to-cross-section timing for both methods.

Timing values may vary with machine load. Scientific arrays must match the
pre-refactor notebook within floating-point tolerance; table values other than
timing should remain identical at the displayed precision.

## Testing and Validation

Before editing the notebook, add failing contract tests that require:

1. every import to occur in the first code cell;
2. no `print(...)` calls or `.head()` inspection calls;
3. the removed ROSE row aliases and tautological equality checks to be absent;
4. the duplicated `cat_error`/`median_error` implementation to be replaced by
   one named median-over-angle metric;
5. the expected extracted function names and annotations to exist;
6. Notebook 02 to retain the required physics literals, ROSE separation,
   observable-only LROM calls, alpha terminology, and five figure markers.

After the static tests pass:

- execute Notebook 02 end to end;
- confirm all code cells execute with zero stored errors;
- confirm five PNG figures are stored and visually inspect them;
- compare alpha selections and all non-timing scientific table values with
  commit `a8fc707`;
- run the focused notebook tests and full project test suite;
- run Ruff on modified Python tests and Notebook 02, allowing only the existing
  intentional import-placement compatibility exception if required;
- verify protected Notebook 01 and scientific archive digests are unchanged.

## Non-Goals

- No package refactor or new dependency.
- No Benchmark 03 cleanup in this change.
- No change to LROM or ROSE mathematics.
- No correction of the deferred momentum-coordinate mismatch.
- No new figures, interactive output, or additional markdown explanation.
