# Notebook 02 Flat Paper-CAT Design

## Goal

Rewrite Notebook 02 in the direct, cell-by-cell style used by Notebook 01.
Remove notebook-local function wrappers, reduce imports, and make the CAT plot
represent the held-out alpha distribution shown in Figure 5 of the ROSE paper.

## Scope

Modify:

- `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
- `tests/test_notebook02_cross_sections.py`
- architecture and memory documentation

Do not modify:

- `lrom/`
- `lrom_legacy/`
- `notebooks/benchmark_helper.py`
- Notebook 01
- Benchmark 01, Benchmark 02, or Benchmark 03
- `scientific_archive/`

## Notebook Structure

Notebook 02 will contain no locally defined functions. Existing large
functions for sampling, LROM evaluation, alpha selection, plotting, and tables
will be replaced by direct sequential code in the relevant cells.

The existing sparse section headings remain. No new Markdown prose or code
comments will be added. Code flow will be communicated by:

- smaller cells;
- blank lines between setup, calculation, validation, and display blocks;
- variables named for the scientific stage they represent.

The intended sequence is:

1. minimal imports;
2. physical and numerical constants;
3. LROM object construction and shared sampling;
4. sampled-array extraction and checks;
5. effective-interaction predictor locations and figure;
6. separate ROSE and LS benchmark calls;
7. old-v2 LROM evaluation;
8. archive-method LROM grid evaluation;
9. alpha selection and parameter table;
10. representative cross sections;
11. representative pointwise errors;
12. split train/test error distributions;
13. configuration summary table;
14. paper-faithful computational-accuracy-versus-time plot;
15. validation table.

## Imports

Remove imports used only by function annotations, wrapper return types, custom
legend classes, or validation metadata:

- `typing.Any`
- `platform`
- `matplotlib.figure.Figure`
- `matplotlib.lines.Line2D`
- `matplotlib.patches.Patch`
- direct `scipy.special`

Retain only:

- `Path` and `sys` for portable local imports;
- `time` for explicit parked-v2 LROM timing;
- `matplotlib.pyplot` for figures;
- NumPy for scientific arrays;
- pandas for result tables;
- `benchmark_helper` for notebook-owned ROSE and LS;
- explicit `lrom_legacy.v2_0` for parked-v2 LROM.

No dependency moves into the LROM package. These retained dependencies support
notebook analysis and presentation rather than the public LROM lifecycle.

## ROSE and LS Boundary

Notebook 02 continues to construct one `CrossSectionBenchmark` object and call:

```python
rose_comparison = cross_section_benchmark.run_rose(...)
ls_comparison = cross_section_benchmark.run_ls(...)
```

ROSE and LS remain separately runnable. ROSE construction, exact/emulated
\(S\)-matrices, cross sections, and ROSE timing remain in
`benchmark_helper.py`. LS remains a separate held-out-wavefunction oracle.
Parked-v2 LROM training, prediction, and timing remain visibly separate in the
notebook.

## Paper-Faithful CAT Plot

Figure 5 of `scientific_archive/ROSE_Guide/ROSEPaper[7945].pdf` plots one point
per held-out parameter vector. A fixed \((n_\phi,n_U)\) ROSE configuration
therefore produces a cluster of test-set points rather than one median point.

Notebook 02 will apply the same interpretation to every tested configuration:

- ROSE: nine \((n_\phi,n_U)\) configurations times 100 test alpha rows;
- LROM: nine \((n_\phi,K)\) configurations times 100 test alpha rows;
- x-coordinate: complete online time for that alpha row;
- y-coordinate: maximum relative cross-section error over angles 1 through
  179 degrees for that alpha row;
- ROSE marker: solid square;
- LROM marker: solid circle;
- color: \(n_\phi\);
- marker size: \(n_U\) for ROSE and \(K\) for LROM;
- transparency: low enough for overlapping clusters to remain legible.

The current one-point-per-configuration median CAT encoding will be removed.
The 10% error and one-million-evaluations-per-hour reference lines remain.
The legend will use empty `scatter` calls, avoiding custom legend-class
imports.

The summary table remains configuration-level and continues to report median
test error, maximum-over-angle diagnostics, and median online time.

## Scientific Preservation

The rewrite must preserve:

- \(^{40}\mathrm{Ca}(n,n)\) at 14.1 MeV lab energy;
- \(\ell=0,1,2,3\);
- ten-parameter full Woods-Saxon interaction;
- 600-point physical-radius mesh in fm;
- angles 1 through 179 degrees;
- closed plus/minus 20% parameter ranges;
- seed 1204;
- 200 training and 100 testing rows;
- basis sizes `(4, 6, 8)`;
- ROSE EIM sizes `(4, 8, 12)`;
- LROM predictor counts `(4, 8, 12)`;
- explicit shared ordered ROSE training rows;
- shared exact snapshots and ROSE free-reference bases;
- alpha selections `test-0002`, `test-0039`, and `test-0019`;
- default errors old v2 `0.043529`, archive LROM `0.000948`, and ROSE
  `0.028713`;
- the five existing scientific figures.

The CAT figure changes its displayed aggregation only. The underlying timing
and error arrays do not change.

## Verification

Tests will require:

- no `FunctionDef` nodes in Notebook 02;
- no removed imports;
- no code-cell comments;
- no new Markdown prose;
- direct ROSE and LS calls;
- solid LROM CAT markers;
- per-alpha CAT arrays using `test_seconds` and
  `test_maximum_over_angle_error`;
- all existing scientific constants and result markers.

The final notebook will be executed in full. Acceptance requires:

- every code cell executed;
- zero error outputs;
- five stored figures;
- visual inspection of all five figures;
- unchanged alpha IDs and scientific summary values;
- full test suite and Ruff success;
- unchanged package, scientific archive, Notebook 01, and benchmark files.
