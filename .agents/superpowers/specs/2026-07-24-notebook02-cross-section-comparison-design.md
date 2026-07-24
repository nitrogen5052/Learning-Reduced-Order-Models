# Notebook 02 ROSE-LROM Cross-Section Comparison Design

## Goal

Create the second Paper Results Map notebook at
`notebooks/02_rose_vs_lrom_cross_sections.ipynb` and adapt
`notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb` into its formal
reproducibility benchmark.

Notebook 02 will compare ROSE and predictor LROM for differential elastic
cross sections generated from the ten-parameter complex Woods-Saxon optical
potential. It will use shared parameter rows, matched wavefunction-basis
sizes, independent method-native reduced bases, an LS-projected diagnostic,
and a paper-style Computational Accuracy versus Time (CAT) analysis.

The interactive HTML exporter requested as an optional later step in the
Paper Results Map is outside this milestone. It will be reconsidered only
after the notebook and benchmark pass scientific validation.

## Approved Scope

- Reaction: \(^{40}\mathrm{Ca}(n,n)\).
- Laboratory energy: \(14.1\) MeV.
- Optical potential: the ten-parameter full Woods-Saxon form used by the
  ROSE study.
- Retained partial waves: \(l=0,1,2,3\), expressed as \(l_{\max}=3\).
- Parameter domain: each parameter varies independently within a closed
  \(\pm20\%\) interval around its central value.
- Sampling: deterministic, independent Latin-hypercube training and testing
  designs.
- Notebook profile: 200 training rows and 100 testing rows.
- Reduced benchmark profile: 120 training rows and 30 testing rows.
- Full benchmark profile: exactly the Notebook 02 profile.
- Physical radial coordinate: every spatial array, basis, potential, and
  predictor visualization is mapped to \(r\) in fm.
- Methods shown: high-fidelity reference, LS-projected cross section, ROSE,
  and predictor LROM.
- Method excluded: linear LROM.
- User prose ownership: generated Markdown is limited to the title and nine
  section headings. The user will write the scientific narrative.

The reduced benchmark uses 120 training rows because the largest selected
LROM system has basis size 8 and predictor count 12. The fit has enough
sample equations when

\[
N_{\mathrm{train}} \ge K(n_\phi + 1) = 12(8+1)=108.
\]

Using 120 rows preserves this determined-system boundary while reducing the
held-out and training workload relative to the full profile.

## Repository Boundaries

Notebook 02 and benchmark 03 import `lrom_legacy.v2_0` explicitly because
cross-section and spin-orbit support remain parked in version 2.0. Public
`import lrom` continues to expose v1.2 and is not changed by this work.

The following remain unchanged:

- `lrom/__init__.py`;
- `lrom_legacy/v1_2/__init__.py`;
- `lrom_legacy/v2_0/__init__.py`, unless a separate scientific defect is
  demonstrated and the user approves a package change;
- Notebook 01;
- archived scientific references and legacy notebooks.

ROSE owns its native free-reference basis and its notebook-local EIM.
LROM owns its central-reference basis and maxvol potential predictors. Their
coefficients are not presented as directly interchangeable.

## Files

### Create

- `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
  - Sparse scientific presentation notebook.
  - Uses the full 200/100 profile.
  - Contains the requested results and figures without authored conclusions.
- `tests/test_notebook02_cross_sections.py`
  - Locks the Notebook 02 structure, scientific constants, method boundaries,
    figure markers, and code-cell compilation.

### Modify

- `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
  - Becomes the formal Notebook 02 reproducibility and timing benchmark.
  - Defaults to the reduced 120/30 profile.
  - Selects the full 200/100 profile when
    `LROM_BENCHMARK_PROFILE=full`.

The new `tests/test_notebook02_cross_sections.py` file owns both the
Notebook 02 and benchmark 03 contracts, including the reduced/full profile
switch, matched configuration grid, per-sample CAT metric, and code-cell
compilation.

No notebook generator is added. The notebook files remain the authoritative
artifacts so later user-authored Markdown cannot be overwritten by
regeneration.

## Notebook Structure

The notebook contains only these Markdown headings at delivery:

1. `# 02. ROSE and LROM Cross-Section Comparison`
2. `## 1. Physical Problem and Optical-Potential Parameters`
3. `## 2. Shared Training and Testing Samples`
4. `## 3. Potential Variation and LROM Predictor Locations`
5. `## 4. Equal-Basis ROSE and LROM Emulators`
6. `## 5. Representative Cross-Section Predictions`
7. `## 6. Cross-Section Error Distributions`
8. `## 7. Basis and Operator-Size Comparison`
9. `## 8. Accuracy Versus Online Time`
10. `## 9. Validation Summary`

Tests do not restrict later Markdown length or cell count. The sparse
delivery structure is a starting point for the user's prose, not a permanent
format constraint.

## Fixed Numerical Configuration

One configuration cell owns all notebook constants:

- target: `(40, 20)`;
- projectile: `(1, 0)`;
- laboratory energy: `14.1`;
- channels: `tuple(range(4))`;
- mesh size: `600`;
- angle grid: integer degrees from \(1^\circ\) through \(179^\circ\),
  inclusive;
- parameter half-width: `0.20`;
- notebook training/testing sizes: `200` and `100`;
- reduced benchmark training/testing sizes: `120` and `30`;
- basis sizes: `(4, 6, 8)`;
- ROSE EIM sizes: `(4, 8, 12)`;
- LROM predictor counts: `(4, 8, 12)`;
- default presentation configuration: basis size 6 and compression size 8;
- timing repeats: `3`;
- sampling seed: `1204`, from which v2.0 spawns independent training and
  testing random streams;
- Runge-Kutta absolute and relative tolerances: `1e-9`;
- relative-error denominator floor: `1e-12`;
- plotting floor: `1e-6`.

The CAT sweep is the full Cartesian product:

\[
n_\phi \in \{4,6,8\},\qquad
n_U \in \{4,8,12\},\qquad
K \in \{4,8,12\}.
\]

For every \(n_\phi\), ROSE evaluates all three \(n_U\) values and LROM
evaluates all three \(K\) values. A ROSE point and an LROM point are described
as equal-basis comparisons only when their \(n_\phi\) values match.
The notebook explicitly states that \(n_U\) and \(K\) are analogous
compression controls, not the same mathematical object.

## Data Flow

### 1. Parameter design

Construct the v2.0 LROM object with the fixed reaction and channel set.
Read its ordered central parameter vector. Build closed \(\pm20\%\) ranges
for every parameter, sorting endpoints so parameters with negative central
values remain valid.

Generate the training and testing Latin-hypercube designs with v2.0
`sampling(..., seed=1204)`. The sampler uses `SeedSequence.spawn(2)` to
create independent deterministic random streams. Reject exact row overlap.
Extract the ordered arrays once and pass those exact arrays to both methods.

ROSE EIM bounds span the complete training/testing domain, preserving the
project's approved benchmark convention. The held-out row values are not
used as ROSE wavefunction snapshots.

### 2. High-fidelity and reduced models

LROM sampling uses exact ROSE Runge-Kutta wavefunctions for channels
\(l=0..3\). It creates a separate central-reference basis for every retained
channel and uses maxvol-selected potential predictors from the central and
spin-orbit radial profiles.

The notebook constructs independent ROSE emulators inline. Each uses the
native free-reference basis, an `InteractionEIMSpace`, the same ordered
training rows, the same radial domain, and the selected \((n_\phi,n_U)\).

High-precision ROSE `exact_dsdo` evaluations at tolerance \(10^{-9}\) provide
the common training and testing cross-section reference.

### 3. LS-projected diagnostic

For each LROM basis size, project exact wavefunctions into the fixed LROM
bases channel by channel and assemble the resulting cross sections.

The notebook labels this method `LS-projected cross section`. Least squares
is optimal only for the chosen wavefunction norm. Once propagated through
the boundary-sensitive S-matrix and cross-section calculation, it is not a
guaranteed lower error bound for the observable. The notebook never labels
it a cross-section floor.

### 4. Representative samples

At the default configuration \((n_\phi,\mathrm{compression})=(6,8)\), rank
each held-out row separately by ROSE and LROM maximum-over-angle relative
error. Average the two ranks and select distinct rows nearest the 25th, 50th,
and 75th percentiles of combined difficulty.

All displayed methods use those identical row indices and parameter values.

### 5. Timings

Warm each trained emulator before measurement. Measure complete online
cross-section evaluations one sample at a time. Repeat each sample three
times and retain its minimum elapsed time to reduce scheduler noise.

Training, EIM construction, high-fidelity reference generation, plotting,
and file I/O are excluded from online timing. The validation summary records
the Python version and platform so timing results are not presented as
hardware-independent constants.

## Figures and Tables

Notebook 02 produces:

1. A full-potential rainbow for the central row plus 12 testing rows at
   equally spaced standardized-distance ranks, shown at \(l=0\) and \(l=3\),
   with selected central and spin-orbit predictor locations marked in
   \(r\) [fm].
2. Three representative cross-section panels for the selected 25th, 50th,
   and 75th percentile held-out rows. Each panel shows the high-fidelity
   reference, LS-projected cross section, ROSE, and LROM.
3. Matching pointwise relative-error panels for the same rows.
4. Training/testing split-violin distributions at the default
   configuration for LS-projected, ROSE, and LROM results.
5. Aligned basis/compression summary tables covering the full Cartesian
   grid.
6. A CAT point cloud with one point per held-out sample and configuration.
7. A compact validation table containing configuration, sample counts,
   median and maximum errors, and timing summaries.

The paper's 10% accuracy and one-million-evaluations-per-hour region is shown
only as a visual reference on the CAT plot. It is not a pass/fail gate.

## Error Metrics

For one reference cross section \(y(\theta)\) and approximation
\(\widehat y(\theta)\), define the stabilized pointwise relative error as

\[
e(\theta)=
\frac{|\widehat y(\theta)-y(\theta)|}
{\max(|y(\theta)|,10^{-12})}.
\]

Use:

- median pointwise error, \(\operatorname{median}_\theta e(\theta)\), for
  sample-level violin distributions and summary medians;
- paper-style CAT accuracy,
  \(\max_{\theta\in[1^\circ,179^\circ]}e(\theta)\), for every CAT point;
- the plotting floor \(10^{-6}\) only when rendering logarithmic axes.

Raw errors remain available without the plotting floor.

## Validation and Failure Behavior

Notebook execution stops with an assertion or exception when:

- a training row exactly matches a testing row;
- ROSE and LROM receive different ordered parameter arrays;
- a sample identifier no longer maps to the same row across methods;
- a comparison uses unequal wavefunction-basis sizes;
- a required channel in \(l=0..3\) is missing;
- a predictor radius is outside the physical radial mesh;
- a wavefunction, coefficient, cross section, error, or timing has an
  unexpected shape or a nonfinite value;
- a cross section is negative beyond floating-point roundoff;
- linear LROM enters a result collection, figure, or table.

Scientific accuracy is reported rather than forced. Tests do not require
ROSE or LROM to win, and they do not encode an expected error ordering
between LROM and the LS-projected cross section.

## Test Contract

`tests/test_notebook02_cross_sections.py` verifies:

- Notebook 02 exists at the required path;
- all ten headings are present;
- the profile constants are 200/100, \(\pm20\%\), and \(l=0..3\);
- the Cartesian configuration values are `(4, 6, 8)` and `(4, 8, 12)`;
- the notebook imports `lrom_legacy.v2_0` explicitly;
- ROSE EIM construction remains notebook-owned;
- no linear LROM label or result is present;
- required figure markers are present;
- the LS label is `LS-projected cross section`;
- every code cell compiles.

The benchmark contract verifies:

- reduced mode is the default;
- `LROM_BENCHMARK_PROFILE=full` selects 200/100 rows;
- reduced mode selects 120/30 rows;
- both profiles use \(\pm20\%\), \(l=0..3\), and the same Cartesian grid;
- CAT accuracy uses maximum-over-angle error;
- CAT timing remains per sample;
- every benchmark code cell compiles.

Runtime validation verifies:

- Notebook 02 completes at 200/100 with no error outputs;
- benchmark 03 completes in reduced mode with no error outputs;
- benchmark 03 completes once in full mode with no error outputs;
- all generated arrays pass the notebook assertions;
- focused notebook tests and the full available project test suite pass;
- all generated figures are visually inspected for labels, physical
  coordinates, sample alignment, clipping, and legibility.

## Acceptance Criteria

The milestone is complete when:

1. Notebook 02 exists under `notebooks/` with the approved sparse structure.
2. It executes the 200/100, \(\pm20\%\), \(l=0..3\) study.
3. ROSE and LROM use identical training/testing parameter rows.
4. Equal-basis comparisons and the full Cartesian grid are present.
5. Potential rainbows, predictor locations, representative cross sections,
   error distributions, grid summaries, and CAT point clouds are rendered.
6. The LS-projected result is described without claiming an observable-space
   lower bound.
7. Benchmark 03 supports reduced and full profiles and passes both execution
   modes.
8. Focused and full tests pass.
9. Figure inspection finds no scientific-label, coordinate, alignment, or
   rendering defect.
10. Public v1.2, parked v2.0 package code, Notebook 01, and archive sources
    remain unchanged.
