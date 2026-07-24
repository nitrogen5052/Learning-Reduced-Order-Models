# Notebook 02 ROSE-LROM Cross-Section Comparison Design

## Goal

Create the second Paper Results Map notebook at
`notebooks/02_rose_vs_lrom_cross_sections.ipynb` and adapt
`notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb` into its formal
reproducibility benchmark.

Notebook 02 will compare ROSE and predictor LROM for differential elastic
cross sections generated from the ten-parameter complex Woods-Saxon optical
potential. It will use shared parameter rows, one authoritative set of
high-fidelity training snapshots, matched wavefunction-basis sizes,
method-native reference states, an LS-projected diagnostic, and a paper-style
Computational Accuracy versus Time (CAT) analysis.

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

Notebook 02 follows Notebook 01's ROSE boundary: ROSE builds its basis from
the exact shared training wavefunctions but subtracts its own free reference.
LROM builds its basis from those same wavefunctions but subtracts the central
high-fidelity state. Sharing snapshots therefore aligns the high-fidelity
conditions without making the two reduced representations identical.

The installed ROSE observable helpers are not used because they interpret
`l_max` as an exclusive loop bound. With `L_MAX=3`, those helpers omit the
\(l=3\) channel even though ROSE constructed it. Notebook-local helpers
instead loop over every constructed row in `sae.rbes`, so the high-fidelity
reference, ROSE, LROM, and LS-projected cross sections all contain
\(l=0,1,2,3\).

## Files

### Modify

- `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
  - Corrects the ROSE construction and all-channel observable assembly while
    preserving the sparse Markdown structure and full 200/100 profile.
- `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
  - Applies the same correction to the formal reproducibility and timing
    benchmark while retaining its reduced/full profile switch.
- `tests/test_notebook02_cross_sections.py`
  - Replaces the obsolete `from_train` and truncating-observable expectations
    with shared-snapshot, free-reference, all-channel, and alpha-selection
    contracts.
- `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
  - Records the diagnosed ROSE truncation behavior, corrected data boundary,
    and validation evidence.

The focused test file owns both the
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

ROSE EIM construction uses `explicit_training=True` with the exact ordered
training rows. This avoids the library's otherwise unseeded internal
Latin-hypercube draw. Held-out rows are not used for EIM construction or
wavefunction-basis construction.

### 2. High-fidelity and reduced models

LROM sampling produces the authoritative exact ROSE Runge-Kutta
wavefunctions for every spin channel in \(l=0..3\). It creates a separate
central-reference basis for every retained channel and uses maxvol-selected
potential predictors from the central and spin-orbit radial profiles.

The notebook constructs ROSE emulators inline in the same explicit style as
Notebook 01. For every partial-wave and spin channel, it creates a ROSE
`CustomBasis` from the corresponding LROM-owned training snapshots, ROSE's
free solution on the shared \(\rho=kr\) mesh, the shared channel solver,
`use_svd=True`, `center=False`, and `scale=False`. The selected \(n_\phi\)
sets the number of retained vectors. A notebook-owned
`InteractionEIMSpace`, trained on the exact ordered training rows, supplies
the selected \(n_U\) operator representation.

Notebook-local `exact_smatrix_all_channels` and
`emulated_smatrix_all_channels` helpers allocate one entry for every row in
`sae.rbes`. For \(l=0\), the plus and minus entries are equal. For every
\(l>0\), index 0 supplies the \(j=l+1/2\) channel and index 1 supplies the
\(j=l-1/2\) channel. Both helpers pass their complete arrays to
`sae.calculate_xs`. High-precision exact evaluations at tolerance \(10^{-9}\)
provide the common training and testing cross-section reference.

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
error and average the two ranks. Select three distinct interior anchor
positions from the ordered combined ranks using
`np.linspace(0, len(ordered_combined) - 1, 5, dtype=int)[1:4]`.

The figures name these rows `alpha selection A`, `alpha selection B`, and
`alpha selection C`. A compact table lists the case identifier and all ten
parameter values for each selection. No representative title, variable,
test, or table uses rank-quantile terminology. All displayed methods use the
same selected row indices and parameter values.

### 5. Timings

Warm each trained emulator before measurement. Measure complete online
cross-section evaluations one sample at a time through the same explicit
all-channel assembly used for accuracy evaluation. Repeat each sample three
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
2. Three representative cross-section panels for alpha selections A, B, and
   C. Each panel shows the high-fidelity reference, LS-projected cross
   section, ROSE, and LROM.
3. Matching pointwise relative-error panels for the same rows.
4. A compact alpha-selection table containing the case identifiers and all
   ten physical parameter values.
5. Training/testing split-violin distributions at the default
   configuration for LS-projected, ROSE, and LROM results.
6. Aligned basis/compression summary tables covering the full Cartesian
   grid.
7. A CAT point cloud with one point per held-out sample and configuration.
8. A compact validation table containing configuration, sample counts,
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
- a ROSE basis is not built from the exact shared training snapshot array for
  its partial-wave and spin channel;
- a sample identifier no longer maps to the same row across methods;
- a comparison uses unequal wavefunction-basis sizes;
- a required channel in \(l=0..3\) is missing;
- an exact, ROSE, LROM, or LS S-matrix array does not contain four partial
  waves;
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
- ROSE `CustomBasis` construction consumes the shared LROM training
  wavefunctions and a free reference;
- ROSE's truncating `exact_dsdo` and `emulate_dsdo` paths are absent;
- exact and emulated ROSE observables loop over every constructed partial
  wave;
- no linear LROM label or result is present;
- required figure markers are present;
- the LS label is `LS-projected cross section`;
- representative labels use `alpha selection A/B/C`, and rank-quantile
  wording is absent;
- every code cell compiles.

The benchmark contract verifies:

- reduced mode is the default;
- `LROM_BENCHMARK_PROFILE=full` selects 200/100 rows;
- reduced mode selects 120/30 rows;
- both profiles use \(\pm20\%\), \(l=0..3\), and the same Cartesian grid;
- both profiles build ROSE bases from the same authoritative snapshots as
  LROM and assemble all four partial waves explicitly;
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

## Scientific Interpretation Boundaries

The correction does not change the approved study conditions. It retains
\(l_{\max}=3\), the 20% parameter ranges, 14.1 MeV laboratory energy, and the
parked v2.0 potential convention.

Those conditions differ from the published ROSE performance study, which
used more partial waves, a broader parameter box, and a 14 MeV
center-of-mass setup. The older ROSE tutorials and successful legacy LROM
script also use a different signed optical-parameter convention. Notebook 02
must record these as external-comparison caveats, but it must not silently
change them during this correction.

The acceptance criterion is internal scientific alignment: every method
uses identical alpha rows, exact snapshots, kinematics, mesh, and retained
partial waves. The benchmark reports the resulting accuracy without encoding
an expected method winner.

## Root-Cause Evidence

A controlled \(l=0..3\), 20% diagnostic reproduced the original failure.
ROSE's built-in observable path omitted \(l=3\) and differed from the
complete-channel reference by approximately \(8.55\times10^{-1}\) in the
median pointwise metric. With all methods assembled from the same four
channels, a rank-8 LS reconstruction achieved a median pointwise
cross-section error of \(3.43\times10^{-5}\) and a median
maximum-over-angle error of \(7.22\times10^{-4}\).

The scientific archive's successful cross-section implementation contains
the same remedy in `exact_smatrix_elements_fixed()` and its explicit LS and
predictor S-matrix loops. This design brings Notebook 02 and benchmark 03
back into that validated methodology while retaining the user's approved
study size.

## Acceptance Criteria

The milestone is complete when:

1. Notebook 02 exists under `notebooks/` with the approved sparse structure.
2. It executes the 200/100, \(\pm20\%\), \(l=0..3\) study.
3. ROSE and LROM use identical parameter rows and authoritative training
   snapshots while retaining their free- and central-reference bases.
4. Every high-fidelity, ROSE, LROM, and LS observable contains all four
   retained partial waves.
5. Equal-basis comparisons and the full Cartesian grid are present.
6. Potential rainbows, predictor locations, representative cross sections,
   error distributions, grid summaries, and CAT point clouds are rendered.
7. Representative cases are alpha selections A, B, and C, and their exact
   ten-parameter rows are shown.
8. The LS-projected result is described without claiming an observable-space
   lower bound.
9. Benchmark 03 supports reduced and full profiles and passes both execution
   modes.
10. Focused and full tests pass.
11. Figure inspection finds no scientific-label, coordinate, alignment, or
   rendering defect.
12. Public v1.2, parked v2.0 package code, Notebook 01, and archive sources
    remain unchanged.
