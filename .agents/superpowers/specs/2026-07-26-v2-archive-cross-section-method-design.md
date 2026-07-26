# V2 Archive Cross-Section Method Design

## Goal

Update the explicitly imported `lrom_legacy.v2_0` implementation and the
second Paper Results Map notebook so predictor LROM uses the stronger
cross-section method preserved in the scientific archive.

The update must improve LROM over the current v2 result on identical alpha
rows and make LROM beat or closely match ROSE at representative
configurations. LROM is not required to beat ROSE for every basis and
compression combination.

## Approved Scope

- Preserve the corrected \(^{40}\mathrm{Ca}(n,n)\) problem at 14.1 MeV
  laboratory energy.
- Preserve \(l=0,1,2,3\), expressed as \(l_{\max}=3\).
- Preserve independent closed ±20% parameter ranges.
- Preserve the deterministic full 200/100 training/testing profile and the
  reduced 120/30 benchmark profile.
- Update the parked `lrom_legacy.v2_0` package.
- Update `notebooks/02_rose_vs_lrom_cross_sections.ipynb`.
- Update
  `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`.
- Add focused package and notebook tests.
- Keep Notebook 01 and public v1.2 unchanged.
- Keep every user-facing spatial coordinate in physical radius \(r\) in fm.
- Keep notebook Markdown sparse so the user can write the scientific
  narrative.

The scientific archive is read-only evidence. Its notebooks and helper code
are not modified.

## Archive Evidence

The archive cross-section workflow differs from current v2 in four material
ways:

1. Each \((l,j)\) channel selects predictors from its own centered effective
   interaction \(\widetilde U_{lj}\), including its \(l\cdot s\) term.
2. Predictor selection starts with QR-residual row selection and then applies
   a max-volume swap refinement.
3. The identity-\(M_0\) RF fit learns a constant source vector \(b_0\).
4. Online cross-section prediction solves packed channel systems and maps
   reduced coefficients directly to \(S_l^\pm\), without reconstructing full
   radial wavefunctions.

The archive CAT result uses the median pointwise relative cross-section error
over angle. Its published execution profile used \(l_{\max}=10\), 100/80
training/testing rows, a 700-point mesh, and older optical-potential sign
conventions. Those numerical outputs are therefore evidence for the method,
not target values for the current \(l_{\max}=3\), ±20% experiment.

## Repository Boundaries

### Changed

- `lrom_legacy/v2_0/__init__.py`
  - channel effective-interaction predictor state;
  - refined max-volume selection;
  - identity-\(M_0\) RF fit with \(b_0\);
  - packed coefficient and \(S\)-matrix evaluation;
  - cross-section-only prediction mode;
  - portable artifact support.
- `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
  - archive-method v2 construction;
  - ablation results;
  - representative cross sections;
  - archive-style CAT result;
  - separate robustness diagnostic.
- `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
  - reduced/full reproducibility mirror of Notebook 02.
- Focused tests under `tests/`.
- `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`
  - final method boundary and verified results.

### Unchanged

- `lrom/__init__.py`;
- `lrom_legacy/v1_2/__init__.py`;
- `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`;
- archived scientific notebooks and code;
- the physical alpha rows, samples, mesh, angles, and partial-wave range of
  the corrected Notebook 02 comparison.

No package split is introduced. V2 remains a flat functional implementation
in its existing single file.

## ROSE and LROM Separation

ROSE reduced-model comparison remains notebook-owned.

The v2 package may use the installed ROSE library for:

- exact high-fidelity scattering solves during sampling;
- the physical interaction objects whose `tilde` method defines
  \(\widetilde U_{lj}\);
- asymptotic and cross-section assembly already required by the nuclear
  scattering backend.

The v2 package must not:

- build a ROSE reduced basis for comparison;
- tune ROSE EIM size;
- time a ROSE reduced emulator;
- calculate ROSE comparison errors;
- select the displayed ROSE result.

Notebook 02 and benchmark 03 construct their own free-reference ROSE bases
and EIM models from the shared exact snapshots. The notebook times ROSE and
LROM through separate functions. Shared alpha rows and cross-section assembly
make the comparison controlled without merging their workflows.

## Effective-Interaction Predictors

### State

Add a channel container whose keys match the existing trained-channel keys:

- `0` for the single \(l=0\) channel;
- `(ell, 0)` and `(ell, 1)` for the two spin channels at \(l>0\).

Each value is a `PredictorState` with:

- `kind="effective-interaction"`;
- selected mesh indices;
- selected radii in fm;
- centered effective-interaction values;
- maximum absolute training scales;
- training and testing features;
- singular values.

The operator-grid points \(\rho=kr\) remain internal. User-facing state and
figures expose only the corresponding physical radii in fm.

### Construction

For each channel:

1. Evaluate `ChannelFOM.interaction.tilde` on the sampled \(\rho\) mesh for
   the central alpha and every training alpha.
2. Form
   \[
   \Delta\widetilde U_{lj}(\rho,\alpha_i)
   =
   \widetilde U_{lj}(\rho,\alpha_i)
   -
   \widetilde U_{lj}(\rho,\alpha_0).
   \]
3. Restrict candidates to physical radii \(r\geq0.5\) fm.
4. Compute the thin SVD of the allowed centered interaction matrix.
5. Select `predictor_count` initial rows with QR residual selection.
6. Apply at most 50 max-volume swaps, stopping when the largest unselected
   coefficient is no greater than \(1+10^{-10}\).
7. Normalize each selected value by
   \[
   s_k=\max_i\left|
   \Delta\widetilde U_{lj}(\rho_k,\alpha_i)
   \right|,
   \]
   with a numerical floor of \(10^{-14}\).

This calculation naturally includes the spin-orbit contribution and its
channel-dependent \(l\cdot s\) factor.

### Compatibility

Existing `"parameters"` and `"potential"` predictors remain available.
The new method is selected explicitly with
`predictor="effective-interaction"`. It becomes the Notebook 02 and benchmark
03 path but does not silently change the older v2 wavefunction benchmark.

Requesting effective-interaction predictors without sampled nuclear
interaction state raises a clear `LROMStateError`. An infeasible predictor
count, insufficient allowed radii, a zero training scale, or a missing
channel raises a deterministic validation error; there is no fallback to a
shared central/spin-orbit predictor.

## Identity-\(M_0\) RF Fit

Keep the current reduced coordinate definition:

\[
\phi(\alpha)\approx\phi_0+Va(\alpha).
\]

Add an intercept-aware RF fit:

\[
\left(I+\sum_{k=1}^{K}p_k(\alpha)M_k\right)a(\alpha)
=
b_0+\sum_{k=1}^{K}p_k(\alpha)b_k.
\]

The fit fixes \(M_0=I\) and solves one complex least-squares system for:

- \(M_1,\ldots,M_K\);
- \(b_0,b_1,\ldots,b_K\).

`RFLROMModel` gains a constant vector. Existing fits receive an exact zero
constant vector, while the new effective-interaction path calls the
intercept-aware fit. The online solve adds the constant vector to the right
hand side before applying the predictor-dependent terms.

The package tests use synthetic coefficients generated from a known
intercept-bearing reduced equation. They must show that the new fit recovers
those coordinates and that a zero-intercept model reproduces the current
solve.

## Packed Online Cross-Section Path

### Data flow

For a batch of alpha rows:

1. Evaluate the selected effective interaction values for every channel.
2. Form a `(cases, channels, predictors)` feature array.
3. Assemble every reduced matrix and right-hand side with batched
   `numpy.einsum`.
4. Solve the `(cases, channels)` stack of small complex systems with
   `numpy.linalg.solve`.
5. Append the implicit central coefficient to each reduced coordinate row.
6. Contract the coefficients with cached asymptotic basis values and
   derivatives.
7. Produce packed \(S_l^+\) and \(S_l^-\) arrays.
8. Assemble differential cross sections on the configured physical angle
   grid.

Add an explicit prediction option to omit full radial reconstruction when
only the observable is required. Coefficients, \(S\)-matrices, and cross
sections remain available; the wavefunction mapping is empty for this mode.
The normal wavefunction prediction behavior remains the default.

The CAT timer uses the observable-only mode and includes all work from an
input alpha row through its differential cross section.

### Equivalence

For deterministic validation rows:

- packed coefficients must match per-channel RF solves;
- packed \(S_l^\pm\) must match the existing coefficient-to-\(S\)-matrix
  calculation;
- packed cross sections must match the existing cross-section assembly;
- observable-only and full-reconstruction prediction must return the same
  coefficients and observables.

Runtime is reported but is not used as a fragile unit-test assertion.

## Portable Artifacts

Bump the parked v2 artifact schema and save:

- the predictor strategy;
- every channel predictor state;
- every RF constant vector;
- data needed by the packed observable path.

Loading the preceding schema assigns a zero constant vector and restores its
single shared predictor state. New channel-predictor artifacts must reproduce
pre-save features, coordinates, \(S\)-matrices, and cross sections.

## Notebook Results Map

Notebook Markdown remains limited to the title and short section headings.
Code and outputs follow this structure:

1. Physical conditions and deterministic alpha design.
2. Shared high-fidelity snapshots and reference cross sections.
3. Full potential variation and channel predictor radii in fm.
4. Separate ROSE and archive-method v2 construction.
5. Old-v2 versus archive-method ablation table.
6. Representative cross-section comparisons for named alpha selections A,
   B, and C.
7. Held-out error distributions.
8. Basis/compression grid and CAT plot.
9. Validation, timing, and method-boundary table.

The representative figures contain exact, LS floor, notebook-owned ROSE, and
archive-method LROM curves. Their companion error panels use the same alpha
selection and angle grid.

## Metrics and CAT Plot

For each held-out alpha row, define:

\[
e_{\mathrm{point}}(\alpha)
=
\operatorname{median}_{\theta}
\frac{|\sigma_{\mathrm{ROM}}(\theta,\alpha)
-\sigma_{\mathrm{FOM}}(\theta,\alpha)|}
{\max(|\sigma_{\mathrm{FOM}}(\theta,\alpha)|,10^{-12})}.
\]

The primary CAT coordinate is:

- x: median complete online time per alpha row;
- y: median of `e_point` across held-out alpha rows.

The maximum-over-angle relative error remains a separate robustness
diagnostic because diffraction minima can strongly amplify it. It is not the
primary CAT y-coordinate.

The CAT sweep remains:

- basis sizes `(4, 6, 8)`;
- ROSE EIM sizes `(4, 8, 12)`;
- LROM predictor counts `(4, 8, 12)`.

ROSE `n_U` and LROM `K` remain method-specific compression controls. Equal
numeric values are not described as equal mathematical complexity.

## Benchmark and Tests

### Package tests

- refined max-volume selection is deterministic and returns unique allowed
  rows;
- every selected public radius is at least 0.5 fm;
- channel features equal direct `interaction.tilde` evaluations;
- \(l>0\) spin channels receive distinct effective-interaction features;
- intercept-aware fitting recovers a known synthetic RF system;
- zero-intercept solving preserves the existing result;
- packed and per-channel coefficients agree;
- packed and scalar \(S\)-matrices agree;
- observable-only and full prediction observables agree;
- old and new artifact schemas load correctly.

### Notebook contract tests

- Notebook 01 hash remains unchanged;
- Notebook 02 and benchmark 03 import parked v2 explicitly;
- ROSE comparison construction exists only in notebook code;
- both notebooks use `predictor="effective-interaction"`;
- both retain \(l=0,1,2,3\), ±20%, shared rows, and explicit all-channel
  assembly;
- both contain predictor, representative cross-section, error-distribution,
  CAT, and validation outputs;
- CAT uses the median pointwise metric;
- maximum-over-angle error remains separately named;
- every code cell compiles.

### Numerical acceptance

On identical deterministic alpha rows:

- archive-method LROM must improve on the current shared-potential v2 result
  at the default `(basis_size, predictor_count)=(6, 8)` configuration;
- the grid must contain representative configurations where archive-method
  LROM beats or closely matches ROSE;
- packed and full LROM results must be numerically equivalent;
- no claim is made that LROM wins every grid point.

Thresholds are recorded only after an initial controlled execution and are
then fixed in the benchmark so later regressions cannot pass through a
relative-to-current comparison.

## Change Ledger

The final handoff will enumerate every modified file and report:

- the previous behavior;
- the new behavior;
- why the change was needed;
- the test or notebook output that validates it;
- measured before/after LROM accuracy and online time;
- representative ROSE comparison results;
- confirmation that public v1.2, Notebook 01, and the scientific archive
  remained unchanged.
