# v1.2-First LROM Package Simplification Design

**Status:** Architecture approved by the user on 2026-07-20, subject to the notebook-preservation contract below. The written specification still requires the user's review before implementation planning.

## Objective

Make the validated v1.2 wavefunction implementation the present public LROM package. Reduce code and avoid work that is not required for RF-LROM training or prediction, while preserving the scientific method, portable artifacts, and the Paper Results Map structure of notebook 01.

The implementation must make its abstraction boundaries readable in the package itself. Comments and docstrings must explain the numerical methodology, especially the two least-squares systems used by RF-LROM.

## Constraints

- Keep the active implementation in one Python file. Do not create a multi-module framework.
- Treat `scientific_archive/` as read-only.
- Keep all spatial interfaces in physical radius `r` measured in fm.
- Do not move plotting into package wrappers.
- Preserve notebook outputs for purely structural simplifications. A scientific correction may change a result only when its before/after effect is measured and documented.
- Preserve the save/load artifact workflow.
- The user owns the final sentence-by-sentence notebook prose review.
- Push each completed milestone directly to `main`.

## Package Version Routing

The public import will expose v1.2:

```text
import lrom  ->  lrom_legacy.v1_2
```

`lrom/__init__.py` becomes a small public entry point for the active v1.2 implementation. There will be one authoritative v1.2 code path rather than a copied implementation.

Before changing the public entry point, preserve the verified current single-file 2.0 implementation as `lrom_legacy.v2_0`. The obsolete modular v2.0 parts-donor files will then be removed. This keeps a future cross-section shell without retaining two different 2.0 implementations.

Notebook version routing will be explicit:

- Notebook 01 and the wavefunction benchmark import public `lrom` and therefore use v1.2.
- The 2.0 comparison and cross-section benchmarks import `lrom_legacy.v2_0` explicitly.
- No notebook silently changes package milestone.

## Active v1.2 Architecture

The single file will use stable numbered section headers that the user can reference:

1. Physical configuration and potential definitions
2. Parameter designs
3. State owned by each lifecycle stage
4. Exact ROSE high-fidelity boundary
5. Centered reduced basis
6. Predictor construction
7. RF-LROM training
8. RF-LROM prediction
9. Portable artifacts
10. Optional analysis utilities
11. Public `LROM` lifecycle

Each section header will state what the section owns and what it does not own. The code remains flat: small dataclasses hold state, and numerical work remains in functions that return NumPy arrays.

The predictive lifecycle remains:

```text
LROM(...) -> sampling(...) -> train(...) -> predict(...) -> save(...)
```

## Required Core

The simplification audit identified these as required:

- Physical configuration validation and KD central-parameter resolution
- Deterministic linspace, Latin-hypercube, and explicit parameter designs
- Mesh and high-fidelity snapshot state
- Exact ROSE Runge-Kutta wavefunction solves
- Central-reference SVD basis construction
- Parameter and potential predictor construction
- RF-LROM fit and online solve
- Wavefunction reconstruction
- Prediction state
- Portable save/load state and provenance
- Focused error types at public trust boundaries

## Candidate Reductions

The implementation plan will prove each removal with tests before deleting it. Expected reductions are:

- Remove `eim_basis_size` and EIM construction from sampling.
- Remove automatic LS baseline evaluation from `train()`.
- Remove benchmark-only LS fields from `TrainingState`.
- Remove `TestingResults`, `TestingCase`, `testing_errors`, and the stateful testing-result convenience properties if no non-notebook consumer remains after migration.
- Flatten the one-implementation `TrainingEngine` wrapper into named training functions.
- Remove the unused public `reduced_basis()` lifecycle path if no approved workflow requires a basis-only state.
- Consolidate repeated imports and blank generated spacing.
- Remove unsupported or unused flexibility only when repository-wide usage and a replacement path have been checked.

The implementation must not delete configuration validation, save/load, or state invalidation merely to reduce line count.

## Least-Squares Responsibilities

The existing code conflates two uses of least squares. They will be separated.

### Training coordinates: part of RF-LROM

For each training wavefunction, RF-LROM requires coordinates in the chosen affine basis:

\[
\phi(\alpha_i) \approx \phi_0 + \Phi a_i.
\]

The training coordinate solves

\[
a_i = \underset{a}{\operatorname{argmin}}\;
\left\|W^{1/2}\left[\phi(\alpha_i)-\phi_0-\Phi a\right]\right\|_2,
\]

where `W` contains trapezoid integration weights on the physical radius mesh. This calculation remains inside the RF-LROM training path because it defines the target reduced coordinates.

The code will use `numpy.linalg.lstsq` directly on the weighted basis matrix. It will not form normal equations such as `A.conj().T @ A`, because normal equations worsen conditioning by squaring the condition number.

### LS baseline: optional analysis

Applying the same projection to high-fidelity testing wavefunctions is a benchmark baseline, not LROM inference. `LROM.train()` will no longer calculate or store this baseline automatically.

An explicit analysis function will accept a basis and wavefunction array and return raw NumPy arrays for coordinates and reconstructed wavefunctions. Notebook cells will call it visibly when an LS comparison is required.

The notebook will identify the norm minimized by LS. A wavefunction least-squares projection will not be called a cross-section floor, because it does not optimize the asymptotic matching quantity or cross section.

## RF-LROM Fit Methodology

For predictors `p_j(alpha)` and known training coordinates `a(alpha)`, RF-LROM assumes

\[
\left(I + \sum_{j=1}^{K}p_j(\alpha)M_j\right)a(\alpha)
= \sum_{j=1}^{K}p_j(\alpha)b_j.
\]

Once the training coordinates and predictors are known, the unknown entries of every `M_j` and `b_j` appear linearly. The implementation stacks all sample/equation rows into one complex least-squares system and solves it once with `numpy.linalg.lstsq`.

Comments beside the implementation will explain:

- the shape of the stacked design matrix;
- why the transformed equation is linear in the unknown operator entries;
- why one solve is preferable to fitting each matrix element independently;
- why `lstsq` is used instead of matrix inversion or normal equations;
- how the fitted flat vector is unpacked into `M_j` and `b_j`;
- how online prediction solves only the small reduced system.

The implementation will favor clear loops for the small reduced dimensions unless a vectorized replacement is both shorter and verified faster. FOM integration, not the four-dimensional reduced algebra, is the dominant cost.

## EIM Removal From Sampling

Current sampling constructs `rose.InteractionEIMSpace`, including a 1,000-snapshot potential ensemble and SVD, even though ROSE's Runge-Kutta equation calls the original coordinate-space potential through `interaction.v_r`.

The package high-fidelity boundary will instead construct `rose.InteractionSpace`. This represents the exact coordinate-space interaction required by the Runge-Kutta solve and removes `eim_basis_size` from both v1.2 and the parked 2.0 shell.

Controlled evidence already obtained for the v1.2 ws_3 setup at mesh 800:

- `InteractionEIMSpace` construction: `0.38782125 s`
- `InteractionSpace` construction: `0.000028375 s`
- construction speedup: approximately `13,668x`
- maximum wavefunction difference: `0.0`
- wavefunctions bitwise equal: yes

ROSE reduced emulators still require EIM for non-affine potentials. Benchmark notebooks will construct their own `InteractionEIMSpace` explicitly, using the methodology appropriate to that comparison. EIM will therefore belong to ROSE benchmarking, not LROM sampling.

## Notebook 01 Preservation Contract

Notebook 01 will continue to follow `scientific_archive/prior_designs/specs/2026-06-16-notebook01-rbm-lrom-design.md` and the Paper Results Map it records. Package simplification must not redesign the notebook.

The following narrative remains fixed:

### Act 1: Vv-only teaching case

1. Scientific setup for the real Woods-Saxon, `l=0` problem
2. Explicit central parameter point and central-reference `phi_0`
3. Vv potential rainbow
4. Vv high-fidelity wavefunction rainbow
5. Reduced basis with singular-value decay
6. Explicit LS baseline
7. ROSE/RBM and LROM coordinate behavior
8. Representative noncentral wavefunction reproduction
9. Testing-error summary

### Act 2: Vv/Rv/av predictor motivation

1. Three-parameter sample definition
2. Visible Vv, Rv, and av variation, including the requested av rainbow
3. Explanation of why raw parameter predictors are insufficient where supported by the results
4. Potential-ensemble spectrum and maxvol-style selected radii
5. Reduced bases with singular-value decay
6. Parameter-based coefficient diagnostics, colored by a second parameter
7. Interpolation and extrapolation wavefunction performance
8. Notebook takeaway leading toward cross sections

Details currently missing or incomplete will be restored rather than replaced:

- Fixed `Rv` and `av` values in ws_1, with KD global-systematics provenance at 14.1 MeV
- Plain comments explaining each public package call
- Narrative before figures and short section summaries
- A noncentral representative Vv case
- Parameter-colored multi-parameter coefficient plots instead of case-index-only plots
- Definition of `alpha`, predictor notation indexed by `j` through `K`, median markers, and case counts
- Consistent method colors and physical-radius axes

## Notebook Simplification Rules

Notebook simplification means removing incidental plumbing, not scientific content.

Allowed:

- Use the public v1.2 entry point instead of version-routing boilerplate.
- Replace stateful automatic LS access with one explicit LS analysis cell.
- Consolidate repeated array extraction and repeated scalar setup.
- Remove duplicate calculations whose outputs are already available.
- Keep short notebook-local numerical setup helpers when they expose, rather than hide, a scientific object.

Not allowed:

- Delete or merge project-map sections solely to reduce cell count.
- Remove required figures.
- Move plots into opaque package functions.
- Replace visible ROSE methodology with a one-call package wrapper.
- Change samples, rank, solver tolerances, or scientific comparisons during a formatting-only simplification.

The generator remains the source of notebook 01. Generated notebook JSON will not be edited directly.

## Other Notebooks

- `benchmark_notebooks/1.0/benchmark_02.ipynb`: use public v1.2; preserve its rebuilt native-ROSE methodology and numerical results.
- `benchmark_notebooks/2.0/benchmark_01.ipynb`: explicitly compare parked v2.0 with v1.2; preserve its parity purpose.
- `benchmark_notebooks/2.0/benchmark_03.ipynb`: explicitly import parked v2.0; preserve the cross-section CAT benchmark.

Simplification of these notebooks is limited to import routing, removed obsolete package arguments, duplicated setup, and explicit analysis calls. Scientific sections and results remain intact.

## Verification

Implementation will be test-first and milestone-based.

Required checks:

1. Characterize current v1.2 wavefunction, coefficient, predictor, and artifact arrays before refactoring.
2. Prove exact-interaction FOM wavefunctions match the prior Runge-Kutta outputs.
3. Prove pure RF-LROM refactoring preserves fitted matrices, vectors, predictions, and saved/loaded predictions.
4. Prove `sampling()` no longer exposes or constructs EIM.
5. Prove `train()` does not calculate testing LS baselines.
6. Prove the explicit LS baseline minimizes the documented wavefunction norm relative to other coefficients in the same basis.
7. Prove public `lrom` resolves to v1.2 and parked v2.0 remains explicitly importable.
8. Generate and execute all affected notebooks with zero error outputs and zero unexecuted code cells.
9. Compare numerical notebook summaries against pre-refactor values for every change classified as structural.
10. Visually inspect all changed figures against the Notebook 01 preservation contract.
11. Run Ruff on the package trees and run the full project test suite.
12. Update `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`, `docs/VERSIONING.md`, project `CLAUDE.md`, and workspace memory with the final boundaries and measured before/after evidence.

## Ownership Handoff

For every functional change, the handoff will state:

1. Methodology
2. Previous behavior
3. New behavior
4. Execution evidence
5. What did not change

The final notebook prose will remain pending until the user completes the sentence-by-sentence ownership pass.
