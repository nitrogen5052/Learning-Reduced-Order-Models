# Parked v2 Fast-Packed Cross-Section Design

## Status

Approved in conversation on 2026-07-27. This document records the design boundary before an implementation plan is written.

## Goal

Make the parked `lrom_legacy.v2_0` cross-section online path materially faster than the separate notebook-owned ROSE comparison by applying the scientific archive's fast-packed execution pattern, while preserving the current coefficients, channel \(S\)-matrix elements, cross sections, accuracy summaries, and public lifecycle.

## Evidence and Root Cause

The current \(l=0\ldots3\), \(n_\phi=6\), \(K=8\) online path was measured after warm-up:

| Stage | Current v2 | Archive-style diagnostic |
|---|---:|---:|
| Effective-interaction predictor evaluation | 32.5 microseconds | 17.8 microseconds |
| Packed reduced-coordinate solve | 33.9 microseconds | 23.2 microseconds |
| Packed coefficient-to-\(S\)-matrix conversion | 29.6 microseconds | 8.1 microseconds |
| Cross-section assembly | 26.2 microseconds | 4.5 microseconds |
| Complete online prediction | 153.7 microseconds | 58.2 microseconds |

The diagnostic fast path agreed with the current cross section to a maximum absolute difference of \(4.6\times10^{-13}\).

The current path repeatedly:

- dispatches effective-interaction evaluation through individual channel objects;
- gathers RF matrices, vectors, constants, and channel features;
- gathers asymptotic and Hankel arrays from ROSE basis objects;
- recreates channel-to-\(S_{lj}\) placement metadata;
- passes the fixed angle grid explicitly, causing ROSE to recompute angular tables; and
- allocates general lifecycle and result objects around a very small numerical kernel.

The archive implementation compiles the channel-independent arrays once, evaluates all channel predictors in one flattened call, solves all reduced systems as one batch, and converts coefficients to \(S_{lj}\) using prepacked arrays.

## Archive Interpretation

The saved scientific-archive demo reports fast-packed LROM times of 157-174 microseconds per sample and ROSE times of 518-761 microseconds per sample, a roughly 3-5x advantage. The May 2026 report gives representative 2.6-3.3x comparisons. A presentation describes the speedup as "almost an order of magnitude"; the separate "almost two orders" statement refers to accuracy at the same basis size.

Benchmark 03 intentionally uses \(l=0\ldots3\), which produces seven channels. The archive study used \(l=0\ldots10\), which produces 21 channels. Under the smaller current study, ROSE already measures about 119-153 microseconds per sample, so the expected post-optimization separation is closer to 2x. Archive-condition \(l=10\) timing must be reported separately and must not replace the user-selected \(l=3\) scientific study.

## Selected Approach

Use a private, lazily compiled collection of raw NumPy arrays inside the existing single-file parked-v2 implementation.

This approach is selected over:

1. a notebook-only optimized function, which would not measure the real package API and would duplicate LROM logic outside the package; and
2. JIT-compiling the entire prediction stack, which would add compilation behavior and maintenance complexity before the known packing overhead is removed.

The implementation remains flat and functional. It does not introduce a new public class hierarchy, split the single-file package, or add a dependency.

## Architecture

### Packed online state

A private compiler will collect the immutable arrays needed after training:

- ordered channel keys;
- flattened predictor evaluation coordinates;
- channel \(\ell\), spin index, and \(l\cdot s\) values;
- predictor centers and scales;
- RF matrices, vectors, and learned constant vectors;
- one reduced-space identity matrix;
- asymptotic basis values and derivatives;
- \(H^\pm\), derivative values, and \(s_0\);
- channel-to-`splus` and channel-to-`sminus` placement indices; and
- the initialized ROSE scattering-amplitude emulator with its precomputed angle tables.

The cache is compiled after cross-section training. A loaded schema-2 inference artifact compiles it lazily on its first cross-section prediction. Sampling and retraining invalidate it. No packed arrays are persisted, because they are derived from existing schema-2 state; the artifact schema does not change.

### Flattened predictor evaluation

For the registered `full_woods-saxon` potential, all channel/radius predictor pairs will be evaluated together with the existing NumPy Woods-Saxon functions. The channel spin-orbit factors are applied as a vector.

The compiler must use the exact physical evaluation locations implied by the current ROSE interaction momentum, so the optimized path reproduces current numerical behavior. Other potential registrations retain the existing generic interaction-dispatch fallback.

### Batched reduced solve

The online function will:

1. evaluate the flattened effective interactions;
2. reshape and normalize them into `(samples, channels, K)`;
3. assemble all `(samples, channels, n_phi, n_phi)` systems;
4. assemble all right-hand sides; and
5. call one batched `numpy.linalg.solve`.

It preserves the current learned intercept and identity-\(M_0\) equation.

### Direct packed \(S\)-matrix conversion

The coefficient tensor will be augmented with the reference coefficient and contracted with prepacked asymptotic values and derivatives. The resulting channel \(S\)-matrix values will be placed into `splus` and `sminus` using fixed index arrays rather than rebuilding ROSE-object lists.

### Cross-section assembly

The LROM and notebook-owned ROSE timing paths will use the angle grid already stored in their initialized scattering-amplitude emulators. Neither timing path will pass the unchanged grid as a new `angles` argument, so precomputed Legendre tables are reused fairly.

ROSE remains a separate notebook workflow. No ROSE reduced-emulator object, basis, or EIM state is moved into the LROM package.

## Public and Scientific Boundaries

The following remain unchanged:

- public `import lrom` and frozen public v1.2;
- `LROM(...)`, `sampling()`, `train()`, `predict()`, `save()`, and `load()` lifecycle meaning;
- Notebook 01;
- scientific archive contents;
- Ca40(n,n), 14.1 MeV lab energy, \(l=0\ldots3\), and the 600-point mesh;
- the user-selected +/-20% alpha ranges;
- seed 1204 and all training/testing alpha rows;
- basis sizes `(4, 6, 8)`;
- ROSE `n_U=(4, 8, 12)` and LROM `K=(4, 8, 12)`;
- explicit all-channel cross-section assembly;
- alpha selections A, B, and C;
- ROSE free-reference basis convention; and
- schema-2 artifact compatibility.

No notebook prose expansion is required. Existing sparse comments remain user-owned.

## Deferred Momentum/Radius Issue

The diagnosis found that the package mesh reports \(k=0.8073091\ \mathrm{fm}^{-1}\), while the ROSE interaction evaluates its potential with \(k=0.8102365\ \mathrm{fm}^{-1}\). This 0.36% difference shifts some displayed predictor radii from their actual interaction-evaluation radii by as much as 0.0246 fm.

This design deliberately preserves current evaluation behavior. Correcting the kinematic mapping could change predictors and scientific outputs, so it requires a separate physics-validation design and is not part of the speed implementation.

## Error Handling and Fallbacks

- The packed path is used only for trained cross-section models with consistent channel basis sizes and predictor counts.
- Inconsistent array shapes or missing cross-section state fail with the existing `LROMStateError`/`ValueError` style before numerical execution.
- Unsupported/custom potentials continue through the current generic predictor path.
- Wavefunction reconstruction behavior remains available and unchanged; the packed observable-only path is used when `reconstruct_wavefunctions=False`.
- Cache compilation failures do not silently change the predicted method.

## Verification

### Numerical parity

Tests will compare the existing reference operations and optimized operations for:

- normalized effective-interaction features;
- reduced coordinates;
- `splus` and `sminus`;
- differential cross sections;
- multi-sample prediction;
- trained and loaded schema-2 artifacts; and
- every \((n_\phi,K)\) configuration used by Notebook 02.

The target is maximum cross-section absolute difference no larger than \(10^{-10}\), with tighter array-level tolerances where stable.

### Scientific regression

Notebook 02 and Benchmark 03 will be re-executed. Their alpha rows, representative selections, accuracy arrays, and validation tables must remain unchanged within numerical tolerance. Only timing values and the CAT x-coordinates may change.

### Performance evidence

Timing will use warmed models and repeated inner calls to reduce scheduler and allocation noise. Performance is recorded rather than enforced by a brittle CI wall-clock assertion.

Expected evidence on the current machine:

- default \(l=3\), \(n_\phi=6\), \(K=8\) LROM near 60-75 microseconds per sample;
- LROM faster than the comparable notebook-owned ROSE configurations for the current study; and
- a separate \(l=10\) archive-condition diagnostic showing a larger separation, with the historical 3-5x range used as context rather than a guaranteed threshold.

## Files Expected to Change

- `lrom_legacy/v2_0/__init__.py`
- `tests/test_v2_archive_method.py`
- `tests/test_notebook02_cross_sections.py`
- `notebooks/02_rose_vs_lrom_cross_sections.ipynb`
- `notebooks/benchmark_notebooks/2.0/benchmark_03.ipynb`
- `docs/LROM_ARCHITECTURE_UNDERSTANDING.md`

No public-v1.2, Notebook 01, or scientific-archive file will change.
