# Benchmark N1 Notebook Design

(Reconstructed 2026-07-12 from the reviewing agent's session context after the
original file was lost to iCloud eviction; content is a faithful restoration
of the approved 2026-06-30 design. Post-dating note: "N1" is version 1.0 and
the notebook now lives at `notebooks/Benchmark_1.0.ipynb` with the snapshot
package importable as `lrom_legacy.v1_0`.)

## Purpose

Create `notebooks/Benchmark_N1.ipynb` as a standalone scientific
reconstruction of the figures in
`scientific_archive/legacy_code/Legacy_benchmark/notebooks/02_lrom_method_walkthrough.ipynb`.

The new notebook must name that archived notebook in its introduction. It
recreates the scientific studies and visual presentation through the public
`lrom_legacy.N1` API; it does not copy the archived implementation code.

## Package Boundary

The notebook imports the snapshot package and the public ROSE package as
separate scientific implementations:

```python
import rose
import lrom_legacy.N1 as n1
```

It must not import: the active `lrom` package; `lrom_bench`;
`scientific_archive...lrom_demo`; any archived notebook helper or
repository-local plotting helper.

All plotting is written inline in the figure's notebook cell with Matplotlib.
The notebook must not define plotting functions such as `plot_*`,
`split_violin`, or a generic scan-plotting abstraction. Each figure cell shows
its complete `plt.subplots`, plotting calls, labels, scales, legend, layout,
and display code. No external notebook generator or helper script owns the
figures.

## N1 And ROSE Ownership

`lrom_legacy.N1` may depend on ROSE for functionality required by N1 itself:
full-order solver construction, full-order wavefunction sampling, and
construction of the centered wavefunction basis used by N1.

N1 must not construct, evaluate, retain, expose, or serialize a ROSE reduced
emulator. The following N1 concepts are removed: `RoseRBMState`;
`TrainingState.rose_rbm`; `LROM.rose_rbm`; ROSE entries in training/testing
coefficients, wavefunctions, and metrics; `TestingCase.rose`; serialized
`rose_rbm` state.

The notebook owns the comparison ROSE ROM. It imports `rose` directly and,
inside each study cell, explicitly constructs its own `InteractionEIMSpace`,
base solver, and `ScatteringAmplitudeEmulator.from_train(...)`. It uses the
same physical parameter rows, basis size, and mesh resolution as N1, but it
does not reuse N1's solver, central wavefunction, basis, or reduced
coordinates. ROSE retains its native free-reference basis convention.
ROSE-ROM coefficients, wavefunctions, and errors exist only as notebook
variables.

ROSE and N1 coefficients are never subtracted from each other because they
use different origins and bases. Every coefficient section keeps two explicit
coordinate comparisons: ROSE ROM versus ROSE least-squares in the
free-reference ROSE basis; N1 LROM versus N1 least-squares in the
central-reference N1 basis. Wavefunctions and their errors remain directly
comparable on the shared physical radial mesh.

## Scientific Studies

Four studies with explicit, deterministic training and testing designs:

1. A one-parameter real-volume-depth (`Vv`) scan.
2. A one-parameter real-volume-radius (`Rv`) scan.
3. A broad two-parameter `Vv`/`Rv` box that exposes the limits of raw linear
   parameter predictors.
4. A three-parameter Woods-Saxon study using potential predictors.

## Approved Figure Set (16 figures)

Vv-only: rainbow; coefficient diagnostics; wavefunction errors.
Rv-only: rainbow; coefficient diagnostics; wavefunction errors.
Broad Vv/Rv: coefficient trajectories; coefficient errors; wavefunction errors.
Potential predictors: selected points; scan rainbows with points; predictor values.
Multiparameter: coefficient trajectories; coefficient errors; wavefunction
errors; split train/test violin.

## Visual Contract

Stay close to the archived figures in organization, axes, labels, units,
titles, legends, line-style encodings, log error scales, color progression,
and split-violin styling. Exact pixel identity is not required.

All eight coefficient-difference and wavefunction-error figures use a visible
plotting floor of `1e-5`: values clipped with
`np.maximum(values, DIFFERENCE_FLOOR)` and log axes use
`bottom=DIFFERENCE_FLOOR`. Underlying arrays are not modified.

(2026-07-09 addendum: global scan windows `VV_SCAN_LIMITS = (30, 70)` MeV and
`RV_SCAN_LIMITS = (3, 5)` fm plus a shared `ERROR_YMAX = 1e1` axis top were
approved and added to the shared configuration cell.)

## Verification

Valid JSON; all code cells compile; separate `rose` and N1 imports; forbidden
imports absent; archived path named in the introduction; 16 inline Matplotlib
figure constructions; no plotting-function definitions; no ROSE workflow
helper; coefficient plots compare only within one basis convention; focused
tests lock in structure, package boundary, and the removed ROSE-ROM state;
executed end-to-end when runtime permits.

## Non-Goals

Comparing N1 against the active `lrom` package or `lrom_bench`; reproducing
archived helper code line-for-line; importing precomputed figures or arrays;
retaining hidden ROSE-ROM evaluation inside N1; a separate adapter package;
modifying the archived notebook.
