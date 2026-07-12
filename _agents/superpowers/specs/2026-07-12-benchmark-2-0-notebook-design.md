# Benchmark 2.0 Notebook Design

## Purpose

Create `notebooks/Benchmark_2.0.ipynb` as a standalone scientific
reconstruction of the cross-section studies in
`scientific_archive/legacy_code/Legacy_benchmark/notebooks/03_cross_section_cat_comparison.ipynb`.

The notebook must name that archived notebook in its introduction. It
recreates the scientific studies and visual presentation through the public
**version 2.0 `lrom` package** (the active package, `lrom.__version__ ==
"2.0.0"`); it does not copy the archived implementation code and does not
load the archived CAT caches — every plotted array is recomputed live.

## Package Boundary

```python
import rose
import lrom
```

- The LROM side runs exclusively through the public `lrom.LROM` API with
  `observable="cross_section"`.
- The comparison ROSE emulator is notebook-owned: it builds its own
  `ScatteringAmplitudeEmulator` inline (native free-reference convention),
  exactly as Benchmark_1.0 does for wavefunctions.
- Must not import `lrom_legacy`, `lrom_bench`, or anything from
  `scientific_archive`.
- No plotting wrapper functions (`plot_*`, `split_violin`, scan helpers).
  Every figure cell contains its complete inline Matplotlib code.

## Physical Setup

- 40Ca(n,n) at 14.1 MeV, `potential="full_woods-saxon"` (10 parameters,
  complex volume + surface + spin-orbit; ROSE KD_simple sign conventions).
- Contiguous exact partial waves `l = 0 .. L_MAX` with spin-orbit split
  channels; `L_MAX` chosen in the shared config cell for convergence at
  14.1 MeV within tolerable runtime.
- Training: Latin-hypercube over all 10 parameters (±10% box).
  Testing: wider box (±15%) for held-out samples. Sizes set in the shared
  config cell.
- Global config cell owns every knob: sizes, meshes, basis sizes, angle
  grid, scan windows, display floor, shared error-axis limits
  (Benchmark_1.0 pattern: `DIFFERENCE_FLOOR`, `ERROR_YMAX`).

## Approved Figure Set

1. **Representative cross sections** — dσ/dΩ(θ) for selected held-out
   samples: FOM curve, 2.0 LROM prediction, notebook-owned ROSE emulator.
   Log y-axis, r-independent angle axis in degrees.
2. **Pointwise relative cross-section error vs angle** for the same
   representative samples (log axis with display floor).
3. **Cross-section error distributions** — split train/test violins of the
   median-over-angle relative error per sample, for ROSE and 2.0 LROM.
4. **CAT plot** — median relative cross-section error vs median online
   evaluation time per sample. Points: ROSE emulator at two or more
   `(n_phi, n_U)` configurations and the 2.0 LROM at two or more predictor
   counts K. Timing covers the online stage only: parameters in, cross
   section out.
5. **Summary table** — method, median online time, median error
   (the legacy cell-14 table, recomputed).

The error metric throughout is the legacy metric: median over angle of
|dσ_pred − dσ_FOM| / dσ_FOM per sample.

## Timing Contract

- Time only the online stage per sample: from receiving parameter values to
  returning a cross-section array, averaged over the held-out set.
- FOM reference cross sections come from the notebook-owned high-fidelity
  path, not from the emulators being compared.
- No caches: everything recomputes in-notebook. Cell-level progress prints
  are acceptable.

## Verification

1. Valid notebook JSON; every code cell compiles.
2. Imports `rose` and `lrom`; forbidden imports absent; archived 03 path in
   the introduction.
3. Contains `# FIGURE:` markers: `representative-cross-sections`,
   `cross-section-errors`, `error-violins`, `cat-plot`, and a summary-table
   cell.
4. No plotting function definitions.
5. Executed end-to-end with all cells and figures rendered.

## Non-Goals

- Reproducing every legacy `(n_phi, n_U)` and packed/fast-packed LROM
  variant from the archived CAT cloud — a representative subset suffices;
  the legacy cache is not loaded.
- Energy/isotope extrapolation (legacy notebook 04 territory; version 3.0).
- Modifying the archived notebook or legacy caches.
