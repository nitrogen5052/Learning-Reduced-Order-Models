# Notebook 01 RBM/LROM Design

## Goal

Create one package-native Notebook 1 that explains the first scientific layer of the project: how a traditional reduced-basis model for a single scattering wavefunction compares with the learned reduced-operator model.

The notebook should be readable as a scientific narrative, not just a benchmark script. It starts with the simplest one-parameter case and then expands to the first multi-parameter case that motivates operator-informed predictors.

## Source Context

The design follows the `Paper Results Map.pdf` direction for Notebook 1:

- Use a simple scattering example: `l = 0`, real Woods-Saxon potential, one wavefunction only.
- Start with varying `Vv` only.
- Show rainbow plots for potentials and wavefunctions.
- Use the central solution as `phi0`.
- Compare coefficients from the traditional RBM/ROSE route and LROM.
- Compare wavefunction reproduction performance.
- Then vary all three real Woods-Saxon parameters: `Vv`, `Rv`, and `av`.
- Add the `av` rainbow plot.
- Introduce maxvol-selected potential predictors.
- For the performance plots, show the LS floor, ROSE/RBM, and LROM. Do not make linear LROM part of the main story unless it is needed as a diagnostic.

## Notebook Structure

The notebook title is:

```text
01. RBM vs LROM for a Single Scattering Wavefunction
```

The notebook has one continuous story with two acts.

### Act 1: Vv-Only Teaching Case

1. **Scientific Setup**
   - Define the `l = 0` one-channel scattering problem.
   - Use the real Woods-Saxon term with parameters `[Vv, Rv, av]`.
   - Identify the central parameter point.
   - Set `phi0 = phi(alpha_c)`.

2. **Vv-Only Samples**
   - Vary only the depth `Vv`.
   - Keep `Rv` and `av` fixed.
   - Plot a rainbow of potentials.
   - Plot a rainbow of FOM wavefunctions.

3. **Reduced Basis And LS Floor**
   - Build the reduced basis from centered snapshots.
   - Project FOM wavefunctions to least-squares coordinates.
   - Treat the LS projection as the basis floor.

4. **RBM/ROSE vs LROM Coordinates**
   - Compute ROSE/RBM reduced coefficients through the Notebook 1 setup.
   - Cache expensive intermediate results only if rerunning the notebook becomes slow.
   - Fit RF-LROM coefficients through the package path.
   - Plot coefficient behavior for LS, ROSE/RBM, and LROM.

5. **Wavefunction Reproduction**
   - Reconstruct wavefunctions.
   - Compare LS floor, ROSE/RBM, and LROM.
   - Use compact error plots plus representative wavefunction overlays.

### Act 2: Vv/Rv/av Predictor Motivation

6. **Three-Parameter Samples**
   - Vary `Vv`, `Rv`, and `av`.
   - Plot rainbow summaries for the potential variation, including `av`.

7. **Why Raw Parameters Are Not Enough**
   - Show the same reduced-coordinate and wavefunction diagnostics with raw centered parameters where useful.
   - Keep this section explanatory rather than exhaustive.

8. **Operator-Informed Potential Predictors**
   - Build potential predictors from centered operator variations.
   - Select predictor locations with delta-maxvol.
   - Visualize selected predictor points on the operator/potential grid.

9. **Three-Parameter Performance**
   - Compare LS floor, ROSE/RBM, and LROM.
   - Focus on interpolation and extrapolation performance across sampled parameter regions.

10. **Notebook 1 Takeaways**
   - Summarize what this notebook proves before moving to cross sections.
   - State that Notebook 2 moves from one wavefunction to cross-section-level comparisons.

## Package Policy

Notebook 1 drives package changes on a need-by-need basis.

Notebook code should stay decomposed and readable:

- Keep plotting logic visible in notebook cells.
- Do not add package functions whose job is to make plots.
- Do not add one whole function that runs the full notebook or full scientific workflow.
- Prefer small package helpers that each produce one scientific object: samples, alphas, potentials, wavefunctions, basis data, coefficients, predictors, fitted LROMs, or metrics.
- Notebook cells may combine these objects into figures so the scientific flow remains visible.

Allowed package changes:

- Add reusable ROSE/FOM helpers needed by Notebook 1.
- Add reusable real Woods-Saxon setup helpers if the notebook would otherwise duplicate fragile solver setup.
- Add plotting-agnostic numerical helpers only when they are useful beyond this notebook.
- Extend existing modules rather than creating a broad abstraction layer.

Disallowed package changes:

- Do not pre-build the full multi-notebook framework.
- Do not move all notebook logic into opaque wrappers.
- Do not delete the legacy notebooks.
- Do not add vector database or graph infrastructure in Notebook 1.

If a package change affects the benchmark architecture, update `docs/architecture/lrom-benchmark-spine.md` in the same implementation slice so the package can be reproduced later.

## Artifacts

Primary artifact:

- `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`

Supporting source if generation is useful:

- `scripts/generate_notebook01.py` or an extension of the existing generator, chosen during implementation.

Architecture documentation, only if package APIs change:

- `docs/architecture/lrom-benchmark-spine.md`

## Verification

Minimum verification for the first Notebook 1 slice:

1. The notebook is generated or saved with the agreed section order.
2. `python -m pytest -v` passes after any package edits.
3. If ROSE-backed cells are implemented, a focused smoke check reaches the `l = 0` central Woods-Saxon setup.
4. If numerical results are produced, the notebook includes at least one small artifact or printed summary that makes RBM/ROSE, LS floor, and LROM comparisons auditable.

## Implementation Defaults

The implementation plan should start from these defaults:

- Use a dedicated generator script: `scripts/generate_notebook01.py`.
- Keep Notebook 1 as one notebook, not `1a` and `1b`.
- Implement the `Vv`-only path as the first executable scientific result.
- Add the three-parameter section in the same notebook, with executable cells when the needed package helpers are available.
- Cache runtime-heavy FOM/ROSE results only after a measured rerun cost makes caching useful.
- Prefer visible notebook cells over opaque wrappers; package helpers should remove fragile repeated setup, not hide the scientific flow.
