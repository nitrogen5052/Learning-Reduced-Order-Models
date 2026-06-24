# Notebook 1 Comparison Figures Design

## Goal

Complete the scientific comparisons in Notebook 1 without changing the `lrom`
package architecture or adding package-level plotting helpers.

## Scope

Modify only:

- `scripts/generate_notebook01.py`
- `notebooks/01_rbm_vs_lrom_single_wavefunction.ipynb`
- Notebook 1 structural tests

Notebook 2 and its in-progress supporting files remain untouched.

## One-Parameter Coefficient Comparison

Add a dedicated two-row figure for the Vv-only emulator. The horizontal axis is
the testing Vv value. The panels show `Re(a1)` and `Re(a2)` respectively, using
the shared reduced basis. Each panel overlays the LS target coordinates, ROSE
coordinates, and LROM coordinates from
`vv_emulator.testing_results.coefficients`.

## One-Parameter Wavefunction Comparison

Select the midpoint of the 41-point Vv testing linspace, which is the central Vv
case. Print its case ID and named parameters. Use `vv_emulator.testing_case` to
plot `Re(phi(r))` for the true high-fidelity testing solution, LS reconstruction,
ROSE approximation, and LROM approximation. The horizontal axis is physical
radius `r [fm]`.

## Three-Parameter Violin Comparison

Replace the final all-case pointwise-error line figure with a violin plot. Each
method contributes 81 values: one relative L2 wavefunction error per ws3 testing
case, read from
`ws3_emulator.testing_results.metrics["relative_l2"][0]`. Plot
`log10(relative L2 error)` for LS, LROM, and ROSE so differences across orders of
magnitude remain visible. Label the sample count and metric explicitly.

## Verification

Structural tests will assert that:

- both first two basis coefficients are plotted for LS, ROSE, and LROM;
- the central Vv testing case is selected through `testing_case`;
- the one-parameter wavefunction figure includes high fidelity, LS, LROM, and ROSE;
- the final figure uses `violinplot` and per-solution relative L2 errors;
- the generator remains deterministic.

Execute Notebook 1 after regeneration and require zero cell errors.
