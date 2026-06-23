# Paper Results Notebook Roadmap

Source reviewed: `Paper Results Map.pdf` (12 pages), provided 2026-06-23.

This roadmap records requirements from the paper-results map while keeping the current architecture effort focused on Notebook 1.

## Current Focus: Notebook 1

Notebook 1 compares traditional ROSE reduced-basis emulation with LROM for the simplest scattering case:

- One partial-wave radial wavefunction, initially `l=0`.
- Real three-parameter Woods-Saxon potential: `Vv`, `Rv`, and `av` (`potential="ws_3"`).
- Use the central FOM solution for `phi0`.
- First experiment varies `Vv` only.
- The `Vv`-only experiment uses evenly spaced samples: 35 training points over central `Vv` +/-10% and 41 testing points over central `Vv` +/-35%.
- Show rainbow plots for potentials and wavefunctions and show the wavefunction reduced basis.
- Compare Galerkin/ROSE coefficients with LROM coefficients.
- Show wavefunction reconstruction performance.
- Second experiment varies `Vv`, `Rv`, and `av`.
- The `ws_3` experiment uses Latin-hypercube samples: 70 training cases over +/-10% for all three parameters and 81 testing cases over +/-22% for `Vv`, +/-20% for `Rv`, and +/-20% for `av`.
- Sampling uses distinct named `training_ranges` and `testing_ranges`; the wider testing domain supports explicit interpolation-versus-extrapolation analysis.
- ROSE RBM uses EIM; LROM supports maxvol-selected potential predictors.
- The `Vv`-only experiment trains with `predictor="parameters"`; the `ws_3` experiment trains with the default `predictor="potential"` and `predictor_count=6`.
- The `ws_3` potential rainbow plot must visibly overlay the maxvol-selected physical radii used by the potential predictor.
- Show interpolation and extrapolation performance across parameter samples.
- Final comparisons show the LS floor, ROSE, and predictor LROM; do not show the older linear-LROM baseline.
- Do not calculate or plot cross sections; compare radial solutions of the scattering equation only.
- Include a representative single-wavefunction comparison among the high-fidelity solution, ROSE, and LROM.
- Plot the real part `Re(phi(r))` for representative wavefunction comparisons.
- Include a logarithmic absolute-difference plot comparing testing-set high-fidelity wavefunctions against ROSE and LROM, modeled on the supplied reference figure.
- Include the least-squares-floor error in that plot when it can reuse existing basis-projection state without a major structural change.

Notebook 1 must determine the reusable API and state required for:

- FOM wavefunctions and potential evaluations.
- ROSE EIM/RBM baseline construction and coefficients.
- Central-reference reduced basis and LS coordinates.
- Potential-predictor RF-LROM training and predictions.
- Human-readable access to arrays needed for notebook-native plots.
- Pointwise error arrays `abs(phi_high_fidelity - phi_rose)` and `abs(phi_high_fidelity - phi_lrom)` on the common radial coordinate.
- Pointwise `abs(phi_high_fidelity - phi_ls)` from the already-computed least-squares coordinates when available.
- Use physical radius `r` in fm as the horizontal axis for all Notebook 1 potential, wavefunction, basis, predictor-location, and pointwise-error figures. Do not use the dimensionless `s` coordinate in the notebook presentation.
- Build one central-reference basis and use the exact same `phi0` and basis vectors for ROSE Galerkin coefficients, LS targets, and LROM reconstruction.
- Use `basis_size=4` and `eim_basis_size=8` as configurable Notebook 1 defaults.

Notebook 1 sampling calls should use this public shape:

```python
emulator.sampling(
    training_ranges={...},
    testing_ranges={...},
    training_size=...,
    testing_size=...,
)
```

## Change-Control Rule

Minor additions that reuse approved state and component boundaries may be incorporated directly. Any change that requires a major structural alteration to the approved `LROM` architecture must be presented to the user and approved before implementation.

## Backlog: Notebook 2

Notebook 2 compares ROSE and LROM at the cross-section level while varying all optical-potential parameters from the ROSE paper:

- Use identical training samples for ROSE and LROM.
- Compare cross-section prediction with the same basis size.
- Compare methods in CAT-plot space across basis and operator sizes.
- Optionally show potential rainbow plots and selected predictor locations for representative `l` channels.
- Do not show the older linear-LROM baseline.
- Explore exporting a trained emulator as an interactive HTML application.

Questions about multi-channel cross-section assembly, observable APIs, CAT metrics, and HTML export are intentionally deferred until Notebook 1 is stable.

## Backlog: Notebook 3

Notebook 3 targets a global Koning-Delaroche emulator across isotopes and energies:

- Vary calcium isotope mass `target_A` and laboratory energy while holding `target_Z=20` and `projectile=(1, 0)` fixed.
- Define distinct training and testing regions in the `(A, E_lab)` plane.
- Evaluate coefficient and wavefunction performance.
- Evaluate cross-section performance in training and testing regions.
- Explore interactive HTML export.
- Export the trained emulator as a callable black box.

The future sampling API should support named physical-system ranges such as:

```python
ranges={
    "target_A": (30, 60),
    "lab_energy": (5.0, 30.0),
}
```

This dynamic-target/energy design is explicitly deferred until Notebooks 1 and 2 establish the fixed-system workflow.

## Related Deferred Capability

External FOM request/response sampling is separately documented in `docs/backlog/external-fom-sampling.md`. Revisit it after the in-process Notebook 1 sampling, training, and prediction contracts are stable.
