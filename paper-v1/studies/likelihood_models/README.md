# Likelihood and outlier study

Date: 2026-11-18

This study was kept separate from the main paper notebooks so exploratory
likelihood choices cannot silently become part of the production calibration.

## Motivation

At the existing refined MAP, using only the reported experimental point errors
gives a marginalized chi-square per observation of about 6.28.  Increasing the
random-walk proposal scale did not solve the frozen, overconfident posterior:
larger proposals were mostly rejected.  The main problem is a likelihood that
does not include enough experimental/model discrepancy, not merely a proposal
step that is too small.

## Models compared

All cases retain the reported point errors and the Gaussian normalization
nuisance for each experimental angular distribution.

1. Gaussian, fixed 10 percent term proportional to the observed data.
2. Gaussian, inferred data-proportional fractional discrepancy.
3. Gaussian, inferred prediction-proportional discrepancy (diagnostic only).
4. Gaussian with the KDUQ scale: fraction times `(data + prediction)/2`.
5. Student-t with four degrees of freedom and inferred data-proportional scale.

For inferred fractions, the diagnostic used the KDUQ differential-observable
prior: a Gaussian centered at 20 percent with 20 percent standard deviation,
truncated above zero.

## Short-chain results

| model | discrepancy median (95 percent interval) | residual RMS | points beyond 3 sigma | worst R-hat |
|---|---:|---:|---:|---:|
| fixed 10 percent, data-scaled Gaussian | 10 percent fixed | 1.54 | 6.2 percent | 3.92 |
| inferred data-scaled Gaussian | 21.6 percent (20.9--22.1) | 1.00 | 0.9 percent | 2.65 |
| inferred prediction-scaled Gaussian | 25.4 percent (24.7--26.0) | 1.19 | 3.5 percent | 4.91 |
| inferred KDUQ-scaled Gaussian | 21.9 percent (20.6--22.8) | 1.09 | 2.5 percent | 3.18 |
| inferred data-scaled Student-t4 | 13.1 percent (12.5--13.7) | 1.38 | 4.8 percent | 2.54 |

These used 16 walkers, 2,600 steps, and a 700-step burn-in.  Their R-hat values
are not publishable.  The table diagnoses the error model; it does not establish
final credible intervals.

## Interpretation

- A fixed 10 percent term helps, but is too small for this global neutron
  differential-cross-section corpus.
- The Gaussian model absorbs the tails by raising the discrepancy to roughly
  22 percent.  Unit residual RMS is therefore partly by construction.
- Student-t4 separates central scatter from tails and supports roughly 13
  percent core discrepancy while retaining a small explicit outlier population.
  It is the most attractive simple robust candidate at this stage.
- Prediction-scaled variance sampled worst and suffers denominator feedback:
  changing the prediction also changes its assigned uncertainty.  It should
  remain a diagnostic, not the leading model.
- The largest remaining record-level residuals concentrate in low-energy data,
  especially Mg-24 at 6.1 MeV, with P-31, Al-27, and low-energy Ca also recurring.
  That pattern suggests structured model defect or inconsistent records rather
  than one homogeneous cloud of Gaussian noise.

## Connection to ROSE and KDUQ

- ROSE used synthetic independent Gaussian noise at about 10 percent of the
  observation and separately included a conservative 1 percent diagonal
  emulator-error term.
- KDUQ inferred one unaccounted-for fractional uncertainty per observable type,
  multiplying the fraction by the average of datum and prediction before adding
  it in quadrature with reported error.
- KDUQ also iterated 3-sigma outlier rejection during MCMC.  Its outlier scale
  combined reported experimental uncertainty with posterior prediction
  variance.  Differential-elastic standardized residual widths remained near
  two, in part because the optical model omitted deformation physics.

## Recommended next run

Carry the Student-t4/data-scaled model as the robust candidate and the
KDUQ-average Gaussian as the literature-matched reference.  Use longer,
better-mixing chains and perform one transparent iterative outlier-refit check.
Inspect flagged records before excluding any points.

The committed CSV files contain record-level diagnostics, the JSON file contains
the compact numerical summary, and `figures/` contains vector comparison plots.
The scripts are provenance copies and still expect the generated global
calibration payload and support modules from the full research workspace.
