# Cross-section accuracy root cause (benchmark_03 finding)

Measured 2026-07-15 with lrom 2.0.0 (spin-orbit-aware predictors, cached
S-matrix assembler), 10-parameter full Woods-Saxon, l=0..6, basis_size=6,
30 training / 20 testing samples:

| quantity | median |
|---|---|
| LROM cross-section error (test) | 3.3e-1 |
| **LS-floor cross-section error (test)** | **3.6e-1** |
| LS-floor wavefunction error | ~1e-3 |
| ROSE(8,12) cross-section error | 1.7e-3 |

The LS floor equals the learned error: the bottleneck is the L2-optimal
reduced representation itself, not the predictors (they respond to all ten
parameters) and not the RF fit. The S-matrix is extracted from phi and
phi' at the matching radius s0; an L2-optimal reconstruction leaves ~1e-3
bulk error but uncontrolled boundary/derivative error, which the R-matrix
amplifies to ~30% in dsigma/dOmega. ROSE's Galerkin projection satisfies
the equation weakly and nails the asymptotics at equal basis size.

## Candidate 2.x fixes (in rough order of promise)

1. Asymptotics-aware coordinate targets: replace plain L2 projection with a
   weighted projection that pins the matching region (weight spike near
   s0), or constrain a(alpha) so that phi(s0), phi'(s0) match the FOM
   exactly (2 complex constraints per channel; solve constrained LS).
2. Learn S-matrix elements directly per channel (RF-LROM on S_l instead of
   or in addition to wavefunction coordinates).
3. Larger basis helps only weakly (the error is boundary-localized, not a
   spectral tail).

The v1 wavefunction results are unaffected (benchmark_01: exact zeros vs
v1_2; benchmark_02: medians bit-identical).
