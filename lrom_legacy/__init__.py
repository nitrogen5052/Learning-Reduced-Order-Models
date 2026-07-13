"""Importable snapshots of LROM package milestones.

Module names encode versions (`1.0` is not a valid Python identifier):

- `v2_0`: the version 2.0 cross-section package (S-matrices, spin-orbit
  channels, `observable="cross_section"`). Parked here 2026-07-13 while
  version 1 physics is re-verified. Known fixes needed before 2.0 resumes:
  potential predictors carry no spin-orbit parameter information
  (Vso/Rso/aso), and the CAT per-sample timing methodology is too noisy.

The version 1 package is the active top-level `lrom` package.
"""

__all__ = ["v2_0"]
