"""Importable snapshots of LROM package milestones.

Module names encode versions (`1.2` is not a valid Python identifier):

- `v1_2`: the validated version 1.2.0 single-file wavefunction package
  (benchmark_02 medians match the legacy notebook exactly; ROSE is
  FOM-solver-only). Snapshot taken before 2.0 development.
- `v2_0`: the first, parked cross-section attempt. Superseded by the 2.0
  work in the active `lrom` package; kept as the parts donor. Known flaws:
  spin-orbit-blind potential predictors and noisy CAT timing.

The active package is the top-level single-file `lrom`.
"""

__all__ = ["v1_2", "v2_0"]
