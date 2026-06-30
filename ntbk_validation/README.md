# Notebook Validation

This directory is for human-reviewed physics validation notebooks.

Validation notebooks should import the active package, generate physics outputs,
and present plots/tables for manual inspection. They should not hide physics
acceptance behind package test helper functions.

Automated tests should cover package mechanics instead:

- importability and public API shape
- array dimensions and named parameter contracts
- serialization and artifact compatibility
- deterministic sample bookkeeping
- small numerical smoke checks that do not decide scientific validity

Use `lrom_legacy.N1` when a notebook needs to compare the active package with
the Notebook 1 package snapshot.

## Notebooks

- `N2_physics_validation.ipynb`: manual review scaffold for the Notebook 2
  package work, including the optical-potential parameter table, active-vs-N1
  import setup, optional wavefunction smoke review, and future cross-section
  review checklist.
