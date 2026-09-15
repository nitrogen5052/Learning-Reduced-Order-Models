# ADR-0009: Separate emulator fidelity from model adequacy

- **Status:** Accepted
- **Date:** 2026-09-08
- **Evidence:** `lrom_git/CleanCode/README.md` § “Scientific scope”; `lrom_git/CleanCode/notebooks/03_curated_data_and_global_lrom.ipynb`; commit `56c0bfd`

## Context

Notebook 3 compares measurements, the KD full-order solver, and its learned
surrogate. Agreement between the surrogate and KD answers a different question
from agreement between KD and experiment. Combining the two would attribute
optical-model inadequacy to the emulator or overstate a faithful emulator's
scientific adequacy.

## Decision

Report emulator fidelity with direct LROM-versus-FOM metrics and report model
adequacy with FOM-versus-experiment metrics. Never label an experimental
mismatch as an emulator failure without the corresponding FOM comparison.

## Consequences

Claims identify whether error comes from emulation or from the underlying
optical model. Evaluations and figures must retain both reference paths, which
adds computation and reporting complexity. A single aggregate score cannot
stand in for both questions.
