# ADR-0005: Route energy through three hard windows

- **Status:** Accepted
- **Date:** 2026-09-08
- **Evidence:** `lrom_git/CleanCode/README.md` § “Current deployment choice”; `lrom_git/CleanCode/models/three_window/training_manifest.json`; `lrom_git/CleanCode/lrom/deployment.py`; commit `18856eb`

## Context

Energy is an experimental or model-input coordinate, not a calibrated optical-
potential coefficient. Evaluating overlapping local models and blending them
would therefore add routine online work without serving the calibration
parameterization. One global learned model would also discard the established
local-window training structure.

## Decision

Deploy three independent LROMs with hard windows `[5, 70)`, `[70, 135)`, and
`[135, 200]` MeV. Evaluate exactly one local model for each energy, while
extending training support beyond internal boundaries where possible.

## Consequences

Routing is deterministic and avoids double evaluation and blending in the
online path. The local bases and models must be trained, stored, and validated
separately. Continuity is not guaranteed at 70 or 135 MeV, so boundary behavior
must be tested explicitly; smoothing the transitions is foreclosed without a
new decision.
