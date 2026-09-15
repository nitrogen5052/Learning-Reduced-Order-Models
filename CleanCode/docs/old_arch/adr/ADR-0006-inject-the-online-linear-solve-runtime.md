# ADR-0006: Inject the online linear-solve runtime

- **Status:** Accepted
- **Date:** 2026-08-31
- **Evidence:** Commit `5031c81`; `lrom_git/CleanCode/lrom/backends.py`; `lrom_git/CleanCode/lrom/deployment.py`; `lrom_git/CleanCode/docs/ARCHITECTURE.md` § “View 3 — Flow”

## Context

CPU and optional JAX/GPU execution must use the same fitted equations and
observable implementation. Letting core physics or emulator modules select a
backend would create a second scientific path and make optional accelerator
dependencies part of normal execution.

## Decision

Resolve a `Runtime` at the caller boundary and inject its linear-solve port into
the batched online path. Keep full-order generation, training, scalar
prediction, and the deployment core independent of backend adapters.

## Consequences

CPU and GPU share one fitted model, and JAX remains optional and function-local.
Callers seeking explicit backend selection or acceleration must select and
pass a runtime, and only supported batched reduced solves are accelerated.
Transparent offload of FOM generation, training, or scalar routed cross
sections is deliberately unavailable.
