# ADR-0001: Organize the live package by responsibility

- **Status:** Accepted
- **Date:** 2026-08-31
- **Evidence:** Commits `5031c81` and `289777e`; `lrom_git/CleanCode/lrom/`; `lrom_git/CleanCode/docs/architecture.json`; `docs/VERSIONING.md` § “Deliberate design decisions”

## Context

The v1.2 lineage kept the full workflow in one authoritative implementation
file and explicitly forbade a module split. The collaborator package grew to
cover the FOM, reduced basis, learned equations, observables, deployment,
backends, curated data, and diagnostics. Keeping those responsibilities in one
file would obscure their boundaries and make independent evolution harder.

## Decision

Organize the live `scattering-lrom` package into modules by responsibility,
with a compact public surface in `lrom/__init__.py` and a directed acyclic
internal dependency graph. Module size alone does not require another split.

## Consequences

Physics, training, deployment, and optional backends can change at explicit
seams, and the package structure is inspectable from source. Cross-module API
coordination and dependency-cycle control are now maintenance costs. For the
live `CleanCode` package, this ADR supersedes `docs/VERSIONING.md`'s rule:
“Do not split the implementation into modules”; that document still describes
the legacy lineage and is not edited by this decision.
