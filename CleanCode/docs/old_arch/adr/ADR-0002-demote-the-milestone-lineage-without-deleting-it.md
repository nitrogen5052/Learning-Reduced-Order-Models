# ADR-0002: Demote the milestone lineage without deleting it

- **Status:** Accepted
- **Date:** 2026-09-08
- **Evidence:** Commit `dc81211`; `lrom_git/legacy_dan_lrom/`; `.agents/superpowers/plans/2026-09-08-claude-md-reconciliation.md`

## Context

The former public `lrom/`, its one-file v1.2 implementation, the v2.0 and v3.0
shells, and their notebooks no longer represent the live package. Deleting
them would erase useful scientific and API history, while leaving them at the
repository root made their authority ambiguous.

## Decision

Move the complete 1.2/2.0/3.0 lineage unchanged under
`legacy_dan_lrom/`, and treat `CleanCode/` as the live package.

## Consequences

The old implementations and notebooks remain available for comparison without
competing for live-package authority. The repository still contains parallel
copies, so searches and fixes require explicit scope, and legacy code does not
receive live-package changes automatically. The permanent live-package path
and the future of the old naming law remain open decisions.
