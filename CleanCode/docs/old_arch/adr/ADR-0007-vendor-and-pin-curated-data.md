# ADR-0007: Vendor and pin curated data

- **Status:** Accepted
- **Date:** 2026-09-01
- **Evidence:** Commit `289777e`; `lrom_git/CleanCode/data/curated/PROVENANCE.md`; `lrom_git/CleanCode/README.md` § “Data and model provenance”

## Context

Notebook 3 depends on a defined subset of KDUQ and held-out neutron-elastic
records. Reading a moving upstream checkout would make the release
irreproducible and could silently change record selection, uncertainties, or
metadata.

## Decision

Vendor the curated neutron-elastic subset without modification and pin its
provenance to upstream commit
`adc8558fe9fcf629af41bc909514f8da63d4f590`. Preserve attribution and update
the pin whenever the vendored data are intentionally refreshed.

## Consequences

Collaborators can identify the exact source revision and work from a stable
snapshot. The repository assumes responsibility for storage, extraction, and
provenance maintenance. Upstream fixes do not arrive automatically, and the
upstream license warning constrains broader redistribution.
