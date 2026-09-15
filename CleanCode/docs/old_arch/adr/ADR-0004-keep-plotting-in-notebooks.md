# ADR-0004: Keep plotting in notebooks

- **Status:** Accepted
- **Date:** 2026-09-08
- **Evidence:** `CLAUDE.md` § “Deliberate decisions — do not fix”; `lrom_git/CleanCode/docs/architecture.json` (`nodes`, `consumer_external_dependencies`)

## Context

Plots in this project explain specific scientific comparisons and belong next
to their assumptions, data selection, and captions. A package plotting layer
would add a presentation dependency and hide choices that readers need to see
in each notebook.

## Decision

Do not import Matplotlib or provide plotting wrappers in `lrom`. Keep all
Matplotlib use inline in notebooks and notebook-generation code.

## Consequences

The reusable `lrom` source exposes no plotting imports or API, and figure
methodology stays visible with each study. Notebook authors must own labels,
styles, and repeated plot setup, and users who want visualizations receive no
package-level plotting API.
