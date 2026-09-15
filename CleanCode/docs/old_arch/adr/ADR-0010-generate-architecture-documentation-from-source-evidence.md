# ADR-0010: Generate architecture documentation from source evidence

- **Status:** Accepted
- **Date:** 2026-09-09
- **Evidence:** Commit `3c9395c`; `lrom_git/CleanCode/docs/architecture.json`; `lrom_git/CleanCode/tools/architecture_graph.py`; `lrom_git/CleanCode/tools/render_architecture.py`

## Context

Hand-maintained architecture diagrams can drift from imports, consumers, and
the deployed workflow. The package needs a reviewable statement of what static
analysis found, including optional edges and caveats, without turning diagram
layout into a second source of architectural truth.

## Decision

Generate `docs/architecture.json` from source and render
`docs/ARCHITECTURE.md` from that evidence. Treat the JSON as evidence and the
Markdown diagrams as a generated artifact that is never edited by hand.

## Consequences

The graph and diagrams are deterministic, checkable, and tied to inspected
source. Architecture-document changes require running and reviewing the
generators, so direct Markdown fixes are foreclosed. Static analysis cannot
prove external callers or dynamic behavior and must retain those limitations
as caveats.
