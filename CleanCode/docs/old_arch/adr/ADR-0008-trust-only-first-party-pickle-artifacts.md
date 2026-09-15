# ADR-0008: Trust only first-party pickle artifacts

- **Status:** Accepted
- **Date:** 2026-09-08
- **Evidence:** `lrom_git/CleanCode/README.md` § “Data and model provenance”; `lrom_git/CleanCode/models/three_window/README.md`; `lrom_git/CleanCode/lrom/pretrained.py`; commit `18856eb`

## Context

The collaborator release distributes trained Python object graphs as pickle
files for direct offline use. Pickle executes object reconstruction logic while
loading and is therefore unsafe as an interchange format for untrusted model
downloads.

## Decision

Treat the checked-in `models/three_window/*.pkl` files as trusted first-party
artifacts. Never replace them with files from an unknown source and load them
through the pretrained-model API.

## Consequences

The release can restore its full trained models without a download or custom
serialization layer. Trust now depends on repository provenance, and loading
arbitrary external models is foreclosed. Pickles add large binary artifacts and
remain sensitive to incompatible Python or class-layout changes.
