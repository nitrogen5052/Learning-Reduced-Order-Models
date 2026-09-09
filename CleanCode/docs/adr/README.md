# Architecture decision records

These records document accepted decisions for the live `CleanCode` package.
Open questions are listed separately and do not have ADRs.

| Number | Title | Status | Date |
|---|---|---|---|
| [ADR-0001](ADR-0001-organize-the-live-package-by-responsibility.md) | Organize the live package by responsibility | Accepted | 2026-08-31 |
| [ADR-0002](ADR-0002-demote-the-milestone-lineage-without-deleting-it.md) | Demote the milestone lineage without deleting it | Accepted | 2026-09-08 |
| [ADR-0003](ADR-0003-keep-rose-outside-the-package.md) | Keep ROSE outside the package | Accepted | 2026-09-01 |
| [ADR-0004](ADR-0004-keep-plotting-in-notebooks.md) | Keep plotting in notebooks | Accepted | 2026-09-08 |
| [ADR-0005](ADR-0005-route-energy-through-three-hard-windows.md) | Route energy through three hard windows | Accepted | 2026-09-08 |
| [ADR-0006](ADR-0006-inject-the-online-linear-solve-runtime.md) | Inject the online linear-solve runtime | Accepted | 2026-08-31 |
| [ADR-0007](ADR-0007-vendor-and-pin-curated-data.md) | Vendor and pin curated data | Accepted | 2026-09-01 |
| [ADR-0008](ADR-0008-trust-only-first-party-pickle-artifacts.md) | Trust only first-party pickle artifacts | Accepted | 2026-09-08 |
| [ADR-0009](ADR-0009-separate-emulator-fidelity-from-model-adequacy.md) | Separate emulator fidelity from model adequacy | Accepted | 2026-09-08 |
| [ADR-0010](ADR-0010-generate-architecture-documentation-from-source-evidence.md) | Generate architecture documentation from source evidence | Accepted | 2026-09-09 |

## Open — no ADR yet

1. **Where does the live package live permanently?** `CleanCode/` is a working title.
   Promoting it to `lrom_git/lrom/` would restore a normal repo shape, but renames the
   import root for every notebook.
2. **Does the 1.2 / 2.0 / 3.0 naming law survive?** `CleanCode/pyproject.toml` declares
   `name = "scattering-lrom"`, `version = "0.3.0"` — a different distribution *and* a
   different numbering scheme from `docs/VERSIONING.md`. One of the two must be retired.
3. **Are `CleanCodeAugust31/` and `forship/` deleted?** Recommended: `git tag` each, then
   delete the directories. **Never delete without explicit confirmation.**
4. **Are the parent `tests/`, `notebooks/`, and `tools/` repointed or retired?** The one-line
   repoint is `pythonpath = ["lrom_git/legacy_dan_lrom", "lrom_git/legacy_dan_lrom/notebooks"]`.
   That revives tests for code just demoted to legacy — retiring them may be the better call.
5. **Is `docs/VERSIONING.md` rewritten or archived?** It currently documents only the legacy
   lineage and contains the superseded "do not split into modules" rule.
