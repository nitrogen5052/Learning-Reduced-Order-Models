# _agents/ — AI-agent working area

This folder holds material used by Claude/Codex agents to plan, validate, and
track the LROM project. **It contains no science results** — researchers can
ignore it. Nothing here is imported by the `lrom` package itself.

- `superpowers/plans/`, `superpowers/specs/` — approved, dated design
  documents and implementation plans. Historical record: do not rewrite;
  add new dated files instead.
- `backlog/` — distilled requirements from the project map
  (`paper-results-roadmap.md`) and deferred work notes.
- `ntbk_validation/` — agent-run physics-validation notebooks and manual
  review checklists (not paper notebooks).
- `scripts/` — notebook-generation helpers exercised by the test suite
  (imported as `_agents.scripts` in tests).

Researcher-facing directories remain at the repository root: `lrom/`,
`lrom_legacy/`, `lrom_bench/`, `notebooks/`, `docs/`, `tests/`,
`scientific_archive/`, `outputs/`.

`tests/` stays at the root deliberately: it is standard Python tooling run by
CI (`.github/workflows/test.yml`) and guards the scientific contracts of the
notebooks, so hiding it in the agent area would break convention.
