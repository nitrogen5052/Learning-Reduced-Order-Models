# LROM Project Cleanup Design

Date: 2026-06-22

## Goal

Make the active LROM workspace easy to navigate before redesigning the package around an object-oriented `lrom` API. Preserve scientifically valuable history without leaving inactive code, superseded plans, and reference material mixed with active work.

## Target Layout

```text
LROM_Project/
├── lrom_bench/              # active implementation until the redesign renames it
├── notebooks/               # active notebooks only
├── scripts/                 # active support and verification scripts only
├── tests/                   # active tests
├── docs/                    # current project direction and approved cleanup spec
├── scientific_archive/
│   ├── legacy_code/         # old package, notebooks, scripts, requirements, and notes
│   ├── references/          # scientific reports and presentation PDFs
│   └── prior_designs/       # superseded benchmark architecture, specs, and plans
├── pyproject.toml
├── README.md
└── .gitignore
```

## Classification Rules

- Keep a file active only when it supports the current `lrom_bench` implementation, its current notebooks, tests, or verification workflow.
- Move old executable material into `scientific_archive/legacy_code/`; preserve its internal relative layout so the historical workflow remains intelligible.
- Move scientific PDFs into `scientific_archive/references/`.
- Move superseded benchmark-oriented architecture documents, specifications, and implementation plans into `scientific_archive/prior_designs/`.
- Keep this cleanup spec active because it documents the archive boundary for the upcoming redesign.
- Do not rename `lrom_bench` or redesign APIs during cleanup. Those changes belong to the next planning cycle.

## Generated Clutter

Delete reproducible local debris throughout the repository:

- `.DS_Store`
- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `.superpowers/` scratch state

Add matching `.gitignore` rules so these files do not return.

## Safety And Verification

- Preserve all existing user-authored changes; reorganize them without reverting or overwriting them.
- Inventory files before and after the move and confirm that every scientifically valuable source remains under either the active tree or `scientific_archive/`.
- Search active files for stale paths after relocation and update only documentation or scripts whose active behavior depends on moved material.
- Run the active test suite after cleanup. Record environment-related failures separately from structural failures.
- Review `git status` and the final directory tree to ensure generated clutter is absent and the active root is concise.

## Acceptance Criteria

- The repository root clearly separates active work from historical scientific material.
- Old code, notebooks, references, and superseded designs remain recoverable in `scientific_archive/`.
- Generated cache files are removed and ignored.
- Active imports, tests, notebooks, and scripts are not intentionally redesigned during this cleanup.
- The workspace is ready for a new outside-in design of the object-oriented `lrom` API.
