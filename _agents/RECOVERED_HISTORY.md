# Reconstructed Project History (post-eviction)

On 2026-07-12, iCloud storage-pressure eviction destroyed the local `.git`
of the original working copy at `~/Documents/Documents-Agent/LROM_Project`
(the base pack, all 1,498 loose objects, and the reflogs became unreadable
dataless placeholders with no recoverable cloud records). The unpushed
commit history from 2026-06-15 to 2026-07-12 was lost as git objects. This
document reconstructs it from Codex session rollouts (`~/.codex/sessions`),
the surviving `_memory` daily notes, and the reviewing agent's session
record. The repository content itself was fully reconstructed (see the
snapshot commit `56ff147`).

## Commit chronology recovered from Codex rollouts

### 2026-06-15 — benchmark spine and Notebook 1 scaffolding
- Add LROM benchmark package scaffold
- Add benchmark configuration and paths
- Add benchmark sampling and portable numerics
- Add LROM predictor utilities
- Add residual-fit LROM core
- Add benchmark artifact parity helpers
- Add lazy ROSE integration boundary
- Add package-native Notebook 02 generator
- Add Notebook 02 parity runner
- Refresh generated Notebook 02 benchmark
- Add central reduced-basis boundary
- Add Notebook 1 config and sampling helpers
- Add centered SVD basis helper
- Add real Woods-Saxon ROSE helpers
- Document notebook-driven helper rule
- Add Notebook 1 RBM LROM generator
- Add real Woods-Saxon problem boundary
- Fill Notebook 1 RBM LROM sections

### 2026-06-22 — stateful `lrom` package and Notebook 1 migration
- Organize LROM scientific archive
- Create lrom package surface
- Add named LROM configuration
- Add named LROM sampling designs
- Add stateful LROM lifecycle
- Add nuclear scattering FOM and shared basis
- Add LROM training and diagnostics
- Add portable LROM inference artifacts
- Migrate Notebook 1 to lrom object workflow
- Document and test supported LROM workflow
- Add Notebook 1 scientific comparison figures
- Add explicit named sampling grids
- Store training and testing diagnostics
- Expand Notebook 1 comparison diagnostics
- Separate ROSE emulation from LROM training
- Run ROSE comparison independently in Notebook 1

### 2026-06-30 — Benchmark N1 (later renamed Benchmark 1.0)
- Test Benchmark N1 notebook contract
- Add Benchmark N1 one-parameter studies
- Add Benchmark N1 predictor diagnostics
- Complete Benchmark N1 figure reconstruction
- Verify Benchmark N1 notebook
- (plus the display-floor commits recorded in the last readable log:
  "Specify difference figure display floor", "Set difference figure floor
  to 1e-4", "... to 1e-5", "Adjust difference figure floor to 1e-5",
  "Use free-reference ROSE workflow in Benchmark N1")

### 2026-07-09 .. 2026-07-12 — version 1.0 consolidation (Claude sessions)
- `79d93fa` Version 1.0: naming consistency, sph_harm fix, and agent-area
  restructure (N1 → 1.0 labels, `lrom_legacy.v1_0` rename,
  `Benchmark_1.0.ipynb`, `_agents/` area, project CLAUDE.md)
- `ee6320e` Benchmark_1.0: global scan windows/axes and coefficient blow-up
  diagnosis (VV 30–70 MeV, RV 3–5 fm, ERROR_YMAX; spurious pole of the
  learned implicit operator at p_Rv ≈ +3.2)
- `33b47ae` Notebook 01: regenerate from fixed generator and execute
  end-to-end (26/26 Paper-Results-Map requirements)
- A `wip/n2-cross-section-v2.0` branch carried the parked cross-section
  drift; its content was preserved by selective restore before the
  eviction and is part of the 2.0 snapshot.

## Recovered artifacts

- `_agents/superpowers/plans/2026-06-30-benchmark-n1-notebook.md` and
  `2026-06-30-n1-rose-rom-separation.md`: extracted verbatim from Codex
  rollout patches.
- `_agents/superpowers/specs/2026-06-30-benchmark-n1-notebook-design.md`:
  restored from the reviewing agent's session context.
- `scientific_archive/recovered_2026-07-13/`: a readable Codex-worktree
  variant of legacy notebook 01 (larger than the archived copy) and the
  `manual-20260601-lrom` presentation outputs.

## Known unrecoverable items

- Git object history (diffs) between 2026-06-15 and 2026-07-12.
- `scientific_archive/references/Paper Results Map.pdf` and
  `ROSEPaper[7945].pdf` (committed locally 2026-07-09, evicted before any
  push; the roadmap distillation `_agents/backlog/paper-results-roadmap.md`
  survives via origin). Re-add from the user's original sources.
- The July working-tree revision of `notebooks/02_lrom_method_walkthrough.ipynb`
  (superseded by the planned version 2.0 notebook 02 anyway).
- The user's `~/Documents/projects/Nuclear_Physics/LROM/my attempt` files
  (evicted; outside this repository).
