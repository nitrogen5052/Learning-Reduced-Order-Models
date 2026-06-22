# LROM Project Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate active LROM work from scientifically valuable history and remove reproducible workspace clutter without redesigning the package.

**Architecture:** Keep the active package, notebooks, scripts, tests, and current cleanup documentation at the repository root. Consolidate legacy executables, scientific references, and superseded benchmark designs beneath `scientific_archive/`, preserving source material rather than deleting it. Ignore and remove machine-generated caches.

**Tech Stack:** Git, POSIX filesystem operations, Python/pytest

---

### Task 1: Record the pre-move inventory

**Files:**
- Read: repository tree and Git working state

- [ ] **Step 1: Capture tracked and untracked state**

Run:

```bash
git status --short
rg --files -g '!*.pyc' | sort
```

Expected: the existing deletions and modifications remain visible; `Legacy_benchmark/`, `project_context_pdfs/`, caches, and prior design documents are present.

- [ ] **Step 2: Confirm archive inputs exist**

Run:

```bash
test -d Legacy_benchmark
test -d project_context_pdfs
test -f docs/architecture/lrom-benchmark-spine.md
test -f outputs/lrom_bench_function_flow.md
```

Expected: all commands exit successfully.

### Task 2: Create the scientific archive

**Files:**
- Move: `Legacy_benchmark/` to `scientific_archive/legacy_code/Legacy_benchmark/`
- Move: `project_context_pdfs/` to `scientific_archive/references/`
- Move: superseded benchmark documents to `scientific_archive/prior_designs/`

- [ ] **Step 1: Create the three archive categories**

Run:

```bash
mkdir -p scientific_archive/legacy_code scientific_archive/references scientific_archive/prior_designs
```

Expected: all three directories exist.

- [ ] **Step 2: Move legacy executable material without flattening it**

Run:

```bash
mv Legacy_benchmark scientific_archive/legacy_code/Legacy_benchmark
```

Expected: the historical package, notebooks, scripts, requirements, and notes remain together under `scientific_archive/legacy_code/Legacy_benchmark/`.

- [ ] **Step 3: Move scientific references**

Run:

```bash
mv project_context_pdfs/* scientific_archive/references/
rmdir project_context_pdfs
```

Expected: both scientific PDFs are under `scientific_archive/references/` and the old directory is gone.

- [ ] **Step 4: Move prior benchmark designs**

Run:

```bash
mkdir -p scientific_archive/prior_designs/architecture scientific_archive/prior_designs/specs scientific_archive/prior_designs/plans scientific_archive/prior_designs/notes
mv docs/architecture/lrom-benchmark-spine.md scientific_archive/prior_designs/architecture/
mv docs/superpowers/specs/2026-06-15-lrom-benchmark-spine-design.md scientific_archive/prior_designs/specs/
mv docs/superpowers/specs/2026-06-16-notebook01-rbm-lrom-design.md scientific_archive/prior_designs/specs/
mv docs/superpowers/plans/2026-06-16-lrom-benchmark-spine.md scientific_archive/prior_designs/plans/
mv docs/superpowers/plans/2026-06-16-notebook01-rbm-lrom.md scientific_archive/prior_designs/plans/
mv docs/superpowers/plans/2026-06-16-notebook01-full-slices.md scientific_archive/prior_designs/plans/
mv outputs/lrom_bench_function_flow.md scientific_archive/prior_designs/notes/
rmdir docs/architecture outputs
```

Expected: only the approved cleanup spec and this cleanup plan remain under `docs/superpowers/`; superseded material is recoverable under the archive.

### Task 3: Add active-root orientation and ignore rules

**Files:**
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Add generated-file exclusions**

Create `.gitignore` with exactly:

```gitignore
.DS_Store
__pycache__/
*.py[cod]
.pytest_cache/
.superpowers/
```

- [ ] **Step 2: Add a concise active-project README**

Create `README.md` explaining:

```text
# LROM

This repository is being reorganized around a small object-oriented LROM API.

- `lrom_bench/`: active implementation pending the `lrom` redesign
- `notebooks/`: active research notebooks
- `scripts/`: active generation and verification utilities
- `tests/`: active regression tests
- `docs/`: current approved design and implementation documents
- `scientific_archive/`: legacy code, scientific references, and superseded designs retained for provenance

Historical material is reference-only. New development should use the active root directories.
```

- [ ] **Step 3: Verify README paths**

Run:

```bash
test -d lrom_bench
test -d notebooks
test -d scripts
test -d tests
test -d scientific_archive
```

Expected: all documented paths exist.

### Task 4: Remove reproducible debris

**Files:**
- Delete: `.DS_Store`, `.pytest_cache/`, `.superpowers/`, all `__pycache__/`, all `*.pyc`

- [ ] **Step 1: Remove known root caches and scratch state**

Run:

```bash
rm -rf .DS_Store .pytest_cache .superpowers
```

- [ ] **Step 2: Remove nested Python caches**

Run:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
```

- [ ] **Step 3: Confirm debris is absent**

Run:

```bash
find . -name .DS_Store -o -name __pycache__ -o -name '*.pyc' -o -name .pytest_cache -o -name .superpowers
```

Expected: no output.

### Task 5: Repair active references and verify behavior

**Files:**
- Modify only active documentation or scripts that still require an archived path
- Test: `tests/`

- [ ] **Step 1: Search active files for relocated paths**

Run:

```bash
rg -n 'Legacy_benchmark|project_context_pdfs|outputs/lrom_bench_function_flow' lrom_bench notebooks scripts tests docs README.md pyproject.toml
```

Expected: no active runtime dependency on relocated paths. Historical references in the cleanup documents are acceptable.

- [ ] **Step 2: Run the active test suite**

Run:

```bash
pytest -q
```

Expected: all active tests pass; dependency or ROSE environment failures, if any, are reported without altering archived code.

- [ ] **Step 3: Inspect the concise final tree**

Run:

```bash
find . -path ./.git -prune -o -maxdepth 3 -print | sort
git status --short
git diff --check
```

Expected: the active root matches the approved design, archive categories are clear, cache debris is absent, and pre-existing user modifications remain represented in Git status.

- [ ] **Step 4: Commit only cleanup-related paths**

Stage `.gitignore`, `README.md`, `scientific_archive/`, the relocated prior-design paths, the root deletions whose contents now live in the legacy archive, and the cleanup plan. Leave the modified active Notebook 02 and metrics test unstaged.

Run:

```bash
git add .gitignore README.md scientific_archive docs/superpowers/plans/2026-06-22-lrom-project-cleanup.md docs/architecture/lrom-benchmark-spine.md docs/superpowers/specs/2026-06-15-lrom-benchmark-spine-design.md docs/superpowers/specs/2026-06-16-notebook01-rbm-lrom-design.md docs/superpowers/plans/2026-06-16-lrom-benchmark-spine.md docs/superpowers/plans/2026-06-16-notebook01-rbm-lrom.md docs/superpowers/plans/2026-06-16-notebook01-full-slices.md CHECKPOINT_NOTES.md lrom_demo notebooks/01_fom_and_rose_basics.ipynb notebooks/03_cross_section_cat_comparison.ipynb notebooks/04_energy_isotope_wavefunction_extrapolation.ipynb requirements.txt scripts/create_clean_demo_notebooks.py scripts/notebook04_source.py scripts/smoke_test_imports.py
git status --short
git commit -m "Organize LROM scientific archive"
```

Expected: the cleanup is committed without reverting or silently absorbing unrelated active changes.
