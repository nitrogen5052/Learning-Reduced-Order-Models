# Architecture mapping methodology

> The philosophy and repeatable method used to produce
> [Architecture V5](architecture-pack-v5.md). This document explains how to
> build and revise the map; V5 remains the description of the current system.

## Purpose

The architecture map is a software mental model for a scientific library. Its
job is to help a researcher or maintainer answer a concrete question without
first learning every source file. It should reveal the boundaries that affect
correctness, ownership, runtime behavior, scientific interpretation, and safe
change.

The map is deliberately selective. Completeness belongs in an evidence audit;
the current map should contain only the views needed to reason accurately about
the system.

## Core philosophy

### Truth before elegance

Begin with current behavior, including awkward seams and missing capabilities.
Do not draw an idealized service, dispatcher, persistence layer, validation
step, or artifact pipeline merely because one would make the picture cleaner.
An absence that affects use or trust is part of the architecture.

For V5 this principle exposed several important facts:

- the package is an in-process Python library, not a service;
- the fast online path produces boundary-derived partial-wave S matrices
  without a Numerov solve or routine full-wavefunction reconstruction;
- runtime injection is supported on specific batch paths rather than through a
  universal backend dispatcher;
- training data have a public portable save/load path, while fitted models do
  not have a public serialization contract; and
- repository-supplied pickle files are trusted deployment artifacts, not a
  general interchange format.

### Reader questions determine the views

Organize the explanation around questions a reader will actually ask:

1. Where does the library run, and what is outside it?
2. Which components perform the major responsibilities?
3. Which objects retain state, derive state, or exist only for one request?
4. What calls what during training and inference?
5. Which artifacts persist, and which can be rebuilt?
6. Which scientific and operational contracts constrain interpretation?
7. Where is each component implemented?

Files and imports remain evidence, but they do not define the narrative.

### One relationship per arrow

An arrow must have a stated meaning. Calls, ownership, construction, selection,
data movement, optional participation, and static dependency are different
relationships and should not be silently mixed in one figure. When a view uses
a different arrow grammar, state it next to the figure.

### Stable concepts across views

Give the principal components stable identifiers and visual roles. A zoomed
view should open an existing component rather than rename or regroup it without
explanation. Repetition is useful when it preserves the reader's orientation;
new terminology is not.

### Scientific architecture includes scientific claims

The architecture must preserve distinctions that determine what the results
mean:

- Numerov-versus-ROSE checks the package's full-order implementation.
- LROM-versus-package-Numerov measures emulator fidelity.
- Optical-model-versus-experiment measures model adequacy.

These are separate validation questions. Likewise, internal coordinates,
displayed physical units, calibrated support, numerical precision, provenance,
and trust boundaries are architectural contracts, not incidental details.

### Current state and possible evolution stay separate

Diagrams describe what exists now. Recommendations belong in a clearly labeled
future-work section and must not appear as present components or flows. Open
questions remain open until an explicit decision and implementation resolve
them.

## Evidence hierarchy

Use evidence in this order:

1. current source code, tests, artifacts, and executable behavior;
2. current project authority and curated state;
3. generated structural evidence and current documentation;
4. full project-scoped memory observations, reconciled for supersession; and
5. legacy documents and historical notes.

Memory supplies rationale, experiments, rejected interpretations, and the path
by which the design evolved. It is append-only and may preserve conclusions that
later work superseded. A remembered statement becomes part of the current map
only after it agrees with higher-authority evidence.

Record contradictions rather than averaging them. If current code and a prior
diagram disagree, describe the current code and preserve the older diagram as
history.

## Construction method

### 1. Define the audience and the questions

State who needs the map and what decisions it should support. For V5, the target
reader is a researcher or maintainer who needs to navigate, evaluate, or change
the live `CleanCode` package without confusing the software implementation with
the scientific method.

Set explicit non-goals. V5 is not a complete import graph, a physics tutorial,
or a claim that every Python module is a C4 container.

### 2. Recover the system as evidence

Inspect the public surface, principal classes, constructors, retained fields,
training path, scalar inference path, batch path, persistence code, optional
backends, tests, notebooks, and supplied artifacts. Use historical records to
identify traps and disputed decisions, then confirm them against the current
tree.

Build a short fact register before drawing. Each fact should be either:

- observed directly in current source or executable behavior;
- an accepted project decision still consistent with the source; or
- an uncertainty that the map must label rather than resolve.

### 3. Choose components by runtime responsibility

Group code into the fewest components that preserve the important boundaries.
A component should answer:

- What job does it perform?
- What does it retain or produce?
- Which other component does it call or depend on?
- Why would a maintainer need to distinguish it?

Do not make one component per file. Do not force the groups into a strict layer
hierarchy when the actual call graph crosses those layers. V5 uses eight
responsibility components because they preserve the meaningful seams: public
surface, problem/FOM, local emulator, reduced model, packed evaluator,
observable kernel, hard router, and solve runtime.

### 4. Present progressive views

Build the explanation in this order:

1. **Boundary view:** process, caller, durable artifacts, and optional external
   runtime.
2. **Main component map:** the smallest useful overview of responsibilities and
   calls.
3. **Ownership zoom:** retained, derived, cached, and request-local state.
4. **Offline sequence:** snapshot generation, fitting, boundary preparation,
   and packing.
5. **Online sequence:** routing, reduced solve, boundary S matrices, and
   observable assembly.
6. **Artifact lifecycle:** creation, persistence, loading, repacking, and
   deployment.
7. **Scientific contracts:** units, validity, verification claims, precision,
   provenance, trust, and online cost.
8. **Source crosswalk:** stable component IDs mapped back to concrete symbols
   and files.

Each view should add one kind of information. If a figure merely restates the
previous one with more boxes, remove it or turn it into a focused zoom.

### 5. Audit the negative space

Before polishing the diagrams, ask what a reader might incorrectly infer:

- Is an optional path drawn as mandatory?
- Is a cache mistaken for a component or durable store?
- Is a derived packed representation mistaken for another learned model?
- Is deterministic routing mistaken for scientific validity checking?
- Is a shipped artifact mistaken for evidence of an export pipeline?
- Is an inspection path mistaken for the fast online path?
- Is model inadequacy mistaken for emulator error?

Add a correction, qualifier, or explicit non-feature wherever the wrong
inference would affect correctness or trust.

### 6. Verify every view

Trace every named component and important arrow back to source. Walk the offline
and online sequences in call order. Check object ownership against constructors,
mutation sites, and load/repack behavior. Check artifact claims against actual
read/write code. Check units and validation claims against their enforcing
boundaries.

Then render every diagram and run the relevant project verification. A valid
Mermaid graph with incorrect semantics is still a failed architecture map.

## Diagram grammar

- Use stable component IDs across the document.
- Label arrows with verbs such as `construct`, `retain`, `select`, `fit`,
  `evaluate`, or `solve`.
- Use solid arrows for active relationships in the depicted path and dashed
  arrows for optional or explicitly non-package handoffs.
- Use cylinders only for durable artifacts.
- Keep component colors semantically stable; use neutral sequence styling.
- Separate ownership diagrams from call diagrams.
- Show loops, alternatives, and fallback behavior in sequence diagrams when
  they materially change execution.
- Prefer a small number of readable diagrams over a comprehensive wall of
  boxes.
- Include a prose sentence stating the shortest useful mental model.

## Lessons from the V1–V5 evolution

### Adapting the method to a scientific figure

- Choose the physicist's question and retain only the inputs, transformations,
  outputs, and feedback needed to answer it.
- Carry over V5's selectivity and stable visual roles, not its inventory of
  software components, diagnostics, or contracts.
- Put details and evidence in an archived companion; do not surround a figure
  with a tutorial when the requested deliverable is the figure itself.
- Judge the rendered figure at reading size. Successful rendering alone does
  not establish readability; cut boxes and labels before shrinking the text.

### Architecture revisions

| Version | Contribution | Lesson carried forward |
| --- | --- | --- |
| Original evidence audit | Comprehensive context, dependencies, workflows, decisions, uncertainty, and verification evidence | Preserve a detailed evidence layer, but do not make it the primary onboarding view. |
| V2 narrative | Top-down explanation built around offline and online work | Start with the whole story and zoom only when the reader needs detail. |
| V3 object map | Containment, function assembly, evaluation paths, lifecycle, remembered state, and file mapping | Ownership and state deserve their own view; they cannot be inferred safely from imports. |
| V4 responsibility map | Stable visual rules and a fixed four-layer frame | Stable orientation helps, but a layer model becomes misleading when real calls cross layers or an entity has several roles. |
| V5 component map | C4-inspired boundary and component views, exact ownership, runtime sequences, artifact lifecycle, contracts, and crosswalk | Use concrete runtime responsibilities without claiming that files are containers or that a strict layer rule exists. |

The separate [big-picture science workflow](legacy/BIG_PICTURE.md) adds another lesson:
software architecture and scientific argument should support each other without
being collapsed into one diagram.

## Definition of done

An architecture revision is ready only when:

- [ ] its audience, questions, scope, and non-goals are explicit;
- [ ] current code and artifacts, not memory alone, support every material
      claim;
- [ ] components are responsibility groups rather than a disguised file list;
- [ ] calls, ownership, persistence, optional paths, and static dependencies are
      not conflated;
- [ ] offline training and online inference are both traceable end to end;
- [ ] retained, derived, cached, request-local, and durable state are distinct;
- [ ] scientific units, validity, fidelity, adequacy, precision, provenance, and
      trust boundaries are represented where relevant;
- [ ] absent capabilities and unresolved questions are stated honestly;
- [ ] every component maps back to concrete symbols and source files;
- [ ] every diagram renders successfully;
- [ ] relevant tests and documentation checks pass; and
- [ ] architecture-affecting code changes update the current architecture
      documentation in the same body of work; and
- [ ] the docs index points readers to the new current map while preserving
      earlier versions as history.

## Architecture documentation during code changes

Use this methodology whenever code work adds, removes, splits, merges, or
materially changes an architectural responsibility, boundary, ownership
relationship, runtime sequence, artifact lifecycle, scientific contract, or
optional execution path.

Before implementation:

1. read the current architecture map and this methodology;
2. name the components and contracts the proposed code may affect; and
3. state the expected architecture delta alongside the implementation plan.

During implementation, keep a short architecture change record. Capture only
material changes: what changed, why, which component or boundary moved, and
which view or contract must be revised. Do not turn ordinary function-level
edits into architectural events.

Before completion:

1. reconcile the change record against the implemented code;
2. update the current architecture documentation in the same body of work;
3. add or revise source crosswalks and scientific contracts where needed;
4. run the negative-space audit so removed or absent behavior is not implied;
5. render changed diagrams and run the relevant code and documentation checks;
   and
6. save the verified change and its supersession boundary in project memory.

## Updating the architecture later

Do not edit a historical version to match new code. Create the next version when
the mental model changes materially; make a small in-place correction only when
the model remains the same and the old wording was inaccurate.

For a new version:

1. copy this method, not the previous diagram layout;
2. reconstruct the fact register from current evidence;
3. identify which reader questions changed;
4. retain stable component identities where the responsibilities truly remain;
5. explain every renamed, split, merged, added, or removed component;
6. rerun the negative-space audit and verification checklist; and
7. update the docs index and save a concise project-scoped memory observation
   containing the rationale, verification evidence, and supersession boundary.

### V6 requirement

Architecture V6 must include an explicit **Changes from V5** record. For every
material delta, identify the V5 component or contract affected, the verified V6
behavior, the reason for the change, and the evidence that supports it.

After V6 is verified, update this methodology with genuinely reusable lessons
from creating it. Preserve the historical V1–V5 lessons; revise the method only
where V6 demonstrates a better general practice, a failed assumption, or a new
class of architectural evidence. The V6 map describes the system, while this
document continues to describe how trustworthy maps are made.
