# Scattering LROM — Architecture Map (V5)

> A software mental model for the live `CleanCode` package. This is a selective,
> C4-inspired set of views, not a claim that Python files or responsibility groups
> are C4 containers. Detailed audit evidence remains in
> [`legacy/architecture-pack.md`](legacy/architecture-pack.md).

## How to read this map

Each component has a stable ID (`C1`–`C8`). A later heading such as
**Zoom: C3 → object ownership** means that the figure opens one box from the main
component map. Component and sequence arrows are calls; ownership arrows are labeled
with the exact relationship. Flowchart component colors stay stable; sequence diagrams
use neutral styling.

The notation is informed by the [C4 diagram types](https://c4model.com/diagrams)
and [C4 notation guidance](https://c4model.com/diagrams/notation), while staying small
enough to match this single-process scientific library.

## Boundary view — where the library runs

The caller hosts the library in one Python process. There is no service, worker, or
runtime dispatcher behind `lrom.__init__`; that module only re-exports public names.

```mermaid
flowchart LR
    subgraph process["One caller-owned Python process"]
        caller["Research code<br/>notebook · script · test"]
        library["scattering-lrom<br/>in-process library"]
        caller -->|imports and calls| library
    end
    train[("TrainingData .npz<br/>optional")]
    model[("Trusted model .pkl<br/>supplied deployment")]
    gpu["Optional JAX/GPU<br/>batched dense solve"]

    library -->|save / load| train
    model -->|trusted unpickle| library
    library -.->|explicit partial-wave solve route| gpu

    classDef person fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef system fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef store fill:#F3E8FF,stroke:#7E22CE,color:#111827
    classDef optional fill:#F3F4F6,stroke:#6B7280,color:#111827,stroke-dasharray:5 5
    class caller person
    class library system
    class train,model store
    class gpu optional
```

## Main component map

This view names the runtime responsibilities that matter when following a request.
It is not an import graph and does not impose a top-to-bottom layer rule.

```mermaid
flowchart TB
    C1["C1 Public surface<br/>re-exports + pretrained loader"]
    C2["C2 Problem + FOM<br/>physics state and snapshots"]
    C3["C3 Local emulator<br/>train · inspect · evaluate"]
    C4["C4 Reduced model<br/>basis + learned equation"]
    C5["C5 Packed evaluator<br/>batched online equation"]
    C6["C6 Observable kernel<br/>partial waves → mb/sr"]
    C7["C7 Hard router<br/>energy-window selection"]
    C8["C8 Solve runtime<br/>NumPy or optional JAX"]

    C1 -->|repack| C3
    C1 -->|construct| C7
    C3 -->|problem + matching| C2
    C3 -->|fit| C4
    C3 -->|evaluate| C5
    C3 -->|uses| C6
    C7 -->|select| C3
    C7 -->|batch S| C5
    C7 -->|share kernel| C6
    C5 -->|potential + k| C2
    C5 -->|solve| C8

    classDef boundary fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef science fill:#FFEDD5,stroke:#C2410C,color:#111827
    classDef model fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef deploy fill:#EDE9FE,stroke:#7C3AED,color:#111827
    class C1 boundary
    class C2,C6 science
    class C3,C4,C5 model
    class C7,C8 deploy
```

The shortest useful mental image is:

> `C7 router → C3 local emulator → channel model → C4 LearnedROM`, with `C5`
> as a derived fast layout and `C6` as the shared observable calculation.

`C2 NumerovFOM` participates in snapshot generation, not online prediction.

## Zoom: C3 → object ownership

This figure separates **retained state** from **calls**. A `ScatteringLROM` retains its
problem, optional training database, channel models, derived packed representation, and kernel
cache. The supplied pickle artifacts have `training_data is None`; loading repacks
their channel models and then constructs the router.

```mermaid
flowchart TB
    router["C7 HardRoutedScatteringLROM"]
    local["C3 ScatteringLROM"]
    problem["ScatteringProblem"]
    data["TrainingData<br/>optional retained state"]
    channel["_ChannelModel<br/>one per partial wave"]
    learned["C4 LearnedROM + ReducedBasis"]
    packed["C5 Packed online model<br/>derived layout"]
    prediction["Prediction<br/>temporary query view"]
    kernel["C6 ElasticCrossSectionKernel<br/>cached / shared"]

    router -->|retains models| local
    router -->|retains one shared kernel| kernel
    local -->|references| problem
    local -->|retains or None| data
    local -->|retains channel models| channel
    channel -->|retains| learned
    local -->|retains derived layout| packed
    local -->|creates| prediction
    prediction -->|references| local
    local -->|caches| kernel

    classDef science fill:#FFEDD5,stroke:#C2410C,color:#111827
    classDef model fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef deploy fill:#EDE9FE,stroke:#7C3AED,color:#111827
    class router deploy
    class local,channel,learned,packed,prediction,data model
    class problem,kernel science
```

`Prediction` is request-local inspection state. It caches reduced coordinates and
S matrices only as its methods are called. The packed object is not another learned
model: it is a contiguous or grouped representation derived from the channel models.

## Sequence: offline snapshot generation and training

This is the actual call ownership. In particular, `ScatteringLROM.train()` directly
uses the matching-index and scaled-derivative helpers while wrapping each fitted
`LearnedROM` in a channel model.

```mermaid
sequenceDiagram
    autonumber
    actor Caller
    participant Problem as C2 ScatteringProblem
    participant FOM as C2 NumerovFOM
    participant Local as C3 ScatteringLROM
    participant ROM as C4 LearnedROM
    participant Match as FOM matching helpers
    participant Packed as C5 Packed evaluator

    Caller->>Problem: generate_training_data(samples)
    loop each channel and sample
        Problem->>FOM: solve(problem, sample, channel)
        FOM-->>Problem: phi(s), S
    end
    Problem-->>Caller: TrainingData(s mesh, potentials, waves, S)
    Caller->>Local: train(data)
    loop each channel
        Local->>ROM: fit(reference, snapshots, potentials, controls)
        ROM-->>Local: ReducedBasis + learned implicit equation
        Local->>Match: matching_index() and scaled_derivative_at()
        Match-->>Local: boundary maps
    end
    Local->>Packed: build from channel models
    Local-->>Caller: same trained ScatteringLROM
```

The input space can reproducibly sample and validate cases, but validation is an
explicit caller action. `generate_training_data()` does not call it automatically.

## Sequence: routed online inference

The packed evaluator calculates potential features, assembles and solves the learned
reduced systems, and applies the boundary-matching formula for `S` itself. It performs
no Numerov solve and normally reconstructs no wavefunction.

```mermaid
sequenceDiagram
    autonumber
    actor Caller
    participant Router as C7 Hard router
    participant Local as C3 Local emulator
    participant Packed as C5 Packed evaluator
    participant Solve as C8 Solve runtime
    participant Kernel as C6 Observable kernel

    alt scalar cross_section(sample)
        Caller->>Router: cross_section(sample)
        Router->>Local: cross_section(sample, fixed angles)
        Local->>Packed: s_matrices(problem, sample, controls)
        Packed->>Packed: features → reduced solve → boundary S formula
        Packed-->>Local: channel S values and k
        Local->>Kernel: evaluate(S+, S-)
        Kernel-->>Local: angular result at unit k
        Local->>Local: divide by k²
        Local-->>Router: cross-section array
        Router-->>Caller: cross-section array
    else routed partial_wave_s_matrices(samples)
        Caller->>Router: partial_wave_s_matrices(samples)
        Router->>Packed: s_matrices_batch(..., linear_solver)
        alt injected runtime exists
            Packed->>Solve: solve stacked reduced systems
            Solve-->>Packed: reduced coordinates
        else no injected runtime
            Packed->>Packed: NumPy solve
        end
        Packed->>Packed: calculate boundary S
        Packed-->>Router: channel S values and k
        Router-->>Caller: S+, S-, k
    end
```

Important API boundaries:

- The router's ordinary `cross_sections()` groups and chunks samples but does **not**
  pass its stored runtime into local cross-section evaluation.
- Runtime injection is used by routed `partial_wave_s_matrices()`.
- `ScatteringLROM.cross_sections(..., linear_solver=...)` is a separate local seam.
- `partial_wave_s_matrices_gpu(..., solver=...)` is a separate explicit GPU method.

## Artifact relationships and model lifecycle

```mermaid
flowchart LR
    fom["C2 FOM generation"]
    data["TrainingData object"]
    npz[("optional .npz")]
    local["C3 trained local model"]
    pickle[("trusted supplied .pkl")]
    repacked["C3 repacked local model"]
    router["C7 routed deployment"]
    prediction["Prediction<br/>temporary"]

    fom -->|creates| data
    data -->|save| npz
    npz -->|load| data
    data -->|train| local
    local -->|repack| repacked
    pickle -->|unpickle then repack| repacked
    repacked -->|construct router| router
    repacked -->|predict| prediction

    classDef science fill:#FFEDD5,stroke:#C2410C,color:#111827
    classDef model fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef deploy fill:#EDE9FE,stroke:#7C3AED,color:#111827
    classDef store fill:#F3E8FF,stroke:#7E22CE,color:#111827
    class fom science
    class data,local,repacked,prediction model
    class router deploy
    class npz,pickle store
```

There is a public, portable `TrainingData.save/load` path. There is currently no public
model-save API; the repository-supplied pickle files are trusted first-party artifacts,
not a general interchange format.

## Scientific software contracts

| Contract | Boundary that enforces or exposes it |
|---|---|
| **Coordinates and units** | Internally, snapshots, bases, predictors, and matching use `s = k r`; `Prediction.physical_wavefunction()` maps back to physical radius `r` in fm. Cross sections are returned in mb/sr. |
| **Model validity** | `InputSpace.validate()` checks declared ranges and integer choices only when called. Hard routing selects a window deterministically; it does not establish scientific validity outside calibrated/tested support. |
| **Verification, fidelity, adequacy** | Numerov-vs-ROSE checks the package FOM implementation. LROM-vs-the-package's own Numerov result measures emulator fidelity. Experiment comparisons additionally test optical-model adequacy. These are different claims. |
| **Precision and backend** | CPU solves follow their NumPy array dtype. The `precision` switch configures the optional JAX/GPU path. Backend substitution changes only supported batched dense reduced solves; it does not change the fitted equation. |
| **Reproducibility and trust** | Sampling is seedable; curated data carry pinned provenance; training metadata live beside supplied models. Only repository-trusted pickle artifacts should be loaded. |
| **Online cost** | Packed inference evaluates selected potential points and small reduced systems. Full wavefunction reconstruction is opt-in through `Prediction`; online Numerov solves are absent. |

## Component crosswalk

| ID | Main classes or functions | Primary source files |
|---|---|---|
| C1 | public re-exports; `load_three_window_lrom()` | `lrom/__init__.py`, `lrom/pretrained.py` |
| C2 | `ScatteringProblem`, `NumerovFOM`, `ScatteringSystem`, potential and matching functions | `lrom/problem.py`, `lrom/fom.py`, `lrom/physics.py` |
| C3 | `ScatteringLROM`, `Prediction`, `_ChannelModel` | `lrom/emulator.py` |
| C4 | `ReducedBasis`, `LearnedROM` | `lrom/reduced.py` |
| C5 | `_PackedOnlineModel`, `_RaggedPackedOnlineModel` | `lrom/emulator.py` |
| C6 | `ElasticCrossSectionKernel`, assembly functions | `lrom/observables.py` |
| C7 | `HardRoutedScatteringLROM` | `lrom/deployment.py` |
| C8 | `Runtime`, `get_runtime()` | `lrom/backends.py` |

## Current architecture vs possible evolution

The diagrams above describe current behavior. Possible future work, deliberately not
shown as present architecture:

- Define a public, versioned model serialization contract if collaborators must create
  and exchange fitted models; do not treat raw pickle as that contract.
- Unify batch-solver injection only if one consistent public behavior is desired across
  router cross sections, partial waves, and local batch calls.
- Attach explicit support-domain metadata to trained artifacts if runtime rejection or
  warnings outside calibrated ranges become a requirement.

These are recommendations, not missing boxes or implied dispatchers in the current code.
