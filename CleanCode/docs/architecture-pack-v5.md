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

## Boundary view — user to package

The notebook or script calls the `lrom` package directly. The library's files
also import code from one another to carry out that work. There is no separate
service or worker.
Training data can stay in memory. Saving them as an `.npz` file is optional;
loading that file later lets a researcher reuse the full-order results.

```mermaid
flowchart LR
    subgraph interaction["User to Package"]
        caller["Research code<br/>notebook · script"]
        library["lrom package"]
        caller -->|"HardRoutedScatteringLROM.cross_section()"| library
    end
    train[("TrainingData .npz<br/>optional")]
    model[("Trained LROM<br/>.pkl files")]
    gpu["Optional JAX/GPU"]

    library -->|"save: TrainingData.save()"| train
    train -->|"load: TrainingData.load()"| library
    model -->|"load only: load_three_window_lrom()<br/>no .pkl save function"| library
    library -.->|"partial_wave_s_matrices_gpu()"| gpu

    classDef person fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef system fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef store fill:#F3E8FF,stroke:#7E22CE,color:#111827
    classDef optional fill:#F3F4F6,stroke:#6B7280,color:#111827,stroke-dasharray:5 5
    class caller person
    class library system
    class train,model store
    class gpu optional
```

`TrainingData` is an object in memory that holds full-order results for the
sampled cases, including wavefunctions, potentials, and S-matrix values. The
optional `.npz` file saves that object for later reuse.
`ScatteringLROM` has no public `.pkl` save or load method. The separate
`load_three_window_lrom()` function reads the three repository-trained LROM
files; only trusted pickle files should be loaded.
`partial_wave_s_matrices()` processes many cases at once. Its reduced solves
use NumPy on the CPU by default, or an explicitly supplied runtime. The
separate `partial_wave_s_matrices_gpu()` method takes a GPU solver directly.
Figure 1 shows the boundary around the package: who uses it, which files it
reads or writes, and which optional hardware it can call. Later figures show
how the calculations work inside that boundary.
`HardRoutedScatteringLROM.cross_section()` on the caller arrow is one example:
it returns the cross section at the chosen angles for one input case. It is
not the only call a notebook can make.

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

## Fig-2-sci — scientific workflow of the LROM

### Mermaid version

```mermaid
%%{init: {"flowchart": {"curve": "linear", "nodeSpacing": 22, "rankSpacing": 28, "wrappingWidth": 360, "subGraphTitleMargin": {"top": 8, "bottom": 18}}, "themeVariables": {"fontSize": "17px"}}}%%
flowchart LR
    subgraph SCI_OFFLINE["Offline stage"]
        direction TB
        Oinput["Input: training cases αₘ and central reference α₀<br/>α = (θ, A, Z, E)"]

        subgraph Obr["Training calculations"]
            direction LR
            subgraph Owave["Wavefunction snapshots"]
                direction TB
                Ofom["1. Solve the radial equation with Numerov<br/>φ″ = [ℓ(ℓ+1)/s² + U(s; α) − 1] φ<br/>s = kr; regular origin φ ∝ s<sup>ℓ+1</sup>"]
                Obasis["2. Center on φ₀ and retain n SVD modes<br/>δφₘ = φ(s; αₘ) − φ₀(s)<br/>Φ = [v₁, …, vₙ]"]
                Ocoords["3. Project the snapshots<br/>aₘ = argminₐ ‖W<sup>1/2</sup>(δφₘ − Φa)‖²<br/>W: trapezoid integration weights"]
                Ofom --> Obasis --> Ocoords
            end
            subgraph Opot["Potential predictors"]
                direction TB
                Ovalues["1. Evaluate scaled training potentials<br/>U(s; α) = V<sub>ℓj</sub>(s/k; α) / E<sub>cm</sub>(α)<br/>ΔUₘ = U(s; αₘ) − U(s; α₀)"]
                Oselect["2. Select n<sub>U</sub> predictor locations s<sub>q</sub><br/>Greedy MaxVol on truncated SVD modes of ΔU"]
                Ofeatures["3. Center and scale the selected values<br/>p<sub>q</sub> = [U(s<sub>q</sub>; α) − U(s<sub>q</sub>; α₀)] / σ<sub>q</sub><br/>Append centered/scaled A, Z, E<br/>σ<sub>q</sub> = max(std<sub>train</sub>, 10⁻¹²)"]
                Ovalues --> Oselect --> Ofeatures
            end
        end

        Oinput --> Ofom
        Oinput --> Ovalues
        Ofit["4. Learn matrices M<sub>q</sub> and vectors b<sub>q</sub><br/>rₘ = aₘ + ∑<sub>q</sub> p<sub>q</sub>(αₘ)(M<sub>q</sub>aₘ − b<sub>q</sub>)<br/>Minimize ∑ₘ ‖rₘ‖² + λ<sub>eff</sub>‖Θ‖²<sub>F</sub><br/>λ<sub>eff</sub> = λ σ<sub>max</sub>(D)²"]
        Oboundary["5. Cache values at the matching point s*<br/>v₀ = φ₀(s*), v = Φ(s*)<br/>d₀ = φ₀′(s*), d = Φ′(s*)"]
        Ocoords -->|training coordinates| Ofit
        Ofeatures -->|training predictors| Ofit
        Obasis -->|reference and basis| Oboundary
        Ostore["Trained local model<br/>Predictor locations, reference values and scales<br/>Learned M<sub>q</sub>, b<sub>q</sub>; cached v₀, v, d₀, d<br/>Supplied models: n = 14, n<sub>U</sub> = 20, n<sub>p</sub> = 23; λ = 10⁻¹⁴"]
        Ofit --> Ostore
        Oboundary --> Ostore
        Ofeatures --> Ostore
    end

    subgraph SCI_ONLINE["Online stage"]
        direction TB
        Qinput["Input: new α = (θ, A, Z, E) and angles ϑ"]
        Qwindow["Select one trained energy window using E<br/>[5, 70), [70, 135), [135, 200] MeV"]
        Qfeatures["1–2. Evaluate and normalize the selected potential values<br/>U<sub>q</sub> = V<sub>ℓj</sub>(s<sub>q</sub>/k; α) / E<sub>cm</sub>(α)<br/>p<sub>q</sub> = [U<sub>q</sub> − U(s<sub>q</sub>; α₀)] / σ<sub>q</sub><br/>Append centered/scaled A, Z, E"]
        Qassemble["3–4. Assemble the learned system<br/>M(p) = I + ∑<sub>q</sub> p<sub>q</sub>M<sub>q</sub><br/>b(p) = ∑<sub>q</sub> p<sub>q</sub>b<sub>q</sub>"]
        Qsolve["5. Solve the n × n system for each partial wave<br/>M(p) a = b(p)"]
        Qboundary["6. Evaluate boundary values and match to free waves<br/>φ* = v₀ + va; φ*′ = d₀ + da; L = φ*′/φ*<br/>S<sub>ℓ</sub><sup>±</sup> = −(h₋′ − Lh₋) / (h₊′ − Lh₊)"]
        Qobservable["Combine partial waves into f and g<br/>dσ/dΩ = 10 (|f|² + |g|²)  [mb/sr]<br/>Fast prediction uses boundary values;<br/>full wavefunctions are optional"]
        Qinput --> Qwindow --> Qfeatures --> Qassemble --> Qsolve --> Qboundary --> Qobservable
    end

    SCI_OFFLINE -->|reuse trained predictors, equation coefficients and boundary data| SCI_ONLINE

    classDef input fill:#FFFFFF,stroke:#20385A,color:#182330,stroke-dasharray:5 4
    classDef wave fill:#F3F8FD,stroke:#4689CF,color:#182330
    classDef potential fill:#FFFAF1,stroke:#D58A14,color:#182330
    classDef fit fill:#FFF6F5,stroke:#BF3745,color:#182330
    classDef online fill:#F7F3FB,stroke:#7352B4,color:#182330
    classDef result fill:#F7FBF4,stroke:#50883C,color:#182330
    classDef stored fill:#F7F9FC,stroke:#20385A,color:#182330
    class Oinput,Qinput input
    class Ofom,Obasis,Ocoords,Oboundary wave
    class Ovalues,Oselect,Ofeatures potential
    class Ofit fit
    class Qfeatures,Qassemble,Qsolve online
    class Qboundary,Qobservable result
    class Ostore,Qwindow stored
    style SCI_OFFLINE fill:#FFFFFF,stroke:#20385A,stroke-width:2px
    style SCI_ONLINE fill:#FFFFFF,stroke:#20385A,stroke-width:2px
    style Obr fill:#FFFFFF,stroke:none
    style Owave fill:#F3F8FD,stroke:#4689CF
    style Opot fill:#FFFAF1,stroke:#D58A14
```

### Illustrated version

![Fig-2-sci: offline construction and online evaluation of the learned reduced-order scattering model](figures/fig-2-sci.png)

[Vector version](figures/fig-2-sci.svg).

**Fig-2-sci.** Offline construction and online evaluation of the supplied
three-window neutron-elastic LROM. Each window has a separate reduced model
for every partial wave. Numerov solutions on the common coordinate \(s=kr\)
provide centered wavefunction modes and training coordinates. Selected
potential values, together with \(A,Z,E\), become normalized predictors for
the learned implicit equation. The illustrated version uses colored arrows
for predictor definitions (orange), fitted equation coefficients (red), and
boundary data (blue); the Mermaid version collects these in the trained-model
box. At a new input, the selected window evaluates these predictors,
solves the reduced equations, obtains \(S_\ell^\pm\) by boundary matching,
and combines partial waves into the differential cross section.

Here \(\phi_0=\phi(s;\alpha_0)\) is the central reference solution, \(W\) contains
trapezoid integration weights, and \(s_*\) is the matching point. \(\Theta\)
collects the fitted \(M_q,b_q\) coefficients; the regression matrix \(D\) has
row blocks \([p_q(\alpha_m)a_m^T,-p_q(\alpha_m)]\).
\(h_\pm=s[j_\ell(s)\pm i y_\ell(s)]\), with primes denoting \(s\)-derivatives.
The amplitudes \(f,g\) are the spin-nonflip and spin-flip partial-wave sums,
including their \(1/(2ik)\) factors; the factor 10 converts fm² to mb.

Layout follows Fig. 3 (p. 7) of the
[ROSE paper](../../../scientific_archive/ROSE_Guide/ROSEPaper%5B7945%5D.pdf).
The equations and steps here follow the current `lrom` implementation:
`problem.py`, `fom.py`, `reduced.py`, `emulator.py`, `deployment.py`, and
`observables.py`. This LROM learns its matrix and right-hand side from
snapshot coordinates; no ROSE EIM inverse or Galerkin operator projection
is used in this path.

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
