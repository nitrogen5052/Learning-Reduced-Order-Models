# Scattering LROM — Architecture Map (V5)

> A software mental model for the live `CleanCode` package. This is a selective,
> C4-inspired set of views, not a claim that Python files or responsibility groups
> are C4 containers. Detailed audit evidence remains in
> [`legacy/architecture-pack.md`](architecture-pack.md).

## How to read this map

Figure 2 groups whole Python files and labels the interactions between them.
Later figures retain the earlier responsibility IDs (`C1`–`C8`) during review.
**Zoom: C3 → object ownership** opens the local-emulator responsibility inside
`emulator.py`. Follow each arrow label for its meaning; arrows can describe
imports, calls, supplied information, or file transfers. Sequence diagrams
show call order.

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

## Internal architecture overview — Figure 2

Figure 2 summarizes the code inside the package. Boxes group whole Python
files; arrows state a short purpose. External callers, artifacts, hardware,
and the package boundary are shown in the surrounding views rather than here.

The notebook-import file is grouped with model loading. Potential and
channel definitions are grouped with the scattering problem. These merge
files that had only one internal connection in the selective Figure 3 map;
they do not claim those files have only one import or caller in the code.
Figure 3 expands these groups, and Figure 4 adds detailed interaction labels.

```mermaid
%%{init: {"flowchart": {"curve": "linear", "nodeSpacing": 25, "rankSpacing": 38, "wrappingWidth": 280}, "themeVariables": {"fontSize": "18px"}}}%%
flowchart TB
    access["<b>\_\_init\_\_.py · pretrained.py</b><br/><small>Notebook access and model loading</small>"]
    dep["<b>deployment.py</b><br/><small>Energy-window selection</small>"]
    em["<b>emulator.py</b><br/><small>Emulator training and prediction<br/>Includes packed evaluators</small>"]
    red["<b>reduced.py</b><br/><small>Reduced basis and learned equation</small>"]
    dat["<b>data.py</b><br/><small>Sampling and training results</small>"]
    prob["<b>problem.py · physics.py<br/>channels.py</b><br/><small>Scattering problem and physics</small>"]
    fom["<b>fom.py</b><br/><small>Full-order solve and boundary matching</small>"]
    obs["<b>observables.py</b><br/><small>Scattering observables</small>"]
    solvers["<b>backends.py · cpu_batched.py<br/>cuda_batched.py</b><br/><small>Optional reduced-system solvers</small>"]
    helpers["<b>curated_data.py · global_kd.py<br/>diagnostics.py</b><br/><small>Research data and comparisons</small>"]

    access -->|Prepare loaded models| em
    access -->|Create energy router| dep
    dep -->|Select and evaluate model| em
    dep -->|Assemble cross sections| obs
    em -->|Fit and evaluate reduced model| red
    em -->|Read training snapshots| dat
    em -->|Obtain physical inputs| prob
    em -->|Match boundary values| fom
    em -->|Assemble cross sections| obs
    prob -->|Solve radial equation| fom
    fom -->|Obtain solver inputs| prob
    prob -->|Store full-order results| dat
    prob -->|Assemble full-order cross sections| obs
    dat -->|Sample input space| red
    em -.->|Solve reduced equations| solvers
    dep -.->|Use supplied solver| solvers
    helpers -->|Inspect trained bases| em
    helpers -->|Read training results| dat
    helpers -->|Match projected waves| fom
    helpers -->|Calculate comparison observables| obs

    classDef accessColor fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef scienceColor fill:#FFEDD5,stroke:#C2410C,color:#111827
    classDef modelColor fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef runtimeColor fill:#EDE9FE,stroke:#7C3AED,color:#111827
    classDef helperColor fill:#F3F4F6,stroke:#6B7280,color:#111827
    class access accessColor
    class prob,fom,obs scienceColor
    class em,red,dat modelColor
    class dep,solvers runtimeColor
    class helpers helperColor
```

## Whole-file overview — Figure 3

Figure 3 expands Figure 2 inside the `lrom package` boundary from Figure 1. Boxes contain whole
Python files; the optional solver and research-helper boxes collect several
whole files. Each file belongs to one box. Arrows give a short purpose for
each interaction. Positions do not prescribe calculation order.

This overview covers all 17 package files. Figure 4 keeps the same boxes and
connections, with more detailed interaction labels for manual code review.

```mermaid
%%{init: {"flowchart": {"curve": "linear", "nodeSpacing": 30, "rankSpacing": 50, "wrappingWidth": 280, "subGraphTitleMargin": {"top": 10, "bottom": 24}}, "themeVariables": {"fontSize": "18px"}}}%%
flowchart TB
    user["Research code<br/>notebook · script"]
    saved[("Trained LROM<br/>.pkl files")]
    archive[("TrainingData .npz<br/>optional")]
    records[("Curated experimental<br/>JSON files")]
    gpu["Optional JAX/GPU"]
    cuda["Optional CUDA libraries<br/>direct loader: Windows"]

    subgraph package["lrom package"]
        direction TB
        init["\_\_init\_\_.py<br/>Notebook imports"]
        pre["pretrained.py<br/>Load trained models"]
        dep["deployment.py<br/>Energy-window selection"]
        em["emulator.py<br/>Train, inspect and predict<br/>Includes packed evaluators"]
        red["reduced.py<br/>Basis and learned equation"]
        dat["data.py<br/>Sampling and TrainingData"]
        prob["problem.py<br/>Define scattering cases<br/>Generate training results"]
        fom["fom.py<br/>Numerov solve, kinematics container<br/>and boundary matching"]
        phys["physics.py<br/>Potentials and kinematics"]
        ch["channels.py<br/>Partial-wave definitions"]
        obs["observables.py<br/>S matrices → cross sections"]
        solvers["Optional reduced-system solvers<br/>backends.py: NumPy / JAX Runtime<br/>cpu_batched.py: threaded CPU<br/>cuda_batched.py: direct CUDA"]
        helpers["Research helpers<br/>curated_data.py: experimental records<br/>global_kd.py: KD coefficient mapping<br/>diagnostics.py: numerical comparisons"]

        init -->|"Expose loader"| pre
        pre -->|"Prepare loaded models"| em
        pre -->|"Create energy router"| dep
        dep -->|"Select and evaluate model"| em
        dep -->|"Assemble cross sections"| obs
        em -->|"Fit and evaluate reduced model"| red
        em -->|"Read training snapshots"| dat
        em -->|"Obtain physical inputs"| prob
        em -->|"Match boundary values"| fom
        em -->|"Assemble cross sections"| obs
        prob -->|"Solve radial equation"| fom
        fom -->|"Obtain solver inputs"| prob
        prob -->|"Evaluate potential and kinematics"| phys
        prob -->|"Define partial waves"| ch
        prob -->|"Store full-order results"| dat
        prob -->|"Assemble full-order cross sections"| obs
        dat -->|"Sample input space"| red
        em -.->|"Solve reduced equations"| solvers
        dep -.->|"Use supplied solver"| solvers
        helpers -->|"Inspect trained bases"| em
        helpers -->|"Read training results"| dat
        helpers -->|"Match projected waves"| fom
        helpers -->|"Calculate comparison observables"| obs

    end

    user -->|"Import package names"| init
    user -->|"Request cross sections"| dep
    saved -->|"Load trained models"| pre
    dat -->|"Save training data"| archive
    archive -->|"Load training data"| dat
    solvers -.->|"Run JAX GPU solves"| gpu
    solvers -.->|"Run direct CUDA solves"| cuda
    records -->|"Read experimental records"| helpers
    user -.->|"Use research tools"| helpers

    classDef accessColor fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef scienceColor fill:#FFEDD5,stroke:#C2410C,color:#111827
    classDef modelColor fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef runtimeColor fill:#EDE9FE,stroke:#7C3AED,color:#111827
    classDef artifactColor fill:#F3E8FF,stroke:#7E22CE,color:#111827
    classDef optionalColor fill:#F3F4F6,stroke:#6B7280,color:#111827
    class init,pre accessColor
    class prob,fom,phys,ch,obs scienceColor
    class em,red,dat modelColor
    class dep,solvers runtimeColor
    class saved,archive,records artifactColor
    class user,gpu,cuda,helpers optionalColor
    style package fill:#FFFFFF,stroke:#15803D,stroke-width:2px
```

The main prediction path is model loading → energy-window selection →
emulator evaluation → scattering observables. Training also uses the
scattering problem, full-order solver, training results, and reduced model.
The two arrows between `problem.py` and `fom.py` represent different calls:
the problem requests a radial solution; the solver obtains its physical
inputs from the problem.

## File interactions — Figure 4

This detailed version names functions or describes the call, data access,
or file transfer. The files and connections are the same as in Figure 3.
Normal function returns and supporting imports are omitted.

```mermaid
%%{init: {"flowchart": {"curve": "linear", "nodeSpacing": 30, "rankSpacing": 50, "wrappingWidth": 280, "subGraphTitleMargin": {"top": 10, "bottom": 24}}, "themeVariables": {"fontSize": "18px"}}}%%
flowchart TB
    user["Research code<br/>notebook · script"]
    saved[("Trained LROM<br/>.pkl files")]
    archive[("TrainingData .npz<br/>optional")]
    records[("Curated experimental<br/>JSON files")]
    gpu["Optional JAX/GPU"]
    cuda["Optional CUDA libraries<br/>direct loader: Windows"]

    subgraph package["lrom package"]
        direction TB
        init["\_\_init\_\_.py<br/>Notebook imports"]
        pre["pretrained.py<br/>Load trained models"]
        dep["deployment.py<br/>Energy-window selection"]
        em["emulator.py<br/>Train, inspect and predict<br/>Includes packed evaluators"]
        red["reduced.py<br/>Basis and learned equation"]
        dat["data.py<br/>Sampling and TrainingData"]
        prob["problem.py<br/>Define scattering cases<br/>Generate training results"]
        fom["fom.py<br/>Numerov solve, kinematics container<br/>and boundary matching"]
        phys["physics.py<br/>Potentials and kinematics"]
        ch["channels.py<br/>Partial-wave definitions"]
        obs["observables.py<br/>S matrices → cross sections"]
        solvers["Optional reduced-system solvers<br/>backends.py: NumPy / JAX Runtime<br/>cpu_batched.py: threaded CPU<br/>cuda_batched.py: direct CUDA"]
        helpers["Research helpers<br/>curated_data.py: experimental records<br/>global_kd.py: KD coefficient mapping<br/>diagnostics.py: numerical comparisons"]

        init -->|"imports/exposes<br/>load_three_window_lrom"| pre
        pre -->|"calls<br/>ScatteringLROM.repack('padded')"| em
        pre -->|"constructs HardRoutedScatteringLROM<br/>models + energy edges + angles"| dep
        dep -->|"selects model; calls cross_section(s)<br/>or packed S-matrix evaluation"| em
        dep -->|"builds and evaluates shared<br/>ElasticCrossSectionKernel"| obs
        em -->|"calls basis/equation fitting<br/>and coordinate reconstruction"| red
        em -->|"train() reads<br/>TrainingData snapshots"| dat
        em -->|"calls system() / omega()<br/>kinematics + optical parameters"| prob
        em -->|"calls boundary matching<br/>for channel inspection"| fom
        em -->|"calls angular-kernel evaluation<br/>and cross-section assembly"| obs
        prob -->|"calls NumerovFOM.solve()<br/>training waves + S matrices"| fom
        fom -->|"solve() calls system() / omega()<br/>for its physical inputs"| prob
        prob -->|"calls potential evaluation<br/>and kinematics"| phys
        prob -->|"calls elastic_channels()"| ch
        prob -->|"constructs TrainingData<br/>waves + potentials + S matrices"| dat
        prob -->|"calls full-order<br/>cross-section assembly"| obs
        dat -->|"sample() calls latin_hypercube()"| red
        em -.->|"optional solve callable<br/>M a = b; default: NumPy"| solvers
        dep -.->|"partial-wave routes<br/>call supplied solver.solve()"| solvers
        helpers -->|"diagnostics reads/projects<br/>trained channel bases"| em
        helpers -->|"diagnostics reads<br/>training waves + inputs"| dat
        helpers -->|"diagnostics calls<br/>boundary matching"| fom
        helpers -->|"diagnostics calls<br/>cross-section assembly"| obs

    end

    user -->|"from lrom import …"| init
    user -->|"cross_section() example<br/>from Figure 1"| dep
    saved -->|"pickle.load() reads<br/>three trained models"| pre
    dat -->|"TrainingData.save()"| archive
    archive -->|"TrainingData.load()"| dat
    solvers -.->|"backends.py configures<br/>JAX GPU solves"| gpu
    solvers -.->|"cuda_batched.py calls<br/>CUDA / cuBLAS solves"| cuda
    records -->|"curated_data.py reads records<br/>and converts b/sr → mb/sr"| helpers
    user -.->|"can call helper functions<br/>and wire their results into a workflow"| helpers

    classDef accessColor fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef scienceColor fill:#FFEDD5,stroke:#C2410C,color:#111827
    classDef modelColor fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef runtimeColor fill:#EDE9FE,stroke:#7C3AED,color:#111827
    classDef artifactColor fill:#F3E8FF,stroke:#7E22CE,color:#111827
    classDef optionalColor fill:#F3F4F6,stroke:#6B7280,color:#111827
    class init,pre accessColor
    class prob,fom,phys,ch,obs scienceColor
    class em,red,dat modelColor
    class dep,solvers runtimeColor
    class saved,archive,records artifactColor
    class user,gpu,cuda,helpers optionalColor
    style package fill:#FFFFFF,stroke:#15803D,stroke-width:2px
```

The notebook can load trained models or construct a scattering problem and
train an emulator from `TrainingData`. During prediction, `deployment.py`
selects a model, `emulator.py` evaluates its reduced equations and boundary
S matrices, and `observables.py` calculates the cross section. The packed
prediction code and the local emulator are both in `emulator.py`.

The problem and Numerov solver communicate in both directions:
`ScatteringProblem.generate_training_data()` calls `NumerovFOM.solve()`,
and that solver calls `problem.system()` and `problem.omega()` to obtain
its inputs. Full-order radial solves are used to generate training results;
shared kinematics and boundary helpers remain useful during prediction.

Optional solvers are supplied as objects or callables. `emulator.py` does
not import the solver implementation files. With no supplied solver, its
batched reduced systems use NumPy directly. `global_kd.py` and
`curated_data.py` are independently callable tools; their outputs are wired
into a research workflow by its caller. Their presence does not mean they
are automatically called by every prediction.

The curated-JSON and direct-CUDA connections extend Figure 1's selective
boundary view. NumPy, SciPy, pandas, and Python standard-library dependencies
are not exhaustively drawn. Bidirectional calls and file transfers are
explicitly labeled; ordinary returned numerical arrays are implicit.

For the later, not-yet-reviewed figures, the earlier responsibility IDs
remain: C1 access/loading; C2 problem/FOM; C3 local emulator; C4 reduced model;
C5 packed evaluator; C6 observables; C7 router; C8 runtime. C3 and C5 both
map to `emulator.py` in this whole-file view.

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
