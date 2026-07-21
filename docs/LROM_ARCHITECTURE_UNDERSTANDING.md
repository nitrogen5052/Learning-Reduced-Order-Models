# LROM Architecture Understanding

## Purpose and ownership

This is Daniel's maintained mental model of the v1.2 active package. It connects each public call to the state it owns and the numerical function that performs the work.

Final prose ownership remains Daniel's. Assistant-written explanations and notebook prose require his sentence-by-sentence review before advisor presentation.

## Version boundary

- `import lrom` is the public v1.2 active package.
- `lrom/__init__.py` is a small public entry point.
- `lrom_legacy.v1_2` contains the one-file authoritative implementation.
- `lrom_legacy.v2_0` is the parked future shell for wavefunctions and cross sections.

The public lifecycle is intentionally rigid. `LROM(...)`, `sampling()`, `train()`, `predict()`, `save()`, and `load()` retain their roles. A major restructure requires user approval before implementation.

## Object and state ownership

```mermaid
flowchart TB
    L["LROM object\npublic lifecycle"]
    C["LROMConfig\nimmutable physical question"]
    S["SamplingState\nFOM arrays and physical mesh"]
    T["TrainingState\nbasis, predictors, RF operators"]
    P["PredictionState\nlatest requested output"]

    L --> C
    L --> S
    L --> T
    L --> P

    C --> F["NuclearScatteringFOM\nExact ROSE high-fidelity boundary"]
    S --> E["_train_state()\nflat training sequence"]
    E --> B["BasisState"]
    E --> R["RFLROMModel"]
    T --> I["predict()"]
    I --> P
```

| Object part | Owns | Does not own |
|---|---|---|
| `LROMConfig` | Immutable physical identity | Solver outputs |
| `SamplingState` | FOM snapshots, potential arrays, designs, and physical-radius mesh | Learned model |
| `TrainingState` | Basis, predictors, RF operators, and RF diagnostics | Latest inference request |
| `PredictionState` | Latest parameters, reduced coefficients, and reconstructed wavefunctions | Training data |
| `LROM` | Lifecycle and public API | Numerical implementation details |

The state boundary separates the physical question, expensive high-fidelity data, a trained reduced model, and the latest prediction.

## Numbered source map

The banners in `lrom_legacy/v1_2/__init__.py` are stable reading landmarks.

| Part | Owns | Main names to read |
|---|---|---|
| 1. Physical configuration and potentials | Validated nuclear problem and potential schema | `LROMConfig`, `PotentialSpec`, `real_woods_saxon()` |
| 2. Parameter designs and lifecycle state | Named cases and stage-specific NumPy containers | `ParameterCases`, `SamplingDesign`, `SamplingState`, `TrainingState`, `PredictionState` |
| 3. Centered reduced basis | Spatial compression around the central solution | `build_basis()`, `project_coordinates()`, `reconstruct()` |
| 4. Optional analysis utilities | Explicit LS benchmark and raw errors | `least_squares_baseline()`, `pointwise_absolute()`, `relative_l2()` |
| 5. Predictor construction | Features describing parameter or potential variation | `build_parameter_predictor()`, `build_potential_predictor()` |
| 6. RF-LROM numerical core | Stacked operator fit and online reduced solve | `fit_rf_lrom()`, `solve_rf_lrom()` |
| 7. Exact ROSE high-fidelity boundary | Authoritative Runge-Kutta wavefunction snapshots | `NuclearScatteringFOM` |
| 8. RF-LROM training orchestration | Basis, predictors, coordinates, and RF fit | `_reduced_basis_state()`, `_train_state()` |
| 9. RF-LROM prediction | Named inputs through reconstructed wavefunctions | `_parameter_rows()`, `predict()` |
| 10. Portable artifacts | Prediction-critical arrays and provenance | `save_artifact()`, `load_artifact()` |
| 11. Public LROM lifecycle | State transitions and user-facing calls | `LROM`, `load()` |

## Public calls mapped to numerical work

| Public call | Requires | Private work | State effect |
|---|---|---|---|
| `LROM(...)` | Physical inputs | `LROMConfig.create()` | Creates immutable configuration |
| `.sampling(...)` | Parameter design | Design builder, then `NuclearScatteringFOM.sample()` | Creates `SamplingState`; clears stale training and prediction |
| `.reduced_basis(...)` | `SamplingState` | `_reduced_basis_state()` and `build_basis()` | Creates basis-only `TrainingState` |
| `.train(...)` | `SamplingState` | `_train_state()` | Creates complete `TrainingState`; clears stale prediction |
| `.predict(...)` | Complete `TrainingState` | `features_for_values()`, `solve_rf_lrom()`, `reconstruct()` | Replaces `PredictionState` |
| `.save(...)` | Trained object | `save_artifact()` | Writes a portable `.lrom` file |
| `lrom.load(...)` | Portable artifact | `load_artifact()` | Restores an inference-only object |

## Notebook-to-package map

Use this table to move from a notebook result to the package code that produced it.

| Notebook section | Public calls or notebook-owned method | Package section to read |
|---|---|---|
| Notebook 01: varying `Vv` | `LROM()`, `sampling()`, `train()`, explicit `least_squares_baseline()` | 7 for exact snapshots; 3 for the basis; 4 for LS; 5-6 for RF-LROM |
| Notebook 01: three-parameter predictors | `sampling()` followed by parameter- or potential-predictor `train()` | 5 for features; 6 for the operator fit |
| Notebook 01: wavefunction emulation | stored results and `testing_case()` | 3 for reconstruction; 8 for evaluation |
| Notebook 01: ROSE comparison | notebook-owned free-reference ROSE basis and EIM | Not package state; see the basis-reference and EIM boundaries below |
| Benchmark 02 | public v1.2 plus explicit LS and notebook-owned ROSE validation | Same v1.2 sections as Notebook 01 |
| Benchmark 01 | public v1.2 compared with explicit parked v2.0 | Version-routing check, not a new algorithm |
| Benchmark 03 | explicit `lrom_legacy.v2_0` cross-section shell | Parked v2.0 only; it does not define public v1.2 behavior |

## Training data flow

```mermaid
flowchart LR
    W["FOM wavefunction snapshots"] --> B["BasisState\ncompress spatial structure"]
    B --> C["training coordinates a(alpha)"]
    U["parameter or potential variation"] --> P["PredictorState\np(alpha)"]
    C --> R["fit_rf_lrom()\nlearn M_j and b_j"]
    P --> R
```

The high-fidelity wavefunctions determine the spatial basis and training coordinates. Parameter or potential samples determine the predictor features. RF-LROM learns a reduced equation connecting these descriptions.

## Prediction data flow

```mermaid
flowchart LR
    A["new named parameters alpha"] --> B["features p_j(alpha)"]
    B --> C["small RF-LROM solve"]
    C --> D["coefficients a(alpha)"]
    D --> E["phi0 + Phi a"]
    E --> F["wavefunction on radius r in fm"]
```

`alpha` is the named physical parameter vector. The index `j` labels predictor terms from 1 through `K`. The online stage solves only the reduced system; it does not call ROSE.

## The two least-squares calculations

### Training coordinates are required by RF-LROM

The centered basis represents a snapshot as

\[
\phi(\alpha_i) \approx \phi_0 + \Phi a_i.
\]

For each training snapshot, `project_coordinates()` solves

\[
a_i = \underset{a}{\operatorname{argmin}}\;
\left\|W^{1/2}\left[\phi(\alpha_i)-\phi_0-\Phi a\right]\right\|_2.
\]

`W` contains trapezoid weights on physical radius. The code forms `A = W**(1/2) Phi` and `b = W**(1/2)(phi - phi0)`, then uses `numpy.linalg.lstsq(A, b)`. It does not form normal equations because that squares the condition number and can lose accuracy. These coordinates are required training data for RF-LROM.

### LS baseline is opt-in analysis

`least_squares_baseline()` applies the same projection to a requested high-fidelity test wavefunction. It sees the target wavefunction and gives the best reconstruction available inside that fixed affine basis.

That result is an oracle basis floor, not an RF-LROM prediction. `LROM.train()` does not calculate or store it automatically. A notebook must call the baseline explicitly.

This explains why LS should normally be at least as accurate as LROM in the same basis and norm: LS receives the wavefunction it is reconstructing, while LROM infers coefficients from predictor features.

## RF-LROM fit methodology

RF-LROM assumes

\[
\left(I + \sum_{j=1}^{K}p_j(\alpha)M_j\right)a(\alpha)
= \sum_{j=1}^{K}p_j(\alpha)b_j.
\]

Once `a(alpha)` and `p_j(alpha)` are known, every unknown entry of `M_j` and `b_j` appears linearly. `fit_rf_lrom()` stacks all sample/equation rows into one complex design matrix and calls `numpy.linalg.lstsq` once. The flat solution is unpacked into the `M_j` matrices and `b_j` vectors.

At prediction time, `solve_rf_lrom()` builds the small matrix for each new predictor row and uses `numpy.linalg.solve`. This is neither another fit nor a high-fidelity solve.

## Exact and EIM boundaries

Package sampling constructs `rose.InteractionSpace` and runs the ROSE Runge-Kutta solver. No EIM basis is calculated in `sampling()`, because an EIM is not required to produce the exact high-fidelity snapshots used by RF-LROM.

A notebook-owned ROSE reduced emulator may require `rose.InteractionEIMSpace`. That EIM belongs visibly in benchmark methodology and is not package state.

- package sampling: exact interaction and authoritative FOM snapshots;
- RF-LROM training: central-reference basis, predictors, and learned operators;
- ROSE benchmark: its own free-reference basis and any required EIM.

## Basis-reference boundary

LROM uses a central-reference affine basis,

\[
\phi(\alpha) \approx \phi_{\mathrm{central}} + \Phi_{\mathrm{LROM}}a_{\mathrm{LROM}}(\alpha).
\]

The notebook-owned ROSE emulator uses a free-reference affine basis,

\[
\phi(\alpha) \approx \phi_{\mathrm{free}} + \Phi_{\mathrm{ROSE}}a_{\mathrm{ROSE}}(\alpha).
\]

Wavefunctions and errors can be compared because both reconstruct the same physical quantity. Raw coefficients cannot be compared as equal coordinates across the two conventions.

## Save/load boundary

The artifact retains configuration, physical mesh, kinematics, reduced bases, predictor transformation, RF-LROM operators, and provenance required for prediction. It omits full sampling arrays and live ROSE objects. Loading restores prediction ability, not sampling.

## How to understand a code change

For each functional change, record:

1. **Methodology:** scientific or architectural reason.
2. **Before:** previous behavior and consequence.
3. **After:** new behavior and the function or state that owns it.
4. **Execution:** commands and measured evidence.
5. **What did not change:** protected methods, physics inputs, data definitions, and unaffected states.

Then ask:

1. Which public call begins the work?
2. Which state should that call create or replace?
3. Which numbered source section owns the transformation?
4. Which arrays enter and leave, and what are their shapes?
5. Which scientific assumption makes it valid?
6. Which characterization test exposes an unintended change?

## Change record

### v1.2 package simplification

- **Methodology:** retain work required by high-fidelity sampling, RF-LROM training, prediction, and artifacts; make optional benchmarking explicit.
- **Before:** public `lrom` exposed 2.0; v1.2 sampling constructed an unnecessary EIM; `train()` automatically projected testing wavefunctions for LS; a one-use `TrainingEngine` wrapper hid the flat sequence.
- **After:** public `lrom` exposes v1.2; sampling uses `InteractionSpace`; LS is explicit; `_train_state()` shows the training sequence. `lrom_legacy.v2_0` remains parked.
- **Execution:** deterministic hashes lock central, training, testing, basis, RF matrices, RF vectors, and prediction arrays. Focused sampling, LS, lifecycle, training, and artifact tests pass.
- **What did not change:** public lifecycle meanings, physical-radius interface, ROSE Runge-Kutta solve, central-reference convention, RF-LROM equations, artifact behavior, scientific archive, or parked v2.0 physics.

### Notebook 01 ROSE reference correction

- **Methodology:** ROSE's reduced equations require the free solution as their affine reference.
- **Before:** notebook 01 supplied LROM's central solution and vectors to ROSE and treated coefficients as if both methods used one convention.
- **After:** notebook 01 constructs a separate ROSE basis around the free solution and compares reconstructed wavefunctions while keeping coefficient conventions separate.
- **Execution:** controlled results are recorded in `.agents/validation/2026-07-20-notebook01-rose-reference-results.md`.
- **What did not change:** installed packages, scientific archive, parameter samples, high-fidelity snapshots, or retained rank.

### Parked v2.0 exact high-fidelity boundary

- **Methodology:** high-fidelity sampling should evaluate the exact interaction; EIM belongs only to a reduced ROSE emulator.
- **Before:** the parked v2.0 sampler built an EIM while preparing its high-fidelity wavefunctions.
- **After:** v2.0 uses the same exact `InteractionSpace` boundary as v1.2 while retaining its spin-orbit and cross-section capabilities. It remains an explicit legacy import and is not the public package.
- **Execution:** shared wavefunction configurations remain array-identical between v1.2 and v2.0. The executed cross-section benchmark has 10 of 10 code cells complete with no error outputs.
- **What did not change:** v2.0's observable API, cross-section workflow, spin-orbit handling, benchmark inputs, or notebook-owned ROSE EIMs.

### Notebook routing and explicit LS benchmarks

- **Methodology:** each notebook names the package milestone it tests, and an LS oracle is requested only where a figure needs it.
- **Before:** benchmark cells read LS coefficients and errors as if `train()` owned them, and versioned benchmarks depended on the public import changing meaning.
- **After:** Notebook 01 and benchmark 02 use public v1.2; benchmark 01 compares public v1.2 with explicit v2.0; benchmark 03 imports explicit v2.0. Notebook LS calls now show the basis and target wavefunctions at the call site.
- **Execution:** Notebook 01 executes 14 of 14 code cells; benchmark 01 executes 4 of 4; benchmark 02 executes 24 of 24; benchmark 03 executes 10 of 10. None contains an error output. Benchmark 02 retains selected ROSE settings `(4, 8)` for `Vv` and `(4, 12)` for both `Rv` and the broad study.
- **What did not change:** notebook section order, scientific cases, figures, physical mesh coordinates, ROSE held-out validation, or the project-map narrative.

Final prose ownership: pending Daniel's review.
