# LROM Benchmark Spine Architecture

This document shows the proposed research benchmark package at three abstraction levels. The diagrams are written in Mermaid so they stay reviewable as plain Markdown.

## Figure 1: Project Context

```mermaid
flowchart TB
    paper["Physics and method context<br/>ROSE paper, LROM report, slides"]
    legacy["Legacy_benchmark/<br/>executed notebooks and lrom_demo helpers"]
    package["lrom_bench/<br/>new Python benchmark package"]
    notebooks["notebooks/<br/>package-native benchmark notebooks"]
    artifacts["outputs/benchmarks/<br/>legacy refs, new outputs, parity reports"]
    future["Future graph/vector memory<br/>indexes stable metadata and explanations"]

    paper --> package
    legacy --> package
    package --> notebooks
    package --> artifacts
    notebooks --> artifacts
    artifacts --> future
    notebooks --> future
```

At this level, the legacy benchmark is an input and comparison source. The new package becomes the executable center of the project. The notebooks and artifacts are review surfaces produced by that package.

## Figure 2: Package Spine

```mermaid
flowchart LR
    config["config.py<br/>benchmark config, paths, tolerances"]
    sampling["sampling.py<br/>Vv/Rv scans, box samples"]
    rose["rose_fom.py<br/>ROSE setup, KD params, FOM wavefunctions"]
    basis["reduced_basis.py<br/>phi0, centered basis, LS coordinates"]
    predictors["predictors.py<br/>raw deltas, potential predictors, delta-maxvol"]
    fit["rf_lrom.py<br/>residual-fit linear solve"]
    predict["prediction.py<br/>online reduced solve, reconstruction"]
    metrics["metrics.py<br/>coefficient and wavefunction errors"]
    artifacts["artifacts.py<br/>NPZ outputs, JSON parity reports"]

    config --> sampling
    config --> rose
    sampling --> rose
    rose --> basis
    basis --> fit
    predictors --> fit
    fit --> predict
    predictors --> predict
    basis --> predict
    predict --> metrics
    basis --> metrics
    metrics --> artifacts
    config --> artifacts
```

This is the benchmark spine: each module owns one scientific stage. The flow is intentionally more concrete than a general framework, because the first goal is reliable Notebook 02 parity.

## Notebook-Driven Helper Rule

Notebook 1 adds only small reusable helpers needed by the notebook:

- `Notebook01Config` records the single-wavefunction benchmark settings.
- `sampling.centered_1d_values` creates the visible `Vv` scan.
- `reduced_basis.build_centered_svd_basis` creates the central-reference basis used by the notebook.
- `rose_fom.central_real_ws_parameters` and `rose_fom.real_woods_saxon_potential` expose the real Woods-Saxon teaching setup.
- `rose_fom.RealWSProblem`, `rose_fom.make_real_ws_problem`, `rose_fom.make_real_ws_custom_basis`, and `rose_fom.make_real_ws_rbe` split the ROSE-backed FOM/RBM setup into small reusable pieces.

Plotting remains in notebook cells. The package should not grow plotting functions or one-call notebook workflow functions.

## Figure 3: Notebook 02 Scientific Flow

```mermaid
flowchart TB
    setup["1. Setup<br/>Notebook02Config, paths, imports"]
    convention["2. Central convention<br/>choose alpha_c, compute phi0 = phi(alpha_c)"]
    vv["3. Vv-only scan<br/>fit RF-LROM for real volume depth"]
    rv["4. Rv-only scan<br/>fit RF-LROM for radius"]
    broad["5. Broad Vv/Rv box<br/>show where raw predictors struggle"]
    potential["6. Potential predictors<br/>delta-maxvol selected operator samples"]
    parity["7. Frozen parity<br/>compare new package artifact to legacy reference"]
    summary["Scientific summary<br/>what passed, what drifted, what to inspect"]

    setup --> convention
    convention --> vv
    vv --> rv
    rv --> broad
    broad --> potential
    potential --> parity
    parity --> summary
```

This diagram is the visible notebook narrative. The notebook should not collapse these steps into one opaque runner.

## Figure 4: RF-LROM Training And Prediction

```mermaid
flowchart LR
    fom["FOM wavefunctions<br/>phi(alpha_i)"]
    central["Central state<br/>phi0 = phi(alpha_c)"]
    svd["Centered SVD basis<br/>Phi"]
    ls["LS target coordinates<br/>a_LS(alpha_i)"]
    pred_train["Predictors<br/>p(alpha_i)"]
    residual["Residual equations<br/>(I + sum p_j M_j) a_LS - sum p_j b_j"]
    solve["One linear least-squares solve<br/>learn M_j and b_j"]
    pred_new["New parameter alpha_*<br/>evaluate p(alpha_*)"]
    online["Online reduced solve<br/>(I + sum p_j M_j) a_* = sum p_j b_j"]
    reconstruct["Reconstruct<br/>phi_hat(alpha_*) = phi0 + Phi a_*"]
    errors["Metrics<br/>coeff error, wavefunction error, LS floor"]

    fom --> central
    fom --> svd
    central --> svd
    svd --> ls
    fom --> ls
    pred_train --> residual
    ls --> residual
    residual --> solve
    solve --> online
    pred_new --> online
    online --> reconstruct
    central --> reconstruct
    svd --> reconstruct
    reconstruct --> errors
    ls --> errors
```

This is the mathematical core: training inserts known LS coordinates into the implicit equation, which turns the fit into a linear least-squares problem.

## Figure 5: Parity And Future Memory

```mermaid
flowchart TB
    legacy_npz["Frozen legacy NPZ<br/>outputs/benchmarks/legacy"]
    new_npz["New package NPZ<br/>outputs/benchmarks/new"]
    compare["Parity comparison<br/>array checks and scientific metrics"]
    report["JSON parity report<br/>metadata, tolerances, pass/fail"]
    notebook["Notebook output<br/>human-readable narrative and figures"]
    graph["Future graph memory<br/>runs, configs, datasets, model fits, artifacts"]
    vector["Future vector memory<br/>notebook summaries, decisions, report excerpts"]

    legacy_npz --> compare
    new_npz --> compare
    compare --> report
    report --> notebook
    report --> graph
    notebook --> vector
    report --> vector
```

The graph/vector layer is intentionally downstream. First the package must produce stable, metadata-rich artifacts; then memory can index them.
