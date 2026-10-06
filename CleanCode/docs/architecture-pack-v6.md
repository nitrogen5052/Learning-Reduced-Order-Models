# LROM package — Architecture V6

Figures reviewed against the code. Figure 1 is reviewed; Figure 2 is still
being reviewed in [V5](legacy/architecture-pack-v5.md).

## Figure 1 — User to Package

The notebook or script calls the `lrom` package to calculate scattering results.

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

`TrainingData` contains wavefunctions, potentials, and S-matrix values.
Saving and loading these results as `.npz` is optional.

`load_three_window_lrom()` reads the three supplied trained LROM `.pkl` files.
The package has no `.pkl` saving function.

`partial_wave_s_matrices()` handles batches on CPU by default.
`partial_wave_s_matrices_gpu()` uses a supplied GPU solver.
The caller arrow shows one example: a cross section at the requested angles.
