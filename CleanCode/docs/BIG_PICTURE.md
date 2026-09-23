# Big picture: a learned reduced-order model for neutron elastic scattering

```mermaid
flowchart TB
    subgraph offline["1 · Offline build — expensive, done once"]
        direction LR
        O1["KD inputs: optical parameters θ, E, A, Z"]
        O2["Training samples"]
        O3["Full radial FOM solves → wavefunction snapshots"]
        O4["Compress snapshots: SVD basis + max-volume predictors"]
        O5["Train reduced equations: normalized ridge"]
        O6["Shipped pretrained window models + training_manifest.json<br/>[5,70), [70,135), [135,200) MeV; basis 14; 20 predictors; ridge 1e-14"]
        O1 --> O2 --> O3 --> O4 --> O5
        O5 -. "artifact export not included in this checkout" .-> O6
    end

    subgraph online["2 · Online emulation — cheap per query"]
        direction LR
        N1["New KD input: θ, E, A, Z"]
        N2["Hard-route E to one energy window"]
        N3["Small dense reduced solves → partial-wave S matrices<br/>CPU/GPU batched when needed"]
        N4["Elastic differential cross section dσ/dΩ"]
        N1 --> N2 --> N3 --> N4
    end

    subgraph fidelity["3 · Emulator fidelity — held-out LROM vs FOM"]
        direction LR
        F1["Held-out test samples"]
        F2["Independently evaluate LROM and FOM on held-out samples"]
        F3["Report wavefunction / cross-section error"]
        F1 --> F2 --> F3
    end

    subgraph adequacy["4 · Model adequacy — KD FOM vs experiment"]
        direction LR
        A1["Curated neutron-elastic data: KDUQ fit corpus + held-out test corpus"]
        A2["Compare KD FOM cross sections with measurements"]
        A3["Profile-normalized χ²"]
        A1 --> A2 --> A3
    end

    %% Invisible links preserve the four-lane reading order; they are not data-flow arrows.
    offline ~~~ online
    online ~~~ fidelity
    fidelity ~~~ adequacy
    classDef offline fill:#E8F1FF,stroke:#2F5E9E,color:#102A4C,stroke-width:2px;
    classDef online fill:#E7F7EE,stroke:#28734A,color:#123B27,stroke-width:2px;
    classDef fidelity fill:#FFF3D9,stroke:#9B6516,color:#513000,stroke-width:2px;
    classDef adequacy fill:#FBE8EE,stroke:#A23C5C,color:#54142A,stroke-width:2px;
    class O1,O2,O3,O4,O5,O6 offline;
    class N1,N2,N3,N4 online;
    class F1,F2,F3 fidelity;
    class A1,A2,A3 adequacy;
```

The offline calculation pays for independent radial FOM snapshots once and trains a
reduced model.  This checkout ships and loads pretrained three-window artifacts, but
does not include their export or manifest-generation pipeline.  A new KD input is routed
to one window, where the LROM replaces the full radial solve with a small learned reduced
system before assembling partial waves into the elastic cross section.  The two validation
lanes answer different questions: held-out LROM–FOM agreement establishes emulator
fidelity, while KD FOM–data agreement tests the adequacy of the underlying optical model.
Thus, a KD–experiment mismatch is not evidence that the emulator failed.

| Concept | Code symbol / file | Notebook |
| --- | --- | --- |
| FOM snapshots from sampled inputs | `ScatteringProblem.generate_training_data()` in `lrom/problem.py`; `NumerovFOM.solve()` in `lrom/fom.py` | `01_single_wavefunction.ipynb`, `02_elastic_cross_sections.ipynb` |
| SVD/max-volume reduction and regularized reduced equations | `ReducedBasis.fit()` and `LearnedROM.fit()` in `lrom/reduced.py`; `ScatteringLROM.train()` in `lrom/emulator.py` | `01_single_wavefunction.ipynb`, `02_elastic_cross_sections.ipynb` |
| Three-window pretrained deployment | `load_three_window_lrom()` in `lrom/pretrained.py` loads checked-in `energy_window_*_b14_p20_ridge1e-14.pkl` artifacts; `models/three_window/training_manifest.json` is provenance/configuration metadata, not loader input | `03_curated_data_and_global_lrom.ipynb` |
| Route, reduced solve, partial waves, and elastic cross section | `HardRoutedScatteringLROM.route_energy()` / `HardRoutedScatteringLROM.cross_section()` in `lrom/deployment.py`; `ScatteringLROM.cross_section()` in `lrom/emulator.py` | `03_curated_data_and_global_lrom.ipynb` |
| Emulator fidelity on held-out samples | `relative_l2()` in `lrom/diagnostics.py`; FOM and LROM calls shown above | `01_single_wavefunction.ipynb`, `02_elastic_cross_sections.ipynb` |
| KD FOM model adequacy against experimental data | `load_neutron_elastic()` in `lrom/curated_data.py`; notebook-local `normalization_profile()` | `03_curated_data_and_global_lrom.ipynb` |

## Brief corrections

The online fast cross-section path does not reconstruct full wavefunctions.  It solves
the learned reduced system and maps its coordinates through precomputed boundary maps
to partial-wave S matrices; the diagram therefore shows partial-wave S matrices rather
than wavefunction reconstruction.

The pretrained pickle files and `training_manifest.json` are checked-in artifacts.  The
checkout loads them through `load_three_window_lrom()`, but contains no production export,
pickle-serialization, or manifest-generation path; the dashed handoff is therefore not a
package data-flow arrow.
