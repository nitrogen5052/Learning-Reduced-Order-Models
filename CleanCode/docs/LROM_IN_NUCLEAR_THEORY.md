# LROM in nuclear-reaction theory

## Nuclear model → fast observables → inference

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 25, "rankSpacing": 30, "subGraphTitleMargin": {"top": 5, "bottom": 15}}}}%%
flowchart TB
    system["Neutron + target<br/>A, Z, E"]
    potential["Optical potential<br/>U(r; θ)"]
    fom["Full-order scattering<br/>H(θ)φ = Eφ"]

    subgraph lrom["LROM"]
        train["Offline · training<br/>basis + reduced equations"]
        emulator["Online · reduced solve<br/>M(p) a = b(p)"]
        train --> emulator
    end

    observable["Elastic scattering<br/>Sℓ± → dσ/dΩ"]
    data["Experiment<br/>d ± σ"]
    prior["Prior<br/>p(θ)"]
    inference["Bayesian inference<br/>p(θ ∣ d) · external"]

    system --> fom
    potential --> fom
    fom -->|φ, U snapshots| train
    potential -->|predictors p| emulator
    emulator --> observable
    observable -->|predictions| inference
    data --> inference
    prior --> inference
    inference -.->|parameter queries| emulator

    classDef physics fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef reference fill:#F3F4F6,stroke:#6B7280,color:#111827
    classDef model fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef result fill:#FFEDD5,stroke:#C2410C,color:#111827
    classDef external fill:#EDE9FE,stroke:#7C3AED,color:#111827
    class system,potential physics
    class fom reference
    class train,emulator model
    class observable result
    class data,prior,inference external
    style lrom fill:#F0FDF4,stroke:#15803D,stroke-width:2px
```

## Package inputs / outputs

```mermaid
flowchart LR
    model["Trusted pretrained models"]
    inputs["Cases: {A, Z, E}<br/>Angles: degrees"]
    lrom["LROM<br/>load_three_window_lrom()<br/>cross_sections(cases)"]
    outputs["dσ/dΩ · mb/sr<br/>array: cases × angles"]

    model --> lrom
    inputs --> lrom
    lrom --> outputs

    classDef physics fill:#DBEAFE,stroke:#2563EB,color:#111827
    classDef model fill:#DCFCE7,stroke:#15803D,color:#111827
    classDef result fill:#FFEDD5,stroke:#C2410C,color:#111827
    classDef artifact fill:#F3F4F6,stroke:#6B7280,color:#111827
    class inputs physics
    class lrom model
    class outputs result
    class model artifact
```
