# Fig-2-sci — scientific workflow of the LROM

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
