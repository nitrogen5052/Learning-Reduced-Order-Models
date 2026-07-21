# LROM Versioning And Naming

## Package versions

| Version | Milestone | Where |
|---|---|---|
| **1.2** | Validated wavefunction-only active package | public `lrom/`, implemented in `lrom_legacy/v1_2/` |
| **2.0** | Parked future shell for wavefunctions and cross sections | `lrom_legacy/v2_0/` |
| **3.0** | Global Koning-Delaroche emulator (future) | — |

Minor digits are fixes or restructures within one scientific milestone. The next notebook milestone is not a minor bump of the previous one.

Python identifiers cannot contain decimal points, so versioned implementation namespaces use names such as `v1_2` and `v2_0`.

## Public routing

`import lrom` exposes version 1.2.0. `lrom/__init__.py` is a thin entry point that re-exports the intentional public surface from the authoritative one-file implementation in `lrom_legacy/v1_2/__init__.py`.

Version 2.0 remains explicit:

```python
import lrom_legacy.v2_0 as lrom_v2
```

This prevents a notebook from silently changing scientific milestones. A major restructure requires user approval.

## Benchmark naming

Benchmark notebooks are named after the legacy notebook they recreate:

- `benchmark_notebooks/1.0/benchmark_02.ipynb` recreates legacy notebook 02.
- `benchmark_notebooks/2.0/benchmark_03.ipynb` recreates legacy notebook 03.
- `benchmark_notebooks/2.0/benchmark_01.ipynb` compares the parked 2.0 shell with the v1.2 notebook-01 workflow.

Benchmarks live under `notebooks/benchmark_notebooks/<version>/`.

## Deliberate design decisions

- **One-file implementation.** `lrom_legacy/v1_2/__init__.py` holds the complete v1.2 workflow. `lrom/__init__.py` only re-exports it. Do not split the implementation into modules.
- **Rigid lifecycle.** The public sequence remains `LROM -> sampling -> train -> predict -> save/load`.
- **Exact sampling.** Package sampling uses the Exact ROSE high-fidelity boundary and does not construct an EIM.
- **Explicit analysis.** The LS baseline is opt-in; required RF-LROM training coordinates remain internal to training.
- **Notebook-owned ROSE.** Benchmark cells own their `InteractionEIMSpace`, training bounds, and free-reference basis.
- **No plotting wrappers.** Benchmark Matplotlib remains visible in notebook cells.
- **Display floors do not alter data.** Clipping for log-scale figures changes display arrays only.
