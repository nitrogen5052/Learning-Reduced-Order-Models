# Explicit Sampling Grids and Notebook 1 Diagnostics Design

## Goal

Allow users to supply named, row-aligned training and testing parameter grids to
`LROM.sampling()`, and expand Notebook 1 so every method-comparison figure shows
its error against the appropriate reference.

## Public Sampling API

`LROM.sampling()` supports two mutually exclusive modes.

Generated mode retains the existing arguments:

```python
emulator.sampling(
    training_ranges={"Vv": (45.0, 55.0)},
    testing_ranges={"Vv": (40.0, 60.0)},
    training_size=35,
    testing_size=41,
    strategy="linspace",
)
```

Explicit-grid mode accepts named columns:

```python
emulator.sampling(
    training_grid={
        "Vv": [45.0, 50.0, 55.0],
        "Rv": [3.8, 4.0, 4.2],
        "av": [0.60, 0.65, 0.70],
    },
    testing_grid={
        "Vv": [42.0, 58.0],
        "Rv": [3.6, 4.4],
        "av": [0.55, 0.75],
    },
)
```

Values at the same index define one FOM case. Grid sizes are inferred. Parameter
columns may be omitted; omitted values are filled from the object's central
parameters. Training and testing grids must use the same non-empty set of named,
sampleable parameters. Values must be finite one-dimensional numeric sequences,
and all columns within one grid must have equal positive length.

Supplying any grid argument together with range, size, strategy, or seed
arguments is rejected with `LROMSamplingError`. Supplying only one of
`training_grid` or `testing_grid` is also rejected. Explicit designs are stored
as ordinary `ParameterCases` with stable `train-*` and `test-*` case IDs and a
`SamplingDesign.strategy` value of `"explicit_grid"`.

## Training and Testing Diagnostics

The training engine evaluates both datasets after fitting. The emulator exposes
`training_results` alongside `testing_results`. Each result contains the true
high-fidelity wavefunctions, LS reconstructions, ROSE reconstructions, LROM
reconstructions, reduced coordinates, pointwise absolute errors, and relative-L2
errors for every exact partial-wave channel.

The implementation factors evaluation into one private helper used for both
datasets. Portable `.lrom` artifacts remain inference-only and do not serialize
training or testing diagnostics.

## Notebook 1 Figures

### Vv coefficient comparison

Plot the first two shared-basis coordinates in two columns. Each column has:

- a top scatter panel containing LS, LROM, and ROSE testing coordinates;
- a bottom panel containing `abs(LS - LROM)` and `abs(LS - ROSE)`;
- shaded test-only extrapolation bands below and above the training Vv interval.

### Wavefunction comparisons

Both the Vv representative case and the ws3 representative case use two rows:

- the top panel overlays the true testing solution, LS, LROM, and ROSE using
  `Re(phi(r))` against physical radius in fm;
- the bottom panel shows pointwise absolute differences from the true testing
  solution for LS, LROM, and ROSE on a logarithmic scale.

The training wavefunction rainbow and basis-vector figures are not method
comparisons and therefore do not receive error panels.

### ws3 coefficient comparison

The existing ws3 shared-coordinate comparison becomes a scatter plot with a
second panel showing `abs(LS - LROM)` and `abs(LS - ROSE)` against testing case
index.

### Split violin comparison

For each of LS, LROM, and ROSE, show a split violin of per-solution relative-L2
errors:

- left half: training cases;
- right half: testing cases.

The violin values are `log10(max(relative_L2, 1e-16))`. Notebook code clips the
two Matplotlib violin bodies at their shared center so the result does not depend
on a recent Matplotlib-only `side` argument. A legend identifies training and
testing halves.

## Verification

Tests cover explicit-grid construction, central-value filling, invalid names,
non-finite values, unequal lengths, incomplete grid pairs, and mixed modes.
Training tests verify matching training/testing diagnostic shapes and metrics.
Notebook structural tests cover scatter usage, extrapolation shading, absolute
difference panels, and split violin inputs. The generator must remain
deterministic, the full suite must pass, and Notebook 1 must execute without cell
errors.
