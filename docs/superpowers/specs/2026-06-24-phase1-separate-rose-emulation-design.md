# Phase 1 Separate ROSE Emulation Design

## Goal

Keep Phase 1 focused on Notebook 1 while establishing a clean architectural
boundary: `lrom` may use ROSE as a full-order solver during sampling, but ROSE
reduced-basis emulation and its comparisons run explicitly and independently in
Notebook 1.

## Package Boundary

The built-in `fom="nucl-scatter-eq"` backend remains ROSE-backed. During
`LROM.sampling()`, the package may use ROSE to resolve kinematics, construct the
interaction/EIM representation, build the full Schrödinger solver, and generate
central, training, and testing wavefunctions. Those full solutions are labeled
as the high-fidelity data used to train and test LROM.

`LROM.train()` must not import or construct a ROSE reduced emulator. It uses
numeric operations owned by `lrom` to:

1. build a basis centered on the central-alpha high-fidelity solution;
2. compute LS coordinates in that basis;
3. fit RF-LROM from its predictors to those coordinates;
4. evaluate high-fidelity, LS, and LROM wavefunctions and errors.

Remove these ROSE-emulation concepts from the package:

- `RoseRBMState`;
- the public `emulator.rose_rbm` property;
- ROSE `CustomBasis` and `ReducedBasisEmulator` construction in training;
- ROSE coefficient, wavefunction, and metric fields in training/testing results;
- ROSE fields in `TestingCase`;
- tests that require package-owned ROSE reduced-emulation results.

Keep `nuclear-rose` as a runtime dependency because the built-in nuclear
scattering FOM backend still uses it. Portable `.lrom` prediction remains
ROSE-free.

## LROM Basis

LROM uses its existing numeric centered-SVD basis implementation. Its reference
solution is the high-fidelity wavefunction evaluated at the central parameter
vector. The basis and LS projection use the physical radial mesh and the
package's trapezoid-weighted coordinate calculation.

The LROM result containers retain:

- high-fidelity wavefunctions;
- LS wavefunctions and coefficients in the LROM central-reference basis;
- LROM wavefunctions and coefficients;
- relative-L2 and pointwise absolute errors for LS and LROM.

## Independent Notebook 1 ROSE Workflow

Notebook 1 imports ROSE directly and performs a visible, independent workflow
for both the Vv-only and ws3 studies.

For each study, the notebook:

1. takes the exact parameter rows from `emulator.samples.design.training` and
   `.testing`;
2. independently resolves ROSE kinematics and reconstructs the interaction/EIM
   space and full Schrödinger solver;
3. independently solves the same training and testing parameter rows;
4. prints the maximum absolute difference between the independent full ROSE
   solutions and the high-fidelity arrays stored by LROM;
5. creates a zero-potential reference solution by setting `Vv=0` while retaining
   finite central `Rv` and `av`;
6. constructs ROSE `CustomBasis` with that zero-potential reference;
7. constructs ROSE `ReducedBasisEmulator`;
8. evaluates ROSE coefficients and wavefunctions for the same training and
   testing rows;
9. computes ROSE-native LS coordinates in the ROSE basis and all wavefunction
   errors against the independently generated full solutions.

Notebook startup installs the existing SciPy spherical-harmonic compatibility
alias before `import rose` so the direct import works across the supported
environment.

## Distinct Coordinate Systems

ROSE and LROM coefficients must not be subtracted from one another:

- LROM coefficients use a central-alpha reference solution and LROM's numeric
  centered-SVD basis.
- ROSE coefficients use a zero-potential reference and ROSE's independently
  constructed basis.

Notebook 1 shows two separate coefficient comparisons:

- LROM coordinates: LROM versus LS in the LROM basis;
- ROSE coordinates: ROSE versus ROSE-native LS in the ROSE basis.

The figures label their coordinate origins explicitly. Cross-method accuracy is
compared at the reconstructed-wavefunction level, where all results are in the
same physical representation.

## Notebook 1 Comparisons

Wavefunction figures compare the independent ROSE reduced approximation, LROM,
and the applicable LS floors against the same full testing solution. Absolute
difference panels use the full testing solution as the reference.

The split violin retains separate training and testing halves. LROM and LS
metrics come from the LROM object; ROSE metrics are calculated in the notebook
from the independent ROSE workflow. All violin entries are per-solution relative
L2 errors.

## Deferred Work

This design does not remove the ROSE-backed built-in FOM or complete the external
`SamplingRequest`/`SolutionBatch` ingestion workflow. That remains a separate
backlog feature. Phase 1 remains limited to the package boundary needed by
Notebook 1.

## Verification

Package tests assert that training does not import or construct ROSE emulation,
that LROM basis and results contain only high-fidelity/LS/LROM state, and that
portable prediction remains unchanged. Notebook tests require a direct
`import rose`, explicit ROSE construction/evaluation calls, same-case alignment,
full-solution parity output, separate coefficient coordinate panels, and
wavefunction-level comparisons. The full generated notebook must execute without
cell errors.
