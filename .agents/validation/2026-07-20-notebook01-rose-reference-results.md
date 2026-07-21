# Notebook 01 ROSE Reference Diagnostic

## Question

Does replacing the invalid central-solution ROSE reference with ROSE's native free reference remove the isolated ws_3 failures?

## Controlled comparison

- LROM implementation: `lrom_legacy.v1_2` 1.2.0.
- Same training/testing rows, high-fidelity snapshots, EIM interaction, mesh, solver, and retained rank.
- Changed variable: ROSE reference/basis convention only.

## Results

| reference | case | interpolation | Vv [MeV] | Rv [fm] | av [fm] | coefficient infinity norm | matrix condition | relative L2 error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| invalid central | `test-0021` | False | 40.6066131676 | 4.41010096545 | 0.610410082723 | 2.966018927621e+02 | 1.248483938636e+04 | 2.195110858074e+01 |
| invalid central | `test-0039` | False | 38.7147895697 | 4.52200910629 | 0.677555683944 | 7.759622751068e+01 | 2.364761126357e+03 | 5.419408419016e+00 |
| invalid central | `test-0065` | False | 41.014421395 | 4.79814643816 | 0.673197895076 | 1.638484287018e+02 | 5.133290439755e+03 | 1.388706269478e+01 |
| native free | `test-0021` | False | 40.6066131676 | 4.41010096545 | 0.610410082723 | 3.269774162376e+01 | 1.257401960956e+02 | 4.162284989076e-02 |
| native free | `test-0039` | False | 38.7147895697 | 4.52200910629 | 0.677555683944 | 3.457339102026e+01 | 1.641055684710e+02 | 1.712474910675e-01 |
| native free | `test-0065` | False | 41.014421395 | 4.79814643816 | 0.673197895076 | 3.145631596169e+01 | 2.154062090795e+02 | 3.642186688592e-02 |

## Interpretation

The controlled comparison supports the wrong ROSE reference as the cause of the three isolated failures. All three points are extrapolation cases. With ROSE's native free reference, their coefficient infinity norms decrease by factors of 9.07, 2.24, and 5.21; their reduced-matrix condition numbers decrease by factors of 99.3, 14.4, and 23.8; and their wavefunction relative-L2 errors decrease by factors of 527, 31.6, and 381, respectively.

The corrected errors are not zero: `test-0039` remains the least accurate of these three at 0.171 relative L2. That is an extrapolation-accuracy limitation of the four-vector free-reference basis, not the previous singular coefficient failure. The final notebook reports its worst corrected ROSE cases by case ID and parameters so this limitation remains visible.
