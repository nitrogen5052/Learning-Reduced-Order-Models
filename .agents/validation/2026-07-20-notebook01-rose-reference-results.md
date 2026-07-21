# Notebook 01 ROSE Reference Diagnostic

## Question

Does replacing the invalid central-solution ROSE reference with ROSE's native free reference remove the isolated ws_3 failures?

## Controlled comparison

- LROM implementation: public `lrom` 1.2.0.
- Same training/testing rows, high-fidelity snapshots, EIM interaction, mesh, solver, and retained rank.
- Changed variable: ROSE reference/basis convention only.
- The controlled rerun uses the final Notebook 01 mesh of 800 points.

## Results

| reference | case | interpolation | Vv [MeV] | Rv [fm] | av [fm] | coefficient infinity norm | matrix condition | relative L2 error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| invalid central | `test-0021` | False | 40.6066131676 | 4.41010096545 | 0.610410082723 | 2.823884478238e+02 | 1.262429154278e+04 | 2.217072377673e+01 |
| invalid central | `test-0039` | False | 38.7147895697 | 4.52200910629 | 0.677555683944 | 7.319434674041e+01 | 2.364586952211e+03 | 5.422510260918e+00 |
| invalid central | `test-0065` | False | 41.014421395 | 4.79814643816 | 0.673197895076 | 1.564862326432e+02 | 5.185986036356e+03 | 1.406652907720e+01 |
| native free | `test-0021` | False | 40.6066131676 | 4.41010096545 | 0.610410082723 | 3.082618211487e+01 | 1.257456105834e+02 | 4.155606831256e-02 |
| native free | `test-0039` | False | 38.7147895697 | 4.52200910629 | 0.677555683944 | 3.259331683734e+01 | 1.641213104075e+02 | 1.711668586402e-01 |
| native free | `test-0065` | False | 41.014421395 | 4.79814643816 | 0.673197895076 | 2.964976190495e+01 | 2.153916263610e+02 | 3.588195444937e-02 |

## Interpretation

The controlled comparison supports the wrong ROSE reference as the cause of the three isolated failures. All three points are extrapolation cases. With ROSE's native free reference, their coefficient infinity norms decrease by factors of 9.16, 2.25, and 5.28; their reduced-matrix condition numbers decrease by factors of 100, 14.4, and 24.1; and their wavefunction relative-L2 errors decrease by factors of 534, 31.7, and 392, respectively.

The corrected errors are not zero: `test-0039` remains the least accurate of these three at 0.171 relative L2. That is an extrapolation-accuracy limitation of the four-vector free-reference basis, not the previous singular coefficient failure. The final notebook reports its worst corrected ROSE cases by case ID and parameters so this limitation remains visible.
