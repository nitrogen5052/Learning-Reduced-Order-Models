# Pretrained global three-window LROM

These trusted artifacts were trained from 3,000 KD-centered Latin-hypercube
snapshots spanning energy, isotope, and local optical-parameter perturbations.
Each hard-routed window owns its reduced bases and implicit equations.

| Window | Core [MeV] | Training support [MeV] | Basis | Predictors | Ridge |
|---:|---:|---:|---:|---:|---:|
| 0 | 5–70 | 5–102.5 | 14 | 20 | 1e-14 |
| 1 | 70–135 | 37.5–167.5 | 14 | 20 | 1e-14 |
| 2 | 135–200 | 102.5–200 | 14 | 20 | 1e-14 |

SHA-256:

```text
6ca61fda61406e75ca4d15e0d6c9b02883e1ed49dec4d8beda48b8a9f408307e  energy_window_0_b14_p20_ridge1e-14.pkl
98d600714e7df619f1339ba6e2288ac5637c2a75dfa9d18f86daf74fc4395b35  energy_window_1_b14_p20_ridge1e-14.pkl
d028d2aaceae53d0a534f435ccafb3e7399ae0d59c9d4137c5f41eb027c5201b  energy_window_2_b14_p20_ridge1e-14.pkl
```

Load these pickle files only from this trusted repository.

