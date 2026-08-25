# LROM for neutron elastic scattering

This repository contains a modular learned reduced-order model (LROM) for
neutron elastic scattering. The executed notebook demonstrates the complete
single-partial-wave workflow: generating full-order solutions, constructing a
reduced basis, learning implicit equations for the reduced coordinates, and
comparing the results with both the internal full-order model and ROSE.

## Code structure

- `01_single_wavefunction.ipynb` presents the scientific calculation and its
  results. It is saved with all cells executed.
- `lrom/physics.py` defines kinematics and Woods--Saxon optical potentials.
- `lrom/fom.py` contains the independent full-order radial Schrodinger solver.
- `lrom/reduced.py` constructs reduced bases and fits the learned implicit
  reduced-coordinate equations.
- `lrom/diagnostics.py` provides numerical error measures.
- `benchmarks/rose.py` provides an optional interface to ROSE for independent
  benchmark calculations. The core `lrom` package does not depend on ROSE.
- `tests/test_core.py` contains basic tests of the numerical components.

The separation between `physics`, `fom`, and `reduced` is intentional: the
full-order solver creates the training and reference solutions, while the LROM
uses those solutions to learn and predict reduced coordinates. ROSE enters
only as an external comparison in the notebook.

## Running the notebook

Create a virtual environment from the repository root and install the listed
dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Open `01_single_wavefunction.ipynb` and select `.venv\Scripts\python.exe` as
the Jupyter kernel. The committed notebook already contains its executed
outputs, so its current results can also be inspected directly on GitHub.

