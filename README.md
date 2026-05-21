# Clean LROM Demonstration

This folder contains a cleaned-up notebook series for the learned reduced-operator model (LROM) experiments.

The notebooks are meant to be portable: they use the public ROSE package from PyPI, installed as:

```bash
pip install nuclear-rose
```

The package imports as `rose`.

## Setup

Use a fresh environment rather than a pre-existing Anaconda/base environment.
The notebooks rely on the public PyPI package `nuclear-rose`; older locally
installed `rose` packages can shadow it and cause import errors.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name clean-lrom-demo --display-name "clean-lrom-demo"
```

Then open the notebooks with the `clean-lrom-demo` kernel.  In VS Code, use the
kernel picker in the upper-right of the notebook.  If you work from WSL, create
and select the kernel inside WSL as well.

If you edit `scripts/create_clean_demo_notebooks.py`, regenerate the notebooks with:

```bash
python scripts/create_clean_demo_notebooks.py
```

## Notebooks

The notebooks are generated from `scripts/create_clean_demo_notebooks.py` and
`scripts/notebook04_source.py`.  Edit those source files and regenerate the
notebooks rather than hand-editing notebook JSON.

1. `notebooks/01_fom_and_rose_basics.ipynb`
   - FOM wavefunctions, phase shifts, cross sections, and a standard ROSE ROM benchmark.

2. `notebooks/02_lrom_method_walkthrough.ipynb`
   - Residual-fit LROM training, central-reference coordinates, raw-parameter predictors,
     potential-collocation predictors, and wavefunction diagnostics.

3. `notebooks/03_cross_section_cat_comparison.ipynb`
   - Cross-section CAT comparison between ROSE and the packed predictor LROM.

4. `notebooks/04_energy_isotope_wavefunction_extrapolation.ipynb`
   - Koning-Delaroche calcium isotope/energy extrapolation with wavefunction and
     cross-section diagnostics, predictor-point rainbows, and `K`/`n` sweeps.

The helper code lives in `lrom_demo/`.  It is intentionally local to this demo so
the notebooks can be handed to collaborators as one folder.

## Cached Outputs

The notebooks are saved with rendered figures.  Some expensive datasets are also
cached under `outputs/`:

- `outputs/cached_cat/`: CAT-plot and representative cross-section data for notebook 03.
- `outputs/cached_ae/`: isotope/energy wavefunction and cross-section sweeps for notebook 04.
- `outputs/notebook_outputs/`: PNG exports of the notebook figures.

These caches are not required for understanding the notebooks, but they make
rerunning the heavier cells much faster.  If repository size is a concern, keep
the executed notebooks and selected PNGs, and share the larger `outputs/cached_ae`
files through a release artifact or separate data link.

## Troubleshooting

If the first cell reports an error such as `cannot import name CustomBasis from
rose.basis`, the active Python environment is probably importing an older `rose`
package.  Create a fresh environment with the commands above and confirm:

```bash
python -c "import rose; print(rose.__file__)"
```

The printed path should point inside the fresh environment's `site-packages`.
