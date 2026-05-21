# May 21 Checkpoint

This checkpoint contains the collaborator-facing LROM demo notebooks and helper code.

The notebooks are saved executed, with figures embedded. Large recomputation
caches are intentionally not included. If a notebook is rerun, it will recreate
missing cache files under `outputs/`.

Recommended files to commit:

- `README.md`
- `requirements.txt`
- `CHECKPOINT_NOTES.md`
- `notebooks/*.ipynb`
- `scripts/*.py`
- `lrom_demo/*.py`
- `.gitignore`

Notebook 04 can take a long time to rerun from scratch because it rebuilds
isotope/energy wavefunction and cross-section caches.
