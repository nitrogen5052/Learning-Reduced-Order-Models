# Draft figure workflows

## Introductory LROM figures

`figure_design_variants.py` trains the two-parameter Woods--Saxon toy LROM and
generates the physical manifold and reduced-coordinate comparisons.  The final
design uses five depths spanning plus/minus 20 percent, three radii, predictor
locations selected by the LROM machinery, and least-squares/LROM/Galerkin
coordinate tracks.  The alternative encodings and panel layouts are retained
in `../../figures/design_variants/`.

## LROM versus ROSE benchmark

`build_rose_paper_cat.py` reads `rose_lrom_paper_cat_metrics.npz` and generates
both wide and one-column CAT plots.  Both methods share the notation
`(n_phi, n_op)` for the wavefunction-basis size and number of learned
operators/predictors.  LROM configurations use filled blue markers; ROSE uses
open orange-red markers.  The separated lower strip reports the much slower
full-order timings without wasting the ROM panel's horizontal range.

## Global cross sections

`plot_stacked_draft_cross_sections.py` records how the five representative
angular distributions were stacked and annotated.  It plots canonical KD FOM,
posterior-median LROM, curated data, and 68/95 percent predictive bands.  It
requires the calibration payload and posterior-predictive arrays, which are not
committed because they are large generated artifacts.

The finalized PDFs and Overleaf snippets are in `../../figures/draft_paper/`.
