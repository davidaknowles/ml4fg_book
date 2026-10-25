# Chapter 8: matrix factorization

Converted from the ML4FG topic 8 lecture slides (2026 deck for content and ordering, 2023 deck for the fuller factor analysis and PCA derivations) and the Fall 2025 lecture transcripts. Handwritten slide annotations were turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, scikit-learn, plotnine
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter08}`.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data, fixed seeds): `pca_2d`, `eigen_scree`, `eigen_mp`, `fa_vs_ppca`, `gamma_ard`, `ica_priors`, `ica_scatter`, `linear_ae`, `nmf_vs_pca`, `svr_losses`, `svr_fit`, `deconvolution_sim`.

TikZ (in `chapter08.tex`): matrix factorization shapes, autoencoder, LIGER schematic (redrawn after Welch et al. 2019).

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `mofa_schematic.png`, `mofa_cll.png` | Argelaguet et al. 2018, Mol Syst Biol (open access; check licence terms) | yes |
| `novembre_pca_europe.png` | Novembre et al. 2008, Nature | yes |
| `knowler_fig1.png`, `knowler_fig3.png` | Knowler et al. 1988, Am J Hum Genet (captions cropped) | yes |
| `liger_tsne_technology.png`, `liger_tsne_celltype.png` | Welch et al. 2019, Cell | yes |
| `cibersort_schematic.png`, `cibersort_coulter.png` | Newman et al. 2015, Nat Methods | yes |
| `gentles_fractions.png`, `gentles_survival.png` | Gentles et al. 2015, Nat Med | yes |

Not carried over: title slides, paper title and author-list screenshots (CIBERSORT, Gentles et al.), the MOFA prior equations (typeset in the text instead), the LIGER figure from the EMF (redrawn in TikZ). The 2023 deck's eQTL and PEER slides are left for the eQTL chapter (Chapter 12), which covers PEER in the 2026 decks.
