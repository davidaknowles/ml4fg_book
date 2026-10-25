# Chapter 1: molecular biology basics and linear models

Converted from the ML4FG topic 1 lecture slides (2023 deck for the detailed derivations, 2024 deck for content and ordering). Handwritten slide annotations were turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, scikit-learn, plotnine, contourpy
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter01}`.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data): `linreg_scatter`, `colinearity_contours`, `prior_likelihood_posterior`, `cv_curve`, `bias_variance`, `penalties`, `objectives_1d`, `shrinkage`, `constraint_geometry`, `sparse_priors`, `risk_vs_p`, `gradient_descent`, `convexity`, `dose_response`, `sigmoid`, `logistic_2d`, `roc_pr`.

TikZ (in `chapter01.tex`): functional genomics workflow, cell/chromosome diagram, DNA strands, RNA-seq pipeline, data matrices, RMSE schematic, train/val/test and CV splits, graphical model, computational graph.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `sequencing_cost.pdf` | own R plot of NHGRI data (public domain) | no |
| `genetic_support_odds_ratio.png` | Nelson et al. 2015, Nat Genet | yes |
| `bulk_vs_singlecell.png` | 10x Genomics blog | yes |
| `primary_culture.png` | unknown, source to be confirmed | yes |
| `ccle_tumour_types.png`, `ccle_data_types.png`, `ccle_elastic_net_pd0325901.png` | Barretina et al. 2012, Nature | yes |
| `pam50_heatmap.png`, `pam50_survival.png` | Parker et al. 2009, J Clin Oncol | yes |

Not carried over: title-slide logos, NYT headline screenshot, ESL book covers, PollEverywhere results screenshot, CCLE paper title screenshot.
