# Chapter 12: molecular quantitative trait loci

Converted from the ML4FG topic 12 lecture slides (2026 deck; there is no 2023 deck for this topic) and the Fall 2025 lecture transcripts. Handwritten slide annotations (PEER model, beta-binomial and negative binomial derivations, combined allele-specific likelihood, mash model) were turned into text and equations; schematics into TikZ; plots into computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, plotnine
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter12}`.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data with fixed seeds): `eqtl_example`, `permutation_null`, `ard_factors`, `hidden_factors_power`, `ld_locus`, `betabinom`, `negbin`, `ase_qq`, `sharing_naive`, `coloc_example`, `mr_egger`.

TikZ (in `chapter12.tex`): variant-molecule-disease mediation, QTL study design, PEER graphical model, cis and trans effects, allele-specific expression, TWAS schematic, Mendelian randomization DAG, direct and total effects in a causal network. Table of QTL types.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `fto_locus.png` | Claussnitzer et al. 2015, N Engl J Med, Fig. 1A-B | yes |
| `fto_expression.png` | Claussnitzer et al. 2015, N Engl J Med | yes |
| `gtex_egenes.pdf` | GTEx Consortium 2020, Science (slide credited to F. Aguet et al.) | yes |
| `mash_tissue_correlation.png` | Urbut et al. 2019, Nat Genet | yes |
| `gaffney_annotation_enrichment.png` | Gaffney et al. 2012, Genome Biol (open access, CC BY) | attribution only, confirm licence |
| `twas_parkinsons.png` | Li, Knowles et al. 2018, Nat Genet (LeafCutter) | yes |

Not carried over: paper title screenshots (EGG childhood obesity, NEJM FTO, WASP, mash, Gaffney, LirNet, TORUS, SuSiE, Sasse et al., coloc, Kim-Hellmuth 2017), the FTO mechanism cartoon (described in text), the review figures on approaches to functional interpretation and on variant annotation (described in text), the AGA eQTL cartoons and plots (replaced by computed `eqtl_example` and `ld_locus`), the MET promoter LD triangle, the WASP QQ plots (replaced by computed `ase_qq`), the GTEx technical and cell type covariate heatmaps, the multiple-eQTL bar chart, the cis/trans tissue-sharing histogram, the xCell heatmap and violin plots, the Gaffney TF enrichment and schematic panels, the Sasse et al. scatter, the coloc example panels (replaced by computed `coloc_example`), the GTEx colocalization summary, the Kim-Hellmuth stimulation figures, the Bowden Egger illustration (replaced by computed `mr_egger`), the Holmes MR diagram (redrawn in TikZ), the bimmer total/direct network figure (redrawn in TikZ) and the inspre simulation results (described in text).
