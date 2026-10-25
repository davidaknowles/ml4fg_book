# Chapter 13: rare variants

Converted from the ML4FG topic 13 lecture slides (2026 deck; there is no 2023 deck for this topic). The deck has no handwritten annotations; its speaker notes were used for DeepRVAT. The topic was not lectured in Fall 2025, so the transcript was used only for related passages (de novo mutations in autism, human genetic variation, effect size and allele frequency, allele-specific expression). Method details were checked against the primary papers.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, plotnine
latexmk -pdf main.tex
```

`make_figures.py` takes optional figure names as arguments (for example `python scripts/make_figures.py skat_power`); `skat_power` and `fst_power` run power simulations (about 15 s and 1 min) and write their results to `scripts/skat_power.csv` and `scripts/fst_power.csv`. `main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter13}`.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data, fixed seeds): `power_single_variant` (analytic), `maf_weights`, `skat_power`, `fst_power`, `outliers`, `aneva_dot`.

TikZ (in `chapter13.tex`): effect size against allele frequency (redrawn after Manolio et al. 2009), LINSIGHT graphical model (redrawn after Huang et al. 2017), autism study design and pipeline (after Zhou et al. 2019), NDEA schematic, regulatory variant schematic (redrawn from a slide schematic), DeepRVAT architecture (after Clarke et al. 2024). Table 13.1 (LINSIGHT notation) is retyped from a table in Huang et al. 2017.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `linsight_fitness.png` | Huang et al. 2017, Nat Genet, Fig. 2 | yes |
| `linsight_hgmd_roc.png` | Huang et al. 2017, Nat Genet (top title strip cropped) | yes |
| `asd_disease_impact.png` | Zhou et al. 2019, Nat Genet, Fig. 1b (panel letter removed) | yes |
| `asd_luciferase.png` | Zhou et al. 2019, Nat Genet | yes |

Not carried over: paper title screenshots (Guo et al. 2018, LINSIGHT, He et al. 2017, SKAT-O, Zhou et al. 2019, Greene et al. 2015, DeepRVAT), gnomAD logo, the He et al. 2017 power bar chart and the enriched-pathway dot plot from Zhou et al. 2019 (both described in the text instead; the power chart is replaced by the computed `fst_power`), the NDEA equations image (retyped), the expression/splicing/allelic outlier panels from GTEx, Cummings et al. 2017 and Mohammadi et al. 2019 (replaced by the computed `outliers` and `aneva_dot`), and the duplicate LINSIGHT model slide.
