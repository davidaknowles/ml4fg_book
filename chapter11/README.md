# Chapter 11: genome-wide association studies

Converted from the ML4FG topic 11 lecture slides (2026 deck for content and ordering, 2023 deck for the extra allele-frequency slide, pedigree and the two-step mixed model) and the Fall 2025 lectures of 14 November and the start of 21 November. Handwritten slide annotations were turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, plotnine
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter11}`.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data, fixed seeds):

| File | Content |
|---|---|
| `sfs.pdf` | neutral folded site frequency spectrum and share of heterozygous sites |
| `manhattan.pdf` | Manhattan plot from z-scores simulated as zhat ~ N(Rz, R) with AR(1) LD |
| `ld_heatmap.pdf` | r^2 in a large and a bottlenecked population (coalescent-tree founders, haplotype mosaic model) |
| `power_maf.pdf` | effect needed for 80% power vs MAF; discoveries with and without negative selection |
| `popstrat_qq.pdf` | QQ plot: linear regression, + PCs, LMM (eigendecomposition) in a structured cohort with sibling pairs |
| `finemap.pdf` | single-causal-variant fine-mapping with uniform and annotation priors |
| `ldsc.pdf` | LD score regression with and without stratification on simulated summary statistics |

TikZ (in `chapter11.tex`): Punnett square, case-control cartoon with contingency table, pedigree and crossing-over diagrams, imputation scenario, LD score schematic.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `fto_locus.png` | Claussnitzer et al. 2015, NEJM, Fig. 1 | yes |
| `fto_region.png`, `fto_expression.png`, `fto_hic.png` | Claussnitzer et al. 2015, NEJM | yes |
| `scz_manhattan.png` | Trubetskoy et al. 2022, Nature | yes |
| `t2d_or_maf.png` | Mahajan et al. 2018, Nat Genet (attribution to be confirmed) | yes |
| `t2d_functional_finemap.png` | Mahajan et al. 2018, Nat Genet | yes |
| `coding_fraction.png` | Pickrell 2014, AJHG | yes |
| `tissue_enrichment.png` | Finucane et al. 2015, Nat Genet (attribution to be confirmed) | yes |

Not carried over: title slide, paper title screenshots (Nature Methods, Nature Genetics, AJHG, NEJM headers), Mendel portrait and Punnett square image (redrawn in TikZ), GWAS Catalog animation (described in text), case-control cartoon image, recombination and HapMap LD triangle images (replaced by TikZ and a computed figure), IMPUTE2 diagram (redrawn), T2D Manhattan plot from Trends in Genetics (replaced by a simulated Manhattan plot), childhood obesity Manhattan plot (described in text), Roadmap chromatin state legend, Elhaik 2012 MAF histogram (2023 deck; replaced by the computed spectrum).
