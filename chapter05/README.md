# Chapter 5: unsupervised learning and clustering

Converted from the ML4FG topic 5 lecture slides (2026 deck for content and ordering; 2023 deck for the handwritten derivations of the mixture model likelihood, K-means and EM, the spectral clustering annotations, and the single-cell RNA-seq technology slides). Handwritten slide annotations were turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, scikit-learn, networkx, plotnine, matplotlib
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter05}`.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data with fixed seeds): `mixture_generative`, `kmeans_iterations`, `kmeanspp`, `choose_k`, `jensen`, `em_vs_kmeans`, `gmm_full_cov`, `hclust_dendro`, `hclust_scatter`, `spectral_moons`, `spectral_embedding`, `method_comparison`, `overclustering`, `significance_data`, `significance_null`.

TikZ (in `chapter05.tex`): droplet-based scRNA-seq workflow, mixture model graphical model.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `als_pca.pdf` | own analysis (PCA of ALS post-mortem RNA-seq), recovered from the EMF on slide 3 | no (own figure; confirm data use terms) |
| `pbmc3k_umap.png` | Seurat PBMC 3k tutorial (Satija Lab) | yes |
| `kiselev2019_fig3.png` | Kiselev, Andrews & Hemberg 2019, Nat Rev Genet, Fig. 3 | yes |
| `tam2019_als_subtypes.png` | Tam et al. 2019, Cell Reports (graphical overview/heatmap) | yes (open access; check licence terms) |

Replaced by computed figures: K-means animation (rpubs.com/sandipan), K-means++ comparison (blog post), EM/weighted K-means and GMM-EM animations, two-moons spectral clustering screenshots, Krishnaswamy lab clustering comparison grid, Grabski et al. 2023 single-population simulation and Ward linkage test panels.

Not carried over: Grabski et al. 2023 results table (summarized in the text), 10x Genomics sample/library preparation schematic from the 2023 deck (redrawn in TikZ), title slide.
