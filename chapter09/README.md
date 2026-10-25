# Chapter 9: biological networks

Converted from the ML4FG topic 9 lecture slides (2026 deck for content and ordering, 2023 deck for the handwritten derivations, which are nearly identical) and the Fall 2025 lecture transcript. Handwritten slide annotations were turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scikit-learn, networkx, plotnine
latexmk -pdf main.tex
```

`make_figures.py` also writes `figures/network_stats.tex`, a set of macros holding numbers quoted in the text (sample correlations, cross-validated lambda, graph statistics, SBM estimates), so the text stays consistent with the figures. `main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter09}`.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data, fixed seeds): `partial_correlation`, `glasso_path`, `betweenness`, `degree_distribution`, `sbm_adjacency`.

TikZ (in `chapter09.tex`): gene regulatory network schematic, yeast two-hybrid assay (redrawn from a slide schematic), TE/IR network pipeline (after Saha et al. 2017), hub motif types (after Saha et al. 2017), tissue-hierarchy sharing cartoon (after Pierson et al. 2015), SBM/MMSB graphical models. Table 9.1 (TE and IR example) is retyped from a slide figure in Saha et al. 2017.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `dixit_perturbseq_network.png` | Dixit et al. 2016, Cell | yes |
| `schizophrenia_interactome.png` | Ganapathiraju et al. 2016, npj Schizophrenia (open access, CC BY) | attribution; confirm license |
| `costanzo_functional_map.png` | Costanzo et al. 2016, Science | yes |
| `metabolic_metro_map.png` | Bert Chan, "Metabolic Metro Map", Wikimedia Commons (believed CC BY-SA) | attribution; confirm license |
| `signaling_pathways.png` | signal transduction pathways diagram, believed from Wikimedia Commons | yes, source to be confirmed |
| `gtex_tissue_dendrogram.png`, `pierson_coronary_network.png`, `pierson_cv_likelihood.png` | Pierson et al. 2015, PLoS Comput Biol (open access, CC BY) | attribution |

Not carried over: title-slide quote (used in the text), paper title screenshots (Perturb-seq, CROP-seq, module networks, Mostafavi et al., Saha et al., Pierson et al., Costanzo et al., Reguly et al., Abbe, Kemp et al., Airoldi et al.), the Mostafavi et al. module network figure (described in text), the Costanzo et al. full network hairball, the phosphorylation cartoon, the GTEx portal sample-size wheel (sample sizes described in text), the hippocampus tissue-network grid (described in text), the Wikipedia betweenness graph and Reguly et al. degree plot (replaced by computed figures), the Abbe SBM illustration and Kemp et al. IRM figure (replaced by the computed SBM figure), and the Airoldi et al. annotation-accuracy histogram (described in text).
