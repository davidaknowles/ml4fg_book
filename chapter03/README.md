# Chapter 3: the chromatin landscape and transformers

Converted from the ML4FG topic 3 lecture slides (2026 deck for content and ordering, 2023 deck for the fuller attention derivation) and the Fall 2025 lecture transcripts. Handwritten slide annotations were turned into text, equations, TikZ diagrams or computed figures. Interpretation methods beyond in silico mutagenesis are left to Chapter 4.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, plotnine
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper (it sets the chapter counter so the chapter is numbered 3); in the book, add its packages to the book preamble and `\include{chapter03}`. It adds the TikZ `3d` library to chapter 1's list.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data or exact functions): `assay_profiles`, `eve_stripe2`, `receptive_field`, `attention_matrices`, `attention_scaling`, `positional_encoding`, `positional_dot`, `hic_map`, `hic_di`.

TikZ (in `chapter03.tex`): nucleosome and histone octamer, ATAC-seq, enhancer-promoter loop (redrawn after a slide schematic attributed to Marand et al.), epigenome data tensor, Basset architecture, CNN receptive field and hybrid convolution-attention, self-attention computation, multi-head attention, Hi-C protocol.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `assay_modalities.jpg` | Meyer and Liu 2014, Nat Rev Genet | yes |
| `roadmap_enhancer_activity.png` | Roadmap Epigenomics Consortium 2015, Nature | yes |
| `basset_auc.png`, `basset_filters.png`, `basset_ism.pdf` | Kelley, Snoek and Rinn 2016, Genome Res | yes |
| `enformer_architecture.png`, `enformer_performance.png`, `enformer_tracks.png`, `enformer_tad_attention_map.png`, `enformer_tad_attention_violin.png` | Avsec et al. 2021, Nat Methods (open access, CC BY) | check licence; attribution required |
| `alphagenome.png` | Avsec et al. 2026, Nature | yes |

Not carried over: title slide, Lander quote slide (quoted in text), chromosome video still, video links, "one genome, many cell types" image collage, fly embryo stripe drawing (replaced by a computed toy model), ENCODE experiment matrix screenshot and logos, Roadmap body map and data cube (replaced by TikZ), Basset title screenshot, TF/DNase/histone cartoon profiles (replaced by a computed figure), enhancer looping cartoon (redrawn in TikZ), d2l convolution vs self-attention figure and complexity table (redrawn as a computed figure and a LaTeX table), d2l multi-head diagram (redrawn in TikZ), sinusoidal encoding heatmap (recomputed), RoPE text screenshots (written as equations), Enformer title screenshot, Hi-C protocol figures from Ulianov et al. 2015 and Active Motif (redrawn as a TikZ flow chart), Dixon et al. 2012 TAD figure (replaced by a simulated contact map and directionality index).
