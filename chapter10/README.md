# Chapter 10: RNA splicing

Converted from the ML4FG topic 10 lecture slides (2026 deck for content and ordering, 2023 deck for the handwritten annotations) and the Fall 2025 lecture transcripts. Handwritten slide content was turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, plotnine
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter10}`.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data, fixed seeds): `dirichlet_simplex`, `dirichlet_draws`, `em_toy`, `vb_vs_gibbs`, `kl_directions`, `meanfield`, `psi_beta`.

TikZ (in `chapter10.tex`): gene anatomy and splice-site consensus, splicing reaction (lariat), tau neuron and MAPT exon 10, types of alternative splicing, split reads, read-to-isoform ambiguity (after Pachter 2011), graphical model, percent spliced in (after Xiong et al. 2015), residual block and dilated convolutions, short versus long reads.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `mrna_processing.png` | Desterro et al. 2020, Nat Rev Drug Discov | yes |
| `neurexin_isoforms.jpg` | Treutlein et al. 2014, PNAS | yes |
| `bitseq_comparison.pdf` | Hensman et al. 2015, Bioinformatics | yes |
| `spliceosome.jpg` | own figure (instructor), drawn from textbook sources | no (confirm) |
| `xiong_psi_prediction.pdf`, `xiong_delta_psi.pdf` | Xiong et al. 2015, Science (Figs 1C, 3C) | yes |
| `spliceai_architecture.png`, `spliceai_cftr.pdf`, `spliceai_saliency.pdf`, `spliceai_pygb.pdf` | Jaganathan et al. 2019, Cell | yes |
| `rosenberg_library.pdf`, `rosenberg_learning_curve.pdf`, `hal_vs_spanr.png` | Rosenberg et al. 2015, Cell | yes |

The PDF and the cropped PNGs were recovered from the PDFs embedded in the slide EMF pictures (PNGs rendered at 300 dpi from the crop box of the panel used on the slide).

Not carried over: gene anatomy textbook figure (Developmental Biology, 9e, Fig. 2.5) and the gene-feature diagram of Barnes 2010 (both redrawn in TikZ); the splicing lariat, alternative splicing modes, isoform cartoon and tau figures (redrawn); Dirichlet simplex and draw figures (recomputed); Minka's alpha-divergence figure and the mean-field figure (recomputed); He et al. ResNet block and Yu & Koltun dilated-convolution figures (redrawn); paper title pages; the SpliceAI MYBPC3 example and MED13 sashimi plot (described in text only); the SpliceAI de novo table (typeset as a table); PacBio marketing figure (redrawn as a schematic); Saberi et al. 2024 tokenization figure (described in text).
