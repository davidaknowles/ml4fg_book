# Chapter 7: recurrent neural networks and DNA methylation

Converted from the ML4FG topic 7 lecture slides (2026 deck for content and ordering, 2023 deck for the fuller vanilla RNN gradient derivation and the DNA methylation enzymes) and the Fall 2025 lecture transcripts. Handwritten slide annotations were turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, plotnine
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter07}`.

## Figure provenance

Computed (`scripts/make_figures.py`, fixed seeds): `tanh` (tanh and sigmoid with derivatives), `gradient_flow` (Jacobian norms for simulated vanilla RNNs and GRUs), `methylation_posterior` (Beta posterior of a CpG methylation level against read depth).

TikZ (in `chapter07.tex`): unrolled RNN, input/output configurations (after Karpathy's blog figure), GRU cell and LSTM cell (redrawn after Olah's blog figures), bidirectional RNN (after Olah), cytosine methylation cycle, CpG island schematic, bisulfite conversion.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `deepcpg_architecture.pdf` | Angermueller et al. 2017, Genome Biol, Fig. 1 (recovered from EMF) | no, CC BY 4.0 (attribution in caption) |
| `deepcpg_auc.png`, `deepcpg_imputation.png`, `deepcpg_motifs.png` | Angermueller et al. 2017, Genome Biol | no, CC BY 4.0 (attribution in caption) |
| `mamba_dna_scaling.png` | Gu and Dao 2023, Mamba (arXiv) | yes |

Not carried over: title slide, OneNote header, screenshots of paper titles (HyenaDNA, Mamba, Caduceus, DeepCpG), and the colah.github.io and karpathy.github.io blog diagrams, which were redrawn in TikZ. The published DNMT/TET figure on the 2023 methylation slide (source not given) was replaced by a TikZ redrawing.
