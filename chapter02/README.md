# Chapter 2: deep neural networks and gene regulation

Converted from the ML4FG topic 2 lecture slides (2026 deck for content and ordering, 2023 deck for the backpropagation and convolution derivations) and the Fall 2025 lecture transcripts. Handwritten slide annotations were turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, scikit-learn, matplotlib, plotnine, contourpy
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper (it sets the chapter counter so numbering starts at 2); in the book, add its packages to the book preamble and `\include{chapter02}`. The chapter `\input`s two generated table fragments, `figures/pwm_example_table.tex` and `figures/pwm_scan_table.tex`, by a path relative to the chapter folder.

`make_figures.py` also prints and checks the numbers quoted in the text: the toy PWM worked example (odds 185), the PWM scan and max pooling values, the hand-set toy CNN (asserts the values on the lecture slide), a finite-difference check of the backpropagation equations, and the Poisson peak-calling p-values.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data or values from the slides): `limitations_nonlinear`, `limitations_interaction`, `activations`, `mlp_random`, `sgd_paths`, `logo_wiki`, `logo_gata`, and the tables `pwm_example_table.tex`, `pwm_scan_table.tex`. The sequence logos are drawn with plotnine from letter outlines.

TikZ (in `chapter02.tex`): one-hidden-layer MLP, backpropagation computational graph, transcriptional regulation, ChIP-seq protocol, motif grammar, sequence-to-function classifier, convolution padding/stride, CNN architecture.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `chipseq_igb.jpg` | Freese et al. 2016, Bioinformatics (Integrated Genome Browser) | yes (check licence; the article may be open access) |
| `sox2_oct4_motifs.jpg` (cropped into `_A.png`, `_B.png` for side-by-side panels) | Rizzino and Wuebben 2016, Biochim Biophys Acta (as credited on the slide) | yes |

Not carried over: title-slide image; embedded YouTube videos on transcription and ChIP-seq; the motif-grammar cartoons cropped from a Nature Reviews Genetics figure (`nrg3682-f2.jpg`, redrawn in TikZ); Wikipedia's PWM matrices and sequence logos (recomputed); the GATA-DNA structure image; the convolution animations of Dumoulin and Visin (redrawn in TikZ); the Amberg convolution animation (Wikipedia, CC BY-SA); the PyTorch notebook screenshots (reproduced as a worked example in the text); the 2023 deck's decision tree (DataCamp) and Gaussian process figures; the discussion-question slides (summarized in the text).
