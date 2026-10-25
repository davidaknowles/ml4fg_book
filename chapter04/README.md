# Chapter 4: interpreting neural networks

Converted from the ML4FG topic 4 lecture slides (2026 deck for content and ordering, 2023 deck for the integrated gradients walk-through), the DeepLIFT exercise sheet and the Fall 2025 lecture transcript. Handwritten slide annotations and node-link diagrams were turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, scikit-learn, plotnine, torch
latexmk -pdf main.tex
```

`make_figures.py` also checks every number quoted in the worked examples and exercises (toy CNN backward passes, DeepLIFT examples, Shapley values) and prints the summary statistics quoted for the simulated-sequence example. Attribution methods (ISM, gradients, deconvnet, guided backprop, DeepLIFT Rescale, integrated gradients, exact Shapley values) are in `scripts/attribution.py`.

`main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter04}`.

## Figure provenance

Computed (`scripts/make_figures.py`): `secant` (sigmoid gradient vs multiplier), `ig_saturation` (integrated gradients on the saturating function), `sequence_attributions` (CNN trained on simulated sequences with GATA and TAL1-like motifs; simulation design after Shrikumar et al. 2017).

TikZ (in `chapter04.tex`): power vs interpretability, local attribution schematic, saturating network, toy CNN backward passes (gradient, deconvnet, guided backprop), 3x3 ReLU rules (own numbers, layout after Springenberg et al. 2015), positive-paths intuition (after M. Guerzhoy's video), threshold network, DeepLIFT on the saturating network.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `occlusion_zeiler_fergus.png` | Zeiler and Fergus 2014, ECCV (occlusion figure, columns a and d) | yes |

Not carried over: Nature news feature header (Castelvecchi 2016; cited instead), sequence-logo screenshots of the DeepLIFT simulation (replaced by the computed `sequence_attributions` figure), the cat photograph used for the guided backprop intuition (replaced by a TikZ schematic), the Springenberg et al. summary figure of ReLU rules (redrawn with new numbers), the sigmoid plot (recomputed), the "interpretation methods" cartoon with a DeepLIFT importance track (redrawn as a schematic).
