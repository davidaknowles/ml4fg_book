# Chapter 6: hidden Markov models

Converted from the ML4FG topic 6 lecture slides (2026 deck for content and ordering, 2023 deck for the derivations of the likelihood, forward-backward messages, factor graphs and the Baum-Welch updates) and the Fall 2025 lecture transcript. Handwritten slide annotations were turned into text, equations, TikZ diagrams or computed figures.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, scipy, plotnine
latexmk -pdf main.tex
```

`scripts/hmm.py` holds the HMM routines (simulation, scaled forward-backward, Viterbi, Baum-Welch with Bernoulli emissions); `make_figures.py` imports it. `make_figures.py` takes optional arguments `posterior`, `baum_welch`, `dwell` to regenerate a subset (the Baum-Welch figures take a few minutes).

`main.tex` is a standalone wrapper; in the book, add its packages to the book preamble and `\include{chapter06}`.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data from a 5-state toy chromatin HMM): `hmm_posterior`, `baum_welch_trace`, `baum_welch_params`, `dwell_time`.

TikZ (in `chapter06.tex`): toy ChromHMM diagram (redrawn after a teaching diagram by the ChromHMM authors), directed graphical models (toy DAG, confounder, collider), HMM graphical model, forward/backward messages, factor graphs.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `roadmap_tissues.png` | Roadmap Epigenomics Consortium 2015, Nature | yes |
| `capza2_states.png` | Ernst and Kellis 2010, Nat Biotechnol | yes |
| `chromatin_state_cartoon.pdf` | unknown, source to be confirmed (from an EMF in the 2026 deck) | yes |
| `chromhmm_emissions.png` | Ernst et al. 2011, Nature (state legend, emission parameters and fold enrichments composited) | yes |
| `chromhmm_transitions.png` | Ernst et al. 2011, Nature | yes |
| `chromhmm_nine_celltypes.png` | Ernst et al. 2011, Nature (main panel and state legend composited) | yes |
| `pax5_states_celltypes.png` | source to be confirmed, likely Roadmap Epigenomics Consortium 2015 | yes |

Not carried over: the ENCODE experiment-matrix screenshot (2023 deck), the binarized-matrix cartoon in the same EMF as the chromatin state schematic (replaced by the computed `hmm_posterior` figure and the TikZ toy diagram), the links to external resources (cited as references instead).
