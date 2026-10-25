# Chapter 14: protein structure prediction with AlphaFold 2

Converted from the ML4FG topic 14 lecture slides (2026 deck; there is no 2023 deck for this topic, and the deck builds on slides by Mohammed AlQuraishi and Nazim Bouatta) and the Fall 2025 lecture transcript. The deck has no handwritten ink: its diagrams are PowerPoint shapes and its equations are embedded PDFs, which were redrawn in TikZ or written as equations. Details beyond the slides (layer equations, sizes, loss terms) follow Jumper et al. 2021 and its supplementary information.

## Build

```
python scripts/make_figures.py   # requires numpy, pandas, plotnine
latexmk -pdf main.tex
```

`main.tex` is a standalone wrapper (it sets the chapter counter so the chapter is numbered 14); in the book, add its packages to the book preamble and `\include{chapter14}`. It uses the same packages as chapter 1.

## Figure provenance

Computed (`scripts/make_figures.py`, simulated data, fixed seeds):
- `coevolution`: MSA evolved down a binary tree under a Potts model with planted contacts; mutual information and APC-corrected MI.
- `fape_vs_rmsd`: FAPE, superposed RMSD and dRMSD for an ideal alpha helix with perturbed torsion angles, and for its mirror image.
- `triangle_violations`: fraction of residue triples violating the triangle inequality when helix distances are perturbed independently.

TikZ (in `chapter14.tex`): backbone with torsion angles and residue triangles; toy MSA with conserved, variable and co-evolving columns; Evoformer two-track schematic (after AlQuraishi/Bouatta slides); triangle inequality and triangle update; residue frame $T_i=(R_i,t_i)$; black hole initialization, residue gas and invariant point attention; equivariance commutative diagram; FAPE schematic.

Taken from the slides (`figures/slides/`):

| File | Source | Permission needed for publication |
|---|---|---|
| `alphafold2_architecture.png` | Jumper et al. 2021, Nature, Fig. 1e (open access, CC BY 4.0) | check licence; attribution required |

Not carried over: title slide image, attention recap animation (Alammar/Kelley "cogito ergo sum" example; covered in chapter 3), section-divider slides, rendered protein images (backbone ball-and-stick, side-chain rotamer, surface and cartoon renderings of predicted and experimental structures, torsion-angle torus), template contact-map image; all replaced by TikZ or text.
