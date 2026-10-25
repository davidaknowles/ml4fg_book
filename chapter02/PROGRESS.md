# Chapter 2 progress

- [x] Read brief, chapter 1 template
- [x] Extracted slide text/notes (2026, 2023) -> scratch text26.txt, text23.txt
- [x] 2026 slides all read (renders in scratch d26_r, pairs p26). Many shapes/tables (Ecker grammar crops, CNN diagram, PWM scan tables) do not render; content taken from text extraction.
- [x] 2023 slides 1-8 read (MLP math, chain rule, backprop graph, SGD)
- [x] 2023 slides 9-34 (mostly duplicates of 2026; extra: output-length derivation H_in-(L-1), stride-1 recommendation, multitask "same TF in two cell types")
- [x] Transcripts read (2025-09-19 from Limitations of linear models; 2025-09-26 to Layers of a CNN)
- [x] EMF figures: image3 = generic MLP diagram (redo in TikZ), image10 = when/where/why boxes (text)
- [x] Slide figures copied: chipseq_igb.jpg (Freese et al. 2016), sox2_oct4_motifs.jpg (Rizzino & Wuebben 2016)
- Verified: slide PWM scan scores = GATA PWM on GCATTACCGATAA with left pad 2, right pad 3. Torch toy CNN numbers (slide 31-33) reproducible deterministically.
- [ ] Sections written
- [ ] Figures
- [ ] Build and check

## Plan
Sections: beyond linear models; MLP; backprop; SGD/Adam; gene regulation; ChIP-seq; motifs & grammar; PWMs; scanning/one-hot; convolution arithmetic; pooling/ReLU; CNN for TF binding (+toy worked example, multitask); PWM vs filters; strengths/weaknesses; further reading; summary; exercises.
Computed figs: limitations_nonlinear, limitations_interaction, activations, mlp_random, sgd_paths, logo_wiki, logo_gata, pwm_scan.tex, toy_cnn.tex
- [x] make_figures.py written and run; figures checked visually (limitations x2, activations, mlp_random, sgd_paths, logo_wiki, logo_gata) + tables pwm_example_rows.tex, pwm_scan_rows.tex; toy CNN reproduces slide numbers (assert)
- [x] main.tex
- [x] Sections written: intro, beyond linear, neural networks, backprop, SGD/Adam
- [x] Sections written: gene regulation, ChIP-seq (+Poisson peaks), motifs/grammar, seq-to-function
- [x] Sections written: PWM (derivation, estimation, worked example, logistic link, logos, scanning, futility)
- [x] Sections written: CNN (conv, padding/stride/dilation, pooling, architecture, worked example, multitask, filters vs PWMs), strengths/weaknesses, summary, exercises
- [x] bib written; build clean (no undefined refs/cites)
- [x] Pages rendered and checked; fixed CNN figure (one-hot fill, flatten lines), chapter counter, table \input
- [x] README.md
- [x] Final build clean, latexmk -c done (29 pages)
