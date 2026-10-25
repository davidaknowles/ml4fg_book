# Chapter 1 revision progress (2026 deck update)

Task: compare the 2026 deck with the 2024 "introbio 2" deck, integrate new or changed content with targeted edits, use the Fall 2025 transcripts (2025-09-05, 2025-09-12, 2025-09-19 up to the topic summary) to strengthen explanations.

## Slide comparison

- Renders of both decks and side-by-side pairs are in the scratch directory (outside the repo).
- Mapping: 2026 slides 1-12 = 2024 slides 1-12; 2026 slide 13 is new; 2026 slides 14-54 = 2024 slides 13-53.
- All 54 pairs read (text, notes, ink, pictures). No new embedded pictures in 2026 (new media files are ink fallbacks).
  - 1-12: same content; ink redrawn. 2026 genome slide is a simplified version of 2024's. RNA-seq ink in 2026 says cDNA ~100 nt (2024: fragments 300-400 nt). Notes added for DNA and RNA slides.
  - 13 (new): "Application: precision medicine for cancer", title only, speaker notes describe the spectrum traditional / targeted / precision / personalized medicine.
  - 14: retitled "Cancer cell lines: model system for cancer"; notes add why cell lines (no systematic experiments in humans, mice expensive).

  - 15 (2024 14): same matrices; notes: prediction is row of X inner product with beta.
  - 18: adds 2D log-likelihood contour sketch. 19: adds definition cor = X^T X / N = I for standardized data.
  - 23: MAP slide shows penalized loss with lambda/2 "regularizer"; empirical Bayes as max over sigma of the integral.
  - 28 CV: 20% test (2024: 10%).
  - 29 L1: adds tangent for L2 and set of subgradient lines at 0 for L1. 30: 1D picture shows small (blue) and large (red) lambda levels.
  - 34 breakout: new student answers (pattern recognition, linear, input-output effect, shallower, architecture vs output).
  - 35 interpretability notes: which data points contributed (kNN, SVM).
  - 37 bias-variance: OSR = bias^2 + variance in terms of beta-hat; 2024 speaker notes removed.
  - 38 GD: L-BFGS as low-rank approx of inverse Hessian; Newton and Adam not mentioned.
  - 39 subgradient: value at 0 "could be anything in [-1,1]".
  - 41 dose response: IC50 marked; 43 elastic net: ink labels left "drug resistant", right "sensitive" (but the figure's activity-area axis is inverted: left is sensitive; checked pixel data, RAS mutations cluster left).
  - 45 logistic: explicit intercept augmentation x_i=[x~_i;1], beta=[beta~;mu].
  - 48 binary metrics: AUPR = average precision; random-performance line.
  - 51 (2024 50): retitled "PAM50 subtypes have different prognosis", Kaplan-Meier label.
  - 52 multiclass: -log p(y_i|x_i) = -log p_{i,y_i}.
  - 54 summary: align to genome (or transcriptome).
- Transcripts read: 2025-09-05, 2025-09-12, 2025-09-19 up to "Summary of the first topic".

## Edits (all in chapter01.tex, targeted)

- Statistical genetics: clinical trial phases and why early failure matters (transcript).
- Functional genomics: reads are substrings of the DNA pool (2026 notes); Alzheimer's brain example (transcript).
- DNA: upstream/downstream, reverse complement example (2026 notes, transcript).
- RNA: single-stranded, secondary structure, messenger role, genetic code/codons (2026 notes).
- RNA-seq: why reverse transcription, reverse transcriptase, align to genome or transcriptome (2026 summary), coverage. Bulk vs single cell: brain example, barcodes, ~1M cells.
- Application section retitled "Application: precision medicine for cancer" (2026 slide 13) with three subsections: precision medicine spectrum (new), cell lines as model system (expanded, 2026 slide 14 notes + transcript, HeLa), drug sensitivity data. Activity area vs IC50 rationale.
- OLS: meaning of "ordinary", global maximum, contour picture (2026 slide 18). Independent covariates: correlation matrix X^T X / N (2026 slide 19).
- Ridge: limits of lambda, effect of sigma^2; BayesianRidge; CV more common.
- Regression metrics: RMSE scale intuition, why R^2, label noise.
- Model selection: shuffling, double dipping, test-set reuse, distribution shift; limits of lambda, complexity; LOOCV, log grid, when to use CV, lambda variability.
- Bias-variance: coefficient-level decomposition (2026 slide 37), OLS unbiased, link to OSR (new label sec:gen_theory).
- Lasso: variable selection, no closed form, coordinate descent; 1D optimality-condition picture (2026 slide 30); Gaussian vs Laplace prior intuition.
- Interpretability: 2026 breakout answers, reproducibility, training-sample attribution (2026 notes), causality vs biomarker.
- Generalization: P>N regime, technical conditions.
- Gradient descent: high dimension, overshooting, momentum equation, backtracking; L-BFGS approximates the inverse Hessian (corrected, 2026 slide 38); subgradient midpoint.
- Elastic net application: DNA feature types, proteins, concatenation; kinase/MEK explanation; FIXED heatmap direction (left = sensitive; figure's activity-area axis is inverted) in text and caption; SPRY2/ACSS3 reading; weight sign note.
- Oncotype DX: tamoxifen, archived tissue, chemotherapy trade-off.
- Logistic regression: name, explicit intercept augmentation (2026 slide 45), Bernoulli coin, BCE on logits.
- Chain rule intuition.
- Binary metrics: TPR/FPR/precision in Oncotype terms, class imbalance, thresholds, random PR line, AUROC values, average precision (2026 slide 48), radar name; practical points expanded (noise, costs, task difficulty, label noise).
- Multiclass: Kaplan-Meier explanation (2026 slide 51) in text and caption, logits, -log p_{i,y_i} (2026 slide 52), top-k example, OvR tables, micro/macro.
- Summary: transcriptome, precision medicine bullet. Exercise: coordinate descent update.
- No figures added or changed; README unchanged.

## Verification

- Built with latexmk: no LaTeX warnings, no undefined references or citations; 38 pages (was 32). Aux files cleaned.
- All pages rendered and checked; no figure changes. New equations (bias-variance for coefficients, intercept augmentation) checked.

## Left

- Instructor checks: heatmap direction and weight sign in the CCLE elastic-net figure; HeLa sentence; Oncotype DX cohort description (already open in LABNOTEBOOK).
