# Chapter 13 progress

## Sources
- 2026 deck (22 slides): rendered and all slides read; text and speaker notes extracted. No handwritten ink in this deck. EMF on slide 21 (regulatory variant schematic) converted and read. No 2023 deck.
- Transcripts: topic not lectured in Fall 2025. Used 2025-11-07 de novo section, 2025-11-14 human variation and effect size vs MAF, 2025-11-21 ASE/beta-binomial.

## Plan (sections)
1 intro; 2 limits of single-variant tests (effect-size vs AF TikZ, power derivation + computed fig); 3 burden tests (collapsing, de novo Poisson with mutation rates, gnomAD constraint, controls/TRAPD);
4 pathogenicity prediction (scores overview, LINSIGHT/INSIGHT model + TikZ + table, slide figs 8, 9);
5 SKAT/SKAT-O (score test derivation, weights, mixture chi2, SKAT-O grid; computed power fig);
6 FST (derivation, computed power fig); 7 noncoding DNMs in autism (Zhou 2019: DeepSEA+lasso, quartet TikZ, signed-rank, slide fig 17);
8 NDEA (weighted Welch t, TikZ); 9 experimental validation (luciferase, slide fig 20); 10 molecular data (outliers, ANEVA-DOT computed, regulatory variant TikZ);
11 DeepRVAT (deep sets, TikZ); 12 summary + exercises.

## Status
- Sections written: none yet
- Figures: none yet

## Update (resumed after usage reset)
- Paper fact-check done via subagent (He 2017, LINSIGHT, Zhou 2019, DeepRVAT). Corrections vs slides: Zhou used L2-regularized logistic regression (not lasso), negatives were 1000 Genomes singletons (slide says gnomAD), Wilcoxon rank-sum (slide says signed-rank). LINSIGHT: rho = Gompertz link exp(-3exp(-x)), gamma logistic, eta dropped, SGD (AdaGrad), 48 features. FST combines by min-p. DeepRVAT: seed genes from burden/SKAT baseline, 5-fold CV, burden test on psi.
- Figures computed: power_single_variant, maf_weights, outliers, aneva_dot (checked visually). skat_power tuned; fst_power running.
- Next: write chapter13.tex section by section.
- Sections written: intro, 13.1 why rare (TikZ effect_freq, power fig), 13.2 burden tests. main.tex done.
- 13.3 pathogenicity / LINSIGHT written (TikZ model, table, slide figs linsight_fitness, linsight_hgmd_roc).
- 13.4 SKAT/SKAT-O written (derivation, null, weights fig, SKAT-O, power fig).
- 13.5 FST written (model, min-p test, He et al. power description, own fst_power fig).
- 13.6 autism case study written (TikZ pedigree+pipeline, rank-sum, slide fig asd_disease_impact).
- 13.7 NDEA (weighted Welch derivation, TikZ) and 13.8 experimental validation (slide fig asd_luciferase) written.
- 13.9 molecular data written (regulatory variant TikZ, outliers fig, ANEVA-DOT model + fig).
- 13.10 DeepRVAT (TikZ), summary, exercises written. All sections drafted. Next: build, check pages, README.
- DONE: build clean (29 pages, no undefined refs/citations, no overfull boxes), pages with figures checked visually and TikZ fixed (effect_freq ellipses, LINSIGHT spacing, ASD pipeline overlap, DeepRVAT width). README written. latexmk -c run.
