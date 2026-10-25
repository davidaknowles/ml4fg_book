# Chapter 8 progress

- [x] Slides read (2026 deck 19 slides, 2023 deck 19 slides; 2023 adds eQTL/PEER slides 4-8, 2026 adds MOFA, CLL, NMF/LIGER)
- [ ] Transcripts read (2025-10-24 from "Matrix factorization models", 2025-10-31 recap)
- [ ] Sections written
- [ ] Figures
- [ ] Build and check

Notation decision: Y (N x P), X (N x K), W (P x K), y_n = W x_n + eps_n, Y ~ X W^T.

## Status (resumed session)
- [x] Transcripts read
- [x] Computed figures done and checked (scripts/make_figures.py): pca_2d, eigen_scree, eigen_mp, fa_vs_ppca, gamma_ard, ica_priors, ica_scatter, linear_ae, nmf_vs_pca, svr_losses, svr_fit, deconvolution_sim
- [x] Slide figures copied to figures/slides (MOFA schematic + CLL, Novembre, Knowler x2, LIGER t-SNE x2, CIBERSORT schematic + Coulter, Gentles x2)
- PEER/eQTL slides (2023 deck only) left to chapter 12 (2026 eQTL deck has PEER); forward reference only.
- [ ] chapter08.tex sections
- [x] main.tex written (setcounter 7)
- [x] Sections written: MF models, factor analysis, fitting FA (EM)
- [x] MOFA section
- [x] PPCA, PCA, population structure sections
- [x] Autoencoders, ICA sections
- [x] NMF + LIGER section
- [x] Deconvolution, summary, exercises sections. Next: bib, README, build
- [x] chapter08.bib
- [x] README; build ok
- [x] Pages rendered and checked (fixed Fig 8.1 label overlap, AE label, ICA layout). latexmk -c done.
- DONE. Remaining: instructor checks listed in final report.
