# Chapter 5 progress

- [x] Rendered 2026 (20 slides) and 2023 (23 slides) decks; extracted text/notes
- [x] Read all slide images (2023 adds: scRNA-seq tech slides 4-6, mixture model II, K-means algorithm, EM derivation slides 13-14, spectral clustering ink)
- [x] Transcript search: 2025-10-10 lecture missing; only brief recaps (HMM lecture 10-17: soft k-means analogy, EM, spectral init; factor analysis link). Little to use.
- [x] Slide 3 EMF -> own ALS PCA plot (emf_to_pdf works)
- [x] Sections written (all; chapter05.tex), bib written
- [x] Figures computed and checked visually (scripts/make_figures.py: mixture_generative, kmeans_iterations, kmeanspp, choose_k, jensen, em_vs_kmeans, gmm_full_cov, hclust_dendro/scatter, spectral_moons/embedding, method_comparison, overclustering, significance_data/null)
- [x] Slide figures copied: figures/slides/{als_pca.pdf (own), kiselev2019_fig3.png, pbmc3k_umap.png, tam2019_als_subtypes.png}
- [x] Build OK (31 pages, no undefined refs)
- [ ] Visual page check: all pages checked at low res, floats fixed ([!htbp]); 29 pages
- [x] README.md
- [x] latexmk -c; final report sent

Plan: sections = supervised vs unsupervised; clustering in genomics (ALS PCA, scRNA-seq TikZ, PBMC UMAP); GMM; K-means (+practical: init, k-means++, choosing K/gap); EM (Jensen, ELBO, responsibilities, learning pi/sigma, K-means limit); generative model extensions (full cov); scRNA-seq clustering methods (Kiselev fig, hierarchical, spectral, Louvain/modularity, method comparison computed); significance (Grabski sc-SHC, computed analog); ALS subtypes application; summary; exercises.
