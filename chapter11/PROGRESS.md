# Chapter 11 progress

## Sources
- 2026 deck: rendered, all 27 slides read. 2023 deck: rendered, all ink slides read (extra: MAF spectrum slide, pedigree, Punnett ink, LMM two-step).
- Transcripts 2025-11-14 and 2025-11-21 recap/FTO read.
- Slide figures extracted to figures/slides/: fto_locus, fto_region, fto_expression, fto_hic, scz_manhattan, t2d_or_maf, t2d_functional_finemap, coding_fraction, tissue_enrichment.

## Plan (sections)
1 variation (+sfs fig) 2 motivation 3 Mendelian/complex (Punnett TikZ) 4 GWAS model (case-control TikZ, Manhattan computed, multiple testing)
5 recombination & LD (pedigree, recombination TikZ, LD decay derivation, LD heatmap computed, FTO LocusZoom slide)
6 imputation (TikZ) 7 architecture (scz, OR-MAF slide, power computed) 8 pop structure & LMM (QQ computed)
9 summary stats (zhat ~ N(Rz,R) derivation, meta-analysis) 10 fine-mapping (computed) 11 LDSC (computed, schematic TikZ)
12 interpretation (Pickrell, Finucane slides) 13 FTO case study 14 summary/exercises

## Status
- Sections written in chapter11.tex: intro, variation (sfs), motivation, Mendelian/complex (Punnett TikZ), GWAS model (case-control TikZ, manhattan), recombination & LD (pedigree+crossover TikZ, LD decay derivation, ld_heatmap, fto_locus slide fig), imputation (TikZ).
- main.tex and chapter11.bib written; builds.
- Written: architecture (scz, OR-MAF, power_maf), pop structure/LMM (popstrat_qq).
- Written: summary stats (meta-analysis, zhat~N(Rz,R)), fine-mapping (ABF derivation, functional prior), LDSC (derivation, stratified).
- Written: interpretation (Pickrell, Finucane), FTO case study, summary, exercises.
- Done: README, all figure pages checked, notation re-read (fine-mapping prior variance renamed omega^2 to avoid clash with S-LDSC tau_c), build clean (34 pages), latexmk -c.
- Status: complete; awaiting instructor checks listed in final report. interpretation (Pickrell, Finucane), FTO, summary, exercises, README.
- Figures computed and checked (scripts/make_figures.py): sfs, manhattan, ld_heatmap, power_maf, popstrat_qq, finemap (seed 26: lead != causal), ldsc (summary-stat simulation)
