# Pre-publication checklist

Open items across all chapters, collected from the chapter READMEs, the `.bib` files and the chapter-writing reports. Tick items off as they are resolved. Chapter READMEs hold the full figure provenance tables; `captions/NOTES.md` (local only, not in the repository) lists reconstructions in the lecture transcripts.

Sections:
1. Slide errors found (fix the slides too)
2. Facts and numbers to check, by chapter
3. Figure permissions
4. Incomplete references

---

## 1. Slide errors found

The chapters already use the corrected version; these are for fixing the decks.

- [ ] **Topic 01:** CCLE elastic-net heatmap (PD-0325901) is labelled resistant on the left in the 2024 and 2026 ink. The activity-area axis is inverted, so the left is sensitive.
- [ ] **Topic 02:** grammar figure credited to "Ecker et al. 2012"; the image (`nrg3682-f2`) is likely Spitz & Furlong 2012, Nat Rev Genet.
- [ ] **Topic 02:** ChIP-seq cross-linking given as UV; ChIP typically uses formaldehyde (UV is used for CLIP).
- [ ] **Topic 02:** the GATA PWM is not on a log2(P/0.25) scale; it is illustrative only.
- [ ] **Topic 04:** SHAP weight 1/(P−|z|)·C(P,|z|)⁻¹ gives (−1)! at |z| = P; use the standard subset form |S|!(P−|S|−1)!/P!.
- [ ] **Topic 08:** "Fitting factor analysis" writes WᵀW; it should be WWᵀ (the 2023 deck is correct). Tipping & Bishop is 1999 (slide 2002); Novembre is 2008 (slide 2009).
- [ ] **Topic 09:** 2023 graphical lasso slide has a sign error (−λ‖Θ‖₁) and uses S for precision and Θ/Σ for covariance.
- [ ] **Topic 10:** Desterro et al. dated 2019 on the slide; published 2020.
- [ ] **Topic 12:** negative binomial variance written "a/b + 1/a"; it is μ + μ²/a. The combined ASE likelihood omits the homozygous term, and the network equation Y = YG + E has the multiplication order wrong.
- [ ] **Topic 13, slide 15:** Zhou et al. 2019 used L2-regularized logistic regression (λ = 10), not a lasso; negatives were 1000 Genomes singletons, not gnomAD; the test is a one-sided Wilcoxon rank-sum, not signed-rank.
- [ ] **Topic 13, slide 22 notes:** in DeepRVAT ψ is the gene impairment score; φ and ρ are the MLPs.
- [ ] **Topic 14, slides 37–38:** the invariant point attention score has + before the squared-distance term; the paper subtracts it. The FAPE slide omits ε inside the square root.
- [ ] **Topic 14:** speaker notes look auto-generated and contain errors (RGN2, nuclear pore complex).

## 2. Facts and numbers to check

Each item was stated from memory, reconstructed, or taken from the slides or lecture without verification.

### Chapter 1: molecular biology basics and linear models
- [ ] Elastic-net weight signs: with the corrected direction, sensitivity-linked features (SPRY2, RAS mutations) have negative weights. Confirm against Barretina et al. 2012. Also the ACSS3 trend and the bootstrap-frequency reading of the numbers next to feature names.
- [ ] Oncotype DX cohort: "about 700 patients, about 10,000 genes" from the slides. The commercial test is a 21-gene RT-PCR assay validated on 668 NSABP B-14 patients.
- [ ] HeLa sentence, including the consent issue.
- [ ] 2022 sequencing-cost claims (Ultima "$100 genome", NovaSeq X about $200), from the speaker notes.
- [ ] Lasso risk rate attributed to Greenshtein & Ritov 2004.

### Chapter 2: deep neural networks and gene regulation
- [ ] About 1,600 human TFs (Lambert 2018); DeepSEA's 919 features; about 1.5×10⁶ chance 6-mer matches vs 10³–10⁵ ChIP-seq peaks.
- [ ] The ChIP-seq browser figure (Freese et al.) does not name the TF; the lecture guessed SOX2.
- [ ] "Resembling the NANOG core" for the toy filter [C/G]ATTA[C/G] is loose.
- [ ] Flat-minima generalization claim (cited to Hochreiter & Schmidhuber 1997, Zhang et al. 2017).

### Chapter 3: the chromatin landscape and transformers
- [ ] Corrections from the lecture: ENCODE "function" claim given as about 80% (lecture 70%); draft genome 2001 (lecture 2002); Enformer test split by genomic region (lecture: chromosome); *eve* stripe 2 posterior border set by Krüppel (lecture: Giant for both).
- [ ] From memory: Basset 600 bp input, 300 width-19 filters, about 2M DHSs; Basenji 131 kb input, 128 bp bins; Basenji2 range about 20 kb; Enformer 196,608 bp input, Poisson loss, relative positional basis functions; about 6,000 fly nuclei at 2–3 h; contact frequency decaying about 1/distance; 4–16 heads.
- [ ] AlphaGenome description (architecture, training stages, track counts) taken from the slide figure.
- [ ] Enhancer-loop TikZ redrawn after a "Marand et al. 2016" cartoon; add a citation if the resemblance stays.
- [ ] Table 3.1 histone-mark associations are textbook generalizations.

### Chapter 4: interpreting neural networks
- [ ] Simulated motif consensus sequences (GATA `AGATAAG`, TAL1-like `AACAGATGGT`) chosen by the agent from slide logos.
- [ ] Deconvnet fails and guided backprop matches the gradient in the simulation (global average pooling). Retrain with max pooling if the slide-like panels are wanted.
- [ ] Enformer ISM cost (about 600,000 forward passes).
- [ ] Reading of the threshold slide's reference (third input kept at 1).

### Chapter 5: unsupervised learning and clustering
- [ ] ALS PCA figure: caption says "From the author's analysis". Confirm the cohort (2023 notes say more than 2,000 post-mortem samples) and data-use terms.
- [ ] Tam et al. 2019 clustering method not named (possibly NMF-based); the caveats about the glia cluster and replication are additions.
- [ ] Grabski et al. cluster-count ranges taken from the slide table; their null model described only as "a model for the counts".
- [ ] Double-dipping point (Gao et al. 2024) is an addition not in the slides.

### Chapter 6: hidden Markov models
- [ ] ChromHMM bin size 200 bp (slides) vs 100 bp (lecture).
- [ ] ChromHMM defaults from memory: Poisson binarization at p < 10⁻⁴, maximum-posterior segmentation.
- [ ] "One to two million" candidate regulatory elements (slide: about 2 million; ENCODE 2020: about 0.9 million cCREs).
- [ ] Alzheimer's risk enrichment in microglial enhancers (Nott et al. 2019).
- [ ] Interpretation of the Ernst et al. 2011 emission and transition figures, and the WLS region size (about 830 kb).

### Chapter 7: recurrent neural networks and DNA methylation
- [ ] DeepCpG's GRU runs across cells, not along the genome (chapter follows the paper; the lecture said otherwise).
- [ ] CpG statistics: 70–80% of CpGs methylated (Jones 2012), about 70% of promoters with an island (Saxonov 2006), 25,000–30,000 islands.
- [ ] scBS-seq coverage 20–40% and scRRBS 1–10% (from the DeepCpG paper); Horvath clock with 353 CpGs.
- [ ] GRU naming: z_t/r_t used instead of the slide's f_t/o_t.

### Chapter 8: matrix factorization
- [ ] MOFA details: spike-and-slab on loadings, Bernoulli likelihood for mutations, 310 drug-response features, Factor 1 = IGHV, Factor 2 = trisomy 12.
- [ ] Novembre sample size (about 1,400 people, about 200,000 SNPs); Knowler association "largely disappeared" after adjusting for heritage.
- [ ] CIBERSORT procedure (standardization, ν ∈ {0.25, 0.5, 0.75}, negatives set to zero, permutation p-value); Gentles "thousands of tumours, 25 malignancies".
- [ ] LIGER fitting by alternating NNLS and its post-factorization normalization; BBP detection threshold ‖w‖² > σ²√γ.

### Chapter 9: biological networks
- [ ] Corrections from the lecture: yeast has about 6,000 genes (lecture 8,000); SBM dates to 1983/2001 and the IRM to 2006.
- [ ] About 23 million double mutants (Costanzo 2016); GTEx about 1,000 donors and 50 tissues; early-release tissue sizes of tens to under 200.
- [ ] Reconstructions: colony-size model terms; Pierson et al. leaf update as graphical lasso plus a quadratic pull; internal-node update as a tree-Laplacian system.
- [ ] Saha et al. penalty scheme and tissue count deliberately not stated.

### Chapter 10: RNA splicing
- [ ] Gene-scale numbers (exon about 150 nt, intron about 3.5 kb, 5′ UTR about 200 nt, 3′ UTR about 800 nt): means or medians?
- [ ] De novo table: "7% (ID) / 16% (ASD)" as on the slide; the SpliceAI abstract says 9–11% of pathogenic mutations overall. Confirm the row label.
- [ ] From memory: SpliceAI held-out chromosomes, GENCODE labels, Δ score as max of four values; MaxEntScan windows 9 and 23 nt; HAL as a linear k-mer model; Xiong 2015 training of about 10,000 exons in 16 tissues.
- [ ] Otari mentioned without a citation.

### Chapter 11: genome-wide association studies
- [ ] Figure credits: `t2d_or_maf.png` to Mahajan et al. 2018; `tissue_enrichment.png` to Finucane 2015 (lecture tied it to Pickrell).
- [ ] About 1 million effective tests behind 5×10⁻⁸ (lecture: maybe 500,000); 4–5 million sites differing from the reference; 50–100 de novo mutations; out-of-Africa about 60,000 years ago; CF carrier frequency about 4%.
- [ ] Finucane: conserved regions hold about 2.6% of SNPs and about a third of heritability.
- [ ] FTO details shared with chapter 12 (rs1421085, ARID5B motif, IRX3/IRX5 distances, CRISPR rescue).
- [ ] Two Claussnitzer FTO panels appear in both chapters 11 and 12; keep one and cross-refer.

### Chapter 12: molecular quantitative trait loci
- [ ] Own methods: MR total-effect rescaling R = D⁻¹(I−G)⁻¹ and recovering G; inspre applied to Perturb-seq.
- [ ] GTEx numbers from slides: 95% of protein-coding genes with a cis-eQTL, 79% with a cis-sQTL, 178 trans-eGenes at 10% FDR; 100 to about 700 samples per tissue; PEER factor counts chosen by sample size.
- [ ] Sasse et al. "about a third wrong sign" (195 of 598).
- [ ] Softened: the interleukin-18 receptor/celiac example and the AGA CRISPR validation are described generically.

### Chapter 13: rare variants
- [ ] LINSIGHT likelihood table is a simplification (ancestral allele assumed known), reconstructed from the authors' code; links, 48 features, 54 genomes, SGD fitting.
- [ ] Functional score test combination by min-p over k and ρ ∈ {0, 1} with a Gaussian-copula null.
- [ ] NDEA weights and GO meta-nodes; the enrichment test behind the reported clusters was not found.
- [ ] Luciferase assay: 230 bp constructs, neuroblastoma line, 57 of 59 active regions with an allelic difference; "sibling" read as the reference allele.
- [ ] Outlier papers: slide's "GTEx 2019" cited as Li et al. 2017 and Ferraro et al. 2020; Li 2017 enrichment for structural, frameshift and splice variants.
- [ ] About 70 de novo mutations per genome (lecture); DeepRVAT about 160,000 individuals, 21 traits, 34 annotations; Eigen-PC description.

### Chapter 14: protein structure prediction with AlphaFold 2
- [ ] Model sizes from memory: S = 512, crop 384, D_m = 256, D_z = 128, D_s = 384, 12 IPA heads, 4 query points, outer product mean D_o = 32.
- [ ] Distogram 64 bins over 2–22 Å; pLDDT 50 bins; masked MSA at 15%; clamping in 90% of examples; self-distillation on about 350k predictions.
- [ ] CASP14 accuracy (about 1 Å vs about 3 Å); accuracy dropping below about 30 effective sequences; PDB about 200,000 structures.
- [ ] "Over 200 million" structures cited to Varadi 2022, which predates that expansion.

## 3. Figure permissions

Third-party figures reproduced from slides, by chapter. "Attribution" means the source is open access (CC BY) and needs only credit, but confirm the licence.

| Ch | Figure files | Source | Status |
|---|---|---|---|
| 1 | `genetic_support_odds_ratio.png` | Nelson et al. 2015, Nat Genet | [ ] permission |
| 1 | `bulk_vs_singlecell.png` | 10x Genomics blog | [ ] permission |
| 1 | `primary_culture.png` | **unknown source** | [ ] identify source, permission |
| 1 | `ccle_tumour_types`, `ccle_data_types`, `ccle_elastic_net_pd0325901` | Barretina et al. 2012, Nature | [ ] permission |
| 1 | `pam50_heatmap`, `pam50_survival` | Parker et al. 2009, J Clin Oncol | [ ] permission |
| 2 | `chipseq_igb.jpg` | Freese et al. 2016, Bioinformatics | [ ] permission (may be open access) |
| 2 | `sox2_oct4_motifs.jpg` | Rizzino & Wuebben 2016, BBA | [ ] permission |
| 3 | `assay_modalities.jpg` | Meyer & Liu 2014, Nat Rev Genet | [ ] permission |
| 3 | `roadmap_enhancer_activity.png` | Roadmap Epigenomics 2015, Nature | [ ] permission |
| 3 | `basset_auc`, `basset_filters`, `basset_ism` | Kelley et al. 2016, Genome Res | [ ] permission |
| 3 | five `enformer_*` panels | Avsec et al. 2021, Nat Methods | [ ] attribution, confirm CC BY |
| 3 | `alphagenome.png` | Avsec et al. 2026, Nature | [ ] permission |
| 4 | `occlusion_zeiler_fergus.png` | Zeiler & Fergus 2014, ECCV | [ ] permission |
| 5 | `pbmc3k_umap.png` | Seurat PBMC 3k tutorial | [ ] permission |
| 5 | `kiselev2019_fig3.png` | Kiselev et al. 2019, Nat Rev Genet | [ ] permission |
| 5 | `tam2019_als_subtypes.png` | Tam et al. 2019, Cell Reports | [ ] permission (open access; check terms) |
| 6 | `roadmap_tissues.png` | Roadmap Epigenomics 2015, Nature | [ ] permission |
| 6 | `capza2_states.png` | Ernst & Kellis 2010, Nat Biotechnol | [ ] permission |
| 6 | `chromatin_state_cartoon.pdf` | **unknown source** (2026 deck EMF) | [ ] identify source, permission |
| 6 | `chromhmm_emissions`, `chromhmm_transitions`, `chromhmm_nine_celltypes` | Ernst et al. 2011, Nature | [ ] permission |
| 6 | `pax5_states_celltypes.png` | **unconfirmed**, likely Roadmap 2015 | [ ] confirm source, permission |
| 7 | four `deepcpg_*` figures | Angermueller et al. 2017, Genome Biol | [ ] attribution (CC BY 4.0) |
| 7 | `mamba_dna_scaling.png` | Gu & Dao 2023, arXiv | [ ] permission |
| 8 | `mofa_schematic`, `mofa_cll` | Argelaguet et al. 2018, Mol Syst Biol | [ ] permission (open access; check terms) |
| 8 | `novembre_pca_europe.png` | Novembre et al. 2008, Nature | [ ] permission |
| 8 | `knowler_fig1`, `knowler_fig3` | Knowler et al. 1988, AJHG | [ ] permission |
| 8 | `liger_tsne_technology`, `liger_tsne_celltype` | Welch et al. 2019, Cell | [ ] permission |
| 8 | `cibersort_schematic`, `cibersort_coulter` | Newman et al. 2015, Nat Methods | [ ] permission |
| 8 | `gentles_fractions`, `gentles_survival` | Gentles et al. 2015, Nat Med | [ ] permission |
| 9 | `dixit_perturbseq_network.png` | Dixit et al. 2016, Cell | [ ] permission |
| 9 | `schizophrenia_interactome.png` | Ganapathiraju et al. 2016, npj Schizophr | [ ] attribution, confirm CC BY |
| 9 | `costanzo_functional_map.png` | Costanzo et al. 2016, Science | [ ] permission |
| 9 | `metabolic_metro_map.png` | Bert Chan, Wikimedia Commons | [ ] attribution, confirm CC BY-SA |
| 9 | `signaling_pathways.png` | **unconfirmed**, believed Wikimedia Commons | [ ] confirm source and licence |
| 9 | `gtex_tissue_dendrogram`, `pierson_coronary_network`, `pierson_cv_likelihood` | Pierson et al. 2015, PLoS Comput Biol | [ ] attribution (CC BY) |
| 10 | `mrna_processing.png` | Desterro et al. 2020, Nat Rev Drug Discov | [ ] permission |
| 10 | `neurexin_isoforms.jpg` | Treutlein et al. 2014, PNAS | [ ] permission |
| 10 | `bitseq_comparison.pdf` | Hensman et al. 2015, Bioinformatics | [ ] permission |
| 10 | `xiong_psi_prediction`, `xiong_delta_psi` | Xiong et al. 2015, Science | [ ] permission |
| 10 | four `spliceai_*` figures | Jaganathan et al. 2019, Cell | [ ] permission |
| 10 | `rosenberg_library`, `rosenberg_learning_curve`, `hal_vs_spanr` | Rosenberg et al. 2015, Cell | [ ] permission |
| 11 | `fto_locus`, `fto_region`, `fto_expression`, `fto_hic` | Claussnitzer et al. 2015, NEJM | [ ] permission (shared with ch12) |
| 11 | `scz_manhattan.png` | Trubetskoy et al. 2022, Nature | [ ] permission |
| 11 | `t2d_or_maf.png` | Mahajan et al. 2018, Nat Genet (**attribution unconfirmed**) | [ ] confirm source, permission |
| 11 | `t2d_functional_finemap.png` | Mahajan et al. 2018, Nat Genet | [ ] permission |
| 11 | `coding_fraction.png` | Pickrell 2014, AJHG | [ ] permission |
| 11 | `tissue_enrichment.png` | Finucane et al. 2015, Nat Genet (**attribution unconfirmed**) | [ ] confirm source, permission |
| 12 | `fto_locus`, `fto_expression` | Claussnitzer et al. 2015, NEJM | [ ] permission (shared with ch11) |
| 12 | `gtex_egenes.pdf` | GTEx Consortium 2020, Science | [ ] permission |
| 12 | `mash_tissue_correlation.png` | Urbut et al. 2019, Nat Genet | [ ] permission |
| 12 | `gaffney_annotation_enrichment.png` | Gaffney et al. 2012, Genome Biol | [ ] attribution, confirm CC BY |
| 12 | `twas_parkinsons.png` | Li, Knowles et al. 2018, Nat Genet | [ ] permission |
| 13 | `linsight_fitness`, `linsight_hgmd_roc` | Huang et al. 2017, Nat Genet | [ ] permission |
| 13 | `asd_disease_impact`, `asd_luciferase` | Zhou et al. 2019, Nat Genet | [ ] permission |
| 14 | `alphafold2_architecture.png` | Jumper et al. 2021, Nature | [ ] attribution, confirm CC BY 4.0 |

Own figures, no permission needed but confirm:
- [ ] `chapter01/figures/slides/sequencing_cost.pdf`: own R plot of public-domain NHGRI data.
- [ ] `chapter05/figures/slides/als_pca.pdf`: own analysis; confirm data-use terms.
- [ ] `chapter10/figures/slides/spliceosome.jpg`: drawn by the instructor; confirm.

## 4. Incomplete references

All bib entries were checked against PubMed, DBLP, Crossref and bioRxiv (October 2026); every `note = {check}` is resolved. Remaining items:

- [x] **Ch 2–13:** all `note = {check}` entries resolved (rizzino2016, avsec2026, guerzhoy_video, nie2018, gao2024, grabski2023, ch 7 author lists, mackay1994, ch 9 pages and article numbers, desterro2020, piovesan2016, ch 12 entries).
- [x] **Ch 13:** "and others" author prefixes, volumes and pages verified.
- [ ] **Ch 12:** `fan2020` is now the ECCO paper (Fan, Zhu, Song, Peng, Zhou, Bioinformatics 37:296--302, 2021). Confirm it is the slide's source and that it supports "eGenes roughly double from 0 to 60 PEER factors" (text line ~317); the full text was not available to check.
- [ ] **Ch 10:** `saberi2024` cites bioRxiv v1; later versions are retitled with 14 authors. No journal version yet.
- [ ] **Unverified in any database:** `mackay1994` (ASHRAE Trans.), `blelloch1990` (CMU tech report), `ghahramani1996`, `minka2005`; pages of `vaswani2017` (5998--6008) and `zaheer2017` (3391--3401) are from the standard citation, not DBLP.
- [ ] **No year (biber warning):** web resources `wetterstrand`, `tenx_blog` (ch 1), `seurat_pbmc` (ch 5).
- [ ] **arXiv only:** `gu2023` (Mamba; published at COLM 2024, not in DBLP), `chung2014`, `feng2024`, `botev2024`, `hendrycks2016`, `dumoulin2016`, `ba2016`, `ho2019`.
- [ ] **Uncited entries:** `murphy2022` (ch 3), `holmes2017` (ch 12), `bishop1999bpca` (ch 12); harmless, as unused entries do not print.
- [ ] **Uncited mentions:** Otari (ch 10) and AlphaGenome (ch 2) are named in the text without a citation.
