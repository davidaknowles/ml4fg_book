# ML4FG book: lab notebook

Converting the ML4FG (machine learning for functional genomics) lecture slides into LaTeX book chapters.

## Layout

- `chapterNN/`: one folder per chapter, with `chapterNN.tex`, `chapterNN.bib`, a standalone `main.tex` wrapper, `figures/` and `scripts/make_figures.py`.
- `figures/slides/` in each chapter holds figures taken from the slides; everything else in `figures/` is generated.
- `tools/`: utilities for reading slide decks.
  - `render_slides.py deck.pptx out/` renders each slide to PNG including handwritten ink.
  - `emf_to_pdf.py` recovers a PDF embedded in an EMF picture.
- Source `.pptx` decks are kept alongside but not tracked in git (large binaries).

## Conventions

- Notation: bold for vectors and matrices (`\vy`, `\mX`, `\vb` macros, defined with `\providecommand` at the top of each chapter), $\vx_i\in\R^P$ is a column vector so row $i$ of $\mX$ is $\vx_i^\top$. Dimensions are stated when variables are introduced.
- Lead-ins to display equations use no colon; no blank line before equation environments.
- Handwritten slide content becomes text or equations where it is math or prose, TikZ where it is a schematic, and a computed plotnine figure where it is a plot of a function or of data.
- Third-party figures are reproduced with a citation in the caption and listed in the chapter README with a permission flag.

## Slide extraction notes

- PowerPoint pen annotations are stored as InkML (`ppt/ink/inkN.xml`) inside `p:contentPart` elements, with a low-resolution PNG fallback that is too coarse to read. The traces use Microsoft's compressed InkML encoding: per-channel prefixes `!` (explicit), `'` (first difference) and `"` (second difference), persistent until changed. Units are 1/1000 cm (360 EMU).
- The 2024 deck stores ink in absolute slide coordinates in a single part; the 2023 deck stores one part per stroke with coordinates relative to the part. `render_slides.py` handles both by mapping each part's stroke bounding box onto its `p14:xfrm` box.
- Slide XML file numbers do not match presentation order; order comes from `p:sldIdLst` in `ppt/presentation.xml`.

## Chapter 1: molecular biology basics and linear models

- Sources: `ML4FG topic 01 regression and introbio.pptx` (2023, fuller derivations) and `... 2.pptx` (2024, newer content and ordering). The chapter follows the 2024 structure and uses the 2023 derivations of the likelihood, OLS and MAP/ridge.
- Added beyond the slides: closed-form logistic and softmax gradients, soft-thresholding derivation, empirical Bayes marginal likelihood, ridge posterior covariance, exercises.
- Computed figures use simulated data with fixed seeds. The bias-variance figure is exact for a fixed design ($N=80$, $P=60$), not Monte Carlo. The lasso risk curve shows the Greenshtein-Ritov rate without constants.
- Open items: permissions for third-party figures (see `chapter01/README.md`); source of the primary-culture figure; check Oncotype DX cohort description and the bootstrap-frequency interpretation of the CCLE elastic-net figure.

## Chapters 2-14 and the full book

- Each chapter was written by an agent following `tools/CHAPTER_BRIEF.md`, from the 2026 deck (content and order), the 2023 deck (fuller derivations, topics 2-11) and the matching Fall 2025 lecture transcripts in `captions/`, which are kept locally and not in the repository because they contain student speech. Topics 5 (10/10 lecture uncaptioned) and 13 rely mostly on the slides. Each `chapterNN/README.md` lists figure provenance and permissions; each agent's report of facts to check is summarized in the chapter README or PROGRESS.md.
- Chapter 1 was later revised from the 2026 deck (new precision-medicine section) and the lectures. The CCLE elastic-net heatmap runs sensitive (left) to resistant (right): its activity-area axis is inverted, and the ink on the 2024 and 2026 slides labels it the wrong way round.
- The full book is `book/main.tex`, built with `latexmk -pdf main.tex`. `\bookchapter{chapterNN}` sets that chapter's graphics and input paths, reads it inside a TeX group so its `\providecommand` macros stay local (e.g. `\va` is alpha in chapter 2 and **a** elsewhere), and prints its bibliography. The book uses biblatex with `refsection=chapter` over all chapter `.bib` files; standalone builds keep natbib/bibtex. An earlier chapterbib version needed a wrapper file per chapter and was dropped. Biber writes accents as combining Unicode, so the book loads `inputenc` explicitly and passes `safeinputenc` to biblatex.
- Labels and bib keys must be unique across chapters. Duplicate labels were renamed with a `-NN` suffix in the later chapter; two bib keys that named different works were renamed (`hochreiter1997lstm` in ch 7, `liu2009chisq` in ch 13). The other 30 shared keys name the same work.
- Each chapter prints its own contents list (etoc `\localtableofcontents`, via `\chaptertoc`) after the chapter heading; the book's main contents lists sections only.
- Standalone chapter builds set `\setcounter{chapter}{N-1}` in `main.tex`; the book build does not use those wrappers.

## Slide extraction notes (additions)

- `render_slides.py` does not draw cropped pictures (`srcRect`), tables, autoshapes/connectors or some grouped shapes; node-link diagrams and tables must be read from `slide_text.py` or the slide XML and redrawn. Some 2026 decks (e.g. topic 14) have no ink and use autoshapes throughout.
- PNGs in a slide's rels can be ink fallback images, not figures.
- EMF-wrapped PDFs carry a CropBox for the pasted panel: rasterize with `pdftoppm -cropbox` (or `pdftocairo -png -transp -cropbox` for white-on-transparent equation images) so whole journal pages are not shipped. ImageMagick here cannot read PDF or TIFF; use `pdftoppm`, and `sips -s format png` or PIL for TIFF.
- Speaker notes in some decks (topic 14) are auto-generated and unreliable.

## LaTeX notes

- Use `[!htbp]` floats; with many `[t]` floats one large float pushes all figures to the chapter end.
- `\protect` fragile commands such as `\overrightarrow` in captions; brace `\foreach` bodies that are macros expanding to `\draw`.
- In bibtex, "and others" must be the last author or abbrvnat prints it as a name.
- Generated tables must be complete `tabular` environments; `\input` of row fragments fails when a row starts with `\multicolumn`. Chapter 9 writes numbers quoted in the text to a macro file from its figure script, which keeps text and figures in sync.
- Code shown in the text uses the `listings` styles `pycode` (code) and `pyout` (its printed output), defined in both `book/main.tex` and the chapter preamble. Outputs are pasted from a run of exactly the printed code, with a fixed seed where weights are random (chapter 2 worked CNN example).

## Website

- `tools/build_site.sh` builds the GitHub Pages site: the book PDF, each chapter's PDF, each chapter's HTML and a landing page (`tools/make_index.py`). `.github/workflows/pages.yml` runs it on every push to `main` with Ubuntu's TeX Live and LaTeXML packages and deploys to Pages.
- HTML is converted from each chapter's standalone `main.tex` by LaTeXML, as arXiv does for its HTML papers, and styled with the ar5iv CSS. Math becomes MathML, PDF figures are rasterized by ImageMagick (the workflow lifts ImageMagick's PDF policy), and natbib citations link to the bibliography read from the chapter `.bib`.
- TikZ figures are rendered by TeX, not by LaTeXML, whose TikZ interpretation looked poor. With the flag file `tikz-external.flag` present, each chapter `main.tex` loads the TikZ external library, so one `pdflatex -shell-escape` pass writes every picture to `tikzext/main-figureN.pdf` in its real context. `pdftocairo -svg` converts these, drawn at 125% so their labels match the web text size, and `tools/latexml/tikz.sty.ltxml` replaces the Nth `tikzpicture` with the Nth SVG. This needs each `\begin{tikzpicture}` and `\end{tikzpicture}` on its own line and no picture inside a macro argument. Without the flag file the PDF builds are unchanged.
- LaTeXML has no etoc support; `tools/latexml/etoc.sty.ltxml` turns the per-chapter contents list into nothing in HTML. Bindings for any other unsupported package go in the same folder, so the `.tex` stays unchanged.
- The lecture transcripts in `captions/` contain student speech, so they were purged from the git history (git filter-repo) before the repository was made public, and the folder is ignored.

## Reusable code

- `chapter04/scripts/attribution.py`: attribution methods (ISM, gradients, DeepLIFT, integrated gradients).
- `chapter06/scripts/hmm.py`: scaled forward-backward, Viterbi, Baum-Welch.
- `chapter12/scripts/make_figures.py` (`mosaic_genotypes`) and `chapter11/scripts/make_figures.py`: genotype simulators with realistic block LD (founders built from coalescent trees).
