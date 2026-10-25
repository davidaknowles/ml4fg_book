# Brief: turning one ML4FG lecture topic into a book chapter

You are writing one chapter of a LaTeX textbook, *Machine Learning for Functional Genomics*, converted from lecture slides and lecture transcripts. Chapter 1 (`chapter01/`) is finished and is the template: match its structure, depth, tone and conventions. Your task message gives the topic number, the slide decks, and which lectures in the transcript cover it.

Repo root: `/Users/daknowles/Dropbox/teaching/2023 slides final/for_book` (git repo; private GitHub `ml4fg_book`).

## Read first

1. `CLAUDE.md` (how to extract content from pptx) and `LABNOTEBOOK.md` (conventions).
2. `chapter01/chapter01.tex`, `chapter01/main.tex`, `chapter01/scripts/make_figures.py`, `chapter01/README.md`. Skim the compiled `chapter01/main.pdf` if present, or build it.

## Sources

- **2026 deck** (`../../2026 finished slides/`): the most up to date. It sets the chapter's content and ordering.
- **2023 deck** (`../` i.e. `2023 slides final/`), when one exists for your topic: often has fuller handwritten derivations. Use its math where the 2026 deck is thinner.
- **Lecture transcripts**: `captions/COMS4762_2025_lectures.md` (Fall 2025, one `# COMS4762 lecture, DATE` section per lecture, `##` topic headings). Read the lectures named in your task. The lecture order follows the slide order, so match transcript passages to slides by topic. Use the transcripts for explanations, intuition, motivation, examples and clarifying student questions that the slides lack. They are LLM-edited auto-captions: `captions/NOTES.md` lists known reconstructions and possible errors. Check technical claims; never copy errors into the book. Write in book prose, not as a transcript.
- Speaker notes in the decks often contain prose explanations; use them.

## Tools

- Render slides including handwritten ink: `~/venv/torch/bin/python tools/render_slides.py "<deck.pptx>" <scratch>/<name>_r/`. Read every slide image. Most math and diagrams are handwritten ink; text extraction alone misses most of the content. To save context, stack two slides per image: `magick a.png b.png -resize 1100x -append pair.png`.
- Slide text and speaker notes: `~/venv/torch/bin/python "/Users/daknowles/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/1afddcdf-29a1-4cf3-a5a6-c90101c4dd8f/9bd56997-5ada-49eb-9c14-ef11da5a69f1/skills/pptx/scripts/slide_text.py" "<deck.pptx>"`.
- Embedded figures: unzip the deck to scratch; images are in `ppt/media/`, mapped to slides via `ppt/slides/_rels/slideN.xml.rels`. Slide XML numbering differs from presentation order (see `p:sldIdLst`). EMF pictures often wrap a PDF: `tools/emf_to_pdf.py`.
- Python: `~/venv/torch/bin/python` (numpy, scipy, pandas, sklearn, plotnine, contourpy, lxml, PIL, python-pptx).
- Scratch space: use the directory given in your task (outside the repo). Never put renders or unzipped decks in the repo.

## Output: `chapterNN/` (NN = two-digit topic number)

Mirror `chapter01/`:
- `chapterNN.tex`: the chapter (`\chapter{...}`, then `\label{ch:<slug>}`, then `\chaptertoc` on its own lines). Labels and bib keys must not clash with other chapters' unless they name the same thing. Copy chapter 1's `\providecommand` macro block at the top and add any new macros the same way.
- `chapterNN.bib`: references cited in the chapter (natbib keys, `\citet`/`\citep`). Cite primary papers for methods and datasets. Only include bibliographic details you are confident of; mark any uncertain field with a `note = {check}`.
- `main.tex`: copy of chapter 1's wrapper with the include and bibliography names changed. Add packages only if needed, and list them in the chapter's header comment.
- `figures/` and `figures/slides/`, `scripts/make_figures.py`, `README.md` (build steps and a figure provenance table with a permission column, like chapter 1's).

## Writing conventions

- Book chapter, not slide notes. Explain from the ground up for a reader who has read the earlier chapters; refer back to them where helpful ("Chapter~\ref{...}" will not resolve across separate builds, so write "Chapter 2" etc. in plain text).
- Turn every handwritten derivation into full math with all steps. Define every variable and its dimension when introduced. Bold vectors/matrices via macros, as in chapter 1.
- No colon before a display equation; no blank line before an equation environment. Avoid excessive adverbs and adjectives. No machine specifics (paths, commands) in the .tex or README prose beyond the build commands.
- Handwritten content: math and prose become text and equations; schematics become TikZ; plots of functions or data become computed plotnine figures (real computations, fixed seeds, `metadata={"CreationDate": None}` on save, as in chapter 1's `save()`).
- Published figures from slides: put them in `figures/slides/`, credit the source in the caption, and list them in the README table with "permission needed". Drop screenshots of paper titles, logos, book covers, polls and similar.
- Add a short summary section and a few exercises at the end, as in chapter 1.
- Do not reproduce long passages from papers, books or videos; describe and cite.

## Verify

- Build: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` in the chapter folder must succeed with no undefined references or citations. Then `latexmk -c`.
- Render the PDF pages (`pdftoppm -r 60`) and look at every page with a figure. Fix overlapping labels, clipped TikZ, unreadable text, and wrong facet orders.
- Check each figure actually shows what its caption and the text claim (e.g. a lasso solution really at a corner, a curve really U-shaped).
- Re-read the math for consistency of notation and dimensions.

## Rules

- Only create and edit files inside your own `chapterNN/` folder (and your scratch directory). Do not edit other chapters, `tools/`, `captions/`, `CLAUDE.md`, `LABNOTEBOOK.md` or `.gitignore`. Do not run git commands that change state (no add/commit/push); the coordinator commits.
- Keep `chapterNN/PROGRESS.md` up to date as you go: which slides are read, which sections are written, which figures are done, what is left. If `PROGRESS.md` already exists when you start, a previous attempt crashed: read it and resume rather than starting over.
- Write the chapter incrementally to disk (section by section), so work survives a crash.

## Final report

Reply with: the chapter title and section list; page count; figures (computed / TikZ / from slides); facts or reconstructions the instructor should check; anything from the slides you left out and why; notes worth adding to `LABNOTEBOOK.md`.
