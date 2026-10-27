# Machine Learning for Functional Genomics

A textbook draft based on the ML4FG course (COMS 4762, Columbia). Read it at https://davidaknowles.github.io/ml4fg_book/, which has the full book as a PDF and each chapter as HTML and PDF.

## Layout

- `chapterNN/`: one chapter, with `chapterNN.tex`, `chapterNN.bib`, a standalone `main.tex`, `figures/` and `scripts/make_figures.py`, which generates the computed figures.
- `book/main.tex`: builds all chapters into one PDF.
- `tools/`: utilities for reading the lecture slides and building the website.
- `LABNOTEBOOK.md`: conventions and implementation notes. `CHECKLIST.md`: open items before publication.

## Building

- One chapter: `latexmk -pdf main.tex` in its folder.
- The book: `latexmk -pdf main.tex` in `book/`.
- The website: `tools/build_site.sh _site` builds the book PDF, chapter PDFs, chapter HTML (converted from the LaTeX with [LaTeXML](https://math.nist.gov/~BMiller/LaTeXML/)) and a landing page. A GitHub Actions workflow runs it on every push to `main` and deploys the result to GitHub Pages.

The LaTeX is the only source; the HTML is generated from it.
