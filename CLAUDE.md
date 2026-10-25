# ML4FG book

Converting ML4FG lecture decks (`*.pptx`, untracked) into LaTeX chapters (`chapterNN/`). See `LABNOTEBOOK.md` for conventions and `chapter01/` as the template.

## Extracting content from a pptx

- **Most content is handwritten ink, not text.** Many slides have only a title. Always render the slides and read the images; don't rely on text extraction alone.
- **Render slides with ink:** `~/venv/torch/bin/python tools/render_slides.py deck.pptx out/` writes `out/sNN.png` in presentation order. Write `out/` to the scratchpad, not the repo. To save context, stack two slides per image with `magick a.png b.png -resize 1100x -append pair.png`; the handwriting is still legible.
- **Don't use the ink fallback PNGs** (`mc:Fallback`): they are too low-resolution to read. `tools/inkml.py` decodes the InkML strokes directly.
- **Ink coordinates differ by year.** 2024 decks use one ink part in absolute slide coordinates; 2023 decks use one part per stroke with relative coordinates. The renderer handles both. If a render shows ink piled in the top-left corner, the coordinate mapping is what broke.
- **Text and speaker notes:** use the pptx skill's `slide_text.py`. It needs python-pptx, which isn't in `~/venv/torch`, so use a uv venv in the scratchpad. Speaker notes often hold the prose explanation; use it.
- **Media:** unzip the pptx; images are in `ppt/media/`. Map a picture to its slide via `ppt/slides/_rels/slideN.xml.rels`. Slide XML numbering does not match presentation order; take the order from `p:sldIdLst` in `ppt/presentation.xml`. One figure can be split into several images inside a group; composite them using their `a:off`/`a:ext` positions.
- **EMF pictures** often wrap a PDF: `tools/emf_to_pdf.py in.emf out.pdf`. ImageMagick can't read EMF.
- **Don't drive PowerPoint via AppleScript** to export a PDF: it hangs on a file-access prompt. LibreOffice is not installed.
- When two versions of a deck exist, compare them. Older decks may have fuller derivations, newer ones updated content.

## Turning slides into a chapter

- Handwritten math or prose becomes text and equations; schematics become TikZ; plots of functions or data become computed plotnine figures in `chapterNN/scripts/make_figures.py`, run with `~/venv/torch/bin/python`.
- After regenerating figures, render them and check them visually. Check that each figure actually shows what its caption claims.
- Third-party figures go in `figures/slides/`, credited in the caption and listed with a permission flag in the chapter README.
- Build with `latexmk -pdf main.tex` in the chapter folder, then `latexmk -c`. Commit and push to origin (private repo `ml4fg_book`).
