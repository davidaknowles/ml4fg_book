#!/usr/bin/env bash
# Build the website: the book PDF, a PDF and an HTML page per chapter, and a landing page.
# The LaTeX sources are the only input; HTML is converted from each chapter's standalone main.tex by LaTeXML.
# Usage: tools/build_site.sh [output_dir] [--html-only|--pdf-only]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/_site}"
MODE="${2:-all}"
AR5IV="https://cdn.jsdelivr.net/gh/dginev/ar5iv-css@0.8.4/css"
mkdir -p "$OUT/pdf"
OUT="$(cd "$OUT" && pwd)"  # absolute, since each chapter is built from inside its folder
failed=()

if [[ "$MODE" != "--html-only" ]]; then
  (cd "$ROOT/book" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /dev/null) \
    && cp "$ROOT/book/main.pdf" "$OUT/pdf/ml4fg_book.pdf" || failed+=("book pdf")
  for dir in "$ROOT"/chapter??; do
    ch="$(basename "$dir")"
    (cd "$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /dev/null) \
      && cp "$dir/main.pdf" "$OUT/pdf/$ch.pdf" || failed+=("$ch pdf")
  done
fi

if [[ "$MODE" != "--pdf-only" ]]; then
  for dir in "$ROOT"/chapter??; do
    ch="$(basename "$dir")"
    # TeX renders each tikzpicture to tikzext/main-figureN.pdf (TikZ external library, switched on by
    # the flag file in main.tex); these become SVGs drawn at 150% so their labels match the web text size
    # (pdftocairo gives sizes in points, with or without the unit depending on its version).
    # tools/latexml/tikz.sty.ltxml puts them in place of the pictures.
    (cd "$dir" && mkdir -p tikzext && touch tikz-external.flag \
      && pdflatex -shell-escape -interaction=nonstopmode main.tex > /dev/null 2>&1; rm -f "${dir:?}/tikz-external.flag")
    for pdf in "$dir"/tikzext/main-figure*.pdf; do
      [[ -e "$pdf" ]] || continue
      pdftocairo -svg "$pdf" "${pdf%.pdf}.svg" \
        && perl -pi -e 's/(width|height)="([\d.]+)(pt)?"/sprintf(q{%s="%.3fpt"}, $1, 1.5 * $2)/ge if /^<svg /' "${pdf%.pdf}.svg"
    done
    (cd "$dir" && latexmlc --path="$ROOT/tools/latexml" --format=html5 \
        --css="$AR5IV/ar5iv.min.css" --css="$AR5IV/ar5iv-fonts.min.css" --css="$ROOT/tools/latexml/ml4fg.css" \
        --log="$OUT/$ch/latexml.log" --dest="$OUT/$ch/index.html" main.tex > /dev/null 2>&1) \
      || failed+=("$ch html")
    # link back to the landing page and to this chapter's PDF
    CH="$ch" perl -pi -e 's|<div class="ltx_page_content">|$&<div class="ml4fg_home"><a href="../">&larr; All chapters</a> &middot; <a href="../pdf/$ENV{CH}.pdf">PDF</a></div>|' \
      "$OUT/$ch/index.html"
    # TikZ SVGs go straight into the page, which browsers draw more smoothly than an embedded <object>
    python3 "$ROOT/tools/inline_svgs.py" "$OUT/$ch" > /dev/null || failed+=("$ch inline svg")
    rm -f "${OUT:?}/$ch/LaTeXML.cache"
  done
fi

python3 "$ROOT/tools/make_index.py" "$ROOT" "$OUT" || failed+=("index")

if (( ${#failed[@]} )); then
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
echo "Site written to $OUT"
