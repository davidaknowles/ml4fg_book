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
    (cd "$dir" && latexmlc --path="$ROOT/tools/latexml" --format=html5 \
        --css="$AR5IV/ar5iv.min.css" --css="$AR5IV/ar5iv-fonts.min.css" \
        --log="$OUT/$ch/latexml.log" --dest="$OUT/$ch/index.html" main.tex > /dev/null 2>&1) \
      || failed+=("$ch html")
    rm -f "$dir/LaTeXML.cache"
  done
fi

python3 "$ROOT/tools/make_index.py" "$ROOT" "$OUT" || failed+=("index")

if (( ${#failed[@]} )); then
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
echo "Site written to $OUT"
