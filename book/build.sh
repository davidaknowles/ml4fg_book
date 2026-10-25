#!/bin/bash
# Build the full book: pdflatex, bibtex per chapter, then pdflatex until references settle.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
for a in ch??.aux; do bibtex "${a%.aux}" > /dev/null; done
for i in 1 2 3 4; do
  pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
  grep -q "Rerun to get" main.log || break
done
grep -E "undefined|multiply defined|Rerun to get" main.log | sort -u || true
echo "built main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
