"""Write the website landing page: links to the book PDF and to each chapter's HTML and PDF.

Chapter titles are read from the \\chapter{...} line of each chapterNN/chapterNN.tex.
Usage: python3 tools/make_index.py <repo_root> <site_dir>
"""
import datetime
import html
import re
import subprocess
import sys
from pathlib import Path


def chapter_title(tex):
    m = re.search(r"\\chapter\{(.*)\}", tex.read_text())
    return re.sub(r"\\[a-zA-Z]+\s*", "", m.group(1).replace("~", " ")) if m else tex.stem


def main(root, site):
    rows = []
    for d in sorted(root.glob("chapter??")):
        num = int(d.name[-2:])
        links = []
        if (site / d.name / "index.html").exists():
            links.append(f'<a href="{d.name}/">HTML</a>')
        if (site / "pdf" / f"{d.name}.pdf").exists():
            links.append(f'<a href="pdf/{d.name}.pdf">PDF</a>')
        rows.append(f'<li><span class="num">{num}</span><span class="title">{html.escape(chapter_title(d / f"{d.name}.tex"))}</span>'
                    f'<span class="links">{" · ".join(links)}</span></li>')
    book = '<p class="book"><a href="pdf/ml4fg_book.pdf">Full book (PDF)</a></p>' if (site / "pdf/ml4fg_book.pdf").exists() else ""
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    stamp = f"Built {datetime.date.today().isoformat()}" + (f" from commit {commit}" if commit else "")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Machine Learning for Functional Genomics</title>
<style>
  :root {{ --fg: #1d1d1f; --muted: #6e6e73; --rule: #d2d2d7; --link: #1b4f8a; --bg: #ffffff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg: #f2f2f2; --muted: #a1a1a6; --rule: #3a3a3c; --link: #8ab4f8; --bg: #161618; }}
  }}
  body {{ background: var(--bg); color: var(--fg); font: 17px/1.5 Georgia, "Times New Roman", serif;
         max-width: 46rem; margin: 3rem auto; padding: 0 16px; }}
  h1 {{ font-size: 2rem; line-height: 1.2; margin-bottom: 0.2rem; }}
  .author {{ color: var(--muted); margin-top: 0; }}
  a {{ color: var(--link); }}
  .book {{ font-size: 1.1rem; margin: 1.5rem 0; }}
  ol {{ list-style: none; padding: 0; border-top: 1px solid var(--rule); }}
  li {{ display: flex; gap: 0.8rem; align-items: baseline; padding: 0.55rem 0; border-bottom: 1px solid var(--rule); }}
  .num {{ color: var(--muted); min-width: 1.6rem; text-align: right; }}
  .title {{ flex: 1; }}
  .links {{ white-space: nowrap; font-size: 0.95rem; }}
  footer {{ color: var(--muted); font-size: 0.85rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>Machine Learning for Functional Genomics</h1>
<p class="author">David A. Knowles · draft</p>
{book}
<ol>
{chr(10).join(rows)}
</ol>
<footer>{stamp}. The HTML chapters are converted from the LaTeX source with LaTeXML; the PDFs are the reference version.</footer>
</body>
</html>
"""
    (site / "index.html").write_text(page)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
