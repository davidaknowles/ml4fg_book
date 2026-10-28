"""Put each TikZ SVG straight into a chapter's HTML page in place of the <object> LaTeXML writes for it.

Browsers draw an SVG <object> as a separate embedded document, which can lose anti-aliasing when the
page scales it; inline SVG is drawn with the page. pdftocairo reuses ids such as glyph-0-0 in every
figure, so each figure's ids are prefixed to keep them unique on the page.
Usage: python3 tools/inline_svgs.py <chapter_site_dir>
"""
import re
import sys
from pathlib import Path

OBJECT = re.compile(r'<object type="image/svg\+xml" data="(tikzext/main-figure(\d+)\.svg)"([^>]*)></object>')


def inline_svg(svg_text, prefix, attrs):
    svg = svg_text[svg_text.index("<svg"):]
    svg = re.sub(r'\bid="([^"]+)"', rf'id="{prefix}\1"', svg)
    svg = re.sub(r'href="#([^"]+)"', rf'href="#{prefix}\1"', svg)
    svg = re.sub(r"url\(#([^)]+)\)", rf"url(#{prefix}\1)", svg)
    # keep LaTeXML's id and classes (used for cross-references and centring) on the root element
    lx_id = re.search(r'\bid="([^"]+)"', attrs)
    lx_class = re.search(r'\bclass="([^"]+)"', attrs)
    root = '<svg role="img"'
    if lx_id:
        root += f' id="{lx_id.group(1)}"'
    root += f' class="{lx_class.group(1) if lx_class else "ltx_graphics"} ml4fg_tikz"'
    return svg.replace("<svg", root, 1)


def main(chapter_dir):
    page = chapter_dir / "index.html"
    html = page.read_text()

    def replace(m):
        path, num, attrs = m.group(1), m.group(2), m.group(3)
        return inline_svg((chapter_dir / path).read_text(), f"f{num}-", attrs)

    html, n = OBJECT.subn(replace, html)
    page.write_text(html)
    print(f"{chapter_dir.name}: inlined {n} SVGs")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
