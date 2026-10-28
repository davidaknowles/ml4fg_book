"""Add a contents list of sections and subsections to a chapter's HTML page.

It is a sidebar on wide screens and a collapsed box at the top of the page on narrow ones (see
tools/latexml/ml4fg.css); a short script opens it to match the screen width and marks the section
being read. Usage: python3 tools/add_toc.py <chapter_site_dir>
"""
import re
import sys
from pathlib import Path

SECTION = re.compile(
    r'<section id="([^"]+)" class="ltx_(section|subsection|bibliography)"[^>]*>\s*'
    r'<h\d class="ltx_title[^"]*">(.*?)</h\d>', re.S)
TAG = re.compile(r'\s*<span class="ltx_tag[^"]*">(.*?)</span>', re.S)
HOME = re.compile(r'<div class="ml4fg_home">.*?</div>', re.S)

SCRIPT = """<script>
(function () {
  var toc = document.getElementById('ml4fg-toc'), wide = matchMedia('(min-width: 72rem)');
  var links = Array.prototype.slice.call(toc.querySelectorAll('a'));
  var heads = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });
  function fit() { toc.open = wide.matches; }
  fit();
  wide.addEventListener('change', fit);
  // on narrow screens, close the box before jumping so that the target stays in place
  toc.addEventListener('click', function (e) { if (!wide.matches && e.target.closest('a')) toc.open = false; });
  var current = null, pending = false;
  function mark() {
    pending = false;
    var i = heads.length - 1;
    while (i > 0 && heads[i].getBoundingClientRect().top > 120) i--;
    if (links[i] === current) return;
    if (current) current.classList.remove('ml4fg_current');
    current = links[i];
    current.classList.add('ml4fg_current');
    if (wide.matches) {  // keep the marked entry visible in the sidebar
      var nav = toc.parentNode, r = current.getBoundingClientRect(), n = nav.getBoundingClientRect();
      if (r.top < n.top || r.bottom > n.bottom) nav.scrollTop += r.top - n.top - n.height / 3;
    }
  }
  addEventListener('scroll', function () { if (!pending) { pending = true; requestAnimationFrame(mark); } });
  mark();
})();
</script>"""


def entry(sec_id, title_html):
    title_html = re.sub(r'\s+id="[^"]*"', "", title_html)  # copies of the title's maths must not repeat ids
    tag = TAG.match(title_html)
    num = f'<span class="ml4fg_toc_num">{tag.group(1).strip()}</span> ' if tag else ""
    title = title_html[tag.end():] if tag else title_html
    return f'<a href="#{sec_id}">{num}{title.strip()}</a>'


def toc_html(html):
    items = []  # (entry, [subsection entries])
    for sec_id, kind, title in SECTION.findall(html):
        if kind == "subsection" and items:
            items[-1][1].append(entry(sec_id, title))
        elif kind != "subsection":
            items.append((entry(sec_id, title), []))
    lines = ['<nav class="ml4fg_toc" aria-label="Contents"><details id="ml4fg-toc"><summary>Contents</summary><ol>']
    for top, subs in items:
        sub = "<ol>" + "".join(f"<li>{s}</li>" for s in subs) + "</ol>" if subs else ""
        lines.append(f"<li>{top}{sub}</li>")
    lines.append("</ol></details></nav>")
    return "\n".join(lines), len(items)


def main(chapter_dir):
    page = chapter_dir / "index.html"
    html = page.read_text()
    toc, n = toc_html(html)
    home = HOME.search(html)
    html = html[:home.end()] + toc + html[home.end():]
    html = html.replace("</body>", SCRIPT + "\n</body>", 1)
    page.write_text(html)
    print(f"{chapter_dir.name}: contents with {n} sections")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
