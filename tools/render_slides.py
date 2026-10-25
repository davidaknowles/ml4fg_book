"""Render pptx slides to PNG, including handwritten ink, for reading slide content.

PowerPoint stores pen annotations as InkML; this draws the strokes at full resolution
together with pictures and (roughly placed) text boxes. Layout is approximate.

Usage:
    python render_slides.py deck.pptx out_dir
Writes out_dir/sNN.png in presentation order.
"""
import sys, os, re
from lxml import etree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkml import parse_ink
from PIL import Image, ImageDraw, ImageFont

import tempfile, zipfile
root = tempfile.mkdtemp()
zipfile.ZipFile(sys.argv[1]).extractall(root)
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
W = 2000
SX, SY = 12192000, 6858000
sc = W / SX
H = int(SY * sc)
ns = {
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
}
P = '{%s}' % ns['p']; A = '{%s}' % ns['a']; R = '{%s}' % ns['r']; MC = '{%s}' % ns['mc']
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 18)
tfont = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 30)


def rels(n):
    t = etree.parse(f'{root}/ppt/slides/_rels/slide{n}.xml.rels')
    return {e.get('Id'): e.get('Target') for e in t.getroot()}


def xfrm(el):
    x = el.find('.//' + A + 'xfrm')
    if x is None:
        return None
    off, ext = x.find(A + 'off'), x.find(A + 'ext')
    if off is None or ext is None:
        return None
    return [int(off.get('x')), int(off.get('y')), int(ext.get('cx')), int(ext.get('cy'))]


def walk(parent, tf, img, draw, rl, n):
    for el in parent:
        tag = el.tag
        if tag == MC + 'AlternateContent':
            ch = el.find(MC + 'Choice')
            cp = ch.find(P + 'contentPart') if ch is not None else None
            if cp is not None:
                tgt = rl.get(cp.get(R + 'id'))
                path = os.path.normpath(f'{root}/ppt/slides/{tgt}')
                strokes = parse_ink(path)
                allp = [p for st in strokes for p in st[0]]
                if not allp:
                    continue
                px = cp.find("{http://schemas.microsoft.com/office/powerpoint/2010/main}xfrm"); o_, e_ = px.find(A + "off"), px.find(A + "ext"); bx = [int(o_.get("x")), int(o_.get("y")), int(e_.get("cx")), int(e_.get("cy"))]
                mnx = min(p[0] for p in allp); mxx = max(p[0] for p in allp)
                mny = min(p[1] for p in allp); mxy = max(p[1] for p in allp)
                wpad = max(st[1] for st in strokes) * 1000
                sx = bx[2] / ((mxx - mnx + wpad) * 360)
                sy = bx[3] / ((mxy - mny + wpad) * 360)
                ox = bx[0] + (bx[2] - (mxx - mnx) * 360 * sx) / 2
                oy = bx[1] + (bx[3] - (mxy - mny) * 360 * sy) / 2
                for pts, wcm, col, tr in strokes:
                    if tr > 0:
                        continue  # skip highlighter for legibility
                    xy = []
                    for x, y in pts:
                        b = tf([ox + (x - mnx) * 360 * sx, oy + (y - mny) * 360 * sy, 0, 0])
                        xy.append((b[0] * sc, b[1] * sc))
                    lw = max(2, int(wcm * 360000 * sc))
                    if len(xy) == 1:
                        xy = xy * 2
                    draw.line(xy, fill=col, width=lw, joint='curve')
                continue
            fb = el.find(MC + 'Fallback')
            if fb is not None:
                walk(fb, tf, img, draw, rl, n)
        elif tag == P + 'grpSp':
            g = el.find(P + 'grpSpPr').find(A + 'xfrm')
            off, ext = g.find(A + 'off'), g.find(A + 'ext')
            choff, chext = g.find(A + 'chOff'), g.find(A + 'chExt')
            ox, oy = int(off.get('x')), int(off.get('y'))
            ex, ey = int(ext.get('cx')), int(ext.get('cy'))
            cx, cy = int(choff.get('x')), int(choff.get('y'))
            cex, cey = max(int(chext.get('cx')), 1), max(int(chext.get('cy')), 1)
            sxg, syg = ex / cex, ey / cey
            def ntf(b, tf=tf, ox=ox, oy=oy, cx=cx, cy=cy, sxg=sxg, syg=syg):
                return tf([ox + (b[0] - cx) * sxg, oy + (b[1] - cy) * syg, b[2] * sxg, b[3] * syg])
            walk(el, ntf, img, draw, rl, n)
        elif tag == P + 'pic':
            b = xfrm(el.find(P + 'spPr'))
            blip = el.find('.//' + A + 'blip')
            if b is None or blip is None:
                continue
            tgt = rl.get(blip.get(R + 'embed'))
            if not tgt:
                continue
            path = os.path.normpath(f'{root}/ppt/slides/{tgt}')
            b = tf(b)
            try:
                im = Image.open(path).convert('RGBA')
                w, h = max(1, int(b[2] * sc)), max(1, int(b[3] * sc))
                im = im.resize((w, h))
                img.alpha_composite(im, (int(b[0] * sc), int(b[1] * sc))) if 0 <= int(b[0]*sc) and 0 <= int(b[1]*sc) else img.paste(im, (int(b[0]*sc), int(b[1]*sc)), im)
            except Exception as e:
                draw.rectangle([b[0]*sc, b[1]*sc, (b[0]+b[2])*sc, (b[1]+b[3])*sc], outline='red')
                draw.text((b[0]*sc+3, b[1]*sc+3), os.path.basename(path), fill='red', font=font)
        elif tag == P + 'sp':
            txb = el.find(P + 'txBody')
            if txb is None:
                continue
            paras = [''.join(t.text or '' for t in p_.iter(A + 't')) for p_ in txb.findall(A + 'p')]
            text = '\n'.join(paras).strip()
            if not text:
                continue
            b = xfrm(el.find(P + 'spPr'))
            ph = el.find('.//' + P + 'ph')
            istitle = ph is not None and ph.get('type') in ('title', 'ctrTitle')
            if b is None:
                b = [400000, 300000, 11000000, 1000000] if istitle else [800000, 1600000, 10500000, 4500000]
            else:
                b = tf(b)
            draw.multiline_text((b[0]*sc, b[1]*sc), text, fill=(0, 0, 160), font=tfont if istitle else font)


def slide_order():
    pres = open(f'{root}/ppt/presentation.xml').read()
    rel = etree.parse(f'{root}/ppt/_rels/presentation.xml.rels').getroot()
    target = {e.get('Id'): e.get('Target') for e in rel}
    ids = re.findall(r'<p:sldId [^>]*r:id="(rId\d+)"', pres)
    return [int(re.search(r'slide(\d+)\.xml', target[r]).group(1)) for r in ids]


for k, n in enumerate(slide_order(), 1):
    tree = etree.parse(f'{root}/ppt/slides/slide{n}.xml')
    img = Image.new('RGBA', (W, H), 'white')
    draw = ImageDraw.Draw(img)
    sp = tree.getroot().find('.//' + P + 'spTree')
    walk(sp, lambda b: b, img, draw, rels(n), n)
    img.convert('RGB').save(f'{out}/s{k:02d}.png')
