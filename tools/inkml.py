"""Decode (Microsoft-flavoured) InkML traces into polylines."""
import re
from lxml import etree

IK = '{http://www.w3.org/2003/InkML}'
tok = re.compile(r"([!'\"]?)\s*(-?\d*\.\d+|-?\d+)")


def decode_trace(s, nch):
    pts = []
    mode = ['!'] * nch
    val = [0.0] * nch
    vel = [0.0] * nch
    for pstr in s.split(','):
        toks = tok.findall(pstr)
        if not toks:
            continue
        for c, (pre, num) in enumerate(toks[:nch]):
            if pre:
                mode[c] = pre
            x = float(num)
            if mode[c] == '!':
                vel[c] = x - val[c]
                val[c] = x
            elif mode[c] == "'":
                vel[c] = x
                val[c] += x
            else:
                vel[c] += x
                val[c] += vel[c]
        pts.append(tuple(val[:2]))
    return pts


def parse_ink(path):
    """Return list of (points, width_cm, color_hex, transparency)."""
    t = etree.parse(path).getroot()
    ctx_nch = {}
    for ctx in t.iter(IK + 'context'):
        nch = len(ctx.findall('.//' + IK + 'channel'))
        ctx_nch[ctx.get('{http://www.w3.org/XML/1998/namespace}id')] = nch or 3
    brushes = {}
    for b in t.iter(IK + 'brush'):
        props = {p.get('name'): p.get('value') for p in b.findall(IK + 'brushProperty')}
        brushes[b.get('{http://www.w3.org/XML/1998/namespace}id')] = props
    out = []
    lastctx, lastbr = None, None
    for tr in t.iter(IK + 'trace'):
        ctx = (tr.get('contextRef') or lastctx or '#ctx0').lstrip('#')
        br = (tr.get('brushRef') or lastbr or '#br0').lstrip('#')
        lastctx, lastbr = '#' + ctx, '#' + br
        nch = ctx_nch.get(ctx, 3)
        pts = decode_trace(tr.text or '', nch)
        props = brushes.get(br, {})
        out.append((pts, float(props.get('width', 0.05)), props.get('color', '#000000'),
                    int(props.get('transparency', 0) or 0)))
    return out
