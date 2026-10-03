# -*- coding: utf-8 -*-
"""QA pass 1 on the PPTX pipeline (exporter + builder + deck CSS)."""
import os
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def patch(path, pairs):
    p = os.path.join(T, path); s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:60])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)

# ---- exporter: content boxes for every block, label/equation rows centred, class + svg order tags
patch("export_layout.js", [
("""      if (el.tagName === 'TD' || el.tagName === 'TH') {
        bx = {x: bx.x + pl, y: bx.y + pt, w: bx.w - pl - pr, h: bx.h - pt - pb};
      }""",
"""      bx = {x: bx.x + pl, y: bx.y + pt, w: bx.w - pl - pr, h: bx.h - pt - pb};"""),
("""      var valign = (el.tagName === 'TD' || el.tagName === 'TH') ? cs.verticalAlign : 'top';""",
"""      var valign = (el.tagName === 'TD' || el.tagName === 'TH') ? cs.verticalAlign : 'top';
      if (el.classList.contains('eqlabel')) {            // label: centre on its equation row
        var row = box(el.parentElement, o);
        bx = {x: bx.x, y: row.y, w: bx.w, h: row.h}; valign = 'middle';
      }
      if (lab) valign = 'middle';"""),
("""      items.push(Object.assign({kind: 'text', align: align, valign: valign, lh: lh, runs: runs}, bx));""",
"""      items.push(Object.assign({kind: 'text', align: align, valign: valign, lh: lh, runs: runs,
                                cls: el.classList[0] || el.tagName.toLowerCase()}, bx));"""),
("""          items.push({kind: 'line', color: stroke, sw: sw, dash: dash, head: head,
                      pts: [[ox""",
"""          items.push({kind: 'line', svg: true, color: stroke, sw: sw, dash: dash, head: head,
                      pts: [[ox"""),
("""          items.push({kind: 'oval', x: ox + cx - r,""", """          items.push({kind: 'oval', svg: true, x: ox + cx - r,"""),
("""          items.push({kind: 'line', color: stroke, sw: sw, dash: dash, head: head, pts: pts});""",
"""          items.push({kind: 'line', svg: true, color: stroke, sw: sw, dash: dash, head: head, pts: pts});"""),
])
# `lab` is declared after the valign line in the original order -> move declaration up
p = os.path.join(T, "export_layout.js"); s = open(p, encoding="utf-8").read()
s = s.replace("""      var lab = el.querySelector(':scope > .eqlabel');
      if (lab) {""", """      if (lab) {""")
s = s.replace("""      bx = {x: bx.x + pl, y: bx.y + pt, w: bx.w - pl - pr, h: bx.h - pt - pb};""",
"""      bx = {x: bx.x + pl, y: bx.y + pt, w: bx.w - pl - pr, h: bx.h - pt - pb};
      var lab = el.querySelector(':scope > .eqlabel');""")
open(p, "w", encoding="utf-8").write(s)

# ---- builder
patch("build_pptx.py", [
("MATH_SCALE = 0.86   # Cambria Math reads larger than KaTeX at the same size",
 "MATH_SCALE = 0.97   # Cambria Math has a smaller x-height than Arial; KaTeX is already 1.1em"),
("""    if not single:
        # a little slack so PowerPoint's metrics do not wrap a line earlier than the browser
        w = w * 1.02 + 4
""",
"""    if it.get("cls") in ("op-oval", "op-txt"):
        y += 3   # the glyph sits high in PowerPoint's line box; CSS centres it with flex
    if not single:
        w = min(w + 2, 1257 - x)   # never past the right margin
"""),
("""    order = {"rect": 0, "oval": 1, "tri": 1, "img": 2, "line": 3, "text": 4}""",
 """    order = {"rect": 0, "oval": 1, "tri": 1, "img": 2, "line": 3, "text": 4}
    key = lambda it: 3 if it.get("svg") else order[it["kind"]]   # SVG keeps its paint order"""),
("""        items = sorted(s["items"], key=lambda it: order[it["kind"]])""",
 """        items = sorted(s["items"], key=key)"""),
("""def convert_norm(texs):
    # pandoc writes \tfrac as a linear fraction ("1/2"); a skewed fraction keeps the
    # compact look of the HTML's text-style fraction without growing the line
    res = convert(texs)
    return {t: res[t].replace('m:val="lin"', 'm:val="skw"') for t in texs}""",
"""VULGAR = {("1", "2"): "½", ("1", "3"): "⅓", ("2", "3"): "⅔", ("1", "4"): "¼", ("3", "4"): "¾"}


def _compact_fractions(omml):
    \"\"\"pandoc writes \\tfrac as a linear fraction (type=lin). Keep it on the text line:
    common numeric ones become single glyphs (½ ¾ ⅓), the rest stay linear a/b.\"\"\"
    om = etree.fromstring(omml)
    for f in list(om.iter(f"{{{NS_M}}}f")):
        typ = f.find(f"{{{NS_M}}}fPr/{{{NS_M}}}type")
        if typ is None or typ.get(f"{{{NS_M}}}val") != "lin":
            continue
        num = "".join(f.find(f"{{{NS_M}}}num").itertext()).strip()
        den = "".join(f.find(f"{{{NS_M}}}den").itertext()).strip()
        glyph = VULGAR.get((num, den))
        if glyph:
            r = etree.Element(f"{{{NS_M}}}r")
            rp = etree.SubElement(r, f"{{{NS_M}}}rPr")
            etree.SubElement(rp, f"{{{NS_M}}}sty").set(f"{{{NS_M}}}val", "p")
            etree.SubElement(r, f"{{{NS_M}}}t").text = glyph
            f.getparent().replace(f, r)
    return etree.tostring(om, encoding="unicode")


def convert_norm(texs):
    res = convert(texs)
    return {t: _compact_fractions(res[t]) for t in texs}"""),
])

# ---- deck CSS: equation grid columns (label column 180px, eq 22px) so row 4 clears the divider
patch(os.path.join("..", "slides", "ADL_2026_Dynamics.html"), [
(".eqgrid{display:grid;grid-template-columns:.8fr 1.2fr;", ".eqgrid{display:grid;grid-template-columns:.9fr 1.1fr;"),
(".eqgrid .k{flex:0 0 210px;", ".eqgrid .k{flex:0 0 180px;"),
(".eqgrid .m{font-size:24px;white-space:nowrap}", ".eqgrid .m{font-size:22px;white-space:nowrap}"),
])
print("ok")
