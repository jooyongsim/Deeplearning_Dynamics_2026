# -*- coding: utf-8 -*-
"""QA pass 3: no-wrap when browser lines == paragraphs; fraction glyphs as normal (unspaced) math text."""
import os
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(T, "build_pptx.py"); s = open(p, encoding="utf-8").read()
pairs = [
("""    single = len(lines) == 1 and (it.get("nlines") == 1 if "nlines" in it else it["h"] <= 1.7 * max(lh, 1))""",
 """    if "nlines" in it:   # every browser line is its own paragraph -> nothing wrapped in the HTML
        single = it["nlines"] <= len(lines)
    else:
        single = len(lines) == 1 and it["h"] <= 1.7 * max(lh, 1)"""),
("""            etree.SubElement(rp, f"{{{NS_M}}}sty").set(f"{{{NS_M}}}val", "p")
            etree.SubElement(r, f"{{{NS_M}}}t").text = glyph""",
 """            etree.SubElement(rp, f"{{{NS_M}}}nor")   # normal text: no operator spacing around ⁄
            etree.SubElement(r, f"{{{NS_M}}}t").text = glyph"""),
]
for a, b in pairs:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
