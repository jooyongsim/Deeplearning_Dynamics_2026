# -*- coding: utf-8 -*-
"""QA pass 1 (part b): remaining builder changes + deck CSS for the equation grid."""
import os
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def patch(path, pairs):
    p = os.path.join(T, path); s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:60])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)

patch("build_pptx.py", [
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
])
patch(os.path.join("..", "slides", "ADL_2026_Dynamics.html"), [
(".eqgrid{display:grid;grid-template-columns:.8fr 1.2fr;", ".eqgrid{display:grid;grid-template-columns:.9fr 1.1fr;"),
(".eqgrid .k{flex:0 0 210px;", ".eqgrid .k{flex:0 0 180px;"),
(".eqgrid .m{font-size:24px;white-space:nowrap}", ".eqgrid .m{font-size:22px;white-space:nowrap}"),
])
print("ok")
