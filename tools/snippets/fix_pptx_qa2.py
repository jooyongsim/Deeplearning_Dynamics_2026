# -*- coding: utf-8 -*-
"""QA pass 2: real line counts from the browser, label/operator nudges, uniform small fractions."""
import os
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def patch(path, pairs):
    p = os.path.join(T, path); s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:60])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)

patch("export_layout.js", [
("""  function box(el, origin) {""",
"""  // Number of rendered line boxes: text-node rects plus whole KaTeX boxes, grouped by
  // vertical overlap (a KaTeX box spans its sub/superscripts, so it stays one line).
  function lineCount(el) {
    var rects = [];
    var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    for (var n = walker.nextNode(); n; n = walker.nextNode()) {
      if (n.parentElement.closest('.katex') || !n.textContent.trim()) continue;
      var r = document.createRange(); r.selectNodeContents(n);
      Array.prototype.forEach.call(r.getClientRects(), function (q) { if (q.width > 0) rects.push(q); });
    }
    el.querySelectorAll('.katex').forEach(function (k) { rects.push(k.getBoundingClientRect()); });
    rects.sort(function (a, b) { return a.top - b.top; });
    var lines = 0, bottom = -1e9;
    rects.forEach(function (q) {
      if (q.top >= bottom - 3) { lines++; bottom = q.bottom; } else bottom = Math.max(bottom, q.bottom);
    });
    return Math.max(lines, 1);
  }

  function box(el, origin) {"""),
("""      items.push(Object.assign({kind: 'text', align: align, valign: valign, lh: lh, runs: runs,
                                cls: el.classList[0] || el.tagName.toLowerCase()}, bx));""",
"""      items.push(Object.assign({kind: 'text', align: align, valign: valign, lh: lh, runs: runs,
                                nlines: lineCount(el),
                                cls: el.classList[0] || el.tagName.toLowerCase()}, bx));"""),
])

patch("build_pptx.py", [
("""    single = len(lines) == 1 and it["h"] <= 1.7 * max(lh, 1)""",
"""    # one rendered line in the browser -> never let PowerPoint wrap it
    single = len(lines) == 1 and (it.get("nlines") == 1 if "nlines" in it else it["h"] <= 1.7 * max(lh, 1))"""),
("""    if it.get("cls") in ("op-oval", "op-txt"):
        y += 3   # the glyph sits high in PowerPoint's line box; CSS centres it with flex""",
"""    # PowerPoint sets these glyphs higher in their line box than CSS does
    y += {"op-oval": 7, "op-txt": 4, "eqlabel": 5}.get(it.get("cls"), 0)"""),
("""        glyph = VULGAR.get((num, den))""",
"""        glyph = VULGAR.get((num, den))
        if not glyph and (num + den).isdigit():   # e.g. 1/12 -> ¹⁄₁₂, same look as ½
            glyph = num.translate(SUP) + "\u2044" + den.translate(SUB)"""),
("""VULGAR = {""", """SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
VULGAR = {"""),
])
print("ok")
