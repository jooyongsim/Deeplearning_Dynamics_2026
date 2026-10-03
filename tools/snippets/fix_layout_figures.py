# -*- coding: utf-8 -*-
"""One-time layout pass after inserting textbook figures (slides 6, 9, 12, 13, 17)."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "slides", "ADL_2026_Dynamics.html")
s = open(p, encoding="utf-8").read()

R = [
    # slide 6 polar: larger figure
    ('<div style="flex:none;width:880px">\n        <div class="eqbox">\n          <div class="eq">\\[\\mathbf v = \\dot r',
     '<div style="flex:none;width:820px">\n        <div class="eqbox">\n          <div class="eq">\\[\\mathbf v = \\dot r'),
    ('meriam_fig2_13_polar.png" style="height:392px"', 'meriam_fig2_13_polar.png" style="height:430px"'),
    # slide 9 impact
    ('<div style="flex:none;width:420px">\n        <p class="sub-h">Definitions</p>',
     '<div style="flex:none;width:390px">\n        <p class="sub-h">Definitions</p>'),
    ('<div style="flex:none;width:410px">\n        <p class="sub-h">Coefficient of restitution</p>',
     '<div style="flex:none;width:370px">\n        <p class="sub-h">Coefficient of restitution</p>'),
    ('meriam_fig3_17_impact.png" style="height:262px"', 'meriam_fig3_17_impact.png" style="width:390px"'),
    # slide 12 relative acceleration
    ('<div style="flex:none;width:900px">\n        <div class="eqbox">\n          <div class="eq">\\[\\mathbf a_B',
     '<div style="flex:none;width:830px">\n        <div class="eqbox">\n          <div class="eq">\\[\\mathbf a_B'),
    ('meriam_fig5_9_rel_accel.png" style="height:420px"', 'meriam_fig5_9_rel_accel.png" style="height:440px"'),
    # slide 17: move the procedure line into the left column
    ('        <p class="def" style="margin-top:16px;align-self:flex-start"><b>Procedure</b> : FBD ≡ kinetic diagram → kinematic link (e.g. \\(a_G = R\\alpha\\)) → solve.</p>\n',
     ''),
    ('        <div class="eq sm">\\[\\sum M_O = I_O\\,\\alpha\\]</div>\n      </div>\n      <div class="tfig">\n        <img src="assets/figures/meriam_fig6_4',
     '        <div class="eq sm">\\[\\sum M_O = I_O\\,\\alpha\\]</div>\n'
     '        <p class="def" style="margin-top:10px"><b>Procedure</b> : FBD ≡ kinetic diagram → kinematic link (e.g. \\(a_G = R\\alpha\\)) → solve.</p>\n'
     '      </div>\n      <div class="tfig">\n        <img src="assets/figures/meriam_fig6_4'),
]
for a, b in R:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)

# slide 13: wider blocks, smaller names so "General Plane Motion" fits
for tone in ("blue", "orange"):
    s = s.replace(f'<div class="blk t-{tone}" style="flex:none;width:270px;height:92px"><span class="nm">',
                  f'<div class="blk t-{tone}" style="flex:none;width:300px;height:92px"><span class="nm" style="font-size:23px">')

open(p, "w", encoding="utf-8").write(s)
print("ok")
