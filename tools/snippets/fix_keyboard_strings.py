# -*- coding: utf-8 -*-
r"""Repair two spots in tutorial/pusht_keyboard_sim.py where escapes were expanded:
line 297 (LaTeX \text, \approx, \frac) and the multi-line key-help string."""
import os
p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "tutorial", "pusht_keyboard_sim.py")
L = open(p, encoding="utf-8", newline="").read().split("\n")

i = next(k for k, l in enumerate(L) if l.startswith("# \mathbf F_{") and "\x07" in l)
L[i] = r"# \mathbf F_{\text{contact}}\approx\frac{1}{\Delta t_{\text{frame}}}\sum_{\text{substeps}}\mathbf J"

a = next(k for k, l in enumerate(L) if l.startswith('        self.fig.text(0.775, 0.36, "arrows'))
b = next(k for k in range(a, len(L)) if L[k].strip().startswith('"Q / Esc'))
L[a:b + 1] = [
    '        help_text = ("arrows / WASD : move target\n"',
    '                     "mouse drag    : target\n"',
    '                     "space         : pause\n"',
    '                     "R             : reset\n"',
    '                     "F             : friction on/off\n"',
    '                     "E             : restitution\n"',
    '                     "Q / Esc       : quit")',
    '        self.fig.text(0.775, 0.36, help_text,',
]
open(p, "w", encoding="utf-8", newline="").write("\n".join(L))
s = "\n".join(L)
print("ctrl chars left:", sum(s.count(c) for c in "\t\b\f\r\x07"))
