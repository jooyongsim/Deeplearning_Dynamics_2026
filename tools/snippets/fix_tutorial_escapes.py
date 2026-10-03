# -*- coding: utf-8 -*-
r"""Repair LaTeX commands that a non-raw Python string turned into control characters
(\t -> TAB, \b -> BS, \f -> FF, \r -> CR) in tutorial/pusht_mini_sim.py. File is LF-only."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tutorial", "pusht_mini_sim.py")
s = open(p, encoding="utf-8", newline="").read()
before = {name: s.count(ch) for name, ch in (("TAB", "\t"), ("BS", "\b"), ("FF", "\f"), ("CR", "\r"))}
for ch, rep in (("\t", "\\t"), ("\b", "\\b"), ("\f", "\\f"), ("\r", "\\r")):
    s = s.replace(ch, rep)
open(p, "w", encoding="utf-8", newline="").write(s)
print("replaced:", before)
