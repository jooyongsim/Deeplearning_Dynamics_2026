"""Convert a jupytext "percent" .py file to .ipynb without any dependency.

    python py2ipynb.py tutorial.py [out.ipynb]

Cells:
  # %%               -> code cell
  # %% [markdown]    -> markdown cell (leading "# " removed from each line)
The jupytext YAML header (the first "# ---" ... "# ---" block) is skipped.
If jupytext is installed, `jupytext --to notebook tutorial.py` gives the same result.
"""
import json
import os
import re
import sys

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".ipynb"
lines = open(src, encoding="utf-8").read().splitlines()

# skip jupytext header
i = 0
if lines and lines[0].strip() == "# ---":
    i = 1
    while i < len(lines) and lines[i].strip() != "# ---":
        i += 1
    i += 1

cells, kind, buf = [], None, []
MARK = re.compile(r"^# %%(.*)$")


def flush():
    if kind is None:
        return
    body = list(buf)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if kind == "markdown":
        body = [l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l for l in body]
    if not body:
        return
    text = [l + "\n" for l in body]
    text[-1] = text[-1].rstrip("\n")
    cell = {"cell_type": kind, "metadata": {}, "source": text}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    cells.append(cell)


for line in lines[i:]:
    m = MARK.match(line)
    if m:
        flush()
        kind = "markdown" if "[markdown]" in m.group(1) else "code"
        buf = []
    else:
        buf.append(line)
flush()

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{out}: {len(cells)} cells "
      f"({sum(c['cell_type'] == 'markdown' for c in cells)} markdown, "
      f"{sum(c['cell_type'] == 'code' for c in cells)} code)")
