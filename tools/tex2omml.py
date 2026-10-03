"""Convert LaTeX math strings to Office Math (OMML) with pandoc, cached.

    from tex2omml import convert
    omml = convert(["a_n = \frac{v^2}{\rho}", ...])   # {tex: '<m:oMath ...>...</m:oMath>'}

Runs pandoc once on a LaTeX document with one \( ... \) per paragraph, then
reads the m:oMath elements back out of the generated DOCX in order.
Cache: build/omml_cache.json
"""
import json
import os
import subprocess
import tempfile
import zipfile

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "build", "omml_cache.json")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _load():
    try:
        return json.load(open(CACHE, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def convert(texs):
    cache = _load()
    todo = [t for t in dict.fromkeys(texs) if t not in cache]
    if todo:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "eq.tex")
            out = os.path.join(tmp, "eq.docx")
            body = "\n\n".join("\(" + t + "\)" for t in todo)
            open(src, "w", encoding="utf-8").write(body + "\n")
            subprocess.run(["pandoc", "-f", "latex", src, "-o", out], check=True)
            xml = zipfile.ZipFile(out).read("word/document.xml")
        root = etree.fromstring(xml)
        paras = [p for p in root.iter(f"{{{W}}}p") if p.find(f".//{{{M}}}oMath") is not None]
        if len(paras) != len(todo):
            raise RuntimeError(f"pandoc returned {len(paras)} equations for {len(todo)} inputs")
        for t, p in zip(todo, paras):
            om = p.find(f".//{{{M}}}oMath")
            cache[t] = etree.tostring(om, encoding="unicode")
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return {t: cache[t] for t in texs}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = json.load(open(os.path.join(HERE, "build", "layout.json"), encoding="utf-8"))
    texs = [r["m"] for s in d["slides"] for it in s["items"] if it["kind"] == "text"
            for r in it["runs"] if "m" in r]
    res = convert(texs)
    print(len(set(texs)), "unique equations converted")
    for t in list(dict.fromkeys(texs))[:3]:
        print(t, "->", res[t][:160])
