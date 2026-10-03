"""Explore the source PDFs for figures to reuse in the lecture deck.

usage:
  python pdf_explore.py toc   PDF            # table of contents
  python pdf_explore.py find  PDF "phrase"   # pages whose text contains the phrase
  python pdf_explore.py page  PDF N [N...]   # render pages (1-based) to review/ as PNG
"""
import os
import sys

import fitz  # PyMuPDF

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.join(HERE, "review")


def toc(doc):
    print(doc.name, doc.page_count, "pages")
    for level, title, page in doc.get_toc():
        print("  " * level + title[:80], page)


def find(doc, phrase):
    phrase = phrase.lower()
    for i, page in enumerate(doc):
        text = page.get_text().lower()
        if phrase in text:
            n_img = len(page.get_images())
            print(f"p{i + 1:4d}  images={n_img}")


def render(doc, pages, zoom=1.3):
    os.makedirs(REVIEW, exist_ok=True)
    tag = os.path.splitext(os.path.basename(doc.name))[0][:12].replace(" ", "_").replace("'", "")
    for n in pages:
        pix = doc[n - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        out = os.path.join(REVIEW, f"{tag}_p{n:04d}.png")
        pix.save(out)
        print(out)


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    d = fitz.open(path)
    if cmd == "toc":
        toc(d)
    elif cmd == "find":
        find(d, sys.argv[3])
    elif cmd == "page":
        render(d, [int(a) for a in sys.argv[3:]])
