"""Crop the lecture figures out of the source PDFs into slides/assets/figures/.

Figures are vector drawings, so they are rendered (not extracted) at high zoom
from a clip rectangle given in PDF points (x0, y0, x1, y1).

usage: python extract_figures.py            # all figures
       python extract_figures.py fig5_1     # only names containing the text
"""
import os
import sys

import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))
LECTURE = os.path.dirname(HERE)
OUT = os.path.join(LECTURE, "slides", "assets", "figures")

MERIAM = r"C:\Users\Sim\Dropbox\[Courses]\[TextBooks]\동역학\Meriam's Engineering Mechanics Dynamics.pdf"
NOTES = os.path.join(LECTURE, "Dynamics_LectureNotes.pdf")

# name: (pdf, 1-based page, clip rect in points[, rects to blank out first])
FIGS = {
    "meriam_fig2_13_polar":        (MERIAM, 53,  (362, 181, 572, 470)),
    "meriam_fig3_17_impact":       (MERIAM, 115, (372, 520, 550, 692)),
    "meriam_fig5_1_plane_motion":  (MERIAM, 165, (70, 40, 402, 392)),
    "meriam_fig5_5_rel_velocity":  (MERIAM, 174, (92, 458, 530, 650)),
    "meriam_fig5_7_ic":            (MERIAM, 181, (66, 518, 456, 692)),
    "meriam_fig5_9_rel_accel":     (MERIAM, 184, (28, 378, 218, 692)),
    "meriam_fig6_4_fbd_kinetic":   (MERIAM, 204, (226, 26, 455, 200), [(224, 24, 238, 40)]),  # blank running-head letter
    "notes_p84_rolling_cycloid":   (NOTES, 88,   (165, 172, 552, 300)),
}

ZOOM = 4.0  # ~288 dpi


def main(filter_text=""):
    os.makedirs(OUT, exist_ok=True)
    docs = {}
    for name, (pdf, page, rect, *blank) in FIGS.items():
        if filter_text not in name:
            continue
        doc = docs.setdefault(pdf, fitz.open(pdf))
        for r in (blank[0] if blank else []):   # in-memory only; the PDF is not saved
            doc[page - 1].draw_rect(fitz.Rect(*r), color=(1, 1, 1), fill=(1, 1, 1))
        pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), clip=fitz.Rect(*rect))
        path = os.path.join(OUT, name + ".png")
        pix.save(path)
        print(f"{name:32s} {pix.width}x{pix.height}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
