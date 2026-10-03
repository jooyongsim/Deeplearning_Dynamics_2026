"""Tile PNGs into one contact sheet for quick visual review.

usage: python contact_sheet.py OUT.png COLS IMG [IMG ...]
"""
import sys

from PIL import Image

out, cols, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
ims = [Image.open(f).convert("RGB") for f in files]
w = max(i.width for i in ims)
h = max(i.height for i in ims)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * (w + 10), rows * (h + 10)), "#888")
for k, im in enumerate(ims):
    sheet.paste(im, ((k % cols) * (w + 10), (k // cols) * (h + 10)))
sheet.save(out)
print(out, sheet.size)
