"""Pre-tint the university mark for PowerPoint (which has no CSS filters).

The HTML deck tints the greyscale mark with CSS filters:
  content slides: blue duotone      -> slides/assets/logo/smwu-mark-blue.png
  cover:          knocked to white  -> slides/assets/logo/smwu-mark-white.png
Both are then placed at 50% opacity, as in the design system.
"""
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, "..", "slides", "assets", "logo")

src = Image.open(os.path.join(LOGO, "smwu-logo-mark-mono.png")).convert("RGBA")
lum = src.convert("L")
alpha = src.getchannel("A")


def duotone(dark, light):
    lut = [tuple(int(d + (l - d) * v / 255) for d, l in zip(dark, light)) for v in range(256)]
    out = Image.new("RGBA", src.size)
    px_l, px_o, px_a = lum.load(), out.load(), alpha.load()
    for y in range(src.height):
        for x in range(src.width):
            r, g, b = lut[px_l[x, y]]
            px_o[x, y] = (r, g, b, px_a[x, y])
    return out


duotone((0x20, 0x38, 0x64), (0xB4, 0xD4, 0xF4)).save(os.path.join(LOGO, "smwu-mark-blue.png"))
duotone((0xE8, 0xEC, 0xF2), (0xFF, 0xFF, 0xFF)).save(os.path.join(LOGO, "smwu-mark-white.png"))
print("wrote smwu-mark-blue.png, smwu-mark-white.png")
