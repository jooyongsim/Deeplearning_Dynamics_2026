"""Build the imitation-learning / RL decks from the Dynamics deck's design system and viewer script.

    python assemble_il_deck.py            # slides/ADL_2026_ImitationRL.html          (snippets/il_slides.html)
    python assemble_il_deck.py physics    # slides/ADL_2026_ImitationRL_Physics.html  (+ snippets/il_physics_slides.html)
    python assemble_il_deck.py steps      # slides/ADL_2026_ImitationRL_StepByStep.html (physics + snippets/il_basics_slides.html)

The physics version inserts each group of il_physics_slides.html after the slide named in its
"<!-- @after: TITLE -->" marker: detailed contact physics and equation <-> code slides.
The step-by-step version adds numbered foundation slides (basic physics, ML, RL) on top of that;
its groups use "@after: TITLE" or "@before: TITLE".
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SLIDES = os.path.join(HERE, "..", "slides")
SECTION = re.compile(r'<section class="slide.*?</section>', re.S)


def sections(html):
    return SECTION.findall(html)


def title(sec):
    return re.search(r'data-title="([^"]*)"', sec).group(1)


def insert(slides, extra):
    groups = re.split(r"<!-- @(after|before): (.*?) -->", extra)[1:]
    for where, anchor, chunk in zip(groups[0::3], groups[1::3], groups[2::3]):
        i = [title(s) for s in slides].index(anchor) + (where == "after")
        slides[i:i] = sections(chunk)


mode = (sys.argv[1:] or ["base"])[0]
src = open(os.path.join(SLIDES, "ADL_2026_Dynamics.html"), encoding="utf-8").read()
body = open(os.path.join(HERE, "snippets", "il_slides.html"), encoding="utf-8").read()
slides = sections(body)
SUB = "Hands-on tutorial: pusht_imitation.py · pusht_rl.py · pusht_rl_realtime.py · pusht_rl_eval.py"

if mode in ("physics", "steps"):
    insert(slides, open(os.path.join(HERE, "snippets", "il_physics_slides.html"), encoding="utf-8").read())
if mode == "steps":
    insert(slides, open(os.path.join(HERE, "snippets", "il_basics_slides.html"), encoding="utf-8").read())

if mode == "physics":
    slides[0] = slides[0].replace(SUB, "Extended edition: contact physics derived and compared line by line with the tutorial code")
    name, label = "ADL_2026_ImitationRL_Physics", "Imitation Learning and RL on PushT — Physics and Code"
elif mode == "steps":
    slides[0] = slides[0].replace(SUB, "Step-by-step edition: from basic physics, learning and RL to the full PushT pipeline")
    name, label = "ADL_2026_ImitationRL_StepByStep", "Imitation Learning and RL on PushT — Step by Step"
else:
    name, label = "ADL_2026_ImitationRL", "Imitation Learning and RL on PushT"

start = src.index("<!-- ====================================================================== 01 -->")
end = src.index("</div></div>\n\n<div id=\"hud\">")
head, tail = src[:start], src[end:]
head = re.sub(r"<title>.*?</title>", f"<title>ADL 2026 · {label}</title>", head)
head = head.replace(
    "/* ======================================================================= cover */",
    # monospace command lines (exported as Consolas text)
    "p.code{margin:0 0 6px;font:400 19px/1.3 var(--font-mono);color:var(--navy-900);"
    "border-left:var(--stroke-diagram) solid var(--navy-200);padding:2px 0 2px 12px}\n"
    # code listings next to equations: one <p class=code> per source line, indentation kept
    ".codebox{border-left:var(--stroke-diagram) solid var(--navy-200);padding:0 0 4px 14px;min-width:0}\n"
    ".codebox p.code{border:none;padding:0;margin:0;white-space:pre;font-size:15.5px;line-height:1.36;color:var(--ink-900)}\n"
    ".codebox p.code-file{margin:0 0 8px;font:700 16px/1.2 var(--font-core);color:var(--navy-700)}\n"
    ".code .cm{color:var(--dia-green-line)}\n\n"
    "/* ======================================================================= cover */")
out = os.path.join(SLIDES, name + ".html")
parts = [f"<!-- {'=' * 70} {i + 1:02d} -->\n{s}\n" for i, s in enumerate(slides)]
open(out, "w", encoding="utf-8").write(head + "\n".join(parts) + "\n" + tail)
print("wrote", out, len(slides), "slides")
