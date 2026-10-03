"""Replace a run of slides in the deck HTML with the markup in a snippet file.

Slides are delimited by comment markers of the form
    <!-- ====...==== NN -->
The region from marker START up to (not including) marker END is replaced.

usage: python replace_slides.py DECK.html START END SNIPPET.html
       e.g. python replace_slides.py ../slides/ADL_2026_Dynamics.html 13 15 snippets/s13_14.html
"""
import re
import sys

deck, start, end, snippet = sys.argv[1:5]
html = open(deck, encoding="utf-8").read()


def marker_pos(tag):
    m = re.search(r"<!-- =+ " + re.escape(tag) + r" -->", html)
    if not m:
        sys.exit(f"marker {tag!r} not found")
    return m.start()


a, b = marker_pos(start), marker_pos(end)
new = open(snippet, encoding="utf-8").read()
if not new.endswith("\n\n"):
    new = new.rstrip("\n") + "\n\n"
open(deck, "w", encoding="utf-8").write(html[:a] + new + html[b:])
print(f"replaced slides {start}..{end} ({b - a} -> {len(new)} chars)")
