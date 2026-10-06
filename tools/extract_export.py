"""Pull the JSON written by export_layout.js out of the dumped DOM → build/layout.json."""
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("BUILD_DIR", os.path.join(HERE, "build"))
dom = open(os.path.join(BUILD, "export_dom.html"), encoding="utf-8").read()
m = re.search(r'<pre id="export">(.*?)</pre>', dom, re.S)
if not m:
    raise SystemExit("no <pre id=export> in DOM — did KaTeX/export finish? raise --virtual-time-budget")
data = json.loads(html.unescape(m.group(1)))
json.dump(data, open(os.path.join(BUILD, "layout.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
n = sum(len(s["items"]) for s in data["slides"])
print(f"layout.json: {len(data['slides'])} slides, {n} items")
