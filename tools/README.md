# tools — Dynamics lecture deck

The HTML deck (`../slides/ADL_2026_Dynamics.html`) is the source. The PDF and the
editable PPTX are generated from it.

| File | What it does |
| --- | --- |
| `render_deck.sh` | `audit` (overflow check) · `shots` (per-slide PNGs + contact sheets in `review/`) · `pdf` (HTML → PDF) · `export` (layout JSON for the PPTX) |
| `export_layout.js` | Loaded by the deck with `?export`: writes every slide's text runs (with LaTeX), shapes, SVG primitives and images with exact positions |
| `extract_export.py` | Pulls that JSON out of the dumped DOM → `build/layout.json` |
| `tex2omml.py` | LaTeX → native Office Math (OMML) via pandoc, cached in `build/omml_cache.json` |
| `build_pptx.py` | Builds `../slides/ADL_2026_Dynamics.pptx` on the AMSL template: native text boxes, shapes, pictures and equations |
| `make_marks.py` | Pre-tints the university mark (blue for content slides, white for the cover) since PowerPoint has no CSS filters |
| `pptx_render.ps1` | Renders the PPTX with PowerPoint itself (PNG per slide, optional PDF) for QA |
| `pdf_explore.py` | `toc` / `find` / `page` on the source PDFs (Meriam, Kochmann notes) |
| `extract_figures.py` | Crops the textbook figures into `../slides/assets/figures/` |
| `replace_slides.py` | Replaces a run of slides (between `<!-- ==== NN -->` markers) with a snippet file |
| `inspect_pptx.py` | Prints layouts/shapes/geometry of a PPTX (used to read the template) |
| `contact_sheet.py` | Tiles PNGs into one sheet for review |
| `py2ipynb.py` | jupytext "percent" `.py` → `.ipynb` with no dependencies (used for `../tutorial/*.py`) |
| `test_keyboard_demo.py` | Drives `../tutorial/pusht_keyboard_sim.py` without a window (Agg): synthetic keys/mouse, wall check, snapshots in `review/` |
| `test_imitation_recorder.py` | Drives the human-demo recorder of `../tutorial/pusht_imitation.py` without a window: 10 Hz recording, idle trimming, auto-save on goal, seed replay, N/Backspace; snapshot in `review/` |
| `snippets/` | Slide markup and one-off layout/QA passes applied during the build |

After editing the HTML deck:

```bash
bash render_deck.sh audit      # 0 flagged = nothing spills past the footer
bash render_deck.sh all        # screenshots + PDF
bash render_deck.sh export     # layout for the PPTX
python build_pptx.py           # -> slides/ADL_2026_Dynamics.pptx
powershell -ExecutionPolicy Bypass -File pptx_render.ps1 -Pptx ../slides/ADL_2026_Dynamics.pptx -OutDir review/pptx
```

Requires Chrome (headless), PowerPoint (for QA renders), pandoc, Python with
python-pptx, lxml, PyMuPDF and Pillow.
