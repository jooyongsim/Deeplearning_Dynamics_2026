#!/usr/bin/env bash
# Linux counterpart of render_deck.sh, for any deck in ../slides.
#
# usage: bash render_deck_linux.sh DECK [audit|shots|pdf|export|pptx|all]   (default: all)
#   DECK   deck basename, e.g. ADL_2026_ImitationRL
#   audit  overflow check via ?audit
#   shots  one PNG per slide into review/DECK/ plus 3x2 contact sheets
#   pdf    print the deck to slides/DECK.pdf
#   export layout JSON -> build/DECK/layout.json
#   pptx   export + build_pptx.py -> slides/DECK.pptx
#   all    audit + shots + pdf + pptx
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SLIDES="$(cd "$HERE/../slides" && pwd)"
NAME="${1:?deck name}"
WHAT="${2:-all}"
DECK="$NAME.html"
CH="${CHROME:-google-chrome}"
PY="${PY:-$HERE/../.venv/bin/python}"
URL="file://$SLIDES/$DECK"
FLAGS=(--headless=new --disable-gpu --no-sandbox --allow-file-access-from-files)
export BUILD_DIR="$HERE/build/$NAME"
export PANDOC="${PANDOC:-$("$PY" -c 'import pypandoc;print(pypandoc.get_pandoc_path())' 2>/dev/null || echo pandoc)}"

n_slides() { grep -c '<section class="slide' "$SLIDES/$DECK"; }

if [[ $WHAT == audit || $WHAT == all ]]; then
  "$CH" "${FLAGS[@]}" --virtual-time-budget=15000 --dump-dom "$URL?audit" 2>/dev/null \
    | grep -E "AUDIT  —|  BAD  " | sed 's/<[^>]*>//g' || true
fi

if [[ $WHAT == shots || $WHAT == all ]]; then
  OUT="$HERE/review/$NAME"; mkdir -p "$OUT"; rm -f "$OUT"/s*.png
  for i in $(seq 1 "$(n_slides)"); do
    "$CH" "${FLAGS[@]}" --hide-scrollbars --window-size=1280,720 --virtual-time-budget=12000 \
      --screenshot="$OUT/s$(printf %02d "$i").png" "$URL#$i" >/dev/null 2>&1
  done
  cd "$OUT"; files=(s*.png)
  for ((k = 0; k < ${#files[@]}; k += 6)); do
    "$PY" "$HERE/contact_sheet.py" "deck_$((k / 6 + 1)).png" 3 "${files[@]:k:6}"
  done
  cd "$HERE"
fi

if [[ $WHAT == pdf || $WHAT == all ]]; then
  "$CH" "${FLAGS[@]}" --no-pdf-header-footer --virtual-time-budget=20000 \
    --print-to-pdf="$SLIDES/$NAME.pdf" "$URL" >/dev/null 2>&1
  echo "pdf: $SLIDES/$NAME.pdf"
fi

if [[ $WHAT == export || $WHAT == pptx || $WHAT == all ]]; then
  mkdir -p "$BUILD_DIR"
  "$CH" "${FLAGS[@]}" --virtual-time-budget=20000 --dump-dom "$URL?export" 2>/dev/null > "$BUILD_DIR/export_dom.html"
  (cd "$HERE" && "$PY" extract_export.py)
fi

if [[ $WHAT == pptx || $WHAT == all ]]; then
  (cd "$HERE" && PPTX_OUT="$SLIDES/$NAME.pptx" "$PY" build_pptx.py)
fi
