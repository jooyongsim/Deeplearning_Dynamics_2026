#!/usr/bin/env bash
# Audit, screenshot and export the lecture deck with headless Chrome.
#
# usage: bash render_deck.sh [audit|shots|pdf|export|all]   (default: all)
#   audit  overflow check via ?audit (prints flagged slides)
#   shots  one PNG per slide into review/shots/, plus 3x2 contact sheets review/deck_N.png
#   pdf    print the deck to slides/ADL_2026_Dynamics.pdf
#   export layout JSON for the PPTX build -> build/layout.json (not part of 'all')
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SLIDES="$HERE/../slides"
DECK="ADL_2026_Dynamics.html"
CH="/c/Program Files/Google/Chrome/Application/chrome.exe"
URL="file:///$(cd "$SLIDES" && pwd -W | sed 's/ /%20/g')/$DECK"
WHAT="${1:-all}"

n_slides() { grep -c '<section class="slide' "$SLIDES/$DECK"; }

if [[ $WHAT == audit || $WHAT == all ]]; then
  "$CH" --headless=new --disable-gpu --virtual-time-budget=15000 --dump-dom "$URL?audit" 2>/dev/null \
    | grep -E "AUDIT  —|  BAD  " | sed 's/<[^>]*>//g' || true
fi

if [[ $WHAT == shots || $WHAT == all ]]; then
  mkdir -p "$HERE/review/shots"
  rm -f "$HERE/review/shots/"*.png
  N=$(n_slides)
  for i in $(seq 1 "$N"); do
    "$CH" --headless=new --disable-gpu --hide-scrollbars --window-size=1280,720 \
      --virtual-time-budget=12000 \
      --screenshot="$(cygpath -w "$HERE/review/shots/s$(printf %02d "$i").png")" "$URL#$i" >/dev/null 2>&1
  done
  cd "$HERE/review"
  files=(shots/s*.png)
  for ((k = 0; k < ${#files[@]}; k += 6)); do
    python ../contact_sheet.py "deck_$((k / 6 + 1)).png" 3 "${files[@]:k:6}"
  done
fi

if [[ $WHAT == pdf || $WHAT == all ]]; then
  "$CH" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=20000 \
    --print-to-pdf="$(cygpath -w "$SLIDES/${DECK%.html}.pdf")" "$URL" >/dev/null 2>&1
  echo "pdf: $SLIDES/${DECK%.html}.pdf"
fi

if [[ $WHAT == export ]]; then
  mkdir -p "$HERE/build"
  "$CH" --headless=new --disable-gpu --allow-file-access-from-files --virtual-time-budget=20000     --dump-dom "$URL?export" 2>/dev/null > "$HERE/build/export_dom.html"
  (cd "$HERE" && python extract_export.py)
fi
