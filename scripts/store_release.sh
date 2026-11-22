#!/usr/bin/env bash
# Draft release reserve-<slug> with the finished video, the chosen cover, metadata, QA and the 480p preview. Nothing is uploaded to YouTube.
set -eu
python -m disasters.meta "$SLUG" > dl/meta.json || { tail -5 dl/meta.json; exit 1; }
SIZE=$(stat -c %s dl/video/final.mp4); echo "video bytes $SIZE"
test "$SIZE" -lt 2000000000
cp dl/covers/cover_CHOSEN.jpg dl/cover.jpg
files=(dl/video/final.mp4 dl/cover.jpg dl/meta.json dl/review/qa_report.md dl/review/verify.json dl/review/preview_480p.mp4 dl/review/contact_sheet.jpg)
for f in dl/covers/cover_[ABC]_*.jpg; do [ -f "$f" ] && files+=("$f"); done
gh release delete "reserve-$SLUG" --yes 2>/dev/null || true
gh release create "reserve-$SLUG" --draft --title "RESERVE $SLUG (${RES}p)"   --notes "Built by pilot run $RUN_ID. Not uploaded (channel not created yet). Files: preview_480p.mp4, contact_sheet.jpg, cover.jpg, qa_report.md." "${files[@]}"
