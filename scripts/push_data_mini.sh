#!/usr/bin/env bash
# usage: push_data_mini.sh "<message>" path [path ...]   (adds the given paths to the orphan branch data-mini)
set -u
msg="$1"; shift
keep=$(mktemp -d)
for p in "$@"; do
  [ -e "$p" ] && mkdir -p "$keep/$(dirname "$p")" && cp -r "$p" "$keep/$(dirname "$p")/"
done
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for i in 1 2 3 4 5; do
  if git fetch -q origin data-mini 2>/dev/null; then
    git checkout -q -f -B data-mini FETCH_HEAD
  else
    git checkout -q -f --orphan data-mini && git rm -rq --cached . >/dev/null 2>&1; git clean -fdxq -e data/raw
  fi
  cp -r "$keep"/. .
  git add -f "$@" 2>/dev/null
  git commit -qm "$msg" || { echo "nothing to commit"; exit 0; }
  git push -q origin data-mini && { echo "pushed"; exit 0; }
  sleep $((i * 10))
done
exit 1
