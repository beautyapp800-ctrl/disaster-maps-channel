#!/usr/bin/env bash
# Auto-chain (owner approved 23.11): once the miniature-world data is on data-mini, render the tornado film, then the hurricane film,
# one at a time (never more than 8 parallel render jobs: the account limit is shared with the other channels).
# 4K first; if a 4K render fails, one automatic retry at 1440p. Results become draft releases reserve-<slug>. Nothing is uploaded to YouTube.
set -u
REPO="${GITHUB_REPOSITORY}"
git fetch -q origin data-mini || { echo "no data-mini yet"; exit 0; }
have() { git ls-tree -r --name-only origin/data-mini data/osm | grep -c "/$1.json.gz"; }
count() { local n=0; for k in "$@"; do n=$((n + $(have "$k"))); done; echo $n; }
TORN="waco flint ruskin jackson xenia moore99 greensburg hackleburg joplin moore13 mayfield"
HURR="galveston1900 laborday1935 newengland1938 camille1969 andrew1992 katrina2005 harvey2017 michael2018 ian2022 helene2024"

active=$(gh run list --workflow pilot.yml --json status -q '[.[]|select(.status!="completed")]|length')
prep=$(gh run list --workflow prep-mini.yml --json status -q '[.[]|select(.status!="completed")]|length')
echo "pilot runs active: $active, prep-mini runs active: $prep"
[ "$active" != "0" ] && exit 0

runs=$(gh run list --workflow pilot.yml --limit 40 --json displayTitle,conclusion,status)
res_of() { echo "$runs" | python3 -c "
import json,sys
eng,res=sys.argv[1],sys.argv[2]
r=[x for x in json.load(sys.stdin) if x['displayTitle']=='pilot %s %s'%(eng,res)]
print('none' if not r else r[0]['conclusion'])" "$1" "$2"; }

launch() {  # engine res slug
  echo "launching $1 at $2p"
  gh workflow run pilot.yml --ref dev -f engine="$1" -f res="$2" -f texw=4096 -f segments=32 -f crf=20 -f slug="$3" -f store=true
  exit 0
}
step() {  # engine slug keys...
  local eng="$1" slug="$2"; shift 2
  local total=$#
  local ok4 ok14 r4 r14
  r4=$(res_of "$eng" 2160); r14=$(res_of "$eng" 1440)
  if [ "$r4" = "success" ] || [ "$r14" = "success" ]; then echo "$eng done"; return 0; fi
  if [ "$r4" = "failure" ] || [ "$r4" = "cancelled" ]; then
    if [ "$r14" = "none" ]; then launch "$eng" 1440 "$slug"; fi
    echo "$eng failed at both resolutions: manual check needed"; return 0
  fi
  # no 4K run yet: data must be complete (all prep jobs finished, at least all but one town present)
  [ "$prep" != "0" ] && { echo "waiting for data jobs"; exit 0; }
  local got; got=$(count "$@")
  if [ "$got" -lt $((total - 1)) ]; then
    missing=""; for k in "$@"; do [ "$(have "$k")" = "0" ] && missing="$missing $k"; done
    tries=$(gh run list --workflow prep-mini.yml --limit 50 --json displayTitle -q '[.[]|select(.displayTitle|test("osm"))]|length')
    if [ "$tries" -lt 8 ]; then echo "retry OSM for:$missing"; gh workflow run prep-mini.yml --ref dev -f what="osm$missing"; fi
    exit 0
  fi
  launch "$eng" 2160 "$slug"
}
step tornado us-tornadoes-1950-2025 $TORN
step hurricane us-hurricane-landfalls-1851-2025 $HURR
echo "all films built"
