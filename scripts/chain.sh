#!/usr/bin/env bash
# Auto-chain (owner approved 23.11): once the data of a film is on data-mini, render it; one film at a time (the account's job limit is
# shared with the other channels). 4K first; if a 4K render fails, one automatic retry at 1440p. Results become draft releases
# reserve-<slug> (video, cover, metadata, QA, 480p preview). Nothing is uploaded to YouTube.
set -u
REPO="${GITHUB_REPOSITORY}"
git fetch -q origin data-mini || { echo "no data-mini yet"; exit 0; }
has_file() { [ -n "$(git ls-tree --name-only origin/data-mini "$1")" ]; }
have() { git ls-tree -r --name-only origin/data-mini data/osm | grep -c "/$1.json.gz"; }
count() { local n=0; for k in "$@"; do n=$((n + $(have "$k"))); done; echo $n; }
TORN="waco flint ruskin jackson xenia moore99 greensburg hackleburg joplin moore13 mayfield dunlap1965 lubbock1970 wichitafalls1979 plainfield1990 jarrell1997 jacksontn2003 vilonia2014 rollingfork2023"
HURR="galveston1900 laborday1935 newengland1938 camille1969 andrew1992 katrina2005 harvey2017 michael2018 ian2022 helene2024 indianola1886 seaislands1893 corpus1919 miami1926 hazel1954 carla1961 eloise1975 hugo1989 charley2004 ike2008"
FLOR="laborday1935 andrew1992 michael2018 ian2022 helene2024 miami1926 eloise1975 charley2004"
QUAKE="sanfrancisco1906 longbeach1933 tehachapi1952 sylmar1971 santacruz1989 yuccavalley1992 northridge1994 ridgecrest2019 santabarbara1925 imperial1940 coalinga1983 ferndale1992 pasorobles2003 napa2014 riodell2022"

active=$(gh run list --workflow pilot.yml --json status -q '[.[]|select(.status!="completed")]|length')
fixing=$(gh run list --workflow fix-segment.yml --json status -q '[.[]|select(.status!="completed")]|length')
active=$((active + fixing))
prep=$(gh run list --workflow prep-mini.yml --json status -q '[.[]|select(.status!="completed")]|length')
echo "pilot runs active: $active, prep-mini runs active: $prep"
[ "$active" != "0" ] && exit 0

runs=$(gh run list --workflow pilot.yml --limit 60 --json displayTitle,conclusion,status)
res_of() { echo "$runs" | python3 -c "
import json,sys
eng,res,tag=sys.argv[1],sys.argv[2],sys.argv[3]
want='pilot %s %s'%(eng,res)+((' '+tag) if tag else '')
r=[x for x in json.load(sys.stdin) if x['displayTitle']==want]
print('none' if not r else r[0]['conclusion'])" "$1" "$2" "$3"; }

launch() {  # engine res slug tag topic texw
  echo "launching $1 at $2p ($4 $5)"
  gh workflow run pilot.yml --ref dev -f engine="$1" -f res="$2" -f texw="$6" -f segments=40 -f crf=20 -f slug="$3" -f store=true -f tag="$4" -f topic="$5"
  exit 0
}
# step engine slug tag topic texw needs(file or -) keys...
step() {  # (a slug listed in state/chain_done.txt is skipped)
  local eng="$1" slug="$2" tag="$3" topic="$4" texw="$5" need="$6"; shift 6
  local total=$#
  local r4 r14
  if [ -f state/chain_done.txt ] && grep -qx "$slug${topic:+:$topic}" state/chain_done.txt; then echo "$slug done (state/chain_done.txt)"; return 0; fi
  r4=$(res_of "$eng" 2160 "$tag"); r14=$(res_of "$eng" 1440 "$tag")
  if [ "$r4" = "success" ] || [ "$r14" = "success" ]; then echo "$slug done"; return 0; fi
  if [ "$r4" = "failure" ] || [ "$r4" = "cancelled" ]; then
    if [ "$r14" = "none" ]; then launch "$eng" 1440 "$slug" "$tag" "$topic" "$texw"; fi
    echo "$slug failed at both resolutions: manual check needed"; return 0
  fi
  if [ "$need" != "-" ] && ! has_file "$need"; then echo "$slug: waiting for $need"; exit 0; fi
  local got; got=$(count "$@")
  if [ "$got" -lt $((total - 2)) ]; then
    [ "$prep" != "0" ] && { echo "waiting for data jobs"; exit 0; }
    missing=""; for k in "$@"; do [ "$(have "$k")" = "0" ] && missing="$missing $k"; done
    tries=$(gh run list --workflow prep-mini.yml --limit 50 --json displayTitle -q '[.[]|select(.displayTitle|test("osm"))]|length')
    if [ "$tries" -lt 8 ]; then echo "retry OSM for:$missing"; gh workflow run prep-mini.yml --ref dev -f what="osm$missing"; fi
    exit 0
  fi
  launch "$eng" 2160 "$slug" "$tag" "$topic" "$texw"
}
# v4 (owner 23.11): state names + crisp borders, short zooms every 30-40 s
step tornado us-tornadoes-1950-2025 v5 "" 4096 - $TORN
step earthquake california-earthquakes-1900-2025 v1 "" 6144 data/world/state_ca.png $QUAKE
step hurricane us-hurricane-landfalls-1851-2025 v4 "" 4096 - $HURR
step hurricane major-hurricanes-us major-v1 major 4096 - $HURR
step hurricane florida-hurricanes florida-v1 florida 4096 data/world/state_fl.png $FLOR
echo "all films built"
