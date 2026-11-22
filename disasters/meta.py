"""YouTube metadata for a built video (plain-text description: no links, no angle brackets, <= 5000 characters)."""
from __future__ import annotations

import json
import re
import sys

from .tornado import Data, fmt

CHANNEL_REF = "everydisastermapped"


def tornado_us(slug):
    D = Data()
    title = f"Every Tornado in US History Mapped: {fmt(D.n)} Tornadoes (1950-2025)"
    famous = ", ".join(f"{e['title'].title().replace(', ', ' ').replace('Ef', 'EF')} {D.yr[e['idx']]}" for e in D.events)
    music = json.load(open("data/music/tracks.json", encoding="utf-8"))["tracks"]
    desc = (f"Every tornado recorded in the United States from 1950 to 2025, mapped path by path from the NOAA Storm Prediction Center database: "
            f"{fmt(D.n)} tornadoes, {fmt(int(D.fat.sum()))} lives lost, and the {int((D.mag == 5).sum())} strongest rated F5 or EF5. "
            f"Up close: {famous}.\n\n"
            "Sources (public data):\n"
            "Tornado paths, ratings and casualties: NOAA / NWS Storm Prediction Center severe weather database (spc.noaa.gov/wcm)\n"
            "Place names of the famous tornadoes: National Weather Service event pages (weather.gov)\n"
            "Satellite colour and night lights: NASA Blue Marble Next Generation and Black Marble (NASA Earth Observatory, NASA GIBS)\n"
            "Close-up imagery: USGS The National Map orthoimagery (NAIP)\n"
            "Buildings: (c) OpenStreetMap contributors, ODbL (openstreetmap.org/copyright)\n"
            "Relief: NOAA ETOPO1. Borders: US Census Bureau, Natural Earth.\n"
            "Music (CC0, OpenGameArt): " + "; ".join(f"{t['title']} by {t['creator']}" for t in music) + ". Sound effects made for this video.\n"
            "2025 data is preliminary. Paths are drawn as straight lines between the official start and end points.\n\n"
            "#tornado #tornadoalley #weather #map #NOAA\n"
            f"[ref: {CHANNEL_REF}/{slug}]")
    tags = ["tornado", "tornadoes", "every tornado", "tornado map", "tornado history", "tornado alley", "F5 tornado", "EF5", "Joplin tornado",
            "Moore tornado", "NOAA", "Storm Prediction Center", "weather history", "map animation", "natural disasters"]
    return dict(title=title, description=desc, tags=tags, categoryId="27", madeForKids=False, containsSyntheticMedia=False, defaultLanguage="en")


def problems(m):
    p = []
    if len(m["title"]) > 70:
        p.append(f"title {len(m['title'])} chars")
    d = m["description"]
    if re.search(r"https?://", d) or "<" in d or ">" in d:
        p.append("link or angle bracket in description")
    if len(d.encode("utf-8")) > 5000:
        p.append("description over 5000 bytes")
    if not (3 <= len(re.findall(r"#\w+", d)) <= 5):
        p.append("hashtags not 3-5")
    return p


if __name__ == "__main__":
    slug = sys.argv[1]
    m = {"us-tornadoes-1950-2025": tornado_us}[slug](slug)
    pr = problems(m)
    print(json.dumps(m, indent=1, ensure_ascii=False))
    if pr:
        print("PROBLEMS:", pr)
        sys.exit(1)
