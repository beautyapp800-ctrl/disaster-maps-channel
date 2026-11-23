"""YouTube metadata for a built video (plain-text description: no links, no angle brackets, <= 5000 characters)."""
from __future__ import annotations

import json
import re
import sys

from .tornado import Data, fmt

CHANNEL_REF = "stormatlasusa"


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
            "2025 data is preliminary. Paths are drawn as straight lines between the official start and end points. In the town close-ups the path is placed through the town named on the card.\n\n"
            "#tornado #tornadoalley #weather #map #NOAA\n"
            f"[ref: {CHANNEL_REF}/{slug}]")
    tags = ["tornado", "tornadoes", "every tornado", "tornado map", "tornado history", "tornado alley", "F5 tornado", "EF5", "Joplin tornado",
            "Moore tornado", "NOAA", "Storm Prediction Center", "weather history", "map animation", "natural disasters"]
    return dict(title=title, description=desc, tags=tags, categoryId="27", madeForKids=False, containsSyntheticMedia=False, defaultLanguage="en")


def hurricanes_us(slug, topic="us"):
    import os
    os.environ["TOPIC"] = topic
    from .hurricane import Data as HData
    D = HData()
    n5 = int((D.cat == 5).sum())
    nmaj = int((D.cat >= 3).sum())
    word = {"us": "Every Hurricane That Hit America", "major": "Every Major Hurricane to Hit the US", "florida": "Every Hurricane to Hit Florida"}[topic]
    unit = {"us": "Hurricanes", "major": "Storms", "florida": "Landfalls"}[topic]
    title = f"{word}, Mapped: {D.n} {unit} (1851-2025)"
    famous = ", ".join(f"{e['title'].title().replace(' · ', ' ')} {int(e['date'])}" for e in D.events)
    music = json.load(open("data/music/tracks.json", encoding="utf-8"))["tracks"]
    scope = {"us": "came ashore on the US mainland", "major": "came ashore on the US mainland as a major hurricane (Category 3 or stronger)",
             "florida": "came ashore in Florida"}[topic]
    desc = (f"Every hurricane that {scope} from 1851 to 2025, mapped track by track from the NOAA National Hurricane Center "
            f"best-track database (HURDAT2): {D.n} {'hurricanes' if topic == 'us' else 'storms' if topic == 'major' else 'hurricane landfalls'}, "
            f"{nmaj} of them major (Category 3 to 5) at landfall, and only {n5} at Category 5.\n"
            f"Up close: {famous}.\n\n"
            "How the map counts: a storm is included when HURDAT2 records a landfall while it was a hurricane" +
            {"us": " on the lower 48 states", "major": " on the lower 48 states and its strongest US landfall was Category 3 or stronger",
             "florida": " in Florida (a storm that hit Florida twice counts once)"}[topic] +
            "; the category is the strongest one at a landfall counted. Storm-surge flooding in the close-ups is illustrative.\n\n"
            "Sources (public data):\n"
            "Hurricane tracks and winds: NOAA National Hurricane Center, HURDAT2 Atlantic best-track data (nhc.noaa.gov/data)\n"
            "Land cover: USGS \nCD 2021 (MRLC). Elevation: USGS 3DEP and NOAA ETOPO1. Outside the US: NASA Blue Marble Next Generation (NASA GIBS)\n"
            "Towns, streets, buildings and place names: (c) OpenStreetMap contributors, ODbL (openstreetmap.org/copyright). Today's streets, shown in illustrative 3D.\n"
            "Borders: US Census Bureau, Natural Earth.\n"
            "Music (CC0, OpenGameArt): " + "; ".join(f"{t['title']} by {t['creator']}" for t in music) + ". Sound effects made for this video.\n\n"
            "#hurricane #weather #map #NOAA #history\n"
            f"[ref: {CHANNEL_REF}/{slug}]")
    tags = ["hurricane", "hurricanes", "every hurricane", "hurricane map", "hurricane history", "Katrina", "Andrew", "Camille", "Michael",
            "Ian", "Helene", "storm surge", "NOAA", "National Hurricane Center", "map animation"]
    if topic == "florida":
        tags = ["Florida hurricanes", "Florida"] + tags
    if topic == "major":
        tags = ["major hurricanes", "Category 5 hurricane"] + tags
    return dict(title=title, description=desc, tags=tags, categoryId="27", madeForKids=False, containsSyntheticMedia=False, defaultLanguage="en")


def earthquakes_ca(slug):
    from .quake import Data as QData, Timeline as QTimeline
    D = QData()
    TL = QTimeline(D)
    cc = D.count_by_class()
    title = f"Every Earthquake in California Since 1900, Mapped: {fmt(D.n)} Quakes"
    famous = ", ".join(f"{e['title'].split(' · ')[-1].title().replace(', California', '')} {e['local'].year}" for e in sorted(D.events, key=lambda e: e["idx"]))
    music = json.load(open("data/music/tracks.json", encoding="utf-8"))["tracks"]
    desc = (f"Every earthquake of magnitude 4.0 or more in California and just off its coast from 1900 to 2025, mapped one by one from the USGS ANSS ComCat catalogue: "
            f"{fmt(D.n)} earthquakes, {cc[7]} of them magnitude 7 or more and {cc[6]} between 6 and 7. The largest: magnitude {D.mag.max():.1f}, San Francisco, 1906.\n"
            f"Up close: {famous}.\n\n"
            "How the map counts: an earthquake is included when the catalogue places it inside California or offshore within about 130 km of the coast. "
            "Small earthquakes before the 1930s are missing from any catalogue, and modern networks find many more, so the early years look quieter than they were. "
            "In the town close-ups the shaking is centred on the town named on the card; damage is illustrative.\n\n"
            "Sources (public data):\n"
            "Earthquakes: USGS Earthquake Hazards Program, ANSS Comprehensive Earthquake Catalog (earthquake.usgs.gov/fdsnws/event)\n"
            "Land cover: USGS \nCD 2021 (MRLC). Elevation and hillshade: USGS 3DEP and NOAA ETOPO1. Outside the US: NASA Blue Marble Next Generation (NASA GIBS)\n"
            "Towns, streets, buildings and place names: (c) OpenStreetMap contributors, ODbL (openstreetmap.org/copyright). Today's streets, shown in illustrative 3D.\n"
            "Borders: US Census Bureau, Natural Earth.\n"
            "Music (CC0, OpenGameArt): " + "; ".join(f"{t['title']} by {t['creator']}" for t in music) + ". Sound effects made for this video.\n\n"
            "#earthquake #california #map #USGS #history\n"
            f"[ref: {CHANNEL_REF}/{slug}]")
    tags = ["earthquake", "earthquakes", "every earthquake", "California earthquakes", "earthquake map", "earthquake history", "San Andreas fault",
            "1906 San Francisco earthquake", "Northridge earthquake", "Loma Prieta earthquake", "Ridgecrest earthquake", "USGS", "seismology", "map animation"]
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
    m = {"us-tornadoes-1950-2025": tornado_us, "us-hurricane-landfalls-1851-2025": hurricanes_us,
         "major-hurricanes-us": lambda s: hurricanes_us(s, "major"), "florida-hurricanes": lambda s: hurricanes_us(s, "florida"),
         "california-earthquakes-1900-2025": earthquakes_ca}[slug](slug)
    pr = problems(m)
    print(json.dumps(m, indent=1, ensure_ascii=False))
    if pr:
        print("PROBLEMS:", pr)
        sys.exit(1)
