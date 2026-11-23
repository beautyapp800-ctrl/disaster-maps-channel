"""Data + timeline for 'Every Hurricane That Hit America, Mapped (1851-2025)'.

Source: NOAA NHC HURDAT2 Atlantic best tracks 1851-2025 (public domain).
A storm counts when HURDAT2 has a landfall record ('L') with status HU (hurricane) on the coast of the lower 48.
Category = Saffir-Simpson category of the maximum sustained wind at that US landfall (the strongest one if several).
"""
from __future__ import annotations

import math
import os
from bisect import bisect_right
from functools import lru_cache

import numpy as np

from .core import ROOT

FPS = 24
TOPIC = os.environ.get("TOPIC") or "us"          # us | major (Cat 3+ at landfall) | florida (hurricane landfalls in Florida)
TOPIC_TITLE = {"us": "EVERY HURRICANE THAT HIT AMERICA, MAPPED", "major": "EVERY MAJOR HURRICANE TO HIT AMERICA, MAPPED",
               "florida": "EVERY HURRICANE TO HIT FLORIDA, MAPPED"}[TOPIC]
TOPIC_CENTER = {"us": None, "major": None, "florida": (-83.4, 28.0)}[TOPIC]
Y0, Y1 = 1851, 2025
YEAR_S = 1.7            # seconds of video per year
EV_S = 22.0             # famous landfall stop (national -> regional -> town -> storm close-up -> back)
EV_HIT = 7.0
EV_RUN = 8.0
SH_S, SH_HIT, SH_RUN = 10.0, 2.4, 5.4     # short zoom to another major landfall (owner 23.11: every 30-40 s)
INTRO_S, HEAT_S, BARS_S, FIN_S = 22.0, 22.0, 22.0, 24.0
CAT_COL = {1: (255, 214, 0), 2: (255, 150, 0), 3: (255, 70, 20), 4: (220, 0, 70), 5: (150, 0, 200), 0: (90, 190, 255)}
CAT_W = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]

# famous US landfalls: (key, storm name in HURDAT2, year, card title, landfall town for the close-up (lat, lon))
# place names to be checked on NHC / NWS pages and recorded in sources.json (event_names_hurricane)
EVENTS = [
    ("galveston1900", "AL011900", 1900, "GALVESTON, TEXAS", (29.30, -94.80)),
    ("laborday1935", "AL031935", 1935, "FLORIDA KEYS", (24.92, -80.63)),
    ("newengland1938", "AL061938", 1938, "LONG ISLAND & NEW ENGLAND", (40.8, -72.66)),
    ("camille1969", "CAMILLE", 1969, "CAMILLE · PASS CHRISTIAN, MISSISSIPPI", (30.318, -89.25)),
    ("andrew1992", "ANDREW", 1992, "ANDREW · HOMESTEAD, FLORIDA", (25.47, -80.45)),
    ("katrina2005", "KATRINA", 2005, "KATRINA · MISSISSIPPI COAST", (30.2866, -89.3762)),
    ("harvey2017", "HARVEY", 2017, "HARVEY · ROCKPORT, TEXAS", (28.02, -97.05)),
    ("michael2018", "MICHAEL", 2018, "MICHAEL · MEXICO BEACH, FLORIDA", (29.94, -85.41)),
    ("ian2022", "IAN", 2022, "IAN · FORT MYERS BEACH, FLORIDA", (26.452, -81.948)),
    ("helene2024", "HELENE", 2024, "HELENE · KEATON BEACH, FLORIDA", (29.82, -83.598)),
]


# flood level used for the close-up picture only (metres above USGS 3DEP ground, never shown as a number);
# conservative values below the peak storm tide in the NHC Tropical Cyclone Reports / NWS event pages (sources.json: surge_visual)
SURGE_M = {"galveston1900": 4.0, "laborday1935": 4.5, "newengland1938": 3.5, "camille1969": 6.0, "andrew1992": 2.0, "katrina2005": 7.0,
           "harvey2017": 3.0, "michael2018": 4.0, "ian2022": 3.5, "helene2024": 3.5}


# short zooms: places checked on NWS / NHC pages (sources.json event_names_hurricane); unnamed storms by HURDAT2 id
SHORTS = [
    ("indianola1886", "AL051886", 1886, "INDIANOLA, TEXAS", (28.530, -96.505)),
    ("seaislands1893", "AL061893", 1893, "SEA ISLANDS · BEAUFORT, SOUTH CAROLINA", (32.431, -80.670)),
    ("corpus1919", "AL021919", 1919, "CORPUS CHRISTI, TEXAS", (27.800, -97.396)),
    ("miami1926", "AL071926", 1926, "GREAT MIAMI HURRICANE · MIAMI, FLORIDA", (25.775, -80.190)),
    ("hazel1954", "HAZEL", 1954, "HAZEL · CALABASH, NORTH CAROLINA", (33.890, -78.560)),
    ("carla1961", "CARLA", 1961, "CARLA · PORT LAVACA, TEXAS", (28.615, -96.626)),
    ("eloise1975", "ELOISE", 1975, "ELOISE · PANAMA CITY BEACH, FLORIDA", (30.176, -85.806)),
    ("hugo1989", "HUGO", 1989, "HUGO · CHARLESTON, SOUTH CAROLINA", (32.770, -79.880)),
    ("charley2004", "CHARLEY", 2004, "CHARLEY · PUNTA GORDA, FLORIDA", (26.930, -82.045)),
    ("ike2008", "IKE", 2008, "IKE · BOLIVAR PENINSULA, TEXAS", (29.470, -94.630)),
]
SHORT_KEYS = {x[0] for x in SHORTS}
SURGE_SHORT = {"indianola1886": 3.5, "seaislands1893": 3.5, "corpus1919": 3.5, "miami1926": 2.5, "hazel1954": 4.0, "carla1961": 4.0,
               "eloise1975": 3.0, "hugo1989": 3.5, "charley2004": 2.0, "ike2008": 3.5}


def category(kt):
    return 5 if kt >= 137 else 4 if kt >= 113 else 3 if kt >= 96 else 2 if kt >= 83 else 1 if kt >= 64 else 0


@lru_cache(None)
def _land():
    """lower-48 coastline test in Albers (buffered 30 km so coastal landfall points count)"""
    from mapviz.core import albers, states
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely import prepared
    u = unary_union([Polygon(rg).buffer(0) for rings in states().values() for rg in rings])
    return prepared.prep(u.buffer(30 / 6371.0)), albers


def in_lower48(lon, lat):
    land, albers = _land()
    from shapely.geometry import Point
    x, y = albers(lon, lat)
    return land.contains(Point(float(x), float(y)))


class Storm:
    __slots__ = ("sid", "name", "year", "t", "lat", "lon", "wind", "status", "rec", "landfalls", "cat", "deaths")

    def __init__(self, sid, name):
        self.sid, self.name = sid, name.strip()
        self.year = int(sid[4:8])
        self.t, self.lat, self.lon, self.wind, self.status, self.rec = [], [], [], [], [], []
        self.landfalls = []


def _dt(date, hhmm):
    y, m, d = int(date[:4]), int(date[4:6]), int(date[6:8])
    import datetime as _d
    a = _d.datetime(y, m, d, int(hhmm[:2]), int(hhmm[2:]))
    b = _d.datetime(y, 1, 1)
    c = _d.datetime(y + 1, 1, 1)
    return y + (a - b).total_seconds() / (c - b).total_seconds()


class Data:
    def __init__(self, path=None):
        path = path or ROOT / "data" / "raw" / "hurdat2.txt"
        storms, cur = [], None
        for line in open(path, encoding="utf-8"):
            p = [x.strip() for x in line.split(",")]
            if p[0].startswith("AL") and len(p) >= 3 and not p[0][2:].isdigit() is False and len(p[0]) == 8:
                cur = Storm(p[0], p[1])
                storms.append(cur)
                continue
            if cur is None or len(p) < 7:
                continue
            lat = float(p[4][:-1]) * (1 if p[4].endswith("N") else -1)
            lon = float(p[5][:-1]) * (-1 if p[5].endswith("W") else 1)
            cur.t.append(_dt(p[0], p[1]))
            cur.lat.append(lat)
            cur.lon.append(lon)
            cur.wind.append(int(p[6]))
            cur.status.append(p[3])
            cur.rec.append(p[2])
        self.all = storms
        hit = []
        for s in storms:
            for k, r in enumerate(s.rec):
                if r == "L" and s.status[k] == "HU" and in_lower48(s.lon[k], s.lat[k]):
                    s.landfalls.append(k)
            if s.landfalls:
                s.cat = max(category(s.wind[k]) for k in s.landfalls)
                for a in ("t", "lat", "lon", "wind"):
                    setattr(s, a, np.array(getattr(s, a)))
                hit.append(s)
        if TOPIC == "major":
            hit = [s for s in hit if s.cat >= 3]
        elif TOPIC == "florida":
            keep = []
            for s in hit:
                s.landfalls = [k for k in s.landfalls if landfall_state(s.lon[k], s.lat[k]) == "FL"]
                if s.landfalls:
                    s.cat = max(category(s.wind[k]) for k in s.landfalls)
                    keep.append(s)
            hit = keep
        self.storms = sorted(hit, key=lambda s: s.t[s.landfalls[0]])
        self.n = len(self.storms)
        self.cat = np.array([s.cat for s in self.storms])
        self.tl = np.array([s.t[s.landfalls[0]] for s in self.storms])      # decimal year of first US landfall
        self.yr = np.array([s.year for s in self.storms])
        # famous events
        self.events = []
        for key, name, year, title, town in EVENTS + SHORTS:
            c = [i for i, s in enumerate(self.storms) if s.year == year and (s.sid == name if name.startswith("AL") else s.name == name)]
            if name == "UNNAMED" and len(c) > 1:
                c = sorted(c, key=lambda i: -self.storms[i].cat)
            if not c and TOPIC != "us":
                continue                      # this storm is not part of the topic
            if TOPIC == "florida" and landfall_state(town[1], town[0]) != "FL":
                continue
            if not c:
                raise ValueError(f"event {key} not found in HURDAT2 landfalls")
            i = c[0]
            s = self.storms[i]
            k = min(s.landfalls, key=lambda k: (s.lat[k] - town[0]) ** 2 + (s.lon[k] - town[1]) ** 2)
            self.events.append(dict(key=key, idx=i, title=title, town=town, lf=k, kt=int(s.wind[k]), cat=category(s.wind[k]),
                                    date=s.t[k], surge_m=SURGE_M.get(key, SURGE_SHORT.get(key, 3.0)), short=key in SHORT_KEYS,
                                    S=SH_S if key in SHORT_KEYS else EV_S, HIT=SH_HIT if key in SHORT_KEYS else EV_HIT,
                                    RUN=SH_RUN if key in SHORT_KEYS else EV_RUN))
        self.events.sort(key=lambda e: e["date"])

    def count_by_cat(self, mask=None):
        c = self.cat if mask is None else self.cat[mask]
        return {k: int((c == k).sum()) for k in (1, 2, 3, 4, 5)}


if __name__ == "__main__":
    D = Data()
    print("storms in HURDAT2:", len(D.all), " US hurricane landfalls (lower 48):", D.n, D.count_by_cat())
    for e in D.events:
        s = D.storms[e["idx"]]
        k = e["lf"]
        print(e["key"], s.sid, s.name, f"{s.lat[k]:.1f},{s.lon[k]:.1f}", e["kt"], "kt cat", e["cat"])


def fmt(n):
    return f"{int(n):,}"


def smooth01(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


STATE_OF = {}


def landfall_state(lon, lat):
    """state whose polygon is nearest to the landfall point (Census shapes)"""
    from mapviz.core import albers, states
    from shapely.geometry import Point, Polygon
    if not STATE_OF:
        for ab, rings in states().items():
            STATE_OF[ab] = [Polygon(rg).buffer(0) for rg in rings]
    x, y = albers(lon, lat)
    p = Point(float(x), float(y))
    return min(STATE_OF, key=lambda ab: min(g.distance(p) for g in STATE_OF[ab]))


class Timeline:
    """phases with t0/t1/mode; ta[i] = video second when storm i starts to be drawn; anim[i] = seconds its track takes;
    tl[i] = video second of its first US landfall (the counter ticks then)."""

    ANIM = 2.4

    def __init__(self, D: Data):
        self.D = D
        n = D.n
        ph = [dict(kind="intro", mode="intro", t0=0.0, t1=INTRO_S)]
        t = INTRO_S
        land_t = np.zeros(n)
        self.year_t0 = {}
        evs = sorted(D.events, key=lambda e: e["date"])
        ei = 0
        for yr in range(Y0, Y1 + 1):
            self.year_t0[yr] = t
            cur = float(yr)
            while ei < len(evs) and evs[ei]["date"] < yr + 1:
                e = evs[ei]
                dte = float(e["date"])
                dur = (dte - cur) * YEAR_S
                ph.append(dict(kind="run", mode="main", t0=t, t1=t + dur, a=cur, b=dte))
                t += dur
                e["t0"] = t
                ph.append(dict(kind="event", mode="main", t0=t, t1=t + e["S"], ev=e, a=dte, b=dte))
                t += e["S"]
                cur = dte
                ei += 1
            dur = (yr + 1 - cur) * YEAR_S
            ph.append(dict(kind="run", mode="main", t0=t, t1=t + dur, a=cur, b=float(yr + 1)))
            t += dur
        self.main_end = t
        for kind, d in (("heat", HEAT_S), ("bars", BARS_S), ("final", FIN_S)):
            ph.append(dict(kind=kind, mode=kind, t0=t, t1=t + d))
            t += d
        self.total = t
        self.phases = ph
        self._starts = [p["t0"] for p in ph]
        ev_by = {e["idx"]: e for e in D.events}
        for i, s in enumerate(D.storms):
            if i in ev_by:
                land_t[i] = ev_by[i]["t0"] + ev_by[i]["HIT"] + ev_by[i]["RUN"] * 0.5
            else:
                land_t[i] = self.video_t(float(D.tl[i]))
        self.tl = land_t
        self.anim = np.full(n, self.ANIM)
        self.flf = np.array([s.landfalls[0] / max(1, len(s.t) - 1) for s in D.storms])
        for i, e in ev_by.items():
            self.anim[i] = e["S"] - 3.0
            self.flf[i] = (e["HIT"] + e["RUN"] * 0.5 - 1.5) / self.anim[i]
        self.ta = self.tl - self.anim * self.flf
        order = np.argsort(self.tl, kind="stable")
        self.tl_sorted = self.tl[order]
        self.cat_sorted = D.cat[order]
        self.cum_year = np.array([int((D.yr <= y).sum()) for y in range(Y0, Y1 + 1)])
        self.captions = self._captions()

    def video_t(self, dec_year):
        """video second for a decimal year in the main run (between phases)"""
        for p in self.phases:
            if p["mode"] == "main" and p["kind"] == "run" and p["a"] <= dec_year <= p["b"] + 1e-9:
                return p["t0"] + (dec_year - p["a"]) / max(p["b"] - p["a"], 1e-9) * (p["t1"] - p["t0"])
        return self.main_end

    def phase_at(self, t):
        return self.phases[min(len(self.phases) - 1, max(0, bisect_right(self._starts, t) - 1))]

    def dt_at(self, t):
        p = self.phase_at(t)
        if p["mode"] == "main":
            return p["a"] + (p["b"] - p["a"]) * ((t - p["t0"]) / max(p["t1"] - p["t0"], 1e-9) if p["kind"] == "run" else 0.0)
        return float(Y1 + 1) if p["mode"] != "intro" else float(Y0)

    def stats(self, t):
        if self.phase_at(t)["mode"] == "intro":
            return 0, 0, {c: 0 for c in (1, 2, 3, 4, 5)}
        k = int(np.searchsorted(self.tl_sorted, t, "right"))
        c = self.cat_sorted[:k]
        cats = {x: int((c == x).sum()) for x in (1, 2, 3, 4, 5)}
        return k, int((c >= 3).sum()), cats

    def heat_year(self, t):
        p = self.phase_at(t)
        return smooth01((t - p["t0"] - 1.0) / (p["t1"] - p["t0"] - 3.0)) * (Y1 - Y0 + 1)

    def _captions(self):
        D = self.D
        per_year = {y: int((D.yr == y).sum()) for y in range(Y0, Y1 + 1)}
        busiest = max(per_year, key=lambda y: (per_year[y], -y))
        five = [D.storms[i] for i in np.where(D.cat == 5)[0]]
        states = {}
        for s in D.storms:
            k = s.landfalls[0]
            st = landfall_state(s.lon[k], s.lat[k])
            states[st] = states.get(st, 0) + 1
        self.state_counts = states
        top = max(states, key=states.get)
        spec = [
            (self.year_t0[1851] + 0.3, "EACH LINE IS ONE HURRICANE TRACK, COLOURED BY STRENGTH"),
            (self.year_t0[1880] + 0.3, "BEFORE SATELLITES, TRACKS COME FROM SHIP LOGS AND COASTAL REPORTS"),
            (self.year_t0[busiest] + 0.3, f"{busiest}: {per_year[busiest]} HURRICANES CAME ASHORE, THE MOST OF ANY YEAR ON THIS MAP"),
            (self.year_t0[1966] + 0.3, "THE 1960S: WEATHER SATELLITES START WATCHING EVERY STORM"),
            (self.year_t0[2000] + 0.3, (f"ONLY {len(five)} HURRICANES HIT FLORIDA AT CATEGORY 5" if TOPIC == "florida"
                                       else f"ONLY {len(five)} HURRICANES HIT THE MAINLAND AT CATEGORY 5")),
        ]
        decs = []
        for d0 in range(1860, 2030, 20):
            n = int(((D.yr >= d0) & (D.yr < d0 + 20)).sum())
            decs.append((self.year_t0[min(d0, Y1)] + 0.3, f"{d0}-{min(d0 + 19, Y1)}: {n} HURRICANE LANDFALLS"))
        ev_win = [(p["t0"] - 0.6, p["t1"] + 0.6) for p in self.phases if p["kind"] == "event"]
        out = []

        def free(a, b):
            if any(a < e1 and b > e0 for e0, e1 in ev_win):
                return False
            return not any(a < c[1] + 1.0 and b > c[0] - 1.0 for c in out)
        for lst in (spec, decs):
            for t0, text in lst:
                cand = t0
                for _ in range(60):
                    if free(cand, cand + 5.0) and cand + 5.0 <= self.main_end:
                        out.append((cand, cand + 5.0, text))
                        break
                    cand += 0.5
        out.append((self.main_end + 1.5, self.main_end + 6.5, "WHERE HURRICANES COME ASHORE MOST OFTEN"))
        b0 = self.main_end + HEAT_S
        from .tornado import STATE_NAMES
        out.append((b0 + 8.0, b0 + 14.0, f"{STATE_NAMES[top]}: {states[top]} OF THE {D.n} LANDFALLS, THE MOST OF ANY STATE"))
        out.sort()
        self.facts = dict(busiest=busiest, busiest_n=per_year[busiest], cat5=[s.name + " " + str(s.year) for s in five], top_state=top, top_n=states[top])
        return out

    def caption_at(self, t):
        for a, b, s in self.captions:
            if a <= t < b:
                return s, min(1.0, (t - a) / 0.4, (b - t) / 0.4)
        return None, 0.0
