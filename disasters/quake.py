"""Data + timeline for 'Every Earthquake in California Since 1900, Mapped'.

Source: USGS ANSS ComCat (earthquake.usgs.gov/fdsnws/event/1), M4.0+ in the box 31.5-42.6 N, 125.5-113.5 W, 1900-01-01 .. 2025-12-31.
An event counts when it lies inside California or offshore within OFFSHORE_KM of the California coast, north of 32.6 N, and not inside
another US state (Census cartographic boundaries); Baja California and the Gulf of California are excluded.
Magnitude classes: M4.0-4.9, M5.0-5.9, M6.0-6.9, M7.0+. Dates of the close-ups are local (Pacific) dates.
"""
from __future__ import annotations

import csv
from bisect import bisect_right
from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np

from .core import ROOT

FPS = 24
Y0, Y1 = 1900, 2025
YEAR_S = 2.1
EV_S, EV_HIT, EV_RUN = 22.0, 7.0, 8.0
SH_S, SH_HIT, SH_RUN = 10.0, 2.4, 5.4
INTRO_S, HEAT_S, LIST_S, FIN_S = 22.0, 22.0, 22.0, 24.0
BARS_S = LIST_S
OFFSHORE_KM = 130.0
CLASS_COL = {4: (255, 214, 0), 5: (255, 140, 0), 6: (230, 30, 50), 7: (150, 0, 200)}
CLASS_W = {4: 2, 5: 3, 6: 5, 7: 8}
CLASS_NAME = {4: "M4.0 - 4.9", 5: "M5.0 - 5.9", 6: "M6.0 - 6.9", 7: "M7.0 OR MORE"}
MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
CSV = ROOT / "data" / "raw" / "quakes" / "ca_m4.csv"
URL = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=1900-01-01&endtime=2025-12-31&minmagnitude=4"
       "&minlatitude=31.5&maxlatitude=42.6&minlongitude=-125.5&maxlongitude=-113.5&orderby=time-asc&limit=20000")

# main stops (long) and short zooms; ids are ComCat ids, places checked on USGS pages (sources.json event_names_quake)
EVENTS = [
    ("sanfrancisco1906", "official19060418131226300_12", "SAN FRANCISCO, CALIFORNIA", (37.7749, -122.4194)),
    ("longbeach1933", "ci3359741", "LONG BEACH, CALIFORNIA", (33.7701, -118.1937)),
    ("tehachapi1952", "ci3319401", "KERN COUNTY · TEHACHAPI, CALIFORNIA", (35.1322, -118.4490)),
    ("sylmar1971", "ci3347678", "SAN FERNANDO · SYLMAR, CALIFORNIA", (34.3074, -118.4490)),
    ("santacruz1989", "nc216859", "LOMA PRIETA · SANTA CRUZ, CALIFORNIA", (36.9741, -122.0308)),
    ("yuccavalley1992", "ci3031111", "LANDERS · YUCCA VALLEY, CALIFORNIA", (34.1145, -116.4322)),
    ("northridge1994", "ci3144585", "NORTHRIDGE, CALIFORNIA", (34.2283, -118.5368)),
    ("ridgecrest2019", "ID_RIDGECREST", "RIDGECREST, CALIFORNIA", (35.6225, -117.6709)),
]
SHORTS = [
    ("santabarbara1925", "ci37161284", "SANTA BARBARA, CALIFORNIA", (34.4208, -119.6982)),
    ("imperial1940", "ci3365279", "IMPERIAL VALLEY · IMPERIAL, CALIFORNIA", (32.8476, -115.5694)),
    ("coalinga1983", "nc1091100", "COALINGA, CALIFORNIA", (36.1397, -120.3601)),
    ("ferndale1992", "nc269151", "PETROLIA QUAKE · FERNDALE, CALIFORNIA", (40.5765, -124.2634)),
    ("pasorobles2003", "nc21323712", "SAN SIMEON · PASO ROBLES, CALIFORNIA", (35.6264, -120.6905)),
    ("napa2014", "ID_NAPA", "SOUTH NAPA · NAPA, CALIFORNIA", (38.2975, -122.2869)),
    ("riodell2022", "nc73821036", "FERNDALE QUAKE · RIO DELL, CALIFORNIA", (40.4986, -124.1058)),
]
SHORT_KEYS = {x[0] for x in SHORTS}
ALL_KEYS = [x[0] for x in EVENTS + SHORTS]
# matches by date (UTC) and magnitude where the id is not known for sure
MATCH = {"ID_RIDGECREST": ("2019-07-06", 7.0, 7.2), "ID_NAPA": ("2014-08-24", 5.9, 6.1)}
PACIFIC = ZoneInfo("America/Los_Angeles")


def category(m):
    return 7 if m >= 7.0 else 6 if m >= 6.0 else 5 if m >= 5.0 else 4


@lru_cache(None)
def _geo():
    """prepared polygons in Albers: California, and the union of all other states"""
    from mapviz.core import albers, states
    from shapely import prepared
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    ca = unary_union([Polygon(r).buffer(0) for r in states()["CA"]])
    other = unary_union([Polygon(r).buffer(0) for ab, rings in states().items() if ab != "CA" for r in rings])
    return ca, prepared.prep(ca), prepared.prep(other), prepared.prep(ca.buffer(OFFSHORE_KM / 6371.0)), albers


def in_california(lon, lat):
    from shapely.geometry import Point
    ca, pca, pother, pbuf, albers = _geo()
    x, y = albers(lon, lat)
    p = Point(float(x), float(y))
    if pca.contains(p):
        return True
    if lat < 32.6 or pother.contains(p):
        return False
    return bool(pbuf.contains(p))


def read_rows(path=None):
    out = []
    for r in csv.DictReader(open(path or CSV, encoding="utf-8")):
        if r["type"] != "earthquake" or not r["mag"]:
            continue
        out.append(r)
    return out


class Data:
    def __init__(self, path=None):
        rows = read_rows(path)
        keep = []
        for r in rows:
            lat, lon = float(r["latitude"]), float(r["longitude"])
            if in_california(lon, lat):
                keep.append(r)
        t, mag, lat, lon, dep, place, eid, loc = [], [], [], [], [], [], [], []
        for r in keep:
            d = datetime.strptime(r["time"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            y0, y1 = datetime(d.year, 1, 1, tzinfo=timezone.utc), datetime(d.year + 1, 1, 1, tzinfo=timezone.utc)
            t.append(d.year + (d - y0).total_seconds() / (y1 - y0).total_seconds())
            mag.append(float(r["mag"]))
            lat.append(float(r["latitude"]))
            lon.append(float(r["longitude"]))
            dep.append(float(r["depth"]) if r["depth"] else float("nan"))
            place.append(r["place"])
            eid.append(r["id"])
            loc.append(d.astimezone(PACIFIC))
        order = np.argsort(np.array(t), kind="stable")
        pick = lambda a: [a[i] for i in order]  # noqa: E731
        self.dt = np.array(pick(t))
        self.mag = np.array(pick(mag))
        self.lat, self.lon, self.depth = np.array(pick(lat)), np.array(pick(lon)), np.array(pick(dep))
        self.place, self.id, self.local = pick(place), pick(eid), pick(loc)
        self.yr = np.array([int(x) for x in np.floor(self.dt)])
        self.cls = np.array([category(m) for m in self.mag])
        self.n = len(self.dt)
        self.events = []
        for key, eid_, title, town in EVENTS + SHORTS:
            if eid_ in MATCH:
                day, lo, hi = MATCH[eid_]
                c = [i for i in range(self.n) if self.local[i].astimezone(timezone.utc).strftime("%Y-%m-%d") == day and lo <= self.mag[i] <= hi]
                if not c:
                    raise ValueError(f"event {key} not found")
                i = max(c, key=lambda i: self.mag[i])
            else:
                if eid_ not in self.id:
                    raise ValueError(f"event {key} ({eid_}) not in the filtered catalogue")
                i = self.id.index(eid_)
            short = key in SHORT_KEYS
            self.events.append(dict(key=key, idx=i, title=title, town=town, short=short, mag=float(self.mag[i]), depth=float(self.depth[i]),
                                    date=float(self.dt[i]), local=self.local[i],
                                    S=SH_S if short else EV_S, HIT=SH_HIT if short else EV_HIT, RUN=SH_RUN if short else EV_RUN))
        self.events.sort(key=lambda e: e["idx"])

    def count_by_class(self, upto=None):
        c = self.cls if upto is None else self.cls[:upto]
        return {k: int((c == k).sum()) for k in (4, 5, 6, 7)}


def fmt(n):
    return f"{int(n):,}"


def smooth01(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Timeline:
    """phases (intro, main run/event, heat, list, final); ta[i] = video second when quake i appears; anim = seconds its pulse lasts."""

    def __init__(self, D: Data):
        self.D = D
        n = D.n
        ph = [dict(kind="intro", mode="intro", t0=0.0, t1=INTRO_S)]
        t = INTRO_S
        ta = np.zeros(n)
        self.year_t0 = {}
        ev_by = {e["idx"]: e for e in D.events}
        order = sorted(D.events, key=lambda e: e["date"])
        ei = 0

        def run(i0, i1, a, b, t0, dur):
            if i1 > i0:
                ta[i0:i1] = t0 + (D.dt[i0:i1] - a) / max(b - a, 1e-9) * dur

        for yr in range(Y0, Y1 + 1):
            lo = int(np.searchsorted(D.dt, yr, "left"))
            hi = int(np.searchsorted(D.dt, yr + 1, "left"))
            self.year_t0[yr] = t
            cur, cur_i = float(yr), lo
            while ei < len(order) and order[ei]["date"] < yr + 1:
                e = order[ei]
                dte = float(D.dt[e["idx"]])
                dur = (dte - cur) * YEAR_S
                ph.append(dict(kind="run", mode="main", t0=t, t1=t + dur, a=cur, b=dte))
                run(cur_i, e["idx"], cur, dte, t, dur)
                t += dur
                e["t0"] = t
                ph.append(dict(kind="event", mode="main", t0=t, t1=t + e["S"], ev=e, a=dte, b=dte))
                ta[e["idx"]] = t + e["HIT"]
                t += e["S"]
                cur, cur_i = dte, e["idx"] + 1
                ei += 1
            dur = (yr + 1 - cur) * YEAR_S
            ph.append(dict(kind="run", mode="main", t0=t, t1=t + dur, a=cur, b=float(yr + 1)))
            run(cur_i, hi, cur, yr + 1, t, dur)
            t += dur
        self.main_end = t
        for kind, d in (("heat", HEAT_S), ("bars", LIST_S), ("final", FIN_S)):
            ph.append(dict(kind=kind, mode=kind, t0=t, t1=t + d))
            t += d
        self.total = t
        self.phases = ph
        self._starts = [p["t0"] for p in ph]
        self.ta = ta
        assert np.all(np.diff(ta) >= -1e-9), "appearance times must be monotone"
        self.anim = np.where(D.mag >= 7.0, 3.4, np.where(D.mag >= 6.0, 2.6, np.where(D.mag >= 5.0, 1.8, 1.2)))
        self.cls_ta = {c: ta[D.cls == c] for c in (4, 5, 6, 7)}
        self.big = np.where(D.mag >= 6.5)[0]
        self.intro_ta = 2.0 + np.arange(len(self.big)) * 0.5
        self.cum_year = np.array([int((D.yr <= y).sum()) for y in range(Y0, Y1 + 1)])
        self.cummax = np.maximum.accumulate(D.mag)
        self.captions = self._captions()

    def phase_at(self, t):
        return self.phases[min(len(self.phases) - 1, max(0, bisect_right(self._starts, t) - 1))]

    def dt_at(self, t):
        p = self.phase_at(t)
        if p["mode"] == "main":
            return p["a"] + (p["b"] - p["a"]) * ((t - p["t0"]) / max(p["t1"] - p["t0"], 1e-9) if p["kind"] == "run" else 0.0)
        return float(Y1 + 1) if p["mode"] != "intro" else float(Y0)

    def stats(self, t):
        if self.phase_at(t)["mode"] == "intro":
            return 0, 0.0, {c: 0 for c in (4, 5, 6, 7)}
        k = int(np.searchsorted(self.ta, t, "right"))
        cls = {c: int(np.searchsorted(a, t, "right")) for c, a in self.cls_ta.items()}
        return k, (float(self.cummax[k - 1]) if k else 0.0), cls

    def heat_year(self, t):
        p = self.phase_at(t)
        return smooth01((t - p["t0"] - 1.0) / (p["t1"] - p["t0"] - 3.0)) * (Y1 - Y0 + 1)

    def intro_count(self, t):
        return int(np.searchsorted(self.intro_ta, t, "right"))

    def _captions(self):
        D = self.D
        per = {y: int((D.yr == y).sum()) for y in range(Y0, Y1 + 1)}
        busiest = max(per, key=lambda y: (per[y], -y))
        big = int((D.mag >= 7.0).sum())
        spec = [
            (2.0, f"THE {len(self.big)} STRONGEST CALIFORNIA EARTHQUAKES: MAGNITUDE 6.5 AND UP"),
            (self.year_t0[1900] + 0.3, "1900: THE CATALOGUE STARTS. SMALL QUAKES BEFORE THE 1930S ARE MISSING"),
            (self.year_t0[1932] + 0.3, "1932: SEISMOGRAPH NETWORKS START COUNTING SMALLER EARTHQUAKES"),
            (self.year_t0[1960] + 0.3, "EACH DOT IS ONE EARTHQUAKE, SIZED AND COLOURED BY MAGNITUDE"),
            (self.year_t0[1975] + 0.3, "MODERN NETWORKS FIND FAR MORE SMALL EARTHQUAKES THAN BEFORE"),
            (self.year_t0[busiest] + 0.3, f"{busiest}: {per[busiest]} EARTHQUAKES OF M4 OR MORE, THE BUSIEST YEAR ON THIS MAP"),
            (self.year_t0[2010] + 0.3, f"ONLY {big} CALIFORNIA EARTHQUAKES SINCE 1900 REACHED MAGNITUDE 7"),
        ]
        decs = []
        for d0 in range(1900, 2030, 20):
            n = int(((D.yr >= d0) & (D.yr < d0 + 20)).sum())
            decs.append((self.year_t0[min(d0, Y1)] + 0.3, f"{d0}-{min(d0 + 19, Y1)}: {fmt(n)} EARTHQUAKES OF M4 OR MORE"))
        ev_win = [(p["t0"] - 0.6, p["t1"] + 0.6) for p in self.phases if p["kind"] == "event"]
        out = []

        def free(a, b):
            if any(a < e1 and b > e0 for e0, e1 in ev_win):
                return False
            return not any(a < c[1] + 1.0 and b > c[0] - 1.0 for c in out)
        for lst in (spec, decs):
            for t0, text in lst:
                cand = t0
                for _ in range(80):
                    if free(cand, cand + 5.0) and cand + 5.0 <= self.main_end + (30 if lst is spec else 0):
                        out.append((cand, cand + 5.0, text))
                        break
                    cand += 0.5
        st0 = max([self.main_end + 1.5] + [c[1] + 1.05 for c in out if c[1] > self.main_end - 8])
        out.append((st0, st0 + 5.0, "WHERE EARTHQUAKES STRIKE MOST OFTEN"))
        b0 = self.main_end + HEAT_S
        out.append((b0 + 7.0, b0 + 13.0, "THE LARGEST EARTHQUAKES ON THIS MAP"))
        out.sort()
        self.facts = dict(busiest=busiest, busiest_n=per[busiest], m7=big)
        return out

    def caption_at(self, t):
        for a, b, s in self.captions:
            if a <= t < b:
                return s, min(1.0, (t - a) / 0.4, (b - t) / 0.4)
        return None, 0.0


if __name__ == "__main__":
    D = Data()
    TL = Timeline(D)
    print("earthquakes in California (M4+, 1900-2025):", D.n, D.count_by_class(), "total video", round(TL.total, 1), "s")
    for e in sorted(D.events, key=lambda e: e["t0"]):
        print(f"{e['t0']:6.1f} {'S' if e['short'] else 'M'} {e['key']:18s} M{e['mag']:.1f} depth {e['depth']:.0f} {e['local']:%Y-%m-%d %H:%M} {D.place[e['idx']][:46]}")
