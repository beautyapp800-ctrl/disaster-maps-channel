"""Tornado pilot: data, events, timeline (one source of truth for every on-screen number).

Data: NOAA/NWS Storm Prediction Center tornado database 1950-2025 (public domain).
Every number shown on screen is computed here from the CSV.
"""
from __future__ import annotations

import csv
import math
from bisect import bisect_right
from datetime import datetime

import numpy as np

from .core import ROOT

FPS = 24
YEAR_S = 3.6          # seconds of video per year
EV_S = 22.0           # a zoom-in stop on a famous tornado (national -> regional -> town from above -> storm close-up -> back)
EV_HIT = 7.0          # seconds after stop start when the close-up run starts
EV_RUN = 8.0          # close-up: the vortex crosses the town
SH_S, SH_HIT, SH_RUN = 10.0, 2.4, 5.4     # short zoom to another strong tornado (owner 23.11: every 30-40 s)
INTRO_S, HEAT_S, BARS_S, FIN_S = 24.0, 24.0, 22.0, 26.0
Y0, Y1 = 1950, 2025

MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
STATE_NAMES = {"AL": "ALABAMA", "AZ": "ARIZONA", "AR": "ARKANSAS", "CA": "CALIFORNIA", "CO": "COLORADO", "CT": "CONNECTICUT", "DE": "DELAWARE",
               "DC": "WASHINGTON, D.C.", "FL": "FLORIDA", "GA": "GEORGIA", "ID": "IDAHO", "IL": "ILLINOIS", "IN": "INDIANA", "IA": "IOWA",
               "KS": "KANSAS", "KY": "KENTUCKY", "LA": "LOUISIANA", "ME": "MAINE", "MD": "MARYLAND", "MA": "MASSACHUSETTS", "MI": "MICHIGAN",
               "MN": "MINNESOTA", "MS": "MISSISSIPPI", "MO": "MISSOURI", "MT": "MONTANA", "NE": "NEBRASKA", "NV": "NEVADA", "NH": "NEW HAMPSHIRE",
               "NJ": "NEW JERSEY", "NM": "NEW MEXICO", "NY": "NEW YORK", "NC": "NORTH CAROLINA", "ND": "NORTH DAKOTA", "OH": "OHIO", "OK": "OKLAHOMA",
               "OR": "OREGON", "PA": "PENNSYLVANIA", "RI": "RHODE ISLAND", "SC": "SOUTH CAROLINA", "SD": "SOUTH DAKOTA", "TN": "TENNESSEE",
               "TX": "TEXAS", "UT": "UTAH", "VT": "VERMONT", "VA": "VIRGINIA", "WA": "WASHINGTON", "WV": "WEST VIRGINIA", "WI": "WISCONSIN", "WY": "WYOMING"}

# famous tornadoes: (key, title, date, state, time, extra) - each row must exist in the CSV (checked on load)
EVENTS = [
    ("waco", "WACO, TEXAS", "1953-05-11", "TX", "16:10", None),
    ("flint", "FLINT, MICHIGAN", "1953-06-08", "MI", "19:30", None),
    ("ruskin", "RUSKIN HEIGHTS, MISSOURI", "1957-05-20", "KS", "18:15", None),
    ("jackson", "JACKSON, MISSISSIPPI", "1966-03-03", "MS", "18:30", None),
    ("xenia", "XENIA, OHIO", "1974-04-03", "OH", "13:30", ("1974-04-03", "1974-04-04", "SUPER OUTBREAK: {n} TORNADOES IN 2 DAYS")),
    ("moore99", "BRIDGE CREEK-MOORE, OKLAHOMA", "1999-05-03", "OK", "17:26", None),
    ("greensburg", "GREENSBURG, KANSAS", "2007-05-04", "KS", "20:03", None),
    ("hackleburg", "HACKLEBURG, ALABAMA", "2011-04-27", "AL", "14:05", ("2011-04-25", "2011-04-28", "APRIL 25-28 OUTBREAK: {n} TORNADOES")),
    ("joplin", "JOPLIN, MISSOURI", "2011-05-22", "MO", "16:34", None),
    ("moore13", "MOORE, OKLAHOMA", "2013-05-20", "OK", "13:56", None),
    ("mayfield", "MAYFIELD, KENTUCKY", "2021-12-10", "TN", "20:54", None),
]


# short zooms (owner 23.11): other strong tornadoes, about every 30-40 s; place names checked on NWS pages (sources.json event_names)
SHORTS = [
    ("dunlap1965", "DUNLAP, INDIANA", "1965-04-11", "IN", "18:10", None),
    ("lubbock1970", "LUBBOCK, TEXAS", "1970-05-11", "TX", "20:35", None),
    ("wichitafalls1979", "WICHITA FALLS, TEXAS", "1979-04-10", "TX", "17:50", None),
    ("plainfield1990", "PLAINFIELD, ILLINOIS", "1990-08-28", "IL", "14:30", None),
    ("jarrell1997", "JARRELL, TEXAS", "1997-05-27", "TX", "14:40", None),
    ("jacksontn2003", "JACKSON, TENNESSEE", "2003-05-04", "TN", "22:35", None),
    ("vilonia2014", "MAYFLOWER-VILONIA, ARKANSAS", "2014-04-27", "AR", "18:06", None),
    ("rollingfork2023", "ROLLING FORK, MISSISSIPPI", "2023-03-24", "MS", "18:57", None),
]
SHORT_KEYS = {x[0] for x in SHORTS}


class Data:
    def __init__(self, path=None):
        path = path or ROOT / "data" / "raw" / "tornadoes.csv"
        rec = []
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dt = datetime(int(r["yr"]), int(r["mo"]), int(r["dy"]), int(r["time"][:2]), int(r["time"][3:5]))
                span = (datetime(dt.year + 1, 1, 1) - datetime(dt.year, 1, 1)).total_seconds()
                dec = dt.year + (dt - datetime(dt.year, 1, 1)).total_seconds() / span
                sl, so, el, eo = float(r["slat"]), float(r["slon"]), float(r["elat"]), float(r["elon"])
                if el == 0 or eo == 0:
                    el, eo = sl, so
                rec.append((dec, dt.year, dt.month, dt.day, int(r["mag"]), int(r["fat"]), int(r["inj"]), sl, so, el, eo,
                            float(r["len"]), float(r["wid"]), r["st"], r["date"], r["time"][:5]))
        rec.sort(key=lambda x: x[0])
        c = list(zip(*rec))
        self.dt = np.array(c[0]); self.yr = np.array(c[1]); self.mo = np.array(c[2]); self.dy = np.array(c[3])
        self.mag = np.array(c[4]); self.fat = np.array(c[5]); self.inj = np.array(c[6])
        self.slat, self.slon, self.elat, self.elon = (np.array(c[7]), np.array(c[8]), np.array(c[9]), np.array(c[10]))
        self.ln = np.array(c[11]); self.wd = np.array(c[12]); self.st = np.array(c[13]); self.date = np.array(c[14]); self.time = np.array(c[15])
        self.n = len(self.dt)
        self.events = []
        for key, title, date, st, tm, extra in EVENTS + SHORTS:
            m = np.where((self.date == date) & (self.st == st) & (self.time == tm))[0]
            assert len(m) >= 1, f"event row not found: {key}"
            i = int(m[np.argmax(self.fat[m])])
            short = key in SHORT_KEYS
            ev = dict(key=key, title=title, idx=i, date=date, extra=None, short=short,
                      S=SH_S if short else EV_S, HIT=SH_HIT if short else EV_HIT, RUN=SH_RUN if short else EV_RUN)
            if extra:
                a, b, fmt = extra
                ev["extra"] = fmt.format(n=int(((self.date >= a) & (self.date <= b)).sum()))
            self.events.append(ev)
        self.events.sort(key=lambda e: e["idx"])

    # ---- aggregates (all from the CSV)
    def year_count(self, y): return int((self.yr == y).sum())
    def year_deaths(self, y): return int(self.fat[self.yr == y].sum())
    def scale_name(self, i): return "EF" if self.date[i] >= "2007-02-01" else "F"


def fmt(n: int) -> str:
    return f"{n:,}"


def ordinal(k: int) -> str:
    return {1: "", 2: "SECOND-", 3: "THIRD-"}.get(k, f"{k}TH-")


class Timeline:
    """phases[] with t0,t1,mode; ta[i] = video second when tornado i appears; anim[i] = seconds its path takes to draw."""

    def __init__(self, D: Data):
        self.D = D
        n = D.n
        ev_by_idx = {e["idx"]: e for e in D.events}
        ph = [dict(kind="intro", mode="intro", t0=0.0, t1=INTRO_S)]
        t = INTRO_S
        ta = np.zeros(n)
        anim = np.zeros(n)
        strong = (D.mag >= 4)
        anim[strong] = 0.9 + np.minimum(D.ln[strong], 60) / 60.0 * 0.9
        self.year_t0 = {}

        def run(i0, i1, a, b, t0, dur):
            if i1 > i0:
                ta[i0:i1] = t0 + (D.dt[i0:i1] - a) / max(b - a, 1e-9) * dur

        for yr in range(Y0, Y1 + 1):
            lo = int(np.searchsorted(D.dt, yr, "left"))
            hi = int(np.searchsorted(D.dt, yr + 1, "left"))
            self.year_t0[yr] = t
            cur, cur_i = float(yr), lo
            for e in [e for e in D.events if lo <= e["idx"] < hi]:
                dte = float(D.dt[e["idx"]])
                dur = (dte - cur) * YEAR_S
                ph.append(dict(kind="run", mode="main", t0=t, t1=t + dur, a=cur, b=dte))
                run(cur_i, e["idx"], cur, dte, t, dur)
                t += dur
                e["t0"] = t
                ph.append(dict(kind="event", mode="main", t0=t, t1=t + e["S"], ev=e, a=dte, b=dte))
                ta[e["idx"]] = t + e["HIT"]
                anim[e["idx"]] = e["RUN"]
                t += e["S"]
                cur, cur_i = dte, e["idx"] + 1
            dur = (yr + 1 - cur) * YEAR_S
            ph.append(dict(kind="run", mode="main", t0=t, t1=t + dur, a=cur, b=float(yr + 1)))
            run(cur_i, hi, cur, yr + 1, t, dur)
            t += dur
        self.main_end = t
        for kind, d in (("heat", HEAT_S), ("bars", BARS_S), ("final", FIN_S)):
            ph.append(dict(kind=kind, mode=kind, t0=t, t1=t + d))
            t += d
        self.total = t
        self.phases = ph
        self._starts = [p["t0"] for p in ph]
        self.ta, self.anim = ta, anim
        assert np.all(np.diff(ta) >= -1e-9), "appearance times must be monotone"
        # cumulative helpers
        self.cum_fat = np.cumsum(D.fat)
        self.cls_ta = {c: ta[D.mag == c] for c in (-9, 0, 1, 2, 3, 4, 5)}
        self.f5 = np.where(D.mag == 5)[0]
        self.intro_ta = 2.0 + np.arange(len(self.f5)) * 0.33
        # per-year cumulative counts (end of year)
        self.cum_year = np.array([int((D.yr <= y).sum()) for y in range(Y0, Y1 + 1)])
        self.cum_year_fat = np.array([int(D.fat[D.yr <= y].sum()) for y in range(Y0, Y1 + 1)])
        self.captions = self._captions()

    def phase_at(self, t):
        return self.phases[min(len(self.phases) - 1, max(0, bisect_right(self._starts, t) - 1))]

    def dt_at(self, t):
        p = self.phase_at(t)
        if p["mode"] == "main":
            return p["a"] + (p["b"] - p["a"]) * ((t - p["t0"]) / max(p["t1"] - p["t0"], 1e-9) if p["kind"] == "run" else 0.0)
        if p["mode"] in ("heat", "bars", "final"):
            return float(Y1 + 1)
        return float(Y0)

    # ---- state at time t (main/heat/bars/final)
    def stats(self, t):
        D = self.D
        k = int(np.searchsorted(self.ta, t, "right"))
        if self.phase_at(t)["mode"] == "intro":
            k = 0
        deaths = int(self.cum_fat[k - 1]) if k else 0
        cls = {c: int(np.searchsorted(a, t, "right")) for c, a in self.cls_ta.items()}
        if self.phase_at(t)["mode"] == "intro":
            cls = {c: 0 for c in cls}
        return k, deaths, cls

    def heat_year(self, t):
        p = self.phase_at(t)
        q = smooth01((t - p["t0"] - 1.0) / (p["t1"] - p["t0"] - 3.0))
        return q * (Y1 - Y0 + 1)    # fractional years elapsed since 1950

    def intro_count(self, t):
        return int(np.searchsorted(self.intro_ta, t, "right"))

    # ---- captions
    def _captions(self):
        D = self.D
        ycount = {y: D.year_count(y) for y in range(Y0, Y1 + 1)}
        ydeath = {y: D.year_deaths(y) for y in range(Y0, Y1 + 1)}
        rank = sorted(ydeath, key=lambda y: -ydeath[y])
        deadliest, busiest = rank[0], max(ycount, key=ycount.get)
        li = int(np.argmax(D.ln))
        day = "1965-04-11"
        day_fat = int(D.fat[D.date == day].sum())
        top_state = max(set(D.st), key=lambda s: (D.st == s).sum())
        top_n = int((D.st == top_state).sum())
        spec = [
            (2.0, f"THE {len(self.f5)} STRONGEST TORNADOES IN THE DATABASE: F5 AND EF5"),
            (self.year_t0[1950] + 0.3, "1950: THE FIRST YEAR OF THE NATIONAL TORNADO DATABASE"),
            (self.year_t0[1954] + 0.3, f"1953: {fmt(ydeath[1953])} LIVES LOST, THE {ordinal(rank.index(1953) + 1)}DEADLIEST YEAR".replace("THE DEADLIEST", "THE DEADLIEST")),
            (self.year_t0[1965] + 2.4, f"PALM SUNDAY 1965: {fmt(day_fat)} LIVES LOST IN ONE DAY"),
            (self.year_t0[1992] + 0.3, "THE 1990S: DOPPLER RADAR STARTS FINDING MORE TORNADOES"),
            (self.year_t0[busiest] + 0.3, f"{busiest}: {fmt(ycount[busiest])} TORNADOES, THE BUSIEST YEAR IN THE DATABASE"),
            (self.year_t0[min(Y1, deadliest + 1)] + 0.3, f"{deadliest}: {fmt(ydeath[deadliest])} LIVES LOST, THE DEADLIEST YEAR IN THE DATABASE"),
            (self.year_t0[2025] + 0.3, "2025 DATA IS PRELIMINARY"),
            (self.year_t0[1957] + 0.3, "EACH LINE IS ONE TORNADO PATH, COLOURED BY STRENGTH"),
            (self.year_t0[int(D.yr[li])] + 0.3 + 4.5, f"LONGEST PATH IN THE DATABASE: {D.ln[li]:.0f} MILES ({D.date[li]})"),
            (self.year_t0[1997] + 0.3, "MORE TORNADOES ARE RECORDED TODAY PARTLY BECAUSE DETECTION IMPROVED"),
        ]
        decs = []
        for d0 in range(1950, 2030, 10):
            n = int(((D.yr >= d0) & (D.yr < d0 + 10)).sum())
            lab = f"THE {d0}S: {fmt(n)} TORNADOES" if d0 < 2020 else f"THE 2020S SO FAR: {fmt(n)} TORNADOES"
            decs.append((self.year_t0[min(d0, Y1)] + 0.3, lab))
        ev_win = [(p["t0"] - 0.6, p["t1"] + 0.6) for p in self.phases if p["kind"] == "event"]
        out = []

        def free(a, b):
            if any(a < e1 and b > e0 for e0, e1 in ev_win): return False
            return not any(a < c[1] + 1.0 and b > c[0] - 1.0 for c in out)

        for pri, lst in ((0, spec), (1, decs)):
            for t0, text in lst:
                dur = 5.0
                cand = t0
                for _ in range(40):
                    if free(cand, cand + dur) and cand + dur <= self.main_end + 0.1 + (30 if pri == 0 else 0):
                        out.append((cand, cand + dur, text))
                        break
                    cand += 0.5
        out.append((self.main_end + HEAT_S * 0.08, self.main_end + HEAT_S * 0.08 + 5.0, "WHERE TORNADOES STRIKE MOST OFTEN"))
        b0 = self.main_end + HEAT_S
        out.append((b0 + 8.0, b0 + 14.0, f"{STATE_NAMES[top_state]}: {fmt(top_n)} TORNADOES, MORE THAN ANY OTHER STATE"))
        out.sort()
        self.facts = dict(deadliest=deadliest, busiest=busiest, top_state=top_state, top_n=top_n)
        return out

    def caption_at(self, t):
        for a, b, s in self.captions:
            if a <= t < b:
                return s, min(1.0, (t - a) / 0.4, (b - t) / 0.4)
        return None, 0.0


def smooth01(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)
