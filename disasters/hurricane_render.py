"""Frame renderer for 'Every Hurricane That Hit America, Mapped' (same miniature-world scene and HUD as the tornado film).

Rules (owner 22.11): no flashes or strobing; no camera shake for hurricanes; counters checked by an independent QA pass.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import core
from .core import FR, ROOT, TH, TW, Cam, font, lerp_cam, smooth, world3
from .hurricane import (BARS_S, CAT_COL, CAT_W, EV_HIT, EV_RUN, EV_S, FPS, HEAT_S, INTRO_S, MONTHS, Y0, Y1, Data, Timeline, category, fmt)
from .renderer import H, K, W, WHITE, WS, YEL, Hud, borders_overlay, lighten, pixel_qa, state_centroids
from .tornado import STATE_NAMES

SRC = "SOURCE: NOAA NATIONAL HURRICANE CENTER, HURDAT2"


def cat_of_wind(kt):
    return category(kt)


# ---------------------------------------------------------------- painter: track overlay
class Painter:
    def __init__(self, D: Data, TL: Timeline):
        self.D, self.TL = D, TL
        self.px = []
        for s in D.storms:
            u, v = FR.px(s.lon, s.lat)
            self.px.append(np.stack([u, v], 1))
        self.base = borders_overlay()
        self.evs = {e["idx"] for e in D.events}
        self.mode, self.comp, self.done = None, None, set()
        self._final = self._faint = self._heat = None
        self._all = None

    def _track(self, dr, i, f0, f1, wmul=1.0):
        P = self.px[i]
        s = self.D.storms[i]
        n = len(P)
        a, b = f0 * (n - 1), f1 * (n - 1)
        if b <= a:
            return
        k0, k1 = int(math.floor(a)), int(math.ceil(b))
        for k in range(k0, min(k1, n - 1)):
            ta, tb = max(a, k) - k, min(b, k + 1) - k
            p0 = P[k] + (P[k + 1] - P[k]) * ta
            p1 = P[k] + (P[k + 1] - P[k]) * tb
            c = cat_of_wind(int(s.wind[k]))
            w = max(1, round((CAT_W[c] + 1) * WS * wmul))
            col = CAT_COL[c] if c > 0 else (150, 210, 255)
            dr.line([tuple(p0), tuple(p1)], fill=(10, 10, 40, 220), width=w + max(1, round(2 * WS)))
            dr.line([tuple(p0), tuple(p1)], fill=col + (255,), width=w)

    def all_tracks(self):
        if self._all is None:
            im = self.base.copy()
            dr = ImageDraw.Draw(im)
            for i in np.argsort(self.D.cat, kind="stable"):
                self._track(dr, int(i), 0.0, 1.0)
            self._all = im
        return self._all

    def faint(self):
        if self._faint is None:
            im = self.all_tracks().copy()
            im.putalpha(im.getchannel("A").point(lambda v: int(v * 0.5)))
            self._faint = im
        return self._faint

    def update(self, t, mode):
        TL = self.TL
        if mode != self.mode:
            self.mode = mode
            self.done = set()
            if mode in ("intro", "main"):
                self.comp = self.base.copy()
            elif mode == "heat":
                self.comp = None
            elif mode == "bars":
                self.comp = self.faint()
            else:
                self.comp = self.all_tracks()
        if mode == "main":
            dr = ImageDraw.Draw(self.comp)
            for i in np.where((TL.ta + TL.anim <= t))[0]:
                if int(i) not in self.done:
                    self._track(dr, int(i), 0.0, 1.0, 1.4 if int(i) in self.evs else 1.0)
                    self.done.add(int(i))

    def texture(self, t, mode, p_heat=None):
        TL = self.TL
        if mode == "heat":
            return self.heat_texture(p_heat)
        if mode != "main":
            return self.comp
        act = np.where((TL.ta <= t) & (t < TL.ta + TL.anim))[0]
        if not len(act):
            return self.comp
        out = self.comp.copy()
        dr = ImageDraw.Draw(out)
        for i in act:
            f = (t - TL.ta[i]) / TL.anim[i]
            self._track(dr, int(i), 0.0, f, 1.4 if int(i) in self.evs else 1.0)
        return out

    def _heat_prep(self):
        if self._heat is not None:
            return
        D = self.D
        S = 8
        gw, gh = TW // S, TH // S
        ny = Y1 - Y0 + 1
        cum = np.zeros((ny + 1, gh, gw), "f4")
        for k in range(ny):
            g = np.zeros((gh, gw), "f4")
            for i in np.where(D.yr == Y0 + k)[0]:
                P = self.px[i] / S
                for j in range(len(P) - 1):
                    for f in np.linspace(0, 1, 6, endpoint=False):
                        x, y = P[j] + (P[j + 1] - P[j]) * f
                        if 0 <= x < gw and 0 <= y < gh:
                            g[int(y), int(x)] += 0.2
            cum[k + 1] = cum[k] + g
        sig = max(2.0, 4.0 * TW / 4096)
        ker = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
        ker /= ker.sum()
        for k in range(1, ny + 1):
            a = np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 1, cum[k])
            cum[k] = np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 0, a)
        self._heat = cum
        self._hmax = float(cum[ny].max()) * 0.6
        stops = [0.0, 0.04, 0.22, 0.5, 0.8, 1.0]
        cols = np.array([(120, 220, 255, 0), (120, 220, 255, 140), (255, 214, 0, 200), (255, 120, 0, 220), (220, 0, 80, 235), (140, 0, 200, 245)], float)
        self._lut = np.stack([np.interp(np.linspace(0, 1, 256), stops, cols[:, c]) for c in range(4)], 1).astype("u1")

    def heat_texture(self, years_elapsed):
        self._heat_prep()
        ny = Y1 - Y0 + 1
        y = min(max(years_elapsed, 0.0), ny)
        i = min(int(y), ny - 1)
        g = self._heat[i] * (1 - (y - i)) + self._heat[i + 1] * (y - i)
        v = np.clip(g / self._hmax, 0, 1) ** 0.6
        ov = Image.fromarray(self._lut[(v * 255).astype(np.uint8)], "RGBA").resize((TW, TH), Image.BILINEAR)
        out = self.base.copy()
        out.alpha_composite(ov)
        return out


# ---------------------------------------------------------------- camera
class Director:
    REG_DIST = 1.4

    def __init__(self, D: Data, TL: Timeline):
        from . import mini
        self.D, self.TL = D, TL
        self.U = mini.M_PER_UNIT
        self.keys = self._keys()
        self.kt = [k[0] for k in self.keys]
        self.ev = {}
        for e in D.events:
            s = D.storms[e["idx"]]
            k = e["lf"]
            lat, lon = e["town"]
            town = world3(lon, lat)
            k0 = max(0, k - 2)
            p0 = world3(s.lon[k0], s.lat[k0])
            p1 = world3(s.lon[min(len(s.t) - 1, k + 1)], s.lat[min(len(s.t) - 1, k + 1)])
            hx, hz = p1[0] - p0[0], p1[2] - p0[2]
            nrm = max(1e-9, math.hypot(hx, hz))
            yaw = math.degrees(math.atan2(-hx / nrm, -hz / nrm))
            osm = ROOT / "data" / "osm" / f"{e['key']}.json.gz"
            self.ev[e["key"]] = dict(town=e["key"] if osm.exists() else None, tw=town, yaw=yaw)

    def _keys(self):
        TL = self.TL
        k = [(0.0, 40, -10, 3.4, 0.35, 0.55), (12.0, 48, 6, 3.3, 0.45, 0.50), (INTRO_S, 54, 0, 3.0, 0.45, 0.40)]
        pat = [(52, -8, 3.1, 0.40, 0.42), (58, 8, 2.9, 0.55, 0.45), (50, -4, 3.0, 0.35, 0.40), (56, 6, 2.95, 0.50, 0.38)]
        t, j = INTRO_S + 14.0, 0
        while t < TL.main_end - 6:
            k.append((t, *pat[j % len(pat)]))
            t += 16.0
            j += 1
        h0 = TL.main_end
        k += [(h0, 76, 0, 3.4, 0.35, 0.30), (h0 + HEAT_S, 70, 6, 3.4, 0.35, 0.32), (h0 + HEAT_S + 3.0, 42, -20, 3.5, 0.2, 0.25),
              (h0 + HEAT_S + BARS_S, 48, 20, 3.4, 0.25, 0.25), (h0 + HEAT_S + BARS_S + 4.0, 60, 0, 3.6, 0.2, 0.2),
              (h0 + HEAT_S + BARS_S + 12.0, 82, 0, 3.9, 0.15, 0.15), (TL.total, 82, 0, 3.9, 0.15, 0.15)]
        return k

    def base_cam(self, t):
        j = max(0, min(len(self.keys) - 2, bisect_right(self.kt, t) - 1))
        a, b = self.keys[j], self.keys[j + 1]
        w = smooth((t - a[0]) / max(b[0] - a[0], 1e-6))
        L = lambda x, y: x + (y - x) * w  # noqa: E731
        return Cam(L(a[4], b[4]), 0.0, L(a[5], b[5]), L(a[3], b[3]), L(a[1], b[1]), L(a[2], b[2]))

    def event_phase(self, t):
        p = self.TL.phase_at(t)
        return p if p["kind"] == "event" else None

    def run_u(self, t, t0, e):
        return min(1.0, max(0.0, (t - (t0 + e["HIT"])) / e["RUN"]))

    def event_cam(self, t, p):
        e = p["ev"]
        S, HIT, RUN = e["S"], e["HIT"], e["RUN"]
        info = self.ev[e["key"]]
        tl = t - p["t0"]
        tw = info["tw"]
        orbit = self.base_cam(t)
        reg = Cam(tw[0], tw[1], tw[2], self.REG_DIST, 52.0, info["yaw"] + 20.0)
        reg2 = Cam(tw[0], tw[1], tw[2], 0.9, 46.0, info["yaw"] + 35.0)
        if not info["town"]:
            if tl < 2.5:
                return lerp_cam(orbit, reg, smooth(tl / 2.5))
            if tl < S - 3.5:
                return lerp_cam(reg, reg2, smooth((tl - 2.5) / (S - 6.0)))
            return lerp_cam(reg2, orbit, smooth((tl - S + 3.5) / 3.5))
        U = self.U
        yaw = info["yaw"] + 70.0

        def city(k):
            return Cam(tw[0], tw[1] + 30 / U * core.EXAG, tw[2], (900 - 90 * k) / U, 38.0 - 3 * k, yaw - 25.0 + 10.0 * k)

        def close(u):
            rise = smooth((u - 0.62) / 0.38)
            return Cam(tw[0], tw[1] + (120 + 200 * rise) / U * core.EXAG, tw[2], (1300 + 1400 * rise) / U, 10.0 + 24.0 * rise, yaw + 14.0 * u)
        if e["short"]:
            if tl < HIT:
                return lerp_cam(orbit, close(0.0), smooth(tl / HIT))
            if tl < HIT + RUN:
                return close(self.run_u(t, p["t0"], e))
            return lerp_cam(close(1.0), orbit, smooth((tl - HIT - RUN) / (S - HIT - RUN)))
        if tl < 2.0:
            return lerp_cam(orbit, reg, smooth(tl / 2.0))
        if tl < 4.5:
            return lerp_cam(reg, city(0.0), smooth((tl - 2.0) / 2.5))
        if tl < 6.0:
            return city((tl - 4.5) / 1.5)
        if tl < HIT:
            return lerp_cam(city(1.0), close(0.0), smooth((tl - 6.0) / (HIT - 6.0)))
        if tl < HIT + RUN:
            return close(self.run_u(t, p["t0"], e))
        if tl < HIT + RUN + 2.5:
            return lerp_cam(close(1.0), reg2, smooth((tl - HIT - RUN) / 2.5))
        return lerp_cam(reg2, orbit, smooth((tl - HIT - RUN - 2.5) / (S - HIT - RUN - 2.5)))

    def cam(self, t):
        p = self.event_phase(t)
        return self.event_cam(t, p) if p is not None else self.base_cam(t)


# ---------------------------------------------------------------- HUD extras
def legend(H_, frame, counts, ident="B"):
    w = 400 * K
    rows = [(5, "CATEGORY 5"), (4, "CATEGORY 4"), (3, "CATEGORY 3"), (2, "CATEGORY 2"), (1, "CATEGORY 1")]
    hgt = (62 + len(rows) * 44 + 54) * K
    im = H_._panel(w, hgt)
    d = ImageDraw.Draw(im)
    d.text((22 * K, 18 * K), "STRENGTH AT LANDFALL", font=font(26 * K), fill=(200, 215, 240))
    y = 62 * K
    for c, lab in rows:
        d.rounded_rectangle((22 * K, y + 6 * K, 22 * K + 64 * K, y + 32 * K), int(6 * K), fill=CAT_COL[c], outline=WHITE, width=max(1, int(2 * K)))
        d.text((100 * K, y + 34 * K), lab, font=font(32 * K), fill=WHITE, anchor="ls")
        d.text((w - 22 * K, y + 34 * K), fmt(counts[c]), font=font(32 * K), fill=YEL, anchor="rs")
        y += 44 * K
    d.text((22 * K, hgt - 14 * K), SRC, font=font(19 * K), fill=(170, 190, 225), anchor="ls")
    H_.put(frame, im, (W - 36 * K - w, 30 * K), ident)


def event_card_lines(D: Data, e):
    s = D.storms[e["idx"]]
    k = e["lf"]
    import datetime as dt
    y = int(s.t[k])
    day = dt.datetime(y, 1, 1) + dt.timedelta(seconds=(s.t[k] - y) * (dt.datetime(y + 1, 1, 1) - dt.datetime(y, 1, 1)).total_seconds())
    mph = round(e["kt"] * 1.15078 / 5) * 5
    return [(e["title"], 84, YEL), (f"{MONTHS[day.month - 1]} {day.day}, {y}  ·  CATEGORY {e['cat']} AT LANDFALL", 46, WHITE),
            (f"{e['kt']} KNOTS  ·  ABOUT {mph} MPH WINDS", 58, WHITE), ("BEST TRACK: NOAA NHC HURDAT2", 34, (200, 215, 240))]


# ---------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, scene=None):
        from .mini import M_PER_UNIT, MiniScene
        self.U = M_PER_UNIT
        self.D = Data()
        self.TL = Timeline(self.D)
        self.painter = Painter(self.D, self.TL)
        self.dir = Director(self.D, self.TL)
        self.scene = scene or MiniScene(W, H)
        from .renderer import StateLabels
        self.states = StateLabels()
        cent = state_centroids()
        counts = self.TL.state_counts
        mx = max(counts.values())
        self.bar_counts = counts
        self.bar_rank = sorted(counts, key=lambda s: -counts[s])
        self.bars = []
        for s, n in counts.items():
            lon, lat = cent[s]
            x, y, z = world3(lon, lat)
            self.bars.append((s, x, float(y), z, n / mx))

    def bar_color(self, v):
        stops = [0, 0.15, 0.4, 0.7, 1.0]
        cols = [(255, 214, 0), (255, 150, 0), (255, 70, 0), (215, 0, 60), (130, 0, 190)]
        return tuple(int(np.interp(v, stops, [c[j] for c in cols])) for j in range(3))

    def storm_pos(self, i, f):
        s = self.D.storms[i]
        n = len(s.t)
        a = f * (n - 1)
        k = min(n - 2, int(a))
        w = a - k
        lon = s.lon[k] + (s.lon[k + 1] - s.lon[k]) * w
        lat = s.lat[k] + (s.lat[k + 1] - s.lat[k]) * w
        kt = s.wind[k] + (s.wind[k + 1] - s.wind[k]) * w
        return lon, lat, kt

    def _hurricanes(self, t, cam, ph):
        TL = self.TL
        out = []
        act = np.where((TL.ta - 0.3 <= t) & (t < TL.ta + TL.anim + 0.5))[0]
        evs = {e["idx"] for e in self.D.events}
        for i in act:
            i = int(i)
            f = min(1.0, max(0.0, (t - TL.ta[i]) / TL.anim[i]))
            lon, lat, kt = self.storm_pos(i, f)
            al = smooth((t - TL.ta[i] + 0.3) / 0.6) * (1 - smooth((t - TL.ta[i] - TL.anim[i]) / 0.5))
            if kt < 64:
                al *= max(0.35, (kt - 25) / 39)
            c = max(1, category(int(kt)))
            big = i in evs and ph["kind"] == "event"
            R = (0.05 + 0.012 * c) * min(cam.dist, 3.4) * (1.4 if big else 1.0)
            R = max(R, 250 / self.U) if big else R
            p = world3(lon, lat)
            out.append(dict(pos=(float(p[0]), float(p[1]) + R * 0.02, float(p[2])), R=R, H=R * 0.11, alpha=float(al), eye=0.06, seed=float(i % 7),
                            spin=0.5))
        return out[:8]

    def local_params(self, t, ph, cam):
        if ph["kind"] != "event":
            return None
        e = ph["ev"]
        info = self.dir.ev[e["key"]]
        if not info["town"]:
            return None
        tl = t - ph["t0"]
        u = self.dir.run_u(t, ph["t0"], e)
        U = self.U
        lead = 1.2 if e["short"] else 2.0
        storm = smooth((tl - (e["HIT"] - lead)) / lead) * (1.0 - smooth((u - 0.7) / 0.3))
        tw = info["tw"]
        loc = dict(key=e["key"], pa=(0.0, 0.0), pb=(1e-6, 0.0), hw=0.0, along=0.0, clen=1e-4, storm=(tw[0], tw[2], 9000 / U, 0.6 * storm),
                   wind=storm, rain=storm if cam.dist < 0.03 else 0.0)
        if cam.dist < 0.03 and storm > 0.01:
            loc["cloud"] = dict(r=7000 / U, base=520 / U, thick=500 / U, alpha=storm, spin=0.02)
            loc["cloud_pos"] = tuple(tw)
            loc["surge"] = dict(level=e.get("surge_m", 0.0) * smooth((u - 0.1) / 0.6), alpha=storm)
        return loc

    def frame(self, n):
        D, TL, P = self.D, self.TL, self.painter
        t = n / FPS
        ph = TL.phase_at(t)
        mode = ph["mode"]
        P.update(t, mode)
        tex = P.texture(t, mode, TL.heat_year(t) if mode == "heat" else None)
        cam = self.dir.cam(t)
        self.scene.set_overlay(tex)
        kz = min(1.0, max(0.0, (3.2 - cam.dist) / 1.6))
        ovl_alpha = (1.0 - 0.6 * kz) * min(1.0, max(0.0, (cam.dist - 0.25) / 0.75)) if mode == "main" else 1.0
        bars, hur = [], []
        loc = self.local_params(t, ph, cam) if mode == "main" else None
        if mode in ("main", "intro") and not (loc is not None and cam.dist < 0.3):
            hur = self._hurricanes(t, cam, ph)
        elif mode == "bars":
            g = smooth((t - ph["t0"] - 3.0) / 9.0)
            for s, x, y, z, v in self.bars:
                bars.append((x, y, z, 0.085, max(0.004, v * 0.62 * g), self.bar_color(v)))
        img = self.scene.render(cam, tm=t, dim=0.92 if mode in ("heat", "bars") else 1.0, ovl_alpha=ovl_alpha, bars=bars, local=loc, hurricanes=hur)
        meta = self.hud(img, t, ph, cam)
        meta["states"] = self.states.draw(img, self.scene.view_cam(cam), self.scene.yscale(cam), [p[1] for p in meta["panels"] if p[2] > 0.02])
        meta.update(n=n, t=round(t, 3), mode=mode, kind=ph["kind"], storms=len(hur))
        return img, meta

    def hud(self, img, t, ph, cam):
        D, TL = self.D, self.TL
        H_ = Hud()
        mode, kind = ph["mode"], ph["kind"]
        meta = dict(count=None, major=None, cats=None, dt=None)
        k, major, cats = TL.stats(t)
        if mode == "intro":
            H_.panel_main(img, str(Y0), "", [("HURRICANES ASHORE", "0"), ("MAJOR (CAT 3-5)", "0")], "A")
            legend(H_, img, cats, "B")
            if t > INTRO_S - 8.0:
                a = min(1.0, (t - (INTRO_S - 8.0)) / 0.8) * (1 - smooth((t - (INTRO_S - 1.0)) / 1.0))
                H_.card(img, [("EVERY HURRICANE THAT HIT AMERICA, MAPPED", 78, YEL), (f"{fmt(D.n)} HURRICANES  ·  {Y0}-{Y1}", 56, WHITE)], a, "TITLE")
            meta.update(count=0, major=0)
        elif mode == "main":
            dt = TL.dt_at(t)
            yr = int(dt)
            month = MONTHS[min(11, int((dt - yr) * 12.0))][:3] if kind == "run" else ""
            H_.panel_main(img, str(yr), month, [("HURRICANES ASHORE", fmt(k)), ("MAJOR (CAT 3-5)", fmt(major))], "A")
            legend(H_, img, cats, "B")
            meta.update(count=k, major=major, cats={str(c): v for c, v in cats.items()}, dt=dt)
            if kind == "event":
                tl = t - ph["t0"]
                ev = ph["ev"]
                c0, c1 = (1.8, ev["S"] - 1.6) if ev["short"] else (4.5, ev["S"] - 2.5)
                if c0 < tl < c1:
                    a = min(1.0, (tl - c0) / 0.5, (c1 - tl) / 0.5)
                    H_.card(img, event_card_lines(D, ev), a, left=bool(self.dir.ev[ev["key"]]["town"]))
            else:
                cap, al = TL.caption_at(t)
                if cap:
                    H_.caption(img, cap, al)
        elif mode == "heat":
            y = TL.heat_year(t)
            yr = Y0 + min(max(int(y), 1), Y1 - Y0 + 1) - 1
            H_.panel_main(img, str(yr), "", [("HURRICANES SINCE 1851", fmt(int(TL.cum_year[yr - Y0])))], "A")
            self.painter._heat_prep()
            H_.heat_legend(img, self.painter._lut, "B")
            meta.update(heat_year=yr, heat_count=int(TL.cum_year[yr - Y0]))
            cap, al = TL.caption_at(t)
            if cap:
                H_.caption(img, cap, al)
        elif mode == "bars":
            H_.panel_main(img, "STATES", "", [("HURRICANES ASHORE", fmt(D.n))], "A")
            mx = max(self.bar_counts.values())
            rows = [(self.bar_color(self.bar_counts[s] / mx), STATE_NAMES[s], fmt(self.bar_counts[s])) for s in self.bar_rank[:8]]
            H_.panel_list(img, "FIRST LANDFALL BY STATE", rows, "B")
            cap, al = TL.caption_at(t)
            if cap:
                H_.caption(img, cap, al)
        else:
            H_.panel_main(img, f"{Y0}-{Y1}", "", [("HURRICANES ASHORE", fmt(D.n)), ("MAJOR (CAT 3-5)", fmt(int((D.cat >= 3).sum())))], "A")
            legend(H_, img, D.count_by_cat(), "B")
            meta.update(count=D.n)
            a = smooth((t - ph["t0"] - 4.0) / 1.0)
            if a > 0:
                H_.card(img, [("EVERY HURRICANE THAT HIT AMERICA, MAPPED", 70, YEL),
                              ("DATA: NOAA NATIONAL HURRICANE CENTER HURDAT2, 1851-2025", 36, WHITE),
                              ("MAP: USGS NLCD, 3DEP · NASA BLUE MARBLE · TOWNS: (C) OPENSTREETMAP CONTRIBUTORS, TODAY'S STREETS IN ILLUSTRATIVE 3D", 28, (200, 215, 240))], a, "END")
        meta["panels"] = H_.items
        return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--crf", default="20")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thumbs").mkdir(exist_ok=True)
    r = Renderer()
    total = int(r.TL.total * FPS)
    per = math.ceil(total / a.of)
    n0, n1 = a.seg * per, min(total, (a.seg + 1) * per)
    print(f"segment {a.seg}/{a.of}: frames {n0}-{n1} of {total} ({r.TL.total:.1f}s) {W}x{H} GL {r.scene.renderer}", flush=True)
    mp4 = out / f"seg_{a.seg:02d}.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", a.crf, "-maxrate", "19M" if H >= 2160 else "14M", "-bufsize", "38M" if H >= 2160 else "28M",
                            "-g", "48", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    qa = open(out / f"qa_{a.seg:02d}.jsonl", "w")
    t0 = time.time()
    for n in range(n0, n1):
        img, meta = r.frame(n)
        enc.stdin.write(img.tobytes())
        if n % FPS == 0:
            pr, sky = pixel_qa(img)
            meta["pixel_problems"], meta["sky"] = pr, round(sky, 3)
            if n % (FPS * 12) == 0:
                img.resize((640, 360), Image.LANCZOS).save(out / "thumbs" / f"t{n // FPS:04d}.jpg", quality=82)
        qa.write(json.dumps(meta) + "\n")
        if (n - n0) % 48 == 0:
            print(f"  frame {n} ({(time.time() - t0) / (n - n0 + 1):.2f}s/frame)", flush=True)
    enc.stdin.close()
    enc.wait()
    qa.close()
    el = time.time() - t0
    (out / f"timing_{a.seg:02d}.json").write_text(json.dumps(dict(seg=a.seg, frames=n1 - n0, seconds=round(el, 1), s_per_frame=round(el / max(1, n1 - n0), 3))))


if __name__ == "__main__":
    main()
