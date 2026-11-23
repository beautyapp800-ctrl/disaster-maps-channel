"""Frame renderer for 'Every Earthquake in California Since 1900, Mapped' (same miniature-world scene and HUD as the other films).

Rules (owner 22-23.11): no flashes or strobing; camera shake only soft and short, only on the strongest events (M7+ main stops);
counters checked by an independent QA pass; state names, crisp borders, long stops plus short zooms every 30-40 s.
In the town close-ups the shaking is centred on the town named on the card.
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
from .quake import (CLASS_COL, CLASS_NAME, CLASS_W, FPS, HEAT_S, INTRO_S, LIST_S, MONTHS, Y0, Y1, Data, Timeline, fmt)
from .renderer import H, K, W, WHITE, WS, YEL, Hud, StateLabels, borders_overlay, pixel_qa, lighten

SRC = "SOURCE: USGS ANSS COMCAT (M4+)"
CA_CENTER = (-119.4, 37.2)
CITIES = [("LOS ANGELES", -118.2437, 34.0522, 0), ("SAN FRANCISCO", -122.4194, 37.7749, 0), ("SAN DIEGO", -117.1611, 32.7157, 0),
          ("SACRAMENTO", -121.4944, 38.5816, 1), ("FRESNO", -119.7871, 36.7378, 1), ("SAN JOSE", -121.8863, 37.3382, 2),
          ("BAKERSFIELD", -119.0187, 35.3733, 1), ("REDDING", -122.3917, 40.5865, 2), ("EUREKA", -124.1637, 40.8021, 2),
          ("PALM SPRINGS", -116.5453, 33.8303, 2), ("SANTA BARBARA", -119.6982, 34.4208, 2), ("LAS VEGAS", -115.1398, 36.1699, 1)]


def clean_name(place):
    import re
    m = re.match(r"^(?:The )?(\d{4}) (.*?),? California,? [Ee]arthquake.*$", place) or re.match(r"^(\d{4}) (.*?) Earthquake.*$", place)
    if m:
        return f"{m.group(1)} {m.group(2).upper()}".replace(", CALIFORNIA", "")
    return place.upper()[:34]


# ---------------------------------------------------------------- painter: dots and pulses on the overlay
class Painter:
    def __init__(self, D: Data, TL: Timeline):
        self.D, self.TL = D, TL
        u, v = FR.px(D.lon, D.lat)
        self.xy = np.stack([u, v], 1)
        self.base = borders_overlay()
        self.mode, self.comp, self.ptr = None, None, 0
        self._all = self._faint = self._heat = None

    def _dot(self, dr, i, scale=1.0):
        c = int(self.D.cls[i])
        r = {4: 2.2, 5: 3.6, 6: 6.0, 7: 9.5}[c] * WS * 0.85 * scale
        x, y = self.xy[i]
        dr.ellipse((x - r - 1.4 * WS, y - r - 1.4 * WS, x + r + 1.4 * WS, y + r + 1.4 * WS), fill=(20, 8, 30, 235))
        dr.ellipse((x - r, y - r, x + r, y + r), fill=CLASS_COL[c] + (255,))

    def all_dots(self):
        if self._all is None:
            im = self.base.copy()
            dr = ImageDraw.Draw(im)
            for i in np.argsort(self.D.cls, kind="stable"):
                self._dot(dr, int(i))
            self._all = im
        return self._all

    def faint(self):
        if self._faint is None:
            im = self.all_dots().copy()
            im.putalpha(im.getchannel("A").point(lambda v: int(v * 0.55)))
            self._faint = im
        return self._faint

    def update(self, t, mode):
        TL = self.TL
        if mode != self.mode:
            self.mode = mode
            if mode in ("intro", "main"):
                self.comp = self.base.copy()
                self.ptr = int(np.searchsorted(TL.ta, t, "right")) if mode == "main" else 0
                dr = ImageDraw.Draw(self.comp)
                if mode == "main":
                    for i in range(self.ptr):
                        self._dot(dr, i)
                else:
                    for k in range(int(np.searchsorted(TL.intro_ta, t, "right"))):
                        self._dot(dr, int(TL.big[k]), 1.3)
                    self.ptr = int(np.searchsorted(TL.intro_ta, t, "right"))
            elif mode == "heat":
                self.comp = None
            elif mode == "bars":
                self.comp = self.faint()
            else:
                self.comp = self.all_dots()
            return
        if mode == "main":
            n = int(np.searchsorted(TL.ta, t, "right"))
            dr = ImageDraw.Draw(self.comp)
            for i in range(self.ptr, n):
                self._dot(dr, i)
            self.ptr = n
        elif mode == "intro":
            n = int(np.searchsorted(TL.intro_ta, t, "right"))
            dr = ImageDraw.Draw(self.comp)
            for k in range(self.ptr, n):
                self._dot(dr, int(TL.big[k]), 1.3)
            self.ptr = n

    def _ring(self, dr, i, f, scale=1.0):
        """one expanding, fading ring (f = 0..1)"""
        c = int(self.D.cls[i])
        x, y = self.xy[i]
        r = (5 + {4: 16, 5: 30, 6: 52, 7: 85}[c] * f) * WS * scale
        a = int(210 * (1.0 - f) ** 1.5)
        col = lighten(CLASS_COL[c], 0.35) + (a,)
        dr.ellipse((x - r, y - r, x + r, y + r), outline=col, width=max(2, int((2.0 + c * 0.5) * WS)))

    def texture(self, t, mode, p_heat=None):
        TL, D = self.TL, self.D
        if mode == "heat":
            return self.heat_texture(p_heat)
        if mode == "bars":
            out = self.comp.copy()
            dr = ImageDraw.Draw(out)
            top = np.argsort(-D.mag, kind="stable")[:8]
            for k, i in enumerate(top):
                f = ((t * 0.45 + k * 0.17) % 1.0)
                self._ring(dr, int(i), f, 1.4)
                self._dot(dr, int(i), 1.5)
            return out
        if mode not in ("main", "intro"):
            return self.comp
        if mode == "intro":
            ks = range(max(0, self.ptr - 3), self.ptr)
            ids = [int(TL.big[k]) for k in ks]
            ages = [t - TL.intro_ta[k] for k in ks]
            durs = [2.4] * len(ids)
        else:
            a = bisect_left(TL.ta, t - 3.6)
            ids = list(range(a, int(np.searchsorted(TL.ta, t, "right"))))
            ages = [t - TL.ta[i] for i in ids]
            durs = [float(TL.anim[i]) for i in ids]
        live = [(i, ag, du) for i, ag, du in zip(ids, ages, durs) if 0 <= ag < du]
        if not live:
            return self.comp
        out = self.comp.copy()
        dr = ImageDraw.Draw(out)
        for i, ag, du in live:
            self._ring(dr, i, ag / du, 1.0 if mode == "main" else 1.4)
        return out

    def _heat_prep(self):
        if self._heat is not None:
            return
        D = self.D
        S = 8
        gw, gh = TW // S, TH // S
        ny = Y1 - Y0 + 1
        gx = np.clip((self.xy[:, 0] / S).astype(int), 0, gw - 1)
        gy = np.clip((self.xy[:, 1] / S).astype(int), 0, gh - 1)
        cum = np.zeros((ny + 1, gh, gw), "f4")
        for k in range(ny):
            sel = D.yr == Y0 + k
            g = np.zeros((gh, gw), "f4")
            np.add.at(g, (gy[sel], gx[sel]), 1.0)
            cum[k + 1] = cum[k] + g
        sig = max(2.0, 4.0 * TW / 4096)
        ker = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
        ker /= ker.sum()
        for k in range(1, ny + 1):
            a = np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 1, cum[k])
            cum[k] = np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 0, a)
        self._heat = cum
        self._hmax = float(cum[ny].max()) * 0.7
        stops = [0.0, 0.04, 0.22, 0.5, 0.8, 1.0]
        cols = np.array([(255, 240, 120, 0), (255, 235, 90, 150), (255, 160, 0, 205), (255, 50, 0, 225), (200, 0, 120, 235), (110, 0, 200, 245)], float)
        self._lut = np.stack([np.interp(np.linspace(0, 1, 256), stops, cols[:, c]) for c in range(4)], 1).astype("u1")

    def heat_texture(self, years_elapsed):
        self._heat_prep()
        ny = Y1 - Y0 + 1
        y = min(max(years_elapsed, 0.0), ny)
        i = min(int(y), ny - 1)
        g = self._heat[i] * (1 - (y - i)) + self._heat[i + 1] * (y - i)
        v = np.clip(g / self._hmax, 0, 1) ** 0.55
        ov = Image.fromarray(self._lut[(v * 255).astype(np.uint8)], "RGBA").resize((TW, TH), Image.BILINEAR)
        out = self.base.copy()
        out.alpha_composite(ov)
        return out


# ---------------------------------------------------------------- city labels (regional shots)
class CityLabels:
    def __init__(self):
        self.pts = [(n, world3(lo, la), pri) for n, lo, la, pri in CITIES]
        self.alpha = {}

    def draw(self, img, vcam, yscale, avoid):
        dist = vcam.dist
        glob = float(np.clip((dist - 0.10) / 0.12, 0.0, 1.0)) * float(1.0 - np.clip((dist - 3.2) / 0.8, 0.0, 1.0))
        size = int(K * (20 + 8 * float(np.clip((3.0 - dist) / 2.0, 0.0, 1.0))))
        f = font(size)
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        placed = [tuple(r) for r in avoid]
        shown = []
        step = 1.0 / (0.35 * FPS)
        for name, p3, pri in sorted(self.pts, key=lambda a: a[2]):
            p = p3.copy()
            p[1] *= yscale
            pp, ok = vcam.project(p[None], W, H)
            x, y = float(pp[0, 0]), float(pp[0, 1])
            target, hard, box = 0.0, False, None
            if glob > 0 and ok[0] and pri <= (0 if dist > 2.6 else 1 if dist > 1.7 else 2):
                tw = d.textlength(name, font=f)
                box = (x - tw / 2 - 8 * K, y - size * 1.6 - 8 * K, x + tw / 2 + 8 * K, y + 8 * K)
                inside = box[0] > 10 * K and box[2] < W - 10 * K and box[1] > 10 * K and box[3] < H - 10 * K
                free = not any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed)
                if inside and free:
                    target = 1.0
                else:
                    hard = True
            a = self.alpha.get(name, target)
            a = min(target, a + step) if target > a else max(target, a - step * (4.0 if hard else 1.0))
            self.alpha[name] = a
            if a <= 0.01 or box is None:
                continue
            if hard:
                self.alpha[name] = 0.0
                continue
            placed.append(box)
            al = int(255 * a * glob * 0.95)
            d.ellipse((x - 5 * K, y - 5 * K, x + 5 * K, y + 5 * K), fill=(255, 255, 255, al), outline=(20, 24, 44, al), width=max(1, int(2 * K)))
            d.text((x, y - 12 * K), name, font=f, fill=(255, 255, 255, al), anchor="ms", stroke_width=max(2, int(2.4 * K)), stroke_fill=(18, 22, 40, int(al * 0.85)))
            shown.append(name)
        if shown:
            img.paste(lay.convert("RGB"), (0, 0), lay.getchannel("A"))
        return shown


# ---------------------------------------------------------------- camera
class Director:
    REG_DIST = 1.0

    def __init__(self, D: Data, TL: Timeline):
        from . import mini
        self.D, self.TL = D, TL
        self.U = mini.M_PER_UNIT
        c = world3(*CA_CENTER)
        self.cx, self.cz = float(c[0]), float(c[2])
        self.keys = self._keys()
        self.kt = [k[0] for k in self.keys]
        self.ev = {}
        self.evd = {e["key"]: e for e in D.events}
        for n, e in enumerate(D.events):
            lat, lon = e["town"]
            tw = world3(lon, lat)
            osm = ROOT / "data" / "osm" / f"{e['key']}.json.gz"
            self.ev[e["key"]] = dict(town=e["key"] if osm.exists() else None, tw=tw, yaw=[-28.0, 20.0, 36.0, -12.0, 28.0, -34.0][n % 6])
        self.hits = sorted((e["t0"] + e["HIT"], 0.0030) for e in D.events if e["mag"] >= 7.0 and not e["short"] and self.ev[e["key"]]["town"])
        self.hit_t = [h[0] for h in self.hits]

    def _keys(self):
        TL = self.TL
        cx, cz = self.cx, self.cz
        k = [(0.0, 52, -12, 2.5, cx + 0.10, cz), (12.0, 56, 10, 2.35, cx - 0.05, cz), (INTRO_S, 56, 0, 2.3, cx, cz)]
        pat = [(54, -10, 2.3, 0.05, 0.12), (58, 12, 2.1, -0.05, -0.10), (52, -4, 2.4, 0.03, 0.05), (57, 8, 2.2, -0.03, 0.10)]
        t, j = INTRO_S + 12.0, 0
        while t < TL.main_end - 6:
            p = pat[j % len(pat)]
            k.append((t, p[0], p[1], p[2], cx + p[3], cz + p[4]))
            t += 14.0
            j += 1
        h0 = TL.main_end
        k += [(h0, 70, 0, 2.5, cx, cz), (h0 + HEAT_S, 64, 6, 2.4, cx, cz + 0.05), (h0 + HEAT_S + 3.0, 48, -18, 2.3, cx, cz),
              (h0 + HEAT_S + LIST_S, 52, 18, 2.2, cx, cz), (h0 + HEAT_S + LIST_S + 6.0, 62, 0, 2.5, cx, cz),
              (TL.total, 70, 0, 2.7, cx, cz)]
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
        reg2 = Cam(tw[0], tw[1], tw[2], 0.7, 46.0, info["yaw"] + 35.0)
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
            return Cam(tw[0], tw[1] + (150 + 120 * rise) / U * core.EXAG, tw[2], (1050 + 1000 * rise) / U, 15.0 + 16.0 * rise, yaw + 26.0 * u)
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
        c = self.event_cam(t, p) if p is not None else self.base_cam(t)
        amp = 0.0
        a = bisect_left(self.hit_t, t - 1.2)
        b = bisect_right(self.hit_t, t)
        for h in self.hits[a:b]:
            amp += h[1] * math.exp(-(t - h[0]) * 3.0)
        d = c.dist * amp
        c.shake = (d * math.sin(t * 9.0), d * 0.5 * math.sin(t * 7.3 + 1.7), d * math.sin(t * 8.1 + 3.1))
        return c


# ---------------------------------------------------------------- HUD extras
def legend(H_, frame, counts, ident="B"):
    w = 430 * K
    rows = [(7, CLASS_NAME[7]), (6, CLASS_NAME[6]), (5, CLASS_NAME[5]), (4, CLASS_NAME[4])]
    hgt = (62 + len(rows) * 44 + 54) * K
    im = H_._panel(w, hgt)
    d = ImageDraw.Draw(im)
    d.text((22 * K, 18 * K), "MAGNITUDE", font=font(26 * K), fill=(200, 215, 240))
    y = 62 * K
    for c, lab in rows:
        d.ellipse((30 * K, y + 7 * K, 30 * K + 26 * K, y + 33 * K), fill=CLASS_COL[c], outline=WHITE, width=max(1, int(2 * K)))
        d.text((74 * K, y + 34 * K), lab, font=font(30 * K), fill=WHITE, anchor="ls")
        d.text((w - 22 * K, y + 34 * K), fmt(counts[c]), font=font(32 * K), fill=YEL, anchor="rs")
        y += 44 * K
    d.text((22 * K, hgt - 14 * K), SRC, font=font(19 * K), fill=(170, 190, 225), anchor="ls")
    H_.put(frame, im, (W - 36 * K - w, 30 * K), ident)


def event_card_lines(D: Data, e):
    loc = e["local"]
    depth = f"  ·  DEPTH {e['depth']:.0f} KM" if e["depth"] == e["depth"] and e["depth"] > 0 else ""
    return [(e["title"], 84, YEL), (f"{MONTHS[loc.month - 1]} {loc.day}, {loc.year}", 46, WHITE),
            (f"MAGNITUDE {e['mag']:.1f}{depth}", 58, WHITE), ("EARTHQUAKE CATALOGUE: USGS COMCAT", 34, (200, 215, 240))]


# ---------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, scene=None):
        from .mini import M_PER_UNIT, MiniScene
        self.U = M_PER_UNIT
        self.D = Data()
        self.TL = Timeline(self.D)
        self.painter = Painter(self.D, self.TL)
        self.dir = Director(self.D, self.TL)
        self.scene = scene or MiniScene(W, H, state="ca")
        self.states = StateLabels()
        self.cities = CityLabels()
        D = self.D
        self.top = list(np.argsort(-D.mag, kind="stable")[:8])

    def local_params(self, t, ph, cam):
        if ph["kind"] != "event":
            return None
        e = ph["ev"]
        info = self.dir.ev[e["key"]]
        if not info["town"]:
            return None
        tl = t - ph["t0"]
        tq = max(0.0, t - (ph["t0"] + e["HIT"]))
        U = self.U
        tw = info["tw"]
        M = e["mag"]
        Rm = min(3500.0, 320.0 * 10 ** ((M - 5.0) * 0.45))
        ramp = smooth(tq / 1.0) * (1.0 - smooth((tl - (e["HIT"] + e["RUN"] + 0.5)) / 1.5))
        on = 1.0 if (tl >= e["HIT"] - 0.05 and ramp > 0.0) else 0.0
        amp = (0.5 + 0.45 * max(0.0, M - 6.0)) * smooth(tq / 1.2) * (1.0 - smooth((tq - (e["RUN"] - 2.0)) / 2.0))
        loc = dict(key=e["key"], pa=(0.0, 0.0), pb=(1e-6, 0.0), hw=0.0, along=0.0, clen=1e-4)
        if on:
            loc["quake"] = dict(pos=(float(tw[0]), float(tw[2])), R=Rm / U, amp=float(min(1.5, amp)), t=float(tq), alpha=float(max(ramp, 0.0)),
                                speed=900.0 / U)
            if cam.dist < 0.03:
                loc["dust"] = dict(pos=(float(tw[0]), float(tw[1]), float(tw[2])), R=min(900.0, Rm * 0.55) / U, H=260.0 / U,
                                   alpha=float(max(ramp, 0.0) * 0.9), grow=float(smooth(tq / 5.5)))
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
        ovl_alpha = (1.0 - 0.7 * kz) * min(1.0, max(0.0, (cam.dist - 0.25) / 0.75)) if mode == "main" else 1.0
        loc = self.local_params(t, ph, cam) if mode == "main" else None
        img = self.scene.render(cam, tm=t, dim=0.95 if mode in ("heat", "bars") else 1.0, ovl_alpha=ovl_alpha, local=loc)
        vcam = self.scene.view_cam(cam)
        labs = []
        if loc is not None and cam.dist < 0.03:
            from .renderer import Renderer as TR
            labs = TR.labels(self, img, vcam, min(1.0, (0.03 - cam.dist) / 0.01))
        ys = self.scene.yscale(cam)
        avoid = [p[1] for p in self.hud(img.copy(), t, ph, cam)["panels"] if p[2] > 0.02]
        shown = self.states.draw(img, vcam, ys, avoid)
        shown += self.cities.draw(img, vcam, ys, avoid + getattr(self.states, "last_boxes", []))
        meta = self.hud(img, t, ph, cam)
        meta.update(states=shown, n=n, t=round(t, 3), mode=mode, kind=ph["kind"], labels=labs)
        return img, meta

    def hud(self, img, t, ph, cam):
        D, TL = self.D, self.TL
        H_ = Hud()
        mode, kind = ph["mode"], ph["kind"]
        meta = dict(count=None, maxmag=None, cls=None, dt=None)
        k, mx, cls = TL.stats(t)
        if mode == "intro":
            n = TL.intro_count(t)
            H_.panel_main(img, str(Y0), "", [("M6.5+ EARTHQUAKES", f"{n} / {len(TL.big)}")], "A")
            legend(H_, img, {c: 0 for c in (4, 5, 6, 7)}, "B")
            cap, al = TL.caption_at(t)
            if cap:
                H_.caption(img, cap, al)
            if t > INTRO_S - 8.0:
                a = min(1.0, (t - (INTRO_S - 8.0)) / 0.8) * (1 - smooth((t - (INTRO_S - 1.0)) / 1.0))
                H_.card(img, [("EVERY EARTHQUAKE IN CALIFORNIA, MAPPED", 78, YEL), (f"{fmt(D.n)} EARTHQUAKES OF M4 OR MORE  ·  {Y0}-{Y1}", 52, WHITE)], a, "TITLE")
            meta.update(intro=n)
        elif mode == "main":
            dt = TL.dt_at(t)
            yr = int(dt)
            month = MONTHS[min(11, int((dt - yr) * 12.0))][:3] if kind == "run" else ""
            if kind == "event":
                month = ph["ev"]["local"].strftime("%b").upper()
            H_.panel_main(img, str(yr), month, [("EARTHQUAKES", fmt(k)), ("LARGEST SO FAR", f"M{mx:.1f}" if k else "-")], "A")
            legend(H_, img, cls, "B")
            meta.update(count=k, maxmag=round(mx, 2), cls={str(c): v for c, v in cls.items()}, dt=dt)
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
            H_.panel_main(img, str(yr), "", [("EARTHQUAKES SINCE 1900", fmt(int(TL.cum_year[yr - Y0])))], "A")
            self.painter._heat_prep()
            H_.heat_legend(img, self.painter._lut, "B")
            meta.update(heat_year=yr, heat_count=int(TL.cum_year[yr - Y0]))
            cap, al = TL.caption_at(t)
            if cap:
                H_.caption(img, cap, al)
        elif mode == "bars":
            H_.panel_main(img, "TOP 8", "", [("LARGEST EARTHQUAKES", "M" + f"{D.mag[self.top[0]]:.1f}")], "A")
            rows = [(CLASS_COL[int(D.cls[i])], f"M{D.mag[i]:.1f}  {clean_name(D.place[i])}"[:34], "") for i in self.top]
            H_.panel_list(img, "LARGEST IN CALIFORNIA, 1900-2025", rows, "B")
            cap, al = TL.caption_at(t)
            if cap:
                H_.caption(img, cap, al)
        else:
            H_.panel_main(img, f"{Y0}-{Y1}", "", [("EARTHQUAKES", fmt(D.n)), ("LARGEST", f"M{D.mag.max():.1f}")], "A")
            legend(H_, img, D.count_by_class(), "B")
            meta.update(count=D.n, maxmag=round(float(D.mag.max()), 2))
            a = smooth((t - ph["t0"] - 4.0) / 1.0)
            if a > 0:
                H_.card(img, [("EVERY EARTHQUAKE IN CALIFORNIA, MAPPED", 70, YEL),
                              ("DATA: USGS ANSS COMCAT, M4+, 1900-2025 (EARLY YEARS INCOMPLETE)", 34, WHITE),
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
