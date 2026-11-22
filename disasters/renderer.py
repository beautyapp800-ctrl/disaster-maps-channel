"""Frame renderer for 'Every Tornado in US History Mapped': painter (map texture), camera, HUD, QA log, segment CLI."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from . import core
from .core import F_COL, F_W, FR, OCEAN, TH, TW, Cam, font, lerp_cam, smooth, world3
from .tornado import (BARS_S, EV_HIT, EV_RUN, EV_S, FIN_S, FPS, HEAT_S, INTRO_S, MONTHS, STATE_NAMES, Y0, Y1, Data, Timeline, fmt)

RES = int(os.environ.get("RES", "2160"))
H = RES
W = RES * 16 // 9
K = H / 1080.0
WS = TW / 4096.0
NAVY = (10, 18, 45)
YEL = (255, 214, 0)
WHITE = (255, 255, 255)


def lighten(c, a):
    return tuple(int(x + (255 - x) * a) for x in c)


# ---------------------------------------------------------------- albers inverse (for state centroids)
def albers_inv(x, y):
    N, C, R0, L0 = core.albers.__globals__["_N"], core.albers.__globals__["_C"], core.albers.__globals__["_R0"], core.albers.__globals__["_L0"]
    rho = math.sqrt(x * x + (R0 - y) ** 2)
    th = math.atan2(x, R0 - y)
    lon = math.degrees(L0 + th / N)
    s = (C - (rho * N) ** 2) / (2 * N)
    return lon, math.degrees(math.asin(max(-1, min(1, s))))


def state_centroids():
    out = {}
    for ab, rings in core.states().items():
        best, area = None, 0
        for r in rings:
            x, y = r[:, 0], r[:, 1]
            a = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
            if abs(a) > abs(area):
                area, best = a, r
        x, y = best[:, 0], best[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        cx = np.sum((x + x1) * cr) / (6 * area)
        cy = np.sum((y + y1) * cr) / (6 * area)
        out[ab] = albers_inv(cx, cy)
    return out


# ---------------------------------------------------------------- painter: the map texture
class Painter:
    def __init__(self, D: Data, TL: Timeline):
        self.D, self.TL = D, TL
        su, sv = FR.px(D.slon, D.slat)
        eu, ev = FR.px(D.elon, D.elat)
        self.p0 = np.stack([su, sv], 1)
        self.p1 = np.stack([eu, ev], 1)
        self.base = core.base_map()
        self.comp = None
        self.mode = None
        self.ptr = 0
        self.active = []
        self._final = None
        self._faint = None
        self._heat = None

    # -- drawing primitives
    def _line(self, dr, i, f0, f1, outline=True, boost=1.0, wmul=1.0):
        a = self.p0[i] + (self.p1[i] - self.p0[i]) * f0
        b = self.p0[i] + (self.p1[i] - self.p0[i]) * f1
        m = int(self.D.mag[i])
        w = max(1, round(F_W[m] * WS * wmul))
        if m >= 3 and outline:
            dr.line([tuple(a), tuple(b)], fill=(25, 0, 40), width=w + max(1, round(2 * WS * wmul)))
        dr.line([tuple(a), tuple(b)], fill=F_COL[m], width=w)

    def _draw_all(self, dr, idx, outline=True):
        for i in idx[np.argsort(self.D.mag[idx], kind="stable")]:
            self._line(dr, int(i), 0.0, 1.0, outline)

    def final_comp(self):
        if self._final is None:
            im = self.base.copy()
            self._draw_all(ImageDraw.Draw(im), np.arange(self.D.n))
            self._final = im
        return self._final

    def faint(self):
        if self._faint is None:
            self._faint = Image.blend(self.base, self.final_comp(), 0.55)
        return self._faint

    def final_f5(self):
        if getattr(self, "_f5", None) is None:
            im = self.final_comp().copy()
            dr = ImageDraw.Draw(im)
            for i in self.TL.f5:
                a, b = tuple(self.p0[i]), tuple(self.p1[i])
                dr.line([a, b], fill=WHITE, width=max(2, round(14 * WS)))
                dr.line([a, b], fill=F_COL[5], width=max(1, round(8 * WS)))
            self._f5 = im
        return self._f5

    # -- main / intro incremental drawing
    def _reset(self, t, mode):
        self.mode = mode
        TL, D = self.TL, self.D
        if mode == "intro":
            self.comp = self.base.copy()
            dr = ImageDraw.Draw(self.comp)
            self.ptr = int(np.searchsorted(TL.intro_ta, t, "right"))
            for k in range(self.ptr):
                self._line(dr, int(TL.f5[k]), 0.0, 1.0, True, wmul=1.4)
        elif mode == "main":
            self.comp = self.base.copy()
            dr = ImageDraw.Draw(self.comp)
            self.ptr = int(np.searchsorted(TL.ta, t, "right"))
            idx = np.arange(self.ptr)
            done = idx[(TL.anim[idx] == 0) | (t >= TL.ta[idx] + TL.anim[idx])]
            self._draw_all(dr, done)
            self.active = []
            for i in idx:
                if TL.anim[i] > 0 and t < TL.ta[i] + TL.anim[i]:
                    f = (t - TL.ta[i]) / TL.anim[i]
                    self._line(dr, int(i), 0.0, f, True, wmul=self._wm(i))
                    self.active.append((int(i), f))
        elif mode == "heat":
            self.comp = None
        else:
            self.comp = {"bars": self.faint, "final": self.final_f5}[mode]()

    def _wm(self, i):
        return 1.6 if self.D.mag[i] >= 4 and i in self.TL_event_idx else 1.0

    @property
    def TL_event_idx(self):
        return {e["idx"] for e in self.D.events}

    def update(self, t, mode):
        if mode != self.mode:
            self._reset(t, mode)
            return
        TL = self.TL
        if mode == "intro":
            dr = ImageDraw.Draw(self.comp)
            n = int(np.searchsorted(TL.intro_ta, t, "right"))
            for k in range(self.ptr, n):
                self._line(dr, int(TL.f5[k]), 0.0, 1.0, True, wmul=1.4)
            self.ptr = n
        elif mode == "main":
            dr = ImageDraw.Draw(self.comp)
            n = int(np.searchsorted(TL.ta, t, "right"))
            evs = self.TL_event_idx
            for i in range(self.ptr, n):
                if TL.anim[i] == 0:
                    self._line(dr, i, 0.0, 1.0, True)
                else:
                    self.active.append((i, 0.0))
            self.ptr = n
            keep = []
            for i, f0 in self.active:
                f1 = min(1.0, (t - TL.ta[i]) / TL.anim[i])
                if f1 > f0:
                    self._line(dr, i, f0, f1, True, wmul=1.6 if i in evs else 1.0)
                if f1 < 1.0:
                    keep.append((i, f1))
            self.active = keep

    # -- per-frame texture (glow on top of comp)
    def texture(self, t, mode, p_heat=None):
        TL, D = self.TL, self.D
        if mode == "heat":
            return self.heat_texture(p_heat)
        if mode not in ("main", "intro"):
            return self.comp
        if mode == "intro":
            ids = [int(TL.f5[k]) for k in range(max(0, self.ptr - 3), self.ptr)]
            ages = [t - TL.intro_ta[k] for k in range(max(0, self.ptr - 3), self.ptr)]
            fr = [min(1, a / 0.6) for a in ages]
            win = 0.9
        else:
            a = bisect_left(TL.ta, t - 6.6)
            ids = list(range(a, self.ptr))
            ages = [t - TL.ta[i] for i in ids]
            fr = [min(1.0, ag / TL.anim[i]) if TL.anim[i] > 0 else 1.0 for i, ag in zip(ids, ages)]
            win = 0.9
        if not ids:
            return self.comp
        S = 4
        gl = Image.new("RGBA", (TW // S, TH // S), (0, 0, 0, 0))
        dr = ImageDraw.Draw(gl)
        for i, ag, f in zip(ids, ages, fr):
            life = 1.0 - ag / (TL.anim[i] + 1.2 if (mode == "main" and TL.anim[i] > 0) else win)
            if life <= 0:
                continue
            m = int(D.mag[i])
            col = lighten(F_COL[m], 0.55) + (int(255 * min(1.0, life * 1.3)),)
            w = max(2, int((6 + 3 * m) * WS / S * (1.8 if m >= 4 else 1.0)))
            a0 = self.p0[i] / S
            b0 = a0 + (self.p1[i] / S - a0) * f
            dr.line([tuple(a0), tuple(b0)], fill=col, width=w)
        gl = gl.filter(ImageFilter.GaussianBlur(3.0 * WS))
        up = gl.resize((TW, TH), Image.BILINEAR)
        out = self.comp.copy()
        out.paste(up.convert("RGB"), (0, 0), up.getchannel("A"))
        return out

    # -- heat map
    def _heat_prep(self):
        if self._heat is not None:
            return
        D = self.D
        S = 8
        gw, gh = TW // S, TH // S
        mid = (self.p0 + self.p1) / 2 / S
        gx = np.clip(mid[:, 0].astype(int), 0, gw - 1)
        gy = np.clip(mid[:, 1].astype(int), 0, gh - 1)
        ny = Y1 - Y0 + 1
        cum = np.zeros((ny + 1, gh, gw), "f4")
        for k in range(ny):
            sel = D.yr == Y0 + k
            g = np.zeros((gh, gw), "f4")
            np.add.at(g, (gy[sel], gx[sel]), 1.0)
            cum[k + 1] = cum[k] + g
        sig = max(2.0, 5.0 * TW / 4096)
        ker = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
        ker /= ker.sum()
        for k in range(1, ny + 1):
            a = cum[k]
            a = np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 1, a)
            a = np.apply_along_axis(lambda r: np.convolve(r, ker, "same"), 0, a)
            cum[k] = a
        self._heat = cum
        self._hmax = float(cum[ny].max()) * 0.75
        stops = [0.0, 0.04, 0.22, 0.5, 0.8, 1.0]
        cols = np.array([(255, 240, 120, 0), (255, 235, 90, 150), (255, 160, 0, 205), (255, 50, 0, 225), (200, 0, 120, 235), (110, 0, 200, 245)], float)
        self._lut = np.stack([np.interp(np.linspace(0, 1, 256), stops, cols[:, c]) for c in range(4)], 1).astype("u1")

    def heat_texture(self, years_elapsed):
        self._heat_prep()
        ny = Y1 - Y0 + 1
        y = min(max(years_elapsed, 0.0), ny)
        i = min(int(y), ny - 1)
        g = self._heat[i] * (1 - (y - i)) + self._heat[i + 1] * (y - i)
        v = np.clip(g / self._hmax, 0, 1) ** 0.6
        rgba = self._lut[(v * 255).astype(np.uint8)]
        ov = Image.fromarray(rgba, "RGBA").resize((TW, TH), Image.BILINEAR)
        out = Image.blend(self.base, self.final_comp(), 0.18)
        out.paste(ov.convert("RGB"), (0, 0), ov.getchannel("A"))
        return out


# ---------------------------------------------------------------- camera + shake + flash
class Director:
    def __init__(self, D: Data, TL: Timeline):
        self.D, self.TL = D, TL
        self.keys = self._keys()
        self.kt = [k[0] for k in self.keys]
        self.cent = state_centroids()
        self.ev = {}
        for e in D.events:
            i = e["idx"]
            p0 = world3(D.slon[i], D.slat[i])
            p1 = world3(D.elon[i], D.elat[i])
            ext = float(np.hypot(p1[0] - p0[0], p1[2] - p0[2]))
            hx, hz = (p1[0] - p0[0]), (p1[2] - p0[2])
            nrm = max(1e-6, math.hypot(hx, hz))
            hx, hz = hx / nrm, hz / nrm
            self.ev[e["key"]] = dict(p0=p0, p1=p1, mid=(p0 + p1) / 2, dist=max(0.9, ext * 3.2 + 0.5), yaw=math.degrees(math.atan2(-hx, -hz)) + 22.0, ext=ext)
        # shake / flash hits: (t, amp_world_per_dist, flash_alpha)
        hits = []
        for k, i in enumerate(TL.f5):
            hits.append((TL.intro_ta[k], 0.0035, 0.22))
        evs = {e["idx"] for e in D.events}
        for i in np.where(D.mag == 5)[0]:
            if int(i) in evs:
                continue
            hits.append((float(TL.ta[i]), 0.003, 0.16))
        for e in D.events:
            i = e["idx"]
            hits.append((float(TL.ta[i]), 0.012 if D.mag[i] == 5 else 0.008, 0.6 if D.mag[i] == 5 else 0.4))
        hits.sort()
        self.hits = hits
        self.hit_t = [h[0] for h in hits]

    def _keys(self):
        TL = self.TL
        k = [(0.0, 34, -16, 4.1, 0.0, 0.30), (12.0, 46, 6, 3.9, 0.0, 0.20), (INTRO_S, 54, 0, 3.6, 0.0, 0.15)]
        pat = [(48, -12, 3.8, 0.0, 0.20), (58, 10, 3.55, 0.0, 0.10), (52, -6, 3.7, 0.0, 0.15), (56, 6, 3.6, 0.0, 0.12), (54, 0, 3.6, 0.0, 0.15)]
        t, j = INTRO_S + 14.0, 0
        while t < TL.main_end - 6:
            k.append((t, *pat[j % len(pat)]))
            t += 16.0
            j += 1
        h0 = TL.main_end
        k += [(h0, 78, 0, 4.0, 0.0, 0.0), (h0 + HEAT_S, 70, 6, 4.0, 0.0, 0.02), (h0 + HEAT_S + 3.0, 40, -24, 3.9, 0.0, 0.25),
              (h0 + HEAT_S + BARS_S, 46, 24, 3.85, 0.0, 0.20), (h0 + HEAT_S + BARS_S + 4.0, 60, 0, 3.8, 0.0, 0.10),
              (h0 + HEAT_S + BARS_S + 12.0, 84, 0, 4.05, 0.0, 0.0), (TL.total, 84, 0, 4.05, 0.0, 0.0)]
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

    def vortex_frac(self, e_key, t, t0):
        return min(1.0, max(0.0, (t - (t0 + EV_HIT)) / EV_RUN))

    def cam(self, t):
        c = self.base_cam(t)
        p = self.event_phase(t)
        if p is not None:
            e = p["ev"]
            info = self.ev[e["key"]]
            tl = t - p["t0"]
            w = smooth(tl / 3.0) * (1 - smooth((tl - (EV_S - 3.0)) / 3.0))
            f = self.vortex_frac(e["key"], t, p["t0"])
            pos = info["p0"] + (info["p1"] - info["p0"]) * f
            tgt = 0.5 * info["mid"] + 0.5 * pos
            evc = Cam(tgt[0], tgt[1], tgt[2], info["dist"], 34.0, info["yaw"])
            c = lerp_cam(c, evc, w)
        # shake
        amp = 0.0
        a = bisect_left(self.hit_t, t - 1.6)
        b = bisect_right(self.hit_t, t)
        for h in self.hits[a:b]:
            amp += h[1] * math.exp(-(t - h[0]) * 4.0)
        if p is not None:
            tl = t - p["t0"]
            if EV_HIT <= tl <= EV_HIT + EV_RUN:
                amp += 0.0035
        d = c.dist * amp
        c.shake = (d * math.sin(t * 57.0), d * 0.7 * math.sin(t * 43.0 + 1.7), d * math.sin(t * 51.0 + 3.1))
        return c

    def flash(self, t):
        a = bisect_left(self.hit_t, t - 0.8)
        b = bisect_right(self.hit_t, t)
        return min(0.85, sum(h[2] * math.exp(-(t - h[0]) * 7.0) for h in self.hits[a:b]))


# ---------------------------------------------------------------- HUD
class Hud:
    def __init__(self):
        self.items = []     # (id, (x0,y0,x1,y1), alpha)

    def _panel(self, w, h, alpha=0.68):
        im = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
        ImageDraw.Draw(im).rounded_rectangle((0, 0, w - 1, h - 1), int(20 * K), fill=NAVY + (int(255 * alpha),), outline=(255, 255, 255, 80), width=max(1, int(2 * K)))
        return im

    def put(self, frame, im, xy, ident, alpha=1.0):
        x, y = int(xy[0]), int(xy[1])
        m = im.getchannel("A")
        if alpha < 0.999:
            m = m.point(lambda p: int(p * alpha))
        frame.paste(im.convert("RGB"), (x, y), m)
        self.items.append((ident, (x, y, x + im.width, y + im.height), round(alpha, 3)))

    def _fit(self, dr, text, size, maxw):
        s = size
        while s > 10 and dr.textlength(text, font=font(s)) > maxw:
            s -= 2
        return s

    # -- panels
    def panel_main(self, frame, year_txt, month_txt, rows, ident="A", sub=None):
        w, hgt = 520 * K, (170 + 28 + len(rows) * 74 + 24) * K
        im = self._panel(w, hgt)
        d = ImageDraw.Draw(im)
        ysz = self._fit(d, year_txt, 150 * K, w - 48 * K)
        d.text((24 * K, 12 * K + (150 * K - ysz) * 0.8), year_txt, font=font(ysz), fill=WHITE)
        if month_txt:
            d.text((w - 24 * K, 20 * K + 105 * K), month_txt, font=font(46 * K), fill=YEL, anchor="rs")
        y = 190 * K
        for lab, val in rows:
            d.text((24 * K, y + 40 * K), lab, font=font(32 * K), fill=(200, 215, 240), anchor="ls")
            d.text((w - 24 * K, y + 56 * K), val, font=font(64 * K), fill=YEL, anchor="rs")
            y += 74 * K
        self.put(frame, im, (36 * K, 30 * K), ident)
        return hgt

    def panel_legend(self, frame, counts, ident="B", title="STRENGTH (F / EF SCALE)"):
        w = 400 * K
        rows = [(5, "F5"), (4, "F4"), (3, "F3"), (2, "F2"), (1, "F1"), (0, "F0"), (-9, "UNRATED")]
        hgt = (62 + len(rows) * 44 + 54) * K
        im = self._panel(w, hgt)
        d = ImageDraw.Draw(im)
        d.text((22 * K, 18 * K), title, font=font(26 * K), fill=(200, 215, 240))
        y = 62 * K
        for c, lab in rows:
            d.rounded_rectangle((22 * K, y + 6 * K, 22 * K + 64 * K, y + 32 * K), int(6 * K), fill=F_COL[c], outline=WHITE, width=max(1, int(2 * K)))
            d.text((100 * K, y + 34 * K), lab, font=font(32 * K), fill=WHITE, anchor="ls")
            d.text((w - 22 * K, y + 34 * K), fmt(counts[c]), font=font(32 * K), fill=YEL, anchor="rs")
            y += 44 * K
        d.text((22 * K, hgt - 14 * K), "SOURCE: NOAA / NWS STORM PREDICTION CENTER", font=font(19 * K), fill=(170, 190, 225), anchor="ls")
        self.put(frame, im, (W - 36 * K - w, 30 * K), ident)

    def panel_list(self, frame, title, rows, ident="B"):
        w = 460 * K
        hgt = (66 + len(rows) * 44 + 52) * K
        im = self._panel(w, hgt)
        d = ImageDraw.Draw(im)
        d.text((22 * K, 18 * K), title, font=font(26 * K), fill=(200, 215, 240))
        y = 62 * K
        for col, lab, val in rows:
            d.rounded_rectangle((22 * K, y + 6 * K, 22 * K + 40 * K, y + 32 * K), int(6 * K), fill=col, outline=WHITE, width=max(1, int(2 * K)))
            d.text((76 * K, y + 34 * K), lab, font=font(30 * K), fill=WHITE, anchor="ls")
            d.text((w - 22 * K, y + 34 * K), val, font=font(30 * K), fill=YEL, anchor="rs")
            y += 44 * K
        d.text((22 * K, hgt - 14 * K), "SOURCE: NOAA / NWS STORM PREDICTION CENTER", font=font(19 * K), fill=(170, 190, 225), anchor="ls")
        self.put(frame, im, (W - 36 * K - w, 30 * K), ident)

    def heat_legend(self, frame, lut, ident="B"):
        w, hgt = 400 * K, 150 * K
        im = self._panel(w, hgt)
        d = ImageDraw.Draw(im)
        d.text((22 * K, 18 * K), "TORNADOES PER AREA", font=font(26 * K), fill=(200, 215, 240))
        for x in range(int(w - 44 * K)):
            c = tuple(int(v) for v in lut[int(x / (w - 44 * K) * 255)][:3])
            d.line([(22 * K + x, 58 * K), (22 * K + x, 92 * K)], fill=c)
        d.text((22 * K, 120 * K), "FEWER", font=font(28 * K), fill=WHITE, anchor="ls")
        d.text((w - 22 * K, 120 * K), "MORE", font=font(28 * K), fill=WHITE, anchor="rs")
        self.put(frame, im, (W - 36 * K - w, 30 * K), ident)

    def caption(self, frame, text, alpha):
        d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        size = self._fit(d0, text, 50 * K, W - 72 * K - 60 * K)
        tw = d0.textlength(text, font=font(size))
        w, hgt = tw + 60 * K, 100 * K
        im = self._panel(w, hgt, 0.74)
        ImageDraw.Draw(im).text((30 * K, hgt / 2 + 18 * K), text, font=font(size), fill=WHITE, anchor="ls")
        self.put(frame, im, (36 * K, H - 36 * K - hgt), "CAP", alpha)

    def card(self, frame, lines, alpha, ident="CARD"):
        """lines: [(text, size, colour)]"""
        d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        maxw = W - 72 * K - 40 * K
        fitted = []
        for text, size, col in lines:
            s = self._fit(d0, text, size * K, maxw - 60 * K)
            fitted.append((text, s, col, d0.textlength(text, font=font(s))))
        w = max(f[3] for f in fitted) + 80 * K
        hgt = sum(f[1] * 1.22 for f in fitted) + 50 * K
        im = self._panel(w, hgt, 0.78)
        d = ImageDraw.Draw(im)
        y = 26 * K
        for text, s, col, _ in fitted:
            d.text((40 * K, y + s), text, font=font(s), fill=col, anchor="ls")
            y += s * 1.22
        self.put(frame, im, ((W - w) / 2, H - 36 * K - hgt), ident, alpha)


def event_card_lines(D: Data, e):
    i = e["idx"]
    dt = f"{MONTHS[D.mo[i] - 1]} {D.dy[i]}, {D.yr[i]}"
    lines = [(e["title"], 84, YEL), (f"{dt}  ·  {D.scale_name(i)}{D.mag[i]}", 46, WHITE),
             (f"{fmt(int(D.fat[i]))} LIVES LOST  ·  {fmt(int(D.inj[i]))} INJURED", 58, WHITE),
             (f"PATH {D.ln[i]:.1f} MI  ·  WIDTH {fmt(int(D.wd[i]))} YD", 40, (200, 215, 240))]
    if e.get("extra"):
        lines.append((e["extra"], 40, YEL))
    return lines


# ---------------------------------------------------------------- frame renderer
class Renderer:
    def __init__(self, scene=None):
        self.D = Data()
        self.TL = Timeline(self.D)
        self.painter = Painter(self.D, self.TL)
        self.dir = Director(self.D, self.TL)
        self.scene = scene or core.Scene(W, H)
        D = self.D
        # bars
        cent = self.dir.cent
        counts = {s: int((D.st == s).sum()) for s in cent}
        self.bar_counts = counts
        self.bar_rank = sorted(counts, key=lambda s: -counts[s])
        mx = max(counts.values())
        self.bars = []
        for s, (lon, lat) in cent.items():
            x, y, z = world3(lon, lat)
            v = counts[s] / mx
            self.bars.append((s, x, float(y), z, v))
        self.hud_log = []

    def bar_color(self, v):
        stops = [0, 0.15, 0.4, 0.7, 1.0]
        cols = [(255, 214, 0), (255, 150, 0), (255, 70, 0), (215, 0, 60), (130, 0, 190)]
        return tuple(int(np.interp(v, stops, [c[j] for c in cols])) for j in range(3))

    def frame(self, n):
        D, TL, P = self.D, self.TL, self.painter
        t = n / FPS
        ph = TL.phase_at(t)
        mode = ph["mode"]
        P.update(t, mode)
        p_heat = TL.heat_year(t) if mode == "heat" else None
        tex = P.texture(t, mode, p_heat)
        cam = self.dir.cam(t)
        kz = min(1.0, max(0.0, (3.2 - cam.dist) / 1.6))
        if mode == "main" and kz > 0.02:
            tex = Image.blend(P.base, tex, 1.0 - 0.88 * kz)
        self.scene.set_texture(tex)
        vort, bars = [], []
        if mode == "main":
            vort = self._vortices(t, ph, cam)
        elif mode == "bars":
            g = smooth((t - ph["t0"] - 3.0) / 9.0)
            for s, x, y, z, v in self.bars:
                hgt = max(0.004, v * 0.62 * g)
                bars.append((x, y, z, 0.085, hgt, self.bar_color(v)))
        dim = 0.9 if mode in ("heat", "bars") else 1.0
        img = self.scene.render(cam, tm=t, dim=dim, vortices=vort, bars=bars)
        if ph["kind"] == "event":
            self.event_overlay(img, t, ph, cam)
        # flash
        fl = self.dir.flash(t)
        if fl > 0.01:
            img = Image.blend(img, Image.new("RGB", img.size, (255, 248, 225)), fl)
        meta = self.hud(img, t, ph, cam)
        meta.update(n=n, t=round(t, 3), mode=mode, kind=ph["kind"])
        return img, meta

    def event_overlay(self, img, t, ph, cam):
        D = self.D
        e = ph["ev"]
        i = e["idx"]
        f = self.dir.vortex_frac(e["key"], t, ph["t0"])
        if f <= 0:
            return
        ss = np.linspace(0, f, 28)
        lon = D.slon[i] + (D.elon[i] - D.slon[i]) * ss
        lat = D.slat[i] + (D.elat[i] - D.slat[i]) * ss
        pts = world3(lon, lat, lift=0.004)
        px, ok = cam.project(pts, W, H)
        if not ok.all():
            return
        sc = min(1.0, max(0.3, 1.8 / cam.dist))
        d = ImageDraw.Draw(img)
        line = [tuple(p) for p in px]
        d.line(line, fill=WHITE, width=max(2, int(17 * K * sc)), joint="curve")
        d.line(line, fill=F_COL[int(D.mag[i])], width=max(1, int(10 * K * sc)), joint="curve")

    def _vortices(self, t, ph, cam):
        D, TL = self.D, self.TL
        out = []
        a = bisect_left(TL.ta, t - 5.8)
        b = bisect_right(TL.ta, t)
        evs = {e["idx"]: e for e in D.events}
        cand = []
        for i in range(a, b):
            an = TL.anim[i]
            if an <= 0:
                continue
            tt = t - TL.ta[i]
            if tt > an + 0.4:
                continue
            cand.append((0 if i in evs else 1, tt, i))
        cand.sort()
        for _, tt, i in cand[:6]:
            an = TL.anim[i]
            f = min(1.0, tt / an)
            al = smooth(tt / 0.25) * (1 - smooth((tt - an) / 0.4))
            lon = D.slon[i] + (D.elon[i] - D.slon[i]) * f
            lat = D.slat[i] + (D.elat[i] - D.slat[i]) * f
            p = world3(lon, lat)
            size = (0.14 if i in evs else 0.032) * cam.dist
            out.append((float(p[0]), float(p[1]), float(p[2]), size, float(al)))
        return out

    # ---- HUD
    def hud(self, img, t, ph, cam):
        D, TL = self.D, self.TL
        H_ = Hud()
        mode, kind = ph["mode"], ph["kind"]
        meta = dict(count=None, deaths=None, cls=None, dt=None, caps=None)
        if mode == "intro":
            n = TL.intro_count(t)
            H_.panel_main(img, "F5", "", [("F5 / EF5 TORNADOES", f"{n} / {len(TL.f5)}")], "A")
            meta.update(f5=n, f5_total=len(TL.f5))
            cap, al = TL.caption_at(t)
            if cap:
                H_.caption(img, cap, al)
            if t > INTRO_S - 7.0:
                a = min(1.0, (t - (INTRO_S - 7.0)) / 0.8) * (1 - smooth((t - (INTRO_S - 1.0)) / 1.0))
                H_.card(img, [("EVERY TORNADO IN U.S. HISTORY, MAPPED", 82, YEL), (f"{fmt(D.n)} TORNADOES  ·  {Y0}-{Y1}", 56, WHITE)], a, "TITLE")
        elif mode == "main":
            k, deaths, cls = TL.stats(t)
            dt = TL.dt_at(t)
            yr = int(dt)
            if kind == "run":
                span = (Y1 - Y0)
                mo = int((dt - yr) * 12.0)
                month = MONTHS[min(11, mo)][:3]
            else:
                i = ph["ev"]["idx"]
                month = MONTHS[D.mo[i] - 1][:3]
            H_.panel_main(img, str(yr), month, [("TORNADOES", fmt(k)), ("LIVES LOST", fmt(deaths))], "A")
            H_.panel_legend(img, cls, "B")
            meta.update(count=k, deaths=deaths, cls={str(c): v for c, v in cls.items()}, dt=dt)
            if kind == "event":
                e = ph["ev"]
                tl = t - ph["t0"]
                if 3.2 < tl < EV_S - 2.2:
                    a = min(1.0, (tl - 3.2) / 0.5, (EV_S - 2.2 - tl) / 0.5)
                    H_.card(img, event_card_lines(D, e), a)
            else:
                cap, al = TL.caption_at(t)
                if cap:
                    H_.caption(img, cap, al)
        elif mode == "heat":
            y = TL.heat_year(t)
            full = int(y)
            yr = Y0 + min(max(full, 1), Y1 - Y0 + 1) - 1
            ky = int(TL.cum_year[yr - Y0]) if full >= 1 else 0
            H_.panel_main(img, str(yr), "", [("TORNADOES SINCE 1950", fmt(ky))], "A")
            H_.heat_legend(img, self.painter._lut if self.painter._heat is not None else np.zeros((256, 4), "u1"), "B")
            meta.update(heat_year=yr, heat_count=ky)
            cap, al = TL.caption_at(t)
            if cap:
                H_.caption(img, cap, al)
        elif mode == "bars":
            H_.panel_main(img, "STATES", "", [("TORNADOES", fmt(D.n))], "A")
            rows = [(self.bar_color(self.bar_counts[s] / max(self.bar_counts.values())), STATE_NAMES[s], fmt(self.bar_counts[s])) for s in self.bar_rank[:8]]
            H_.panel_list(img, "MOST TORNADOES, 1950-2025", rows, "B")
            cap, al = TL.caption_at(t)
            if cap:
                H_.caption(img, cap, al)
        else:
            k, deaths, cls = TL.stats(t)
            H_.panel_main(img, f"{Y0}-{Y1}", "", [("TORNADOES", fmt(k)), ("LIVES LOST", fmt(deaths))], "A")
            H_.panel_legend(img, cls, "B")
            meta.update(count=k, deaths=deaths, cls={str(c): v for c, v in cls.items()}, dt=float(Y1 + 1))
            a = smooth((t - ph["t0"] - 4.0) / 1.0)
            if a > 0:
                H_.card(img, [("EVERY TORNADO IN U.S. HISTORY, MAPPED", 70, YEL),
                              ("DATA: NOAA / NWS STORM PREDICTION CENTER, 1950-2025 (2025 PRELIMINARY)", 36, WHITE),
                              (f"{fmt(D.n)} TORNADOES  ·  {fmt(int(D.fat.sum()))} LIVES LOST  ·  WHITE OUTLINES: THE {len(TL.f5)} STRONGEST", 34, (200, 215, 240))], a, "END")
        meta["panels"] = H_.items
        return meta


# ---------------------------------------------------------------- pixel QA (every second)
def pixel_qa(frame: Image.Image):
    a = np.asarray(frame)[::6, ::6]
    problems = []
    if a.std() < 8:
        problems.append("uniform")
    mx_col = a.max(axis=(0, 2))
    mx_row = a.max(axis=(1, 2))
    if (mx_col < 14).any():
        problems.append("black column stripe")
    if (mx_row < 14).any():
        problems.append("black row stripe")
    h, w = a.shape[:2]
    for by in range(4):
        for bx in range(8):
            blk = a[by * h // 4:(by + 1) * h // 4, bx * w // 8:(bx + 1) * w // 8]
            if blk.mean() < 25:
                problems.append(f"dark block {bx},{by}")
    ocean = (np.abs(a.astype(int) - np.array(OCEAN)).max(axis=2) < 14).mean()
    if ocean > 0.62:
        problems.append(f"map too empty (ocean {ocean:.2f})")
    return problems, float(ocean)


# ---------------------------------------------------------------- segment CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--maxframes", type=int, default=0)
    ap.add_argument("--crf", default="20")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thumbs").mkdir(exist_ok=True)
    r = Renderer()
    total = int(r.TL.total * FPS)
    per = math.ceil(total / a.of)
    n0, n1 = a.seg * per, min(total, (a.seg + 1) * per)
    if a.maxframes:
        n1 = min(n1, n0 + a.maxframes)
    print(f"segment {a.seg}/{a.of}: frames {n0}-{n1} of {total} ({r.TL.total:.1f}s) {W}x{H} tex {TW} GL {r.scene.renderer}", flush=True)
    mp4 = out / f"seg_{a.seg:02d}.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", a.crf, "-maxrate", "26M" if H >= 2160 else "10M", "-bufsize", "52M" if H >= 2160 else "20M",
                            "-g", "48", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    qa = open(out / f"qa_{a.seg:02d}.jsonl", "w")
    t0 = time.time()
    for n in range(n0, n1):
        img, meta = r.frame(n)
        enc.stdin.write(img.tobytes())
        if n % FPS == 0:
            pr, oc = pixel_qa(img)
            meta["pixel_problems"], meta["ocean"] = pr, round(oc, 3)
            if n % (FPS * 12) == 0:
                img.resize((640, 360), Image.LANCZOS).save(out / "thumbs" / f"t{n // FPS:04d}.jpg", quality=82)
        qa.write(json.dumps(meta) + "\n")
        if (n - n0) % 48 == 0:
            print(f"  frame {n} ({(time.time() - t0) / (n - n0 + 1):.2f}s/frame)", flush=True)
    enc.stdin.close()
    enc.wait()
    qa.close()
    print(f"done {n1 - n0} frames in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
