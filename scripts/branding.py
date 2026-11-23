"""Branding options (3 avatars + 3 banners) for the Storm Atlas USA channel (@StormAtlasUSA).

Real data only: NOAA SPC tornado database 1950-2025 (data/raw/tornadoes.csv),
NOAA NHC HURDAT2 1851-2025 (data/raw/hurdat2.txt), US Census state shapes.
Run:  python scripts/branding.py  ->  branding/options/*.png
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mapviz.core import albers, states  # noqa: E402

FONT = str(ROOT / "assets" / "fonts" / "Anton-Regular.ttf")
OUT = ROOT / "branding" / "options"
OUT.mkdir(parents=True, exist_ok=True)

OCEAN = (64, 170, 235)
OCEAN_DEEP = (20, 92, 170)
LAND = (250, 240, 214)
BORDER = (40, 40, 60)
YEL = (255, 214, 0)
F_COL = {0: (120, 200, 90), 1: (255, 225, 40), 2: (255, 150, 0), 3: (240, 60, 30), 4: (205, 0, 90), 5: (130, 0, 170)}
NAME = "STORM ATLAS USA"
TAG = "U.S. disasters, mapped from real NOAA & USGS data"


def tornadoes():
    rows = []
    with open(ROOT / "data" / "raw" / "tornadoes.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                m = int(r["mag"])
                sl, so, el, eo = map(float, (r["slat"], r["slon"], r["elat"], r["elon"]))
            except ValueError:
                continue
            if m < 0 or sl == 0:
                continue
            if el == 0:
                el, eo = sl, so
            rows.append((m, sl, so, el, eo))
    return rows


def hurricanes():
    tracks, cur = [], None
    for line in open(ROOT / "data" / "raw" / "hurdat2.txt", encoding="utf-8"):
        p = [x.strip() for x in line.split(",")]
        if p[0].startswith("AL") and len(p) < 8:
            cur = []
            tracks.append(cur)
            continue
        lat = float(p[4][:-1]) * (1 if p[4][-1] == "N" else -1)
        lon = float(p[5][:-1]) * (-1 if p[5][-1] == "W" else 1)
        cur.append((lat, lon, int(p[6])))
    return tracks


class Frame:
    """Albers -> pixel mapping fitted to a lon/lat box."""

    def __init__(self, w, h, lon0, lat0, lon1, lat1, fit=False):
        xs, ys = albers([lon0, lon1, lon0, lon1, (lon0 + lon1) / 2], [lat0, lat0, lat1, lat1, lat1])
        self.x0, self.x1, self.y0, self.y1 = min(xs), max(xs), min(ys), max(ys)
        self.s = (min if fit else max)(w / (self.x1 - self.x0), h / (self.y1 - self.y0))
        self.cx, self.cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        self.w, self.h = w, h

    def px(self, lon, lat):
        x, y = albers(lon, lat)
        return (np.asarray(x) - self.cx) * self.s + self.w / 2, self.h / 2 - (np.asarray(y) - self.cy) * self.s

    def xy(self, x, y):
        return (x - self.cx) * self.s + self.w / 2, self.h / 2 - (y - self.cy) * self.s


def base_map(fr: Frame, border_w=2, ocean=OCEAN):
    img = Image.new("RGB", (fr.w, fr.h), ocean)
    d = ImageDraw.Draw(img)
    for rings in states().values():
        for r in rings:
            x, y = fr.xy(r[:, 0], r[:, 1])
            d.polygon(list(zip(x, y)), fill=LAND)
    lines = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dl = ImageDraw.Draw(lines)
    for rings in states().values():
        for r in rings:
            x, y = fr.xy(r[:, 0], r[:, 1])
            pts = list(zip(x, y))
            dl.line(pts + pts[:1], fill=BORDER + (255,), width=border_w)
    return img, lines


def draw_tornadoes(img, fr, rows, minmag=0, width_k=1.0):
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    core = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dg, dc = ImageDraw.Draw(glow), ImageDraw.Draw(core)
    for m, sl, so, el, eo in sorted(rows, key=lambda r: r[0]):
        if m < minmag:
            continue
        x, y = fr.px([so, eo], [sl, el])
        w = max(1, int((1 + m * 1.2) * width_k))
        seg = [(x[0], y[0]), (x[1], y[1])]
        if m >= 3:
            dg.line(seg, fill=F_COL[m] + (150,), width=w * 4)
        dc.line(seg, fill=F_COL[m] + (255,), width=w)
    img.paste(glow.filter(ImageFilter.GaussianBlur(4 * width_k)), (0, 0), glow.filter(ImageFilter.GaussianBlur(4 * width_k)))
    img.paste(core, (0, 0), core)


def outlined(d, xy, text, size, fill=(255, 255, 255), stroke=10, anchor="mm", sfill=(15, 15, 30)):
    f = ImageFont.truetype(FONT, size)
    d.text(xy, text, font=f, fill=fill, stroke_width=stroke, stroke_fill=sfill, anchor=anchor)


def circle_crop(img):
    s = img.size[0]
    m = Image.new("L", (s * 4, s * 4), 0)
    ImageDraw.Draw(m).ellipse((0, 0, s * 4, s * 4), fill=255)
    m = m.resize((s, s), Image.LANCZOS)
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    out.paste(img, (0, 0), m)
    return out


def preview(av, name):
    """Save the avatar + how it reads at 98 px and 40 px (YouTube sizes)."""
    av.save(OUT / f"{name}.png")
    sheet = Image.new("RGB", (800 + 98 + 40 + 60, 800), (245, 245, 245))
    sheet.paste(av, (0, 0), av)
    s98 = av.resize((98, 98), Image.LANCZOS)
    s40 = av.resize((40, 40), Image.LANCZOS)
    sheet.paste(s98, (820, 20), s98)
    sheet.paste(s40, (940, 20), s40)
    sheet.save(OUT / f"{name}_preview.png")


# ---------------- avatars (800x800) ----------------
def avatar_a(rows):
    """A: Tornado Alley at night-blue, every F2+ track glowing, big 'X' pin."""
    s = 800
    fr = Frame(s, s, -103, 30, -84, 44)
    img, lines = base_map(fr, 3, OCEAN_DEEP)
    img = Image.blend(img, Image.new("RGB", img.size, (30, 40, 90)), 0.55)
    img.paste(lines, (0, 0), lines)
    draw_tornadoes(img, fr, rows, minmag=3, width_k=1.1)
    d = ImageDraw.Draw(img)
    outlined(d, (s / 2, s / 2 + 10), "X", 520, fill=YEL, stroke=22)
    return circle_crop(img)


def avatar_b(tracks):
    """B: spiral 'eye' symbol in hurricane colours over the Gulf, real HURDAT2 tracks behind."""
    s = 800
    fr = Frame(s, s, -100, 18, -70, 42)
    img, lines = base_map(fr, 3)
    img.paste(lines, (0, 0), lines)
    d = ImageDraw.Draw(img, "RGBA")
    for t in tracks:
        if max(p[2] for p in t) < 96:
            continue
        x, y = fr.px([p[1] for p in t], [p[0] for p in t])
        d.line(list(zip(x, y)), fill=(255, 255, 255, 110), width=3)
    # spiral arms, drawn as overlapping discs = smooth tapered strokes
    cx, cy = s / 2, s / 2
    for pass_ in (0, 1):
        for arm in range(3):
            for i in range(900):
                k = i / 900
                a = k * 2.1 * math.pi + arm * 2 * math.pi / 3
                r = 50 + k * 330
                x, y = cx + r * math.cos(a), cy + r * math.sin(a)
                w = (34 - 22 * k) + (9 if pass_ == 0 else 0)
                col = (20, 20, 40, 255) if pass_ == 0 else (255, int(225 - 195 * k), int(50 * (1 - k)), 255)
                d.ellipse((x - w, y - w, x + w, y + w), fill=col)
    d.ellipse((cx - 46, cy - 46, cx + 46, cy + 46), fill=(255, 255, 255, 255), outline=(20, 20, 40, 255), width=10)
    return circle_crop(img)


def avatar_c(rows):
    """C: US silhouette filled with all 73k tracks + shock-wave rings, 'SA' badge."""
    s = 800
    fr = Frame(s, s, -125, 24, -66, 50, fit=True)
    img = Image.new("RGB", (s, s), YEL)
    d = ImageDraw.Draw(img)
    for i, r in enumerate(range(520, 0, -70)):
        d.ellipse((s / 2 - r, s / 2 - r, s / 2 + r, s / 2 + r), outline=(255, 120, 0), width=10)
    m = Image.new("L", (s, s), 0)
    dm = ImageDraw.Draw(m)
    land = Image.new("RGB", (s, s), (35, 45, 95))
    for rings in states().values():
        for r in rings:
            x, y = fr.xy(r[:, 0], r[:, 1])
            dm.polygon(list(zip(x, y)), fill=255)
    draw_tornadoes(land, fr, rows, minmag=0, width_k=0.6)
    sh = m.filter(ImageFilter.MaxFilter(15))
    img.paste((20, 20, 40), (0, 0), sh)
    img.paste(land, (0, 0), m)
    d = ImageDraw.Draw(img)
    outlined(d, (s / 2, s * 0.80), "SA", 230, fill=(255, 255, 255), stroke=14)
    return circle_crop(img)


# ---------------- banners (2560x1440, safe area 1546x423 centred) ----------------
def banner_common(img, title_fill=(255, 255, 255)):
    d = ImageDraw.Draw(img)
    W, H = img.size
    outlined(d, (W / 2, H / 2 - 50), NAME, 170, fill=title_fill, stroke=14)
    outlined(d, (W / 2, H / 2 + 80), TAG, 54, fill=YEL, stroke=8)
    outlined(d, (W / 2, H / 2 + 160), "2 NEW MAPS EVERY WEEK", 46, fill=(255, 255, 255), stroke=7)
    return img


def banner_a(rows):
    """A: full US, every tornado 1950-2025 coloured by F-scale."""
    W, H = 2560, 1440
    fr = Frame(W, H, -118, 27, -72, 48)
    img, lines = base_map(fr, 3)
    img.paste(lines, (0, 0), lines)
    draw_tornadoes(img, fr, rows, minmag=0, width_k=1.3)
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rounded_rectangle((W / 2 - 860, H / 2 - 170, W / 2 + 860, H / 2 + 215), 40, fill=(15, 20, 50, 150))
    img.paste(shade, (0, 0), shade)
    return banner_common(img)


def banner_b(tracks):
    """B: every Atlantic major hurricane (Cat 3+) track 1851-2025, colour = wind speed."""
    W, H = 2560, 1440
    fr = Frame(W, H, -105, 12, -55, 46)
    img, lines = base_map(fr, 3, OCEAN_DEEP)
    img.paste(lines, (0, 0), lines)
    d = ImageDraw.Draw(img, "RGBA")

    def col(v):
        return (130, 220, 255) if v < 64 else (255, 230, 60) if v < 96 else (255, 130, 0) if v < 113 else (240, 30, 40) if v < 137 else (200, 0, 200)

    for t in tracks:
        if max(p[2] for p in t) < 96:
            continue
        x, y = fr.px([p[1] for p in t], [p[0] for p in t])
        for j in range(len(t) - 1):
            d.line([(x[j], y[j]), (x[j + 1], y[j + 1])], fill=col(t[j][2]) + (230,), width=4)
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rounded_rectangle((W / 2 - 860, H / 2 - 170, W / 2 + 860, H / 2 + 215), 40, fill=(10, 15, 40, 170))
    img.paste(shade, (0, 0), shade)
    return banner_common(img)


def banner_c(rows):
    """C: split hazard strip - tornado tracks (left) | shock-wave rings (centre) | hurricane spiral (right) on bright map."""
    W, H = 2560, 1440
    fr = Frame(W, H, -125, 25, -66, 50)
    img, lines = base_map(fr, 3)
    img.paste(lines, (0, 0), lines)
    draw_tornadoes(img, fr, rows, minmag=3, width_k=1.6)
    d = ImageDraw.Draw(img, "RGBA")
    # shock-wave rings at Northridge 1994 epicentre (USGS: 34.213 N, 118.537 W)
    x, y = fr.px(-118.537, 34.213)
    for r in range(40, 420, 60):
        d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 60, 30, max(40, 255 - r // 2)), width=10)
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shade).rounded_rectangle((W / 2 - 860, H / 2 - 170, W / 2 + 860, H / 2 + 215), 40, fill=(15, 20, 50, 150))
    img.paste(shade, (0, 0), shade)
    return banner_common(img, title_fill=YEL)


def banner_preview(img, name):
    img.save(OUT / f"{name}.jpg", quality=92)
    # TV / desktop / phone crops
    W, H = img.size
    desk = img.crop((W / 2 - 1273, H / 2 - 211, W / 2 + 1273, H / 2 + 212)).resize((1273, 211))
    phone = img.crop((W / 2 - 773, H / 2 - 211, W / 2 + 773, H / 2 + 212)).resize((773, 211))
    sheet = Image.new("RGB", (1300, 830), (240, 240, 240))
    sheet.paste(img.resize((640, 360)), (10, 10))
    sheet.paste(desk, (10, 385))
    sheet.paste(phone, (10, 610))
    sheet.save(OUT / f"{name}_preview.jpg", quality=88)


if __name__ == "__main__":
    rows = tornadoes()
    tracks = hurricanes()
    print("tornado tracks:", len(rows), "hurricane tracks:", len(tracks))
    preview(avatar_a(rows), "avatar_A_tornado_x")
    preview(avatar_b(tracks), "avatar_B_hurricane_eye")
    preview(avatar_c(rows), "avatar_C_usa_edm")
    banner_preview(banner_a(rows), "banner_A_every_tornado")
    banner_preview(banner_b(tracks), "banner_B_major_hurricanes")
    banner_preview(banner_c(rows), "banner_C_hazard_mix")
    print("done ->", OUT)
