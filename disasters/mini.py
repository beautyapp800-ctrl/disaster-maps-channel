"""Stylized 'miniature world' OpenGL scene (CPU OpenGL / Mesa llvmpipe on GitHub runners).

Look: clean, warm and saturated, sharp sun with real shadow mapping, no haze or fog.
  nation  : NLCD 2021 land cover with a cartographic palette + USGS 3DEP hillshade (outside the US: graded NASA Blue Marble)
  region  : the same at 73 m and 16 m per texel around each famous tornado
  town    : OpenStreetMap ground (land use, parks, water, roads with markings, rails) at ~1 m per texel,
            extruded OSM buildings (real footprints, procedural facades with windows, pitched or flat roofs, rooftop units),
            procedural trees on parks / yards / woods, 3D bridge decks, place labels
  events  : ray-marched rotating funnel with a dust cloud, tumbling debris, dark rotating mother cloud, collapsing buildings,
            damage swath on the ground
"""
from __future__ import annotations

import gzip
import json
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import core
from .core import FR, KM_PER_UNIT, TH, TW, WORLD_H, WORLD_W, height_at, height_world, relief

ROOT = core.ROOT
M_PER_UNIT = KM_PER_UNIT * 1000.0
SUN = np.array([-0.5845, 0.5632, 0.5845])
SKY_TOP, SKY_HOR = (0.36, 0.62, 0.93), (0.78, 0.88, 0.98)
L1_KM, L2_KM, L2_N, L1_N = 16.0, 6.0, 6144, 4096
TREE_KM = 7.0

# ------------------------------------------------------------------ palettes (NLCD classes)
NAT_PAL = {11: (52, 148, 222), 12: (238, 244, 250), 21: (172, 202, 120), 22: (214, 196, 160), 23: (206, 182, 152), 24: (190, 166, 142),
           31: (226, 206, 166), 41: (98, 166, 74), 42: (56, 130, 68), 43: (82, 152, 72), 52: (204, 192, 126), 71: (198, 212, 122),
           81: (178, 208, 98), 82: (228, 206, 120), 90: (94, 154, 102), 95: (128, 182, 150)}
LOC_PAL = {**NAT_PAL, **{21: (122, 174, 86), 22: (118, 172, 84), 23: (132, 176, 98), 24: (198, 194, 186), 41: (84, 150, 64), 42: (54, 120, 60),
                           43: (72, 140, 62), 82: (220, 200, 120), 81: (138, 184, 88), 71: (150, 188, 96)}}


def _lut(pal):
    lut = np.zeros((256, 3), np.float32)
    lut[:] = (176, 206, 104)
    for k, v in pal.items():
        lut[k] = v
    return lut


def _shore(mask, blur_px):
    """water mask -> shore field: 0.5 exactly on the shoreline, >0.5 water."""
    im = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_px))
    return np.asarray(im)


def _rgba(rgb, shore):
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), shore.astype(np.uint8)]), "RGBA")


def _hs_mult(hs):
    h = hs.astype(np.float32) / 180.0
    return np.clip(1.0 + (h - 1.0) * 0.85, 0.55, 1.28)[..., None]


@lru_cache(None)
def nation_image(w, h):
    wd = ROOT / "data" / "world"
    sat = _stylize_sat(np.asarray(Image.open(wd / "sat_8192.jpg").convert("RGB").resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0)
    rgb = sat
    nf = wd / "nlcd_8192.png"
    cls = None
    if nf.exists():
        cls = np.asarray(Image.open(nf).resize((w, h), Image.NEAREST))
        col = _lut(NAT_PAL)[cls]
        hs = wd / "hs_8192.jpg"
        if hs.exists():
            col *= _hs_mult(np.asarray(Image.open(hs).convert("L").resize((w, h), Image.BILINEAR)))
        us = (cls > 0)[..., None]
        rgb = np.where(us, col, sat)
    water = np.asarray(Image.open(wd / "water_4096.png").convert("L").resize((w, h), Image.BILINEAR)) > 127
    if cls is not None:
        water |= cls == 11
    return _rgba(rgb, _shore(water, max(1.5, w / 3000)))


def _stylize_sat(sat):
    lum = sat @ np.array([0.3, 0.59, 0.11], np.float32)
    sat = lum[..., None] + (sat - lum[..., None]) * 1.35
    return np.power(np.clip(sat, 0, 1), 0.58) * 255 * np.array([1.02, 1.02, 0.86], np.float32)


@lru_cache(None)
def outer_image():
    f = ROOT / "data" / "world" / "outer_4096.jpg"
    if not f.exists():
        return None
    a = np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    water = (b > r * 1.25) & (b > g * 1.02) & ((r + g + b) < 1.2)
    return _rgba(_stylize_sat(a), _shore(water, 1.5))


def region_image(key, lvl):
    d = ROOT / "data" / "region"
    cls = np.asarray(Image.open(d / f"{key}_{lvl}.png"))
    col = _lut(NAT_PAL if lvl == "reg" else LOC_PAL)[cls]
    if lvl == "mid":                      # 30 m land-cover cells: field patchwork + soft edges instead of visible pixel blocks
        col = _fields(col, cls, cls.shape[0], 64.0)
        col = np.asarray(Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6))).astype(np.float32)
    hf = d / f"{key}_{lvl}_hs.jpg"
    if hf.exists():
        col *= _hs_mult(np.asarray(Image.open(hf).convert("L")))
    return _rgba(col, _shore(cls == 11, 2.0)), cls


def noise_image(n=1024, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(3):
        f = np.fft.fft2(rng.standard_normal((n, n)))
        fx = np.fft.fftfreq(n)[:, None]
        fy = np.fft.fftfreq(n)[None, :]
        amp = 1.0 / np.maximum(np.hypot(fx, fy), 1.0 / n) ** (1.6 + 0.25 * k)
        a = np.real(np.fft.ifft2(f * amp))
        a = (a - a.min()) / (a.max() - a.min())
        out.append((a * 255).astype("u1"))
    return Image.merge("RGB", [Image.fromarray(o) for o in out])


# ------------------------------------------------------------------ OpenStreetMap town model
ROAD_W = {"motorway": 26, "trunk": 22, "primary": 17, "secondary": 14, "tertiary": 12, "unclassified": 9, "residential": 9,
          "living_street": 7, "service": 5, "pedestrian": 6, "motorway_link": 9, "trunk_link": 9, "primary_link": 9,
          "secondary_link": 8, "tertiary_link": 8}
GROUND = {  # land use -> (rgb, tree density per 100 m2, blocks trees)
    "residential": ((118, 172, 84), 0.24), "commercial": ((204, 200, 192), 0.0), "retail": ((206, 200, 190), 0.0),
    "industrial": ((196, 192, 184), 0.0), "construction": ((206, 180, 140), 0.0), "farmland": ((222, 202, 122), 0.0),
    "farmyard": ((200, 190, 140), 0.02), "meadow": ((172, 206, 104), 0.01), "grass": ((128, 192, 84), 0.02),
    "recreation_ground": ((122, 190, 80), 0.04), "cemetery": ((132, 184, 98), 0.06), "religious": ((180, 190, 150), 0.03),
    "forest": ((70, 128, 58), 0.75), "wood": ((70, 128, 58), 0.75), "scrub": ((150, 168, 96), 0.18), "grassland": ((178, 204, 108), 0.01),
    "wetland": ((110, 168, 128), 0.05), "sand": ((232, 214, 168), 0.0), "beach": ((236, 220, 172), 0.0),
    "park": ((116, 194, 78), 0.12), "garden": ((120, 196, 84), 0.10), "golf_course": ((126, 200, 86), 0.05), "pitch": ((96, 176, 76), 0.0),
    "playground": ((214, 190, 150), 0.0), "stadium": ((150, 160, 150), 0.0), "sports_centre": ((160, 190, 140), 0.0),
    "track": ((196, 112, 84), 0.0), "nature_reserve": ((96, 156, 74), 0.4), "parking": ((118, 120, 126), 0.0),
    "school": ((214, 204, 176), 0.02), "hospital": ((214, 204, 186), 0.02), "university": ((212, 204, 182), 0.04), "college": ((212, 204, 182), 0.04),
    "orchard": ((140, 180, 90), 0.3), "vineyard": ((160, 180, 100), 0.0), "allotments": ((150, 180, 100), 0.02), "railway": ((170, 160, 150), 0.0),
    "quarry": ((200, 190, 170), 0.0), "landfill": ((180, 170, 150), 0.0), "apron": ((180, 180, 180), 0.0), "runway": ((110, 112, 118), 0.0),
    "taxiway": ((120, 122, 128), 0.0),
}
WATER_RGB = (52, 148, 222)
LABEL_KINDS = {"city": 0, "town": 0, "village": 1, "suburb": 1, "quarter": 1, "neighbourhood": 2, "hamlet": 2, "hospital": 2, "university": 2,
               "college": 3, "stadium": 2, "park": 3, "golf_course": 3, "water": 3, "river": 3}


def _kind(t):
    for k in ("natural", "landuse", "leisure", "amenity", "aeroway"):
        if k in t:
            v = t[k]
            if v in ("water", "reservoir", "basin", "pond", "riverbank"):
                return "water"
            return v
    if t.get("waterway") == "riverbank":
        return "water"
    return None


def _fields(img, cls, n, km):
    """cropland and pasture as a patchwork of fields (about 400 m parcels, mile-grid roads) instead of one flat colour"""
    crop = np.isin(cls, (81, 82, 71, 52, 31))
    if not crop.any():
        return img
    cell = max(8, int(round(400.0 * n / (km * 1000.0))))
    ij = np.indices((n, n))
    yy, xx = ij[0] + 0.37 * ij[1] * 0.0, ij[1]
    gx = (xx + 0.5 * cell * ((yy // cell) % 2)) // cell          # brick-like parcels
    gy = yy // cell
    h = np.sin(gx * 12.9898 + gy * 78.233) * 43758.5453
    h = h - np.floor(h)
    h2 = np.sin(gx * 39.3468 + gy * 11.135) * 24634.6345
    h2 = h2 - np.floor(h2)
    bright = (0.84 + 0.30 * h)[..., None]
    tint = np.stack([1.0 + 0.16 * (h2 - 0.5), 1.0 + 0.05 * (h2 - 0.5), 1.0 - 0.22 * (h2 - 0.5)], -1)
    f = crop[..., None]
    out = np.where(f, img * bright * tint, img)
    if cell >= 10:                            # faint field boundaries only where a field is big enough to read
        edge = ((xx % cell) < 1) | ((yy % cell) < 1)
        out = np.where(f & edge[..., None], out * 0.90, out)
    return out.astype(np.float32)


class Town:
    """OSM model of one town in world coordinates, relative to the window centre O (float64 on the CPU)."""

    def __init__(self, key):
        self.key = key
        osm = json.load(gzip.open(ROOT / "data" / "osm" / f"{key}.json.gz", "rt", encoding="utf-8"))
        self.osm = osm
        clat, clon = osm["center"]
        rj = ROOT / "data" / "region" / f"{key}.json"
        self.region = json.loads(rj.read_text()) if rj.exists() else None
        cx, cz = (float(v) for v in FR.world(clon, clat))
        self.O = np.array([cx, 0.0, cz])
        self.clat, self.clon = clat, clon
        self.feats = []
        for f in osm["feats"]:
            g = np.asarray(f["g"], np.float64).reshape(-1, 2)
            x, z = FR.world(g[:, 0], g[:, 1])
            self.feats.append((f["t"], np.stack([x - cx, z - cz], 1)))
        self.labels = []
        for lb in osm["labels"]:
            k = LABEL_KINDS.get(lb["kind"])
            nm = lb["name"]
            if k is None or any(ch.isdigit() for ch in nm) or "noname" in nm.lower() or "holding pond" in nm.lower() or len(nm) > 28:
                continue
            x, z = FR.world(lb["lon"], lb["lat"])
            self.labels.append(dict(name=nm.upper(), pri=k, x=float(x) - cx, z=float(z) - cz, lon=lb["lon"], lat=lb["lat"]))
        seen, lab = set(), []
        for lb in sorted(self.labels, key=lambda a: a["pri"]):
            if lb["name"] in seen:
                continue
            seen.add(lb["name"])
            lab.append(lb)
        self.labels = lab

    # ---- ground rasters (colour + shore field), axis-aligned in world x/z around O
    def ground(self, km, n, markings, nlcd=None):
        size = km / KM_PER_UNIT
        sc = n / size                       # px per world unit
        mpp = km * 1000.0 / n               # metres per pixel
        if nlcd is not None:
            base = nlcd
        else:
            base = np.zeros((n, n, 3), np.float32) + (118, 172, 84)
        img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
        wat = Image.new("L", (n, n), 0)
        d, dw = ImageDraw.Draw(img), ImageDraw.Draw(wat)

        def P(xy):
            return [((x + size / 2) * sc, (z + size / 2) * sc) for x, z in xy]

        def W(m):
            return max(1, int(round(m / mpp)))

        lim = size * 0.75
        feats = [(t, g) for t, g in self.feats if np.abs(g).min(0).max() < lim]
        # land use / leisure polygons: big first
        polys = []
        for t, g in feats:
            if len(g) >= 4 and np.allclose(g[0], g[-1]) and ("highway" not in t) and ("waterway" not in t or t.get("waterway") == "riverbank"):
                k = _kind(t)
                if k is None:
                    continue
                area = abs(np.dot(g[:-1, 0], g[1:, 1]) - np.dot(g[1:, 0], g[:-1, 1])) / 2
                polys.append((area, k, g))
        polys.sort(key=lambda a: -a[0])
        for area, k, g in polys:
            if k == "water":
                d.polygon(P(g), fill=WATER_RGB)
                dw.polygon(P(g), fill=255)
            elif k in GROUND:
                d.polygon(P(g), fill=GROUND[k][0])
        # waterways
        for t, g in feats:
            ww = t.get("waterway")
            if ww in ("river", "canal", "stream"):
                wm = {"river": 34, "canal": 14, "stream": 4}[ww]
                d.line(P(g), fill=WATER_RGB, width=W(wm), joint="curve")
                dw.line(P(g), fill=255, width=W(wm), joint="curve")
        # rails
        for t, g in feats:
            if t.get("railway") == "rail" and t.get("bridge") != "yes":
                d.line(P(g), fill=(150, 140, 130), width=W(6))
                d.line(P(g), fill=(92, 84, 80), width=W(2.2))
        # roads: kerbs, asphalt, markings (motorways on top)
        roads = [(t, g) for t, g in feats if t.get("highway") in ROAD_W and t.get("bridge") != "yes" and t.get("layer", "0") in ("0", "-1", "-2")]
        order = {k: i for i, k in enumerate(["service", "living_street", "pedestrian", "residential", "unclassified", "tertiary_link", "secondary_link",
                                             "primary_link", "trunk_link", "motorway_link", "tertiary", "secondary", "primary", "trunk", "motorway"])}
        roads.sort(key=lambda r: order.get(r[0]["highway"], 0))
        for t, g in roads:
            hw = t["highway"]
            if hw not in ("motorway", "trunk", "service"):
                d.line(P(g), fill=(214, 212, 204), width=W(ROAD_W[hw] + 3.5), joint="curve")
        for t, g in roads:
            hw = t["highway"]
            col = (88, 90, 98) if hw in ("motorway", "trunk", "primary", "secondary") else (104, 106, 112)
            d.line(P(g), fill=col, width=W(ROAD_W[hw]), joint="curve")
        if markings:
            for t, g in roads:
                hw = t["highway"]
                p = P(g)
                if hw in ("primary", "secondary", "tertiary", "trunk"):
                    d.line(p, fill=(244, 206, 64), width=1)
                if hw in ("motorway", "trunk"):
                    self._dashes(d, g, size, sc, offs=(-ROAD_W[hw] / 4, ROAD_W[hw] / 4), mpp=mpp)
                    self._dashes(d, g, size, sc, offs=(-ROAD_W[hw] / 2 + 1.2, ROAD_W[hw] / 2 - 1.2), mpp=mpp, solid=True)
        shore = _shore(np.asarray(wat) > 127, max(1.2, 5.0 / mpp))
        return _rgba(np.asarray(img).astype(np.float32), shore), size

    def _dashes(self, d, g, size, sc, offs, mpp, solid=False):
        for o in offs:
            seg = np.diff(g, axis=0)
            L = np.linalg.norm(seg, axis=1) + 1e-12
            nrm = np.stack([-seg[:, 1], seg[:, 0]], 1) / L[:, None]
            a = g[:-1] + nrm * o / M_PER_UNIT
            b = g[1:] + nrm * o / M_PER_UNIT
            for (ax, az), (bx, bz), l in zip(a, b, L):
                lm = l * M_PER_UNIT
                if solid:
                    d.line([((ax + size / 2) * sc, (az + size / 2) * sc), ((bx + size / 2) * sc, (bz + size / 2) * sc)], fill=(240, 240, 236), width=1)
                    continue
                k = int(lm // 12)
                for j in range(k):
                    f0, f1 = j * 12 / lm, (j * 12 + 4) / lm
                    d.line([((ax + (bx - ax) * f0 + size / 2) * sc, (az + (bz - az) * f0 + size / 2) * sc),
                            ((ax + (bx - ax) * f1 + size / 2) * sc, (az + (bz - az) * f1 + size / 2) * sc)], fill=(240, 240, 236), width=1)

    def nlcd_local(self, km, n):
        """mid-level NLCD classes resampled into the town window (local palette), or None"""
        if not self.region or "mid" not in self.region["levels"]:
            return None
        lv = self.region["levels"]["mid"]
        cls = np.asarray(Image.open(ROOT / "data" / "region" / f"{self.key}_mid.png"))
        N = cls.shape[0]
        size_mid = lv["size"]
        frac = (km / KM_PER_UNIT) / size_mid
        a = (0.5 - frac / 2) * N
        crop = Image.fromarray(cls).crop((int(a), int(a), int(a + frac * N), int(a + frac * N))).resize((n // 4, n // 4), Image.NEAREST)
        lut = _lut(LOC_PAL)
        lut[11] = LOC_PAL[21]          # water comes from OpenStreetMap at this scale (crisp shorelines)
        col = lut[np.asarray(crop)]
        im = Image.fromarray(col.astype(np.uint8)).resize((n, n), Image.BILINEAR).filter(ImageFilter.GaussianBlur(n / 520))
        out = np.asarray(im).astype(np.float32)
        crop_full = np.asarray(Image.fromarray(np.asarray(crop)).resize((n, n), Image.NEAREST))
        out = _fields(out, crop_full, n, km)
        return out, np.asarray(crop)

    # ---- buildings: extruded footprints
    def building_mesh(self):
        import shapely
        rng = np.random.default_rng(11)
        rows = []
        for b in self.osm["buildings"]:
            g = np.asarray(b["g"], np.float64).reshape(-1, 2)
            x, z = FR.world(g[:, 0], g[:, 1])
            rows.append((b["t"], np.stack([x - self.O[0], z - self.O[2]], 1)))
        polys = []
        for _, g in rows:
            try:
                pg = shapely.Polygon(g * M_PER_UNIT) if len(g) >= 4 else None
                if pg is not None and not pg.is_valid:
                    pg = pg.buffer(0)
            except Exception:  # noqa: BLE001
                pg = None
            polys.append(pg)
        V, wuv, info, bc = [], [], [], []

        def emit(tris, nrm, uv, inf, cen):
            V.append(tris)
            wuv.append(uv)
            n = len(tris)
            info.append(np.repeat(np.asarray(inf, np.float32)[None], n, 0))
            bc.append(np.repeat(np.asarray(cen, np.float32)[None], n, 0))
            NN.append(np.repeat(np.asarray(nrm, np.float32)[None], n, 0) if np.ndim(nrm) == 1 else nrm.astype(np.float32))

        NN = []
        flat_polys, flat_meta = [], []
        lon = np.array([np.mean(np.asarray(b["g"]).reshape(-1, 2)[:, 0]) for b in self.osm["buildings"]])
        lat = np.array([np.mean(np.asarray(b["g"]).reshape(-1, 2)[:, 1]) for b in self.osm["buildings"]])
        base_y = height_at(lon, lat) if len(lon) else np.zeros(0)
        for k, ((t, g), poly) in enumerate(zip(rows, polys)):
            if poly is None or poly.is_empty:
                continue
            if poly.geom_type != "Polygon":
                parts = [p for p in shapely.get_parts(poly) if p.geom_type == "Polygon"]
                if not parts:
                    continue
                poly = max(parts, key=lambda p: p.area)
            area = poly.area
            if area < 12:
                continue
            bt = t.get("building", "yes")
            hgt = None
            try:
                hgt = float(str(t.get("height", "")).split()[0].replace("m", ""))
            except (ValueError, IndexError):
                hgt = None
            lv = None
            try:
                lv = float(t.get("building:levels"))
            except (TypeError, ValueError):
                lv = None
            h01 = rng.random()
            house = bt in ("house", "detached", "semidetached_house", "duplex", "residential", "bungalow", "terrace", "farm") or (bt == "yes" and area < 330)
            small = bt in ("shed", "garage", "garages", "carport", "roof", "hut", "cabin")
            if hgt is None:
                if lv is not None:
                    hgt = lv * 3.1 + (0.0 if house else 1.0)
                elif small:
                    hgt = 2.6 + h01 * 0.6
                elif house:
                    hgt = 3.0 + h01 * 0.9 + (2.8 if rng.random() < 0.18 else 0.0)
                elif bt == "apartments":
                    hgt = 9.5 + h01 * 4
                elif bt in ("industrial", "warehouse", "manufacture"):
                    hgt = 8 + h01 * 3
                elif bt in ("church", "cathedral"):
                    hgt = 10 + h01 * 4
                elif bt in ("school", "hospital", "university", "college", "civic", "public"):
                    hgt = 7.5 + h01 * 3
                else:
                    hgt = 5.5 + h01 * 2.5 if area < 2500 else 8 + h01 * 3
            hgt = float(min(max(hgt, 2.4), 300))
            typ = 0 if (house or small) else (3 if hgt >= 28 else (2 if bt in ("industrial", "warehouse", "manufacture") or area > 6000 else 1))
            pitched = (house or bt in ("church",)) and area < 600 and hgt < 12
            ring = np.asarray(poly.exterior.coords)[:-1]          # metres
            if len(ring) < 3:
                continue
            A = 0.5 * np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])
            if A < 0:
                ring = ring[::-1]
            p0, p1 = ring, np.roll(ring, -1, 0)
            e = p1 - p0
            el = np.linalg.norm(e, axis=1)
            good = el > 0.05
            p0, p1, e, el = p0[good], p1[good], e[good], el[good]
            nrm = np.stack([e[:, 1], np.zeros(len(e)), -e[:, 0]], 1) / el[:, None]
            u0 = np.concatenate([[0], np.cumsum(el)[:-1]])
            u1 = u0 + el
            by = float(base_y[k])
            hw_ = hgt / M_PER_UNIT
            P0 = np.c_[p0[:, 0] / M_PER_UNIT, np.full(len(p0), by), p0[:, 1] / M_PER_UNIT]
            P1 = np.c_[p1[:, 0] / M_PER_UNIT, np.full(len(p1), by), p1[:, 1] / M_PER_UNIT]
            T0, T1 = P0 + [0, hw_, 0], P1 + [0, hw_, 0]
            quads = np.stack([P0, P1, T1, P0, T1, T0], 1).reshape(-1, 3)
            uvq = np.stack([np.c_[u0, np.zeros_like(u0)], np.c_[u1, np.zeros_like(u0)], np.c_[u1, np.full_like(u0, hgt)],
                            np.c_[u0, np.zeros_like(u0)], np.c_[u1, np.full_like(u0, hgt)], np.c_[u0, np.full_like(u0, hgt)]], 1).reshape(-1, 2)
            cxz = np.asarray(poly.centroid.coords[0]) / M_PER_UNIT
            cen = (cxz[0], cxz[1], by)
            inf_w = (h01, typ, hgt, 0.0)
            emit(quads, np.repeat(nrm, 6, 0), uvq, inf_w, cen)
            if pitched:
                rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]
                s0, s1 = np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[1])
                if s0 < s1:
                    rect = np.roll(rect, -1, 0)
                    s0, s1 = s1, s0
                c = rect.mean(0)
                ax_l = (rect[1] - rect[0]) / max(s0, 1e-6)
                ax_w = (rect[2] - rect[1]) / max(s1, 1e-6)
                ov = 0.45
                L2_, W2 = s0 / 2 + ov, s1 / 2 + ov
                rise = min(s1 * 0.36, 4.5)
                inset = min(W2, L2_ * 0.9) if bt != "church" else 0.0
                cs = [c - ax_l * L2_ - ax_w * W2, c + ax_l * L2_ - ax_w * W2, c + ax_l * L2_ + ax_w * W2, c - ax_l * L2_ + ax_w * W2]
                r0, r1 = c - ax_l * (L2_ - inset), c + ax_l * (L2_ - inset)
                Y = by + hw_
                Yr = Y + rise / M_PER_UNIT

                def v3(p, y):
                    return np.array([p[0] / M_PER_UNIT, y, p[1] / M_PER_UNIT])

                faces = [[v3(cs[0], Y), v3(cs[1], Y), v3(r1, Yr), v3(cs[0], Y), v3(r1, Yr), v3(r0, Yr)],
                         [v3(cs[2], Y), v3(cs[3], Y), v3(r0, Yr), v3(cs[2], Y), v3(r0, Yr), v3(r1, Yr)],
                         [v3(cs[1], Y), v3(cs[2], Y), v3(r1, Yr)], [v3(cs[3], Y), v3(cs[0], Y), v3(r0, Yr)]]
                for f in faces:
                    f = np.asarray(f)
                    nn = np.cross(f[1] - f[0], f[2] - f[0])
                    nn[1] *= 1.0
                    nn = nn / (np.linalg.norm(nn) + 1e-12)
                    if nn[1] < 0:
                        nn = -nn
                    emit(f, nn, np.zeros((len(f), 2)), (h01, typ, hgt, 1.0), cen)
            else:
                flat_polys.append(poly)
                flat_meta.append((by + hw_, (h01, typ, hgt, 1.0), cen, area))
        # flat roofs: constrained Delaunay triangulation of the real footprint
        if flat_polys:
            tri = shapely.constrained_delaunay_triangles(np.array(flat_polys, dtype=object))
            for (Y, inf, cen, area), gc, poly in zip(flat_meta, tri, flat_polys):
                parts = shapely.get_parts(gc)
                if len(parts) == 0:
                    continue
                co = shapely.get_coordinates(parts).reshape(len(parts), 4, 2)[:, :3]
                co = co.reshape(-1, 2) / M_PER_UNIT
                v = np.c_[co[:, 0], np.full(len(co), Y), co[:, 1]]
                # make all triangles CCW seen from above (y up, z south)
                vv = v.reshape(-1, 3, 3)
                cr = np.cross(vv[:, 1] - vv[:, 0], vv[:, 2] - vv[:, 0])[:, 1]
                vv[cr < 0] = vv[cr < 0][:, [0, 2, 1]]
                emit(vv.reshape(-1, 3), np.array([0.0, 1.0, 0.0]), np.zeros((len(co), 2)), inf, cen)
                if area > 700 and inf[1] in (1, 2):
                    for j in range(1 + int(area > 3000) + int(area > 8000)):
                        pt = shapely.Point(poly.representative_point().x + rng.uniform(-1, 1) * math.sqrt(area) * 0.25,
                                           poly.representative_point().y + rng.uniform(-1, 1) * math.sqrt(area) * 0.25)
                        if not poly.contains(pt):
                            continue
                        s = rng.uniform(2.5, 5.0)
                        bx = self._box(pt.x, pt.y, s, s * rng.uniform(0.8, 1.6), 1.8, Y)
                        for f, nn in bx:
                            emit(f, nn, np.zeros((len(f), 2)), (inf[0], inf[1], inf[2], 2.0), cen)
        if not V:
            return None
        return dict(p=np.concatenate(V).astype(np.float32), n=np.concatenate(NN).astype(np.float32), uv=np.concatenate(wuv).astype(np.float32),
                    info=np.concatenate(info).astype(np.float32), bc=np.concatenate(bc).astype(np.float32))

    @staticmethod
    def _box(x, z, w, d, h, Y):
        """axis-aligned box on a roof at height Y (world), metres in, world units out: list of (6 verts, normal)"""
        u = 1.0 / M_PER_UNIT
        x0, x1, z0, z1 = (x - w / 2) * u, (x + w / 2) * u, (z - d / 2) * u, (z + d / 2) * u
        y0, y1 = Y, Y + h * u
        out = []
        q = lambda a, b, c, dd: np.array([a, b, c, a, c, dd])  # noqa: E731
        out.append((q([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]), np.array([0, 1.0, 0])))
        out.append((q([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]), np.array([0, 0, 1.0])))
        out.append((q([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]), np.array([0, 0, -1.0])))
        out.append((q([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]), np.array([1.0, 0, 0])))
        out.append((q([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]), np.array([-1.0, 0, 0])))
        return out

    def bridges(self):
        out = []
        for t, g in self.feats:
            if t.get("highway") in ROAD_W and t.get("bridge") == "yes" and np.abs(g).max() < TREE_KM / KM_PER_UNIT:
                w = ROAD_W[t["highway"]]
                for a, b in zip(g[:-1], g[1:]):
                    out.append((a, b, w))
            elif t.get("railway") == "rail" and t.get("bridge") == "yes" and np.abs(g).max() < TREE_KM / KM_PER_UNIT:
                for a, b in zip(g[:-1], g[1:]):
                    out.append((a, b, 7.0))
        return out

    def trees(self, cls_mid, max_n=80000):
        """tree instances from land use (OSM) and NLCD forest, avoiding roads, water and buildings."""
        rng = np.random.default_rng(5)
        km = TREE_KM
        n = int(km * 1000 / 5)              # 5 m cells
        size = km / KM_PER_UNIT
        sc = n / size
        dens = Image.new("L", (n, n), 14)   # default: yard lawn, few trees (units: 0.01 trees per cell)
        block = Image.new("L", (n, n), 0)
        d, db = ImageDraw.Draw(dens), ImageDraw.Draw(block)

        def P(xy):
            return [((x + size / 2) * sc, (z + size / 2) * sc) for x, z in xy]

        if cls_mid is not None:
            c = Image.fromarray(cls_mid).resize((n, n), Image.NEAREST)
            a = np.asarray(c)
            base = np.full(a.shape, 6, np.uint8)
            base[np.isin(a, (41, 42, 43))] = 70
            base[np.isin(a, (90,))] = 40
            base[np.isin(a, (21, 22))] = 16
            base[np.isin(a, (23,))] = 9
            base[np.isin(a, (24, 81, 82, 11))] = 0
            dens = Image.fromarray(base)
            d = ImageDraw.Draw(dens)
        lim = size * 0.75
        polys = []
        for t, g in self.feats:
            if np.abs(g).min(0).max() > lim:
                continue
            if len(g) >= 4 and np.allclose(g[0], g[-1]) and "highway" not in t:
                k = _kind(t)
                if k is None:
                    continue
                area = abs(np.dot(g[:-1, 0], g[1:, 1]) - np.dot(g[1:, 0], g[:-1, 1])) / 2
                polys.append((area, k, g))
        polys.sort(key=lambda a: -a[0])
        for area, k, g in polys:
            if k == "water":
                db.polygon(P(g), fill=255)
            elif k in GROUND:
                d.polygon(P(g), fill=int(min(255, GROUND[k][1] * 100)))
        for t, g in self.feats:
            hw = t.get("highway")
            if hw in ROAD_W:
                db.line(P(g), fill=255, width=max(1, int((ROAD_W[hw] + 4) / 5)))
            if t.get("waterway") in ("river", "canal", "stream") or t.get("railway") == "rail":
                db.line(P(g), fill=255, width=2)
        for b in self.osm["buildings"]:
            gg = np.asarray(b["g"], np.float64).reshape(-1, 2)
            x, z = FR.world(gg[:, 0], gg[:, 1])
            pts = np.stack([x - self.O[0], z - self.O[2]], 1)
            if np.abs(pts).min(0).max() < lim:
                db.polygon(P(pts), fill=255, outline=255)
        p = np.asarray(dens, np.float32) / 100.0 * (np.asarray(block) < 128)
        p = np.clip(p * 0.25, 0, 1)        # per 25 m2 cell
        pick = rng.random(p.shape) < p
        iz, ix = np.nonzero(pick)
        if len(ix) > max_n:
            sel = rng.choice(len(ix), max_n, replace=False)
            ix, iz = ix[sel], iz[sel]
        x = (ix + rng.random(len(ix))) / sc - size / 2
        z = (iz + rng.random(len(iz))) / sc - size / 2
        forest = np.asarray(dens)[iz, ix] >= 60
        lon, lat = self.lonlat(x, z)
        by = height_at(lon, lat)
        hm = np.where(forest, rng.uniform(9, 17, len(x)), rng.uniform(6, 13, len(x)))
        crown = hm * rng.uniform(0.28, 0.4, len(x))
        conifer = (rng.random(len(x)) < np.where(forest, 0.25, 0.08)).astype(np.float32)
        hue = rng.random(len(x))
        rot = rng.uniform(0, 6.283, len(x))
        u = 1.0 / M_PER_UNIT
        return np.stack([x, z, by, hm * u, crown * u, conifer, hue, rot], 1).astype(np.float32)



def _inv_world(x, z):
    """world x/z -> lon/lat (inverse of FR.world)"""
    import mapviz.core as mc
    fr = FR
    u = (np.asarray(x) / WORLD_W + 0.5) * TW
    v = (np.asarray(z) / WORLD_H + 0.5) * TH
    ax = (u - TW / 2) / fr.s + fr.cx
    ay = fr.cy - (v - TH / 2) / fr.s
    N, C, R0, L0 = mc._N, mc._C, mc._R0, mc._L0
    rho = np.sqrt(ax * ax + (R0 - ay) ** 2)
    th = np.arctan2(ax, R0 - ay)
    lon = np.degrees(L0 + th / N)
    s = np.clip((C - (rho * N) ** 2) / (2 * N), -1, 1)
    return lon, np.degrees(np.arcsin(s))


Town.lonlat = lambda self, x, z: _inv_world(np.asarray(x) + self.O[0], np.asarray(z) + self.O[2])


# ------------------------------------------------------------------ shaders
COMMON = """
uniform float tm; uniform vec3 eye; uniform float camdist; uniform float mpu;
uniform sampler2D noise;
uniform float shadow_on; uniform mat4 lmvp; uniform sampler2DShadow smap; uniform float sbias; uniform float stex;
uniform vec4 storm;   // x, z (same frame as vp), radius, strength
uniform vec2 qk; uniform vec4 qr;   // earthquake: epicentre (x, z), (damage radius, wave speed, seconds since start, alpha)
uniform sampler2D dem; uniform float demsize; uniform float surge;   // storm surge: water level (m) over USGS 3DEP ground (m)
float demAt(vec2 lxz){
  vec2 uv = lxz/demsize + 0.5;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1e4;
  return texture(dem, uv).r;
}
vec3 floodCol(vec2 m, float tmv){
  float w = texture(noise, m/55.0 + vec2(tmv*0.02, tmv*0.013)).r;
  return mix(vec3(0.30,0.38,0.38), vec3(0.42,0.50,0.50), w);
}
const vec3 SUN = vec3(-0.5845, 0.5632, 0.5845);
const vec3 SUNC = vec3(1.0, 0.96, 0.88);
const vec3 AMB = vec3(0.70, 0.78, 0.92);
float shadowAt(vec3 wp, vec3 n){
  if (shadow_on < 0.5) return 1.0;
  vec4 l = lmvp*vec4(wp,1.0); vec3 q = l.xyz/l.w*0.5+0.5;
  if (q.x<0.002||q.x>0.998||q.y<0.002||q.y>0.998||q.z>1.0) return 1.0;
  float b = sbias*(1.0 + 3.0*(1.0 - clamp(dot(n, SUN), 0.0, 1.0)));
  float s = 0.0;
  for (int i=-1;i<=1;i+=2) for (int j=-1;j<=1;j+=2) s += texture(smap, vec3(q.xy + vec2(i,j)*0.5*stex, q.z - b));
  return s*0.25;
}
float stormAt(vec3 wp){
  if (storm.w <= 0.0) return 0.0;
  float d = length(wp.xz - storm.xy)/storm.z;
  return storm.w*(1.0 - smoothstep(0.55, 1.0, d));
}
vec3 light(vec3 base, vec3 n, float sh, float occ){
  float d = max(dot(n, SUN), 0.0);
  return base*(AMB*0.62*occ + SUNC*0.62*d*sh);
}
"""

VS_TER = """#version 330
uniform mat4 mvp; uniform float yscale; in vec3 p; in vec2 t; in vec3 n; in float ed; out vec2 vt; out vec3 vn; out vec3 vp; out float ved;
void main(){ vec3 q = vec3(p.x, p.y*yscale, p.z); gl_Position = mvp*vec4(q,1.0); vt=t; vn=normalize(vec3(n.x*yscale, n.y, n.z*yscale)); vp=q; ved=ed; }"""
FS_TER = """#version 330
""" + COMMON + """
uniform sampler2D nat; uniform sampler2D lv0; uniform sampler2D lv1; uniform sampler2D lv2; uniform sampler2D lv3;
uniform vec3 box0; uniform vec3 box1; uniform vec3 box2; uniform vec3 box3; uniform vec4 lvon;
uniform sampler2D ovl; uniform float ovl_alpha; uniform float dim;
uniform vec2 sa; uniform vec2 sb; uniform float shw; uniform float salong; uniform float swath_on; uniform vec3 O;
uniform sampler2D outer; uniform float outer_on;
uniform sampler2D stt; uniform vec3 sbox; uniform float son;
in vec2 vt; in vec3 vn; in vec3 vp; in float ved; out vec4 c;
vec4 lvl(sampler2D s, vec3 b, vec2 xz, out float w){
  vec2 uv = (xz - b.xy)/b.z + 0.5;
  float e = min(min(uv.x, 1.0-uv.x), min(uv.y, 1.0-uv.y));
  w = smoothstep(0.0, 0.06, e);
  return texture(s, clamp(uv, 0.0, 1.0));
}
void main(){
  float e = min(min(vt.x, 1.0-vt.x), min(vt.y, 1.0-vt.y));
  vec2 uv = clamp(vt, 0.0, 1.0);
  vec4 g = texture(nat, uv);
  if (outer_on > 0.5) {
    vec4 go = texture(outer, (vt + 0.3)/1.6);
    g = mix(go, g, smoothstep(0.0, 0.012, e));
  } else if (e < 0.0) discard;
  vec2 lxz = vp.xz - O.xz;
  vec2 wm = lxz*mpu;
  vec2 warp = (texture(noise, wm/90.0).rg - 0.5)*2.0*(22.0/mpu) + (texture(noise, wm/23.0).gb - 0.5)*2.0*(7.0/mpu);   // organic class boundaries
  lxz += warp*(1.0 - smoothstep(0.03, 0.30, camdist));
  float w; vec4 g2;
  if (lvon.x > 0.5) { g2 = lvl(lv0, box0, lxz, w); g = mix(g, g2, w); }
  if (lvon.y > 0.5) { g2 = lvl(lv1, box1, lxz, w); g = mix(g, g2, w); }
  if (lvon.z > 0.5) { g2 = lvl(lv2, box2, lxz, w); g = mix(g, g2, w); }
  if (lvon.w > 0.5) { g2 = lvl(lv3, box3, lxz, w); g = mix(g, g2, w); }
  if (son > 0.5) {
    vec2 su = (vp.xz - sbox.xy)/sbox.z + 0.5;
    float sw = smoothstep(0.0, 0.03, min(min(su.x, 1.0 - su.x), min(su.y, 1.0 - su.y)));
    vec4 sc = texture(stt, clamp(su, 0.0, 1.0));
    g = vec4(mix(g.rgb, sc.rgb, sc.a*sw*(1.0 - smoothstep(0.6, 1.4, camdist*0.0 + 0.0))), g.a);
  }
  vec3 n = normalize(vn);
  vec3 lp = vec3(lxz.x, vp.y, lxz.y);
  float sh = shadowAt(lp, n);
  vec3 col = g.rgb;
  float sf = g.a;
  vec2 m = lxz*mpu;
  float detail = 1.0 - smoothstep(0.03, 0.12, camdist);
  if (sf > 0.5) {
    float lam = max(6.0, camdist*mpu*0.0035);
    float h1 = texture(noise, m/(lam*22.0) + vec2(tm*0.011, tm*0.007)).r;
    float h2 = texture(noise, m/(lam*9.0) - vec2(tm*0.017, -tm*0.012)).g;
    float wamp = 1.0 - 0.85*smoothstep(0.05, 0.4, camdist);
    vec3 wn = normalize(vec3(((h1-0.5)*0.55 + (h2-0.5)*0.45)*wamp, 1.0, ((h2-0.5)*0.55 - (h1-0.5)*0.35)*wamp));
    vec3 v = normalize(eye - vp);
    float fres = pow(1.0 - max(dot(wn, v), 0.0), 3.0);
    col = mix(vec3(0.30,0.71,0.90), vec3(0.11,0.47,0.82), smoothstep(0.5, 0.8, sf));
    col = mix(col, vec3(0.78,0.89,0.98), fres*0.55);
    float spec = pow(max(dot(reflect(-SUN, wn), v), 0.0), 160.0);
    col += vec3(1.0,0.98,0.92)*spec*1.4*sh*(1.0 - smoothstep(0.02, 0.15, camdist));
    col *= mix(0.72, 1.0, sh);
    float fl = 1.0 - smoothstep(0.0, 0.035, sf - 0.5);
    col = mix(col, vec3(0.96,0.98,1.0), fl*0.55);
  } else {
    float gn = texture(noise, m/45.0).r*0.6 + texture(noise, m/9.0).g*0.4;
    float lo = texture(noise, m/700.0).b;
    float green = (col.g > col.r*1.15 && col.g > col.b*1.4) ? 1.0 : 0.0;
    col = mix(col, col*vec3(1.08, 1.02, 0.80), green*smoothstep(0.45, 0.8, lo)*0.7*detail);
    col = mix(col, col*vec3(0.86, 0.95, 0.86), green*smoothstep(0.55, 0.25, lo)*0.6*detail);
    col *= mix(1.0, 0.92 + 0.16*gn, detail);
    float d = max(dot(n, SUN), 0.0);
    vec3 lit = col*(0.80 + 0.30*d);
    col = mix(lit*vec3(0.62,0.68,0.82), lit, sh);
  }
  if (swath_on > 0.5) {
    vec2 ab = sb - sa; float L = max(length(ab), 1e-9); vec2 dir = ab/L;
    vec2 P = vp.xz - sa; float s = dot(P, dir); float dd = abs(dot(P, vec2(-dir.y, dir.x)));
    float r1 = texture(noise, m/70.0).r, r2 = texture(noise, m/6.0).b;
    float inside = (1.0 - smoothstep(shw*(0.55+0.4*r1), shw*(0.8+0.4*r1), dd)) * step(0.0, s) * (1.0 - smoothstep(salong - shw*0.15, salong, s));
    vec3 rub = mix(vec3(0.52,0.43,0.34), vec3(0.70,0.64,0.56), r2);
    rub = mix(rub, vec3(0.62,0.30,0.22), step(0.82, r2)*0.6);
    col = mix(col, rub*mix(0.65, 1.0, sh), inside*0.85);
  }
  if (surge > 0.0) {
    float dh = surge - demAt(lxz);
    if (dh > 0.0) {
      vec3 fc = floodCol(m, tm) * mix(0.70, 1.0, sh);
      float foam = (1.0 - smoothstep(0.0, 0.35, dh)) * (0.6 + 0.4*texture(noise, m/8.0 + tm*0.05).g);
      col = mix(fc, vec3(0.92,0.93,0.90), foam*0.7);
    }
  }
  if (qr.w > 0.0) {
    float dq = length(lxz - qk);
    float rr = qr.y*qr.z;
    float wr = max(rr*0.05, 4.0/mpu);
    float ring = exp(-pow((dq - rr)/wr, 2.0)) * (1.0 - smoothstep(2.0, 5.0, rr/max(qr.x, 1e-9)));
    col = mix(col, vec3(1.0, 0.72, 0.30), ring*0.50*qr.w);
    float inten = exp(-dq/max(qr.x, 1e-9));
    col = mix(col, vec3(0.50, 0.42, 0.34), smoothstep(0.50, 0.90, inten)*smoothstep(1.5, 4.0, qr.z)*0.55*qr.w);
  }
  vec4 o = texture(ovl, uv);
  col = mix(col, o.rgb, o.a*ovl_alpha*step(0.0, e));
  col *= 1.0 - 0.45*stormAt(lp);
  col *= dim;
  if (outer_on < 0.5) col = mix(vec3(0.78,0.88,0.98), col, smoothstep(0.0, 0.025, e));
  col = mix(vec3(0.42,0.66,0.94), col, smoothstep(0.0, 1.0, ved));
  c = vec4(col, 1.0);
}"""

VS_SKY = """#version 330
in vec2 p; out float vy; void main(){ gl_Position = vec4(p, 0.9999, 1.0); vy = p.y*0.5+0.5; }"""
FS_SKY = """#version 330
uniform float storm_sky; in float vy; out vec4 c;
void main(){ vec3 s = mix(vec3(0.78,0.88,0.98), vec3(0.36,0.62,0.93), smoothstep(0.30, 1.0, vy));
  s = mix(s, vec3(0.44,0.48,0.55)*mix(1.0, 0.8, vy), storm_sky); c = vec4(s, 1.0); }"""

VS_BLD = """#version 330
uniform mat4 mvp; uniform float yscale; uniform vec2 pa; uniform vec2 pb; uniform float hw; uniform float along; uniform float clen;
uniform float wind; uniform float mpu;
uniform vec2 qk; uniform vec4 qd;   // earthquake epicentre (x, z) and (damage radius, shake amplitude m, seconds since start, alpha)
in vec3 p; in vec3 n; in vec2 wuv; in vec4 info; in vec3 bc;
out vec3 vn; out vec2 vwuv; flat out vec4 vinfo; out vec3 vp; out float vdmg; out float vag; flat out float vlost;
void main(){
  vec2 ab = pb - pa; float L = max(length(ab), 1e-12); vec2 dir = ab/L;
  vec2 rel = bc.xy - pa; float s = dot(rel, dir); float d = abs(dot(rel, vec2(-dir.y, dir.x)));
  float sev = 1.0 - smoothstep(hw*0.55, hw*1.0, d);
  float k = clamp((along - s + hw*0.25)/clen, 0.0, 1.0) * step(0.0, s) * step(0.001, hw);
  float dmg = k*sev;
  float yl = p.y - bc.z;
  float hk = fract(sin(info.x*91.7)*437.5);
  float qint = exp(-length(bc.xy - qk)/max(qd.x, 1e-9))*qd.w;
  dmg = max(dmg, smoothstep(0.55, 0.92, qint*(0.60 + 0.75*hk))*smoothstep(1.0, 3.5, qd.z)*step(0.5, qd.w));
  float lost = (info.w > 0.5 && info.y < 1.5) ? step(hk, wind*0.42) : 0.0;
  if (lost > 0.5) yl = min(yl, info.z/mpu);
  vec3 q = vec3(p.x, bc.z*yscale + yl*mix(1.0, 0.10 + 0.25*hk, dmg), p.z);
  q.xz += (q.xz - bc.xy)*0.15*dmg + dir*yl*0.35*dmg*hk;
  q.xz += vec2(sin(qd.z*33.0 + info.x*13.0), cos(qd.z*27.0 + info.x*7.0))*qd.y*qint*(0.4 + 0.6*clamp(yl*mpu/12.0, 0.0, 1.0))/mpu;
  vp = q; gl_Position = mvp*vec4(q, 1.0);
  vn = lost > 0.5 ? vec3(0.0, 1.0, 0.0) : n; vwuv = wuv; vinfo = info; vdmg = dmg; vag = (q.y - bc.z*yscale)*mpu; vlost = lost;
}"""
FS_BLD = """#version 330
""" + COMMON + """
uniform float alpha;
in vec3 vn; in vec2 vwuv; flat in vec4 vinfo; in vec3 vp; in float vdmg; in float vag; flat in float vlost; out vec4 c;
vec3 pick(float h, vec3 a, vec3 b, vec3 cc, vec3 d, vec3 e){ return h < 0.2 ? a : (h < 0.4 ? b : (h < 0.6 ? cc : (h < 0.8 ? d : e))); }
float grid(float x, float period, float w, float fw){ float f = fract(x/period); float hw = w/period*0.5;
  float aa = clamp(fw/period, 0.0005, 0.5); return smoothstep(0.5-hw-aa, 0.5-hw+aa, f) * (1.0 - smoothstep(0.5+hw-aa, 0.5+hw+aa, f)); }
void main(){
  float id = vinfo.x, typ = vinfo.y, H = vinfo.z, part = vinfo.w;
  float h2 = fract(id*7.31), h3 = fract(id*13.7);
  vec3 n = normalize(vn);
  vec3 col;
  float occ = 1.0;
  if (part > 0.5) {
    if (typ < 0.5) col = pick(h2, vec3(0.36,0.37,0.40), vec3(0.47,0.33,0.26), vec3(0.70,0.38,0.24), vec3(0.38,0.43,0.50), vec3(0.55,0.52,0.48));
    else if (typ < 1.5) col = pick(h2, vec3(0.80,0.79,0.76), vec3(0.70,0.70,0.70), vec3(0.62,0.60,0.58), vec3(0.85,0.84,0.80), vec3(0.55,0.57,0.60));
    else if (typ < 2.5) col = pick(h2, vec3(0.78,0.79,0.80), vec3(0.68,0.70,0.72), vec3(0.84,0.84,0.82), vec3(0.60,0.66,0.70), vec3(0.74,0.72,0.68));
    else col = vec3(0.62,0.64,0.66);
    if (part > 1.5) col = vec3(0.86,0.86,0.84);
    if (n.y < 0.99 && part < 1.5) col *= 0.92 + 0.08*grid(vwuv.x + vp.y*mpu*3.0, 0.6, 0.12, fwidth(vp.y*mpu*3.0));
  } else {
    float u = vwuv.x, v = vwuv.y;
    float fu = fwidth(u), fv = fwidth(v);
    vec3 wall, glass = vec3(0.24,0.33,0.44);
    float win = 0.0;
    if (typ < 0.5) {
      wall = pick(h2, vec3(0.95,0.92,0.84), vec3(0.93,0.93,0.90), vec3(0.86,0.80,0.70), vec3(0.78,0.84,0.88), vec3(0.74,0.48,0.38));
      float rows = step(0.9, mod(v, 2.9)) * (1.0 - step(2.0, mod(v, 2.9)));
      win = grid(u + h3*3.0, 3.8, 1.3, fu) * rows * step(v, H - 0.5);
    } else if (typ < 1.5) {
      wall = pick(h2, vec3(0.93,0.90,0.84), vec3(0.86,0.83,0.78), vec3(0.96,0.95,0.92), vec3(0.80,0.72,0.62), vec3(0.72,0.78,0.82));
      float fr = fract(v/3.3);
      win = grid(u, 3.0, 1.7, fu) * smoothstep(0.24, 0.24+fv/3.3, fr) * (1.0 - smoothstep(0.74, 0.74+fv/3.3, fr)) * step(v, H - 0.8);
    } else if (typ < 2.5) {
      wall = pick(h2, vec3(0.86,0.86,0.84), vec3(0.80,0.82,0.84), vec3(0.90,0.88,0.84), vec3(0.72,0.76,0.80), vec3(0.84,0.80,0.74));
      win = grid(u, 7.0, 4.5, fu) * step(H - 2.6, v) * step(v, H - 1.4);
      wall *= 0.94 + 0.06*grid(u, 1.2, 0.15, fu);
    } else {
      vec3 vdir = normalize(eye - vp);
      float f = pow(1.0 - abs(dot(n, vdir)), 2.0);
      wall = mix(pick(h2, vec3(0.42,0.60,0.78), vec3(0.36,0.56,0.66), vec3(0.55,0.66,0.78), vec3(0.30,0.46,0.60), vec3(0.62,0.72,0.80)), vec3(0.82,0.90,0.98), f*0.6);
      float mull = max(grid(u, 1.6, 0.14, fu), grid(v, 3.6, 0.25, fv));
      wall = mix(wall, vec3(0.86,0.88,0.90), mull*0.7);
      glass = wall;
    }
    float fade = clamp(1.0 - fu*0.6, 0.0, 1.0);
    col = mix(wall, mix(wall, glass, 0.65), (1.0 - fade)*0.25);
    col = mix(col, glass, win*fade);
    occ = 0.72 + 0.28*smoothstep(0.0, 2.5, v);
  }
  col = mix(col, mix(vec3(0.46,0.38,0.30), vec3(0.62,0.58,0.52), h3), vdmg*0.85);
  if (vlost > 0.5) col = mix(vec3(0.55,0.44,0.32), vec3(0.66,0.58,0.48), h3);
  float sh = shadowAt(vp, n);
  col = light(col, n, sh, occ) * 1.12;
  col *= 1.0 - 0.45*stormAt(vp);
  if (surge > 0.0) {
    float gh = demAt(vp.xz);
    float dh = surge - (gh + vag);
    if (dh > 0.0) col = mix(col, floodCol(vp.xz*mpu, tm)*0.85, 0.88);
    col = mix(col, vec3(0.92,0.93,0.90), (1.0 - smoothstep(0.0, 0.25, abs(dh)))*0.5*step(0.0, surge - gh));
  }
  c = vec4(col, alpha);
}"""

VS_TREE = """#version 330
uniform mat4 mvp; uniform float yscale; uniform vec2 pa; uniform vec2 pb; uniform float hw; uniform float along; uniform float clen;
uniform float wind; uniform vec2 wdir; uniform float tm; uniform float mpu;
in vec3 p; in vec3 n; in float part;            // unit tree: trunk part 0, crown part 1 (crown centred at y=0.62, radius 1)
in vec4 ia; in vec4 ib;                          // ia = (x, z, base_y, height); ib = (crown, conifer, hue, rot)
out vec3 vn; out vec3 vp; flat out float vpart; flat out float vhue; out float vdmg; out float vag;
void main(){
  vec2 ab = pb - pa; float L = max(length(ab), 1e-12); vec2 dir = ab/L;
  vec2 rel = ia.xy - pa; float s = dot(rel, dir); float d = abs(dot(rel, vec2(-dir.y, dir.x)));
  float sev = 1.0 - smoothstep(hw*0.6, hw*1.05, d);
  float dmg = clamp((along - s + hw*0.25)/clen, 0.0, 1.0) * step(0.0, s) * sev * step(0.001, hw);
  float ca = cos(ib.w), sa = sin(ib.w);
  vec3 q;
  if (part < 0.5) q = vec3(p.x*ib.x*0.18, p.y*ia.w*0.45, p.z*ib.x*0.18);
  else q = vec3(p.x*ib.x, ia.w*0.58 + p.y*ib.x*1.1, p.z*ib.x);
  q.xz = vec2(q.x*ca - q.z*sa, q.x*sa + q.z*ca);
  float fall = dmg*1.35;
  vec2 fd = dir;
  float yy = q.y;
  q.y = yy*cos(fall) * mix(1.0, 0.6, dmg);
  q.xz += fd*yy*sin(fall);
  float hf = clamp(yy/max(ia.w, 1e-9), 0.0, 1.5);
  q.xz += wdir*wind*ia.w*hf*hf*(0.16 + 0.07*sin(tm*2.7 + ia.x*9.0e4 + ia.y*7.0e4));
  vec3 w = vec3(ia.x + q.x, ia.z*yscale + q.y, ia.y + q.z);
  vp = w; gl_Position = mvp*vec4(w, 1.0);
  vn = vec3(n.x*ca - n.z*sa, n.y, n.x*sa + n.z*ca); vpart = part; vhue = ib.z; vdmg = dmg; vag = q.y*mpu;
}"""
FS_TREE = """#version 330
""" + COMMON + """
in vec3 vn; in vec3 vp; flat in float vpart; flat in float vhue; in float vdmg; in float vag; out vec4 c;
void main(){
  vec3 n = normalize(vn);
  if (surge > 0.0 && surge - (demAt(vp.xz) + vag) > 0.0) { c = vec4(floodCol(vp.xz*mpu, tm)*0.8, 1.0); return; }
  vec3 col = vpart < 0.5 ? vec3(0.40,0.30,0.22) : mix(mix(vec3(0.30,0.56,0.20), vec3(0.42,0.66,0.24), vhue), vec3(0.22,0.46,0.22), step(0.8, vhue));
  col = mix(col, vec3(0.48,0.40,0.28), vdmg*0.8);
  float sh = shadowAt(vp, n);
  float wrap = max(dot(n, SUN)*0.6 + 0.4, 0.0);
  col = col*(AMB*0.55*(0.75+0.25*n.y) + SUNC*0.75*wrap*sh) * 1.08;
  col *= 1.0 - 0.45*stormAt(vp);
  c = vec4(col, 1.0);
}"""

VS_DEPTH_FS = """#version 330
void main(){}"""

VS_FUN = """#version 330
uniform mat4 mvp; uniform vec3 cen; uniform float R; uniform float Hh; in vec3 p; out vec3 vp;
void main(){ vp = cen + vec3(p.x*R, p.y*Hh, p.z*R); gl_Position = mvp*vec4(vp, 1.0); }"""
FS_FUN = """#version 330
uniform float tm; uniform vec3 eye; uniform sampler2D noise; uniform vec3 cen; uniform float R; uniform float Hh; uniform float r0;
uniform float alpha; uniform float dark; uniform float dusty; uniform float flare;
in vec3 vp; out vec4 c;
const vec3 SUN = vec3(-0.5845, 0.5632, 0.5845);
void main(){
  vec3 ro = eye - cen; vec3 rd = normalize(vp - eye);
  // ray / box [-R,R]x[0,Hh]x[-R,R]
  vec3 bmin = vec3(-R, 0.0, -R), bmax = vec3(R, Hh, R);
  vec3 inv = 1.0/rd; vec3 t0 = (bmin - ro)*inv, t1 = (bmax - ro)*inv;
  vec3 tmn = min(t0, t1), tmx = max(t0, t1);
  float ta = max(max(tmn.x, tmn.y), max(tmn.z, 0.0)), tb = min(min(tmx.x, tmx.y), tmx.z);
  if (tb <= ta) discard;
  const int N = 56;
  float dt = (tb - ta)/float(N);
  float jit = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233)))*43758.5453);
  vec3 C = vec3(0.0); float A = 0.0;
  for (int i = 0; i < N; i++) {
    float t = ta + (float(i) + jit)*dt;
    vec3 q = ro + rd*t;
    float h = q.y/Hh;
    vec2 x = q.xz/r0;
    vec2 wob = vec2(sin(tm*0.8 + h*3.0), cos(tm*0.7 + h*2.6))*0.30*h*h;
    x -= wob;
    float dist = length(x);
    float ang = atan(x.y, x.x);
    float sw = ang/6.2831853 + tm*(0.85 - 0.45*h);
    float n1 = texture(noise, vec2(sw*2.0, h*1.1 - tm*0.05)).r;
    float n2 = texture(noise, vec2(sw*5.0 + h*1.5, h*4.0 - tm*0.22)).g;
    float n3 = texture(noise, vec2(sw*12.0 + h*4.0, h*4.5 - tm*0.4)).b;
    float rad = 0.30 + 1.15*pow(h, 1.6) + 1.5*flare*smoothstep(0.80, 1.0, h);
    float edge = rad*(0.84 + 0.22*n1 + 0.14*n3);
    float dfun = (1.0 - smoothstep(edge*0.86, edge, dist)) * (1.0 - smoothstep(0.94, 1.0, h));
    float hd = h/0.34;
    float ddust = 0.0;
    float dn = 0.0;
    if (hd < 1.0) {
      dn = texture(noise, vec2(sw*3.0 + dist*0.05, dist*0.08 - tm*0.16 + hd*0.3)).b;
      float dn2 = texture(noise, vec2(sw*9.0 + dist*0.2, hd*3.0 - tm*0.4)).r;
      float drad = 3.1*(1.0 - 0.50*hd) * (0.8 + 0.4*dn2);
      ddust = (1.0 - smoothstep(drad*0.30, drad, dist)) * pow(1.0 - hd, 1.1) * (0.30 + 1.0*dn*dn2) * dusty;
    }
    float sig = dfun*3.2 + ddust*0.9;
    if (sig < 0.002) continue;
    float a = 1.0 - exp(-sig*dt/r0*2.2);
    vec3 nr = dist > 1e-4 ? vec3(x.x, 0.0, x.y)/dist : vec3(0.0, 1.0, 0.0);
    float lit = 0.40 + 0.85*max(dot(nr, SUN), 0.0);
    float band = 0.5 + 0.5*sin(6.2831853*(sw*4.0 + h*2.2) + n1*3.0);
    vec3 fc = mix(vec3(0.33,0.34,0.37), vec3(0.82,0.82,0.84), band*0.40 + n3*0.35 + 0.12) * mix(0.80, 1.0, h);
    vec3 dc = mix(vec3(0.42,0.32,0.22), vec3(0.76,0.64,0.48), dn*0.6 + band*0.25 + n3*0.15);
    vec3 col = mix(fc, dc, clamp(ddust/(dfun + ddust + 1e-4), 0.0, 1.0)) * lit * dark;
    C += (1.0 - A)*a*col; A += (1.0 - A)*a;
    if (A > 0.985) break;
  }
  if (A < 0.004) discard;
  c = vec4(C*alpha, A*alpha);
}"""

FS_CLOUDV = """#version 330
uniform float tm; uniform vec3 eye; uniform sampler2D noise; uniform vec3 cen; uniform float R; uniform float Hh;
uniform float alpha; uniform float spin;
in vec3 vp; out vec4 c;
const vec3 SUN = vec3(-0.5845, 0.5632, 0.5845);
float dens(vec3 q, out float hf){
  float r = length(q.xz)/R;
  float a = tm*spin*(1.0 - 0.5*clamp(r, 0.0, 1.0));
  float ca = cos(a), sa = sin(a);
  vec2 p = vec2(q.x*ca - q.z*sa, q.x*sa + q.z*ca)/R;
  float n1 = texture(noise, p*0.9 + vec2(0.13, 0.71)).r;
  float n2 = texture(noise, p*2.7 + vec2(0.37, 0.11)).g;
  float n3 = texture(noise, p*7.5 + vec2(q.y/Hh*0.3, 0.5)).b;
  float wall = 1.0 - smoothstep(0.10, 0.30, r);
  float base = 0.30 - 0.26*wall + (n2 - 0.5)*0.30 + (n3 - 0.5)*0.16;
  float top = 0.62 + (n1 - 0.5)*0.45 + (n2 - 0.5)*0.2 - 0.25*smoothstep(0.5, 1.0, r);
  float h = q.y/Hh;
  hf = clamp((h - base)/max(top - base, 0.05), 0.0, 1.0);
  float edge = 1.0 - smoothstep(0.62 + 0.25*n1, 1.0, r);
  float d = smoothstep(base, base + 0.05, h) * (1.0 - smoothstep(top - 0.06, top, h)) * edge;
  return d*(0.55 + 0.8*n3);
}
void main(){
  vec3 ro = eye - cen; vec3 rd = normalize(vp - eye);
  vec3 bmin = vec3(-R, 0.0, -R), bmax = vec3(R, Hh, R);
  vec3 inv = 1.0/rd; vec3 t0 = (bmin - ro)*inv, t1 = (bmax - ro)*inv;
  vec3 tmn = min(t0, t1), tmx = max(t0, t1);
  float ta = max(max(tmn.x, tmn.y), max(tmn.z, 0.0)), tb = min(min(tmx.x, tmx.y), tmx.z);
  if (tb <= ta) discard;
  const int N = 60;
  float dt = (tb - ta)/float(N);
  float jit = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233)))*43758.5453);
  vec3 C = vec3(0.0); float A = 0.0;
  for (int i = 0; i < N; i++) {
    vec3 q = ro + rd*(ta + (float(i) + jit)*dt);
    float hf;
    float d = dens(q, hf);
    if (d < 0.01) continue;
    float a = 1.0 - exp(-d*dt/Hh*26.0);
    float hf2;
    float occ = dens(q + SUN*Hh*0.18, hf2);
    float lit = exp(-occ*1.6);
    float rr = length(q.xz)/R;
    float nn = texture(noise, q.xz/R*1.7 + vec2(0.5, 0.2)).g;
    float n4 = texture(noise, q.xz/R*5.0 + vec2(tm*0.004, 0.0)).r;
    vec3 col = mix(vec3(0.20,0.22,0.26), vec3(0.66,0.68,0.72), pow(hf, 0.8)) * (0.55 + 0.75*lit) * (0.62 + 0.55*nn) * (0.8 + 0.4*n4);
    col += vec3(0.20,0.21,0.23)*smoothstep(0.50, 1.0, rr);
    a *= 1.0 - 0.55*smoothstep(0.6, 1.0, rr);
    C += (1.0 - A)*a*col; A += (1.0 - A)*a;
    if (A > 0.98) break;
  }
  if (A < 0.004) discard;
  c = vec4(C*alpha, A*alpha);
}"""

FS_DUST = """#version 330
uniform float tm; uniform vec3 eye; uniform sampler2D noise; uniform vec3 cen; uniform float R; uniform float Hh;
uniform float alpha; uniform float grow;
in vec3 vp; out vec4 c;
const vec3 SUN = vec3(-0.5845, 0.5632, 0.5845);
float hsh(float x){ return fract(sin(x*127.1)*43758.5453); }
float dens(vec3 q){
  float h = q.y/Hh;
  if (h < 0.0 || h > 1.0) return 0.0;
  float d = 0.0;
  for (int k = 0; k < 9; k++) {
    float fk = float(k);
    vec2 pc = (vec2(hsh(fk*1.7), hsh(fk*3.1 + 1.0)) - 0.5)*1.6*R;
    float hgt = grow*(0.40 + 0.60*hsh(fk*5.3));
    float w = R*(0.10 + 0.16*hsh(fk*7.7))*(0.7 + 1.3*h);
    vec2 dd = q.xz - pc - vec2(sin(tm*0.6 + fk), cos(tm*0.5 + fk*2.0))*R*0.04*h;
    float r = length(dd)/w;
    float col = exp(-r*r)*(1.0 - smoothstep(hgt*0.65, hgt, h))*smoothstep(0.0, 0.05, h);
    float n = texture(noise, dd/(R*0.7) + vec2(h*0.6 - tm*0.03, fk*0.31)).g;
    d += col*(0.45 + n);
  }
  return d*0.8;
}
void main(){
  vec3 ro = eye - cen; vec3 rd = normalize(vp - eye);
  vec3 bmin = vec3(-R, 0.0, -R), bmax = vec3(R, Hh, R);
  vec3 inv = 1.0/rd; vec3 t0 = (bmin - ro)*inv, t1 = (bmax - ro)*inv;
  vec3 tmn = min(t0, t1), tmx = max(t0, t1);
  float ta = max(max(tmn.x, tmn.y), max(tmn.z, 0.0)), tb = min(min(tmx.x, tmx.y), tmx.z);
  if (tb <= ta) discard;
  const int N = 48;
  float dt = (tb - ta)/float(N);
  float jit = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233)))*43758.5453);
  vec3 C = vec3(0.0); float A = 0.0;
  for (int i = 0; i < N; i++) {
    vec3 q = ro + rd*(ta + (float(i) + jit)*dt);
    float d = dens(q);
    if (d < 0.01) continue;
    float a = 1.0 - exp(-d*dt/Hh*7.0);
    float occ = dens(q + SUN*Hh*0.15);
    vec3 col = mix(vec3(0.50,0.41,0.31), vec3(0.80,0.73,0.63), clamp(q.y/Hh*1.2, 0.0, 1.0))*(0.62 + 0.45*exp(-occ*1.2));
    C += (1.0 - A)*a*col; A += (1.0 - A)*a;
    if (A > 0.97) break;
  }
  if (A < 0.004) discard;
  c = vec4(C*alpha, A*alpha);
}"""

FS_HURR = """#version 330
uniform float tm; uniform vec3 eye; uniform sampler2D noise; uniform vec3 cen; uniform float R; uniform float Hh;
uniform float alpha; uniform float spin; uniform float eyer; uniform float seed;
in vec3 vp; out vec4 c;
const vec3 SUN = vec3(-0.5845, 0.5632, 0.5845);
float dens(vec3 q){
  vec2 p = q.xz/R;
  float r = length(p);
  if (r > 1.0) return 0.0;
  float th = atan(p.y, p.x);
  float a = th + tm*spin;              // rigid rotation (differential rotation wound the arms into rings over time)
  float arms = 0.5 + 0.5*sin(3.0*a - 6.0*log(r + 0.04) + seed);
  vec2 pr = vec2(cos(a), sin(a))*r;
  float n1 = texture(noise, pr*1.6 + vec2(seed*0.13, 0.3)).r;
  float n2 = texture(noise, pr*5.0 + vec2(0.7, seed*0.07)).g;
  float wall = exp(-pow((r - eyer*1.25)/(eyer*0.7), 2.0));
  arms = smoothstep(0.30, 0.85, arms);
  float core = 1.0 - smoothstep(0.22, 0.48, r);
  float d = (core*0.95 + arms*(1.0 - core)*1.0) * (0.55 + 0.7*n1) * (0.75 + 0.5*n2) + wall*1.3;
  d *= smoothstep(eyer*0.75, eyer*1.1, r) * (1.0 - smoothstep(0.6 + 0.3*n1, 1.0, r));
  float top = Hh*(0.62 + 0.25*n2 + 0.13*n1)*(1.0 - 0.35*smoothstep(0.55, 1.0, r));
  float bot = Hh*0.08*(1.0 + n1);
  return d * smoothstep(bot, bot + Hh*0.06, q.y) * (1.0 - smoothstep(top - Hh*0.08, top, q.y));
}
void main(){
  vec3 ro = eye - cen; vec3 rd = normalize(vp - eye);
  vec3 bmin = vec3(-R, 0.0, -R), bmax = vec3(R, Hh, R);
  vec3 inv = 1.0/rd; vec3 t0 = (bmin - ro)*inv, t1 = (bmax - ro)*inv;
  vec3 tmn = min(t0, t1), tmx = max(t0, t1);
  float ta = max(max(tmn.x, tmn.y), max(tmn.z, 0.0)), tb = min(min(tmx.x, tmx.y), tmx.z);
  if (tb <= ta) discard;
  const int N = 72;
  float dt = (tb - ta)/float(N);
  float jit = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233)))*43758.5453);
  vec3 C = vec3(0.0); float A = 0.0;
  for (int i = 0; i < N; i++) {
    vec3 q = ro + rd*(ta + (float(i) + jit)*dt);
    float d = dens(q);
    if (d < 0.01) continue;
    float a = 1.0 - exp(-d*dt/Hh*3.2);
    float occ = dens(q + SUN*Hh*0.35);
    float lit = exp(-occ*1.2);
    vec3 col = mix(vec3(0.55,0.58,0.64), vec3(1.0,1.0,1.0), clamp(q.y/Hh*1.4, 0.0, 1.0)) * (0.62 + 0.42*lit);
    C += (1.0 - A)*a*col; A += (1.0 - A)*a;
    if (A > 0.985) break;
  }
  if (A < 0.004) discard;
  c = vec4(C*alpha, A*alpha);
}"""

FS_RAIN = """#version 330
uniform vec2 res; uniform float tm; uniform float amount; uniform float slant;
in float vy; out vec4 c;
float h1(float x){ return fract(sin(x*127.1)*43758.5453); }
void main(){
  vec2 uv = gl_FragCoord.xy/res;
  float asp = res.x/res.y;
  float a = 0.0;
  for (int L = 0; L < 3; L++) {
    float sc = 70.0 + 55.0*float(L);
    vec2 q = vec2((uv.x*asp + uv.y*slant)*sc, uv.y*sc*0.10 + tm*(1.6 + 0.6*float(L)));
    float col = floor(q.x);
    float y = fract(q.y + h1(col + float(L)*31.0)*7.0);
    float w = abs(fract(q.x) - 0.5);
    a += smoothstep(0.0, 0.03, y)*(1.0 - smoothstep(0.05, 0.14, y)) * (1.0 - smoothstep(0.06, 0.18, w)) * (0.3 + 0.7*h1(col*3.1 + float(L)));
  }
  c = vec4(vec3(0.86,0.89,0.93)*a*amount*0.30, a*amount*0.30);
}"""

VS_BORDER = """#version 330
uniform mat4 mvp; uniform float yscale; uniform float hw; uniform float lift;
in vec3 p; in vec2 pr; in float side; out float vs;
void main(){ vec3 q = vec3(p.x + pr.x*side*hw, p.y*yscale + lift, p.z + pr.y*side*hw); gl_Position = mvp*vec4(q, 1.0); vs = side; }"""
FS_BORDER = """#version 330
uniform float alpha; in float vs; out vec4 c;
void main(){ float s = abs(vs); float core = 1.0 - smoothstep(0.30, 0.45, s);
  vec3 col = mix(vec3(1.0), vec3(0.16,0.18,0.26), core);
  float a = alpha*mix(0.80, 0.85, core)*(1.0 - smoothstep(0.85, 1.0, s));
  c = vec4(col*a, a); }"""

VS_PUFF = """#version 330
uniform mat4 mvp; uniform vec3 cen; uniform float spin; uniform float tm;
in vec3 p; in vec4 ia; in float ib;          // ia = (radius_frac_x, y, radius_frac_z, size), ib = shade
out vec3 vn; out vec3 vp; flat out float vsh; out float vy;
void main(){
  float a = tm*spin*(1.0 - 0.4*clamp(length(ia.xz)/0.0015, 0.0, 1.0));
  float ca = cos(a), sa = sin(a);
  vec2 xz = vec2(ia.x*ca - ia.z*sa, ia.x*sa + ia.z*ca);
  vec3 w = cen + vec3(xz.x, ia.y, xz.y) + p*vec3(ia.w, ia.w*0.62, ia.w);
  vp = w; vn = p; vsh = ib; vy = p.y;
  gl_Position = mvp*vec4(w, 1.0);
}"""
FS_PUFF = """#version 330
uniform vec3 eye; uniform float alpha; uniform sampler2D noise;
in vec3 vn; in vec3 vp; flat in float vsh; in float vy; out vec4 c;
const vec3 SUN = vec3(-0.5845, 0.5632, 0.5845);
void main(){
  vec3 n = normalize(vn);
  vec3 v = normalize(eye - vp);
  float id = floor(vsh);
  float sh = 0.85 + 0.2*fract(vsh);
  vec2 q = n.xz*0.35 + n.y*0.2 + vec2(id*0.137, id*0.071);
  float b1 = texture(noise, q).r, b2 = texture(noise, q*3.1).g;
  float bump = b1*0.6 + b2*0.4;
  float lit = clamp(dot(n, SUN)*0.5 + 0.5 + (bump - 0.5)*0.6, 0.0, 1.0);
  vec3 col = mix(vec3(0.17,0.19,0.23), vec3(0.70,0.72,0.76), lit*lit) * mix(0.70, 1.12, smoothstep(-0.7, 0.8, n.y)) * sh;
  float a = smoothstep(0.0, 0.45, dot(n, v) - (bump - 0.5)*0.25);
  c = vec4(col*a*alpha, a*alpha);
}"""

VS_DEB = """#version 330
uniform mat4 mvp; uniform vec3 cen; uniform float r0; uniform float Hh; uniform float tm;
in vec3 p; in vec3 n; in vec4 ia; in vec4 ib;   // ia = (rfrac, hfrac, omega, phase), ib = (size_world, colour, tumble, axis)
out vec3 vn; flat out float vcol;
mat3 rot(vec3 ax, float a){ ax = normalize(ax); float s = sin(a), cc = cos(a), oc = 1.0 - cc;
  return mat3(oc*ax.x*ax.x + cc, oc*ax.x*ax.y + ax.z*s, oc*ax.z*ax.x - ax.y*s,
              oc*ax.x*ax.y - ax.z*s, oc*ax.y*ax.y + cc, oc*ax.y*ax.z + ax.x*s,
              oc*ax.z*ax.x + ax.y*s, oc*ax.y*ax.z - ax.x*s, oc*ax.z*ax.z + cc); }
void main(){
  float r = r0*(0.45 + 1.9*ia.x*ia.x);
  float y = Hh*0.50*pow(ia.y, 2.3) + r0*0.04;
  float ang = ia.w + tm*ia.z;
  vec3 ax = vec3(sin(ib.w*7.0), cos(ib.w*3.0), sin(ib.w*5.0 + 1.0));
  mat3 R = rot(ax, tm*ib.z);
  vec3 sz = vec3(1.0, 0.35, 0.6)*ib.x;
  vec3 w = cen + vec3(cos(ang)*r, y, sin(ang)*r) + R*(p*sz);
  vn = R*n; vcol = ib.y;
  gl_Position = mvp*vec4(w, 1.0);
}"""
FS_DEB = """#version 330
uniform float alpha;
in vec3 vn; flat in float vcol; out vec4 c;
const vec3 SUN = vec3(-0.5845, 0.5632, 0.5845);
void main(){
  vec3 col = vcol < 0.3 ? vec3(0.55,0.42,0.30) : (vcol < 0.5 ? vec3(0.50,0.50,0.52) : (vcol < 0.7 ? vec3(0.86,0.85,0.82) : (vcol < 0.85 ? vec3(0.30,0.30,0.33) : vec3(0.62,0.32,0.24))));
  float l = 0.55 + 0.45*abs(dot(normalize(vn), SUN));
  c = vec4(col*l*0.85, alpha);
}"""
VS_BAR = core.VS_BAR
FS_BAR = core.FS_BAR


def _set(prog, name, value):
    if name in prog:
        prog[name].value = value


def _icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v, np.float32), np.array(f, np.int32)


def mvp64(cam, W, H, origin=(0.0, 0.0, 0.0)):
    p, y = math.radians(min(cam.pitch, 89.0)), math.radians(cam.yaw)
    tgt = np.array([cam.tx, cam.ty, cam.tz], float) + np.array(cam.shake, float)
    eye = tgt + cam.dist * np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)])
    o = np.asarray(origin, float)
    e, t_ = eye - o, tgt - o
    f = t_ - e
    f /= np.linalg.norm(f)
    s = np.cross(f, [0, 1, 0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    V = np.eye(4)
    V[0, :3], V[1, :3], V[2, :3] = s, u, -f
    V[:3, 3] = -V[:3, :3] @ e
    n_, f_ = max(1e-6, cam.dist * 0.02), cam.dist * 40 + 2
    tt = 1 / math.tan(math.radians(cam.FOV) / 2)
    P = np.array([[tt / (W / H), 0, 0, 0], [0, tt, 0, 0], [0, 0, (f_ + n_) / (n_ - f_), 2 * f_ * n_ / (n_ - f_)], [0, 0, -1, 0]])
    return P @ V, eye


def _vao(ctx, prog, parts, index=None):
    """vertex_array that tolerates attributes the linker removed (depth-only programs)"""
    sizes = {"f": 4, "i": 4}
    out = []
    for buf, fmt, *names in parts:
        toks = fmt.split()
        inst = toks[-1].endswith("/i")
        if inst:
            toks[-1] = toks[-1][:-2]
        nf, nn = [], []
        for tk, nm in zip(toks, names):
            if nm in prog:
                nf.append(tk)
                nn.append(nm)
            else:
                nf.append(f"{int(tk[:-1]) * sizes[tk[-1]]}x")
        if nn:
            out.append((buf, " ".join(nf) + ("/i" if inst else ""), *nn))
    return ctx.vertex_array(prog, out, index) if index is not None else ctx.vertex_array(prog, out)


def _b(m):
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


# ------------------------------------------------------------------ the scene
class MiniScene:
    SHADOW = 4096

    def __init__(self, W: int, H: int, msaa: int = 4, state: str | None = None):
        import moderngl
        self.mgl = moderngl
        try:
            self.ctx = moderngl.create_standalone_context(backend="egl")
        except Exception:
            self.ctx = moderngl.create_standalone_context()
        ctx = self.ctx
        self.W, self.H = W, H
        self.renderer = ctx.info.get("GL_RENDERER", "?")
        la, lo, z = relief()
        LO, LA = np.meshgrid(lo, la)
        x, zz = FR.world(LO, LA)
        y = height_world(z)
        pos = np.dstack([x, y, zz]).astype("f4")
        nrm = np.cross(np.gradient(pos, axis=1), np.gradient(pos, axis=0))
        nrm /= np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-9
        nrm[nrm[..., 1] < 0] *= -1
        u, v = (x / WORLD_W + 0.5), (zz / WORLD_H + 0.5)
        nr, nc = z.shape
        ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        edge = np.minimum(np.minimum(ii, nr - 1 - ii), np.minimum(jj, nc - 1 - jj)).astype("f4")
        edge = np.clip(edge / 40.0, 0, 1)
        lim = (-0.3, 1.3) if outer_image() is not None else (-0.03, 1.03)
        edge = np.minimum(edge, np.clip(np.minimum.reduce([u - lim[0], lim[1] - u, v - lim[0], lim[1] - v]) / 0.06, 0, 1))
        verts = np.dstack([pos, u, v, nrm, edge]).astype("f4").reshape(-1, 9)
        inside = (u > lim[0]) & (u < lim[1]) & (v > lim[0]) & (v < lim[1])
        i = np.arange(nr - 1)[:, None] * nc + np.arange(nc - 1)[None, :]
        keep = inside[:-1, :-1] | inside[1:, :-1] | inside[:-1, 1:] | inside[1:, 1:]
        i = i[keep]
        idx = np.stack([i, i + 1, i + nc, i + 1, i + nc + 1, i + nc], -1).astype("i4").ravel()
        self.p_ter = ctx.program(vertex_shader=VS_TER, fragment_shader=FS_TER)
        self._tb, self._ti = ctx.buffer(verts.tobytes()), ctx.buffer(idx.tobytes())
        self.vao_ter = _vao(ctx, self.p_ter, [(self._tb, "3f 2f 3f 1f", "p", "t", "n", "ed")], self._ti)
        sw, sh = min(8192, TW * 2), min(4608, TH * 2)
        self.t_nat = self._tex(nation_image(sw, sh), mip=True)
        self.t_state, self.sbox = None, (0.0, 0.0, 1.0)
        if state:
            sj = ROOT / "data" / "world" / f"state_{state}.json"
            sp = ROOT / "data" / "world" / f"state_{state}.png"
            if sj.exists() and sp.exists():
                meta = json.loads(sj.read_text())
                self.t_state = self._tex(Image.open(sp).convert("RGBA"), mip=True)
                self.sbox = (meta["cx"], meta["cz"], meta["size"])
        oi = outer_image()
        self.t_outer = self._tex(oi, mip=True) if oi is not None else None
        self.t_noise = self._tex(noise_image(), mip=True, repeat=True)
        self.t_ovl = ctx.texture((TW, TH), 4, b"\x00" * (TW * TH * 4))
        self.t_ovl.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.t_ovl.repeat_x = self.t_ovl.repeat_y = False
        self.t_ovl.build_mipmaps()
        self.t_dummy = ctx.texture((4, 4), 4, b"\x00" * 64)
        self.p_sky = ctx.program(vertex_shader=VS_SKY, fragment_shader=FS_SKY)
        self.vao_sky = ctx.vertex_array(self.p_sky, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes()), "2f", "p")])
        # unit meshes
        self.box_v, self.box_i = self._box()
        cube = self._cube()
        self.p_bar = ctx.program(vertex_shader=VS_BAR, fragment_shader=FS_BAR)
        self.vao_bar = ctx.vertex_array(self.p_bar, [(ctx.buffer(self.box_v.tobytes()), "3f 3f", "p", "n")], ctx.buffer(self.box_i.tobytes()))
        self.p_bld = ctx.program(vertex_shader=VS_BLD, fragment_shader=FS_BLD)
        self.p_bld_d = ctx.program(vertex_shader=VS_BLD, fragment_shader=VS_DEPTH_FS)
        self.p_tree = ctx.program(vertex_shader=VS_TREE, fragment_shader=FS_TREE)
        self.p_tree_d = ctx.program(vertex_shader=VS_TREE, fragment_shader=VS_DEPTH_FS)
        self.tree_mesh = self._tree_meshes()
        self.p_fun = ctx.program(vertex_shader=VS_FUN, fragment_shader=FS_FUN)
        self.p_cloudv = ctx.program(vertex_shader=VS_FUN, fragment_shader=FS_CLOUDV)
        self.p_hurr = ctx.program(vertex_shader=VS_FUN, fragment_shader=FS_HURR)
        self.p_dust = ctx.program(vertex_shader=VS_FUN, fragment_shader=FS_DUST)
        self.p_border = ctx.program(vertex_shader=VS_BORDER, fragment_shader=FS_BORDER)
        bv = self._border_mesh()
        self.vao_border = ctx.vertex_array(self.p_border, [(ctx.buffer(bv.tobytes()), "3f 2f 1f", "p", "pr", "side")])
        self.n_border = len(bv)
        self.p_rain = ctx.program(vertex_shader=VS_SKY, fragment_shader=FS_RAIN)
        self.vao_rain = ctx.vertex_array(self.p_rain, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes()), "2f", "p")])
        self.t_dem0 = ctx.texture((2, 2), 1, np.zeros(4, "f4").tobytes(), dtype="f4")
        cb = np.array([[-1, 0, -1], [1, 0, -1], [1, 1, -1], [-1, 1, -1], [-1, 0, 1], [1, 0, 1], [1, 1, 1], [-1, 1, 1]], "f4")
        ci = np.array([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5], "i4")
        tri = cb[ci.reshape(-1, 3)]
        ctr = np.array([0, 0.5, 0], "f4")
        nrm_ = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        flip = np.einsum("ij,ij->i", nrm_, tri.mean(1) - ctr) < 0
        ci = ci.reshape(-1, 3)
        ci[flip] = ci[flip][:, [0, 2, 1]]
        ci = ci.ravel().astype("i4")
        self.vao_fun = ctx.vertex_array(self.p_fun, [(ctx.buffer(cb.tobytes()), "3f", "p")], ctx.buffer(ci.tobytes()))
        self.vao_cloudv = ctx.vertex_array(self.p_cloudv, [(ctx.buffer(cb.tobytes()), "3f", "p")], ctx.buffer(ci.tobytes()))
        self.vao_hurr = ctx.vertex_array(self.p_hurr, [(ctx.buffer(cb.tobytes()), "3f", "p")], ctx.buffer(ci.tobytes()))
        self.vao_dust = ctx.vertex_array(self.p_dust, [(ctx.buffer(cb.tobytes()), "3f", "p")], ctx.buffer(ci.tobytes()))
        sv, sf = _icosphere(2)
        self.p_puff = ctx.program(vertex_shader=VS_PUFF, fragment_shader=FS_PUFF)
        self.puff_vb = ctx.buffer(sv.tobytes())
        self.puff_ib = ctx.buffer(sf.ravel().tobytes())
        self.puff_inst = ctx.buffer(reserve=4 * 5 * 1200)
        self.vao_puff = ctx.vertex_array(self.p_puff, [(self.puff_vb, "3f", "p"), (self.puff_inst, "4f 1f/i", "ia", "ib")], self.puff_ib)
        self.p_deb = ctx.program(vertex_shader=VS_DEB, fragment_shader=FS_DEB)
        rng = np.random.default_rng(3)
        nd = 1100
        deb = np.c_[rng.random(nd) ** 0.8, rng.random(nd), rng.uniform(0.7, 2.2, nd), rng.uniform(0, 6.283, nd),
                    np.zeros(nd), rng.random(nd), rng.uniform(2, 7, nd), rng.random(nd)].astype("f4")
        self.deb_base = deb
        self.deb_inst = ctx.buffer(deb.tobytes())
        self.vao_deb = ctx.vertex_array(self.p_deb, [(ctx.buffer(cube[0].tobytes()), "3f 3f", "p", "n"), (self.deb_inst, "4f 4f/i", "ia", "ib")],
                                        ctx.buffer(cube[1].tobytes()))
        self.n_deb = nd
        self.puffs = None
        self.town = None
        self.levels = {}
        self.msaa = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), samples=msaa)], depth_attachment=ctx.depth_renderbuffer((W, H), samples=msaa))
        self.out = ctx.simple_framebuffer((W, H))
        self.smap = ctx.depth_texture((self.SHADOW, self.SHADOW))
        self.smap.compare_func = "<="
        self.smap.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.smap.repeat_x = self.smap.repeat_y = False
        self.sfb = ctx.framebuffer(depth_attachment=self.smap)

    @staticmethod
    def _border_mesh():
        """state borders (US Census cartographic boundaries) as ribbons: two triangles per segment, widened in the shader"""
        from mapviz.core import states
        out = []
        for rings in states().values():
            for rg in rings:
                u, v = FR.xy(rg[:, 0], rg[:, 1])
                wx, wz = (np.asarray(u) / TW - 0.5) * WORLD_W, (np.asarray(v) / TH - 0.5) * WORLD_H
                lon, lat = _inv_world(wx, wz)
                wy = height_at(lon, lat)
                P = np.stack([wx, wy, wz], 1)
                a, b = P[:-1], P[1:]
                d = b[:, [0, 2]] - a[:, [0, 2]]
                L = np.linalg.norm(d, axis=1)
                ok = L > 1e-9
                a, b, d, L = a[ok], b[ok], d[ok], L[ok]
                n = np.stack([-d[:, 1], d[:, 0]], 1) / L[:, None]
                one = np.ones((len(a), 1))
                va = lambda P_, s_: np.concatenate([P_, n, one * s_], 1)  # noqa: E731
                quad = np.stack([va(a, -1), va(b, -1), va(b, 1), va(a, -1), va(b, 1), va(a, 1)], 1).reshape(-1, 6)
                out.append(quad)
        return np.concatenate(out).astype("f4")

    def _borders(self, M, cam, yscale):
        ctx, mgl = self.ctx, self.mgl
        al = float(np.clip((cam.dist - 0.03) / 0.06, 0.0, 1.0))
        if al <= 0.01:
            return
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
        self.msaa.depth_mask = False
        ctx.depth_func = "<="
        px = 3.4 * self.H / 1080.0
        hw = px / 2 * 2 * math.tan(math.radians(cam.FOV) / 2) * cam.dist / self.H
        p = self.p_border
        p["mvp"].write(_b(M))
        for k, v in dict(yscale=float(yscale), hw=float(hw), lift=float(cam.dist * 0.0015), alpha=al).items():
            _set(p, k, v)
        self.vao_border.render()
        ctx.depth_func = "<"
        self.msaa.depth_mask = True
        ctx.disable(mgl.BLEND)

    def _tex(self, img, mip=False, repeat=False):
        mgl = self.mgl
        t = self.ctx.texture(img.size, len(img.getbands()), img.tobytes())
        if mip:
            t.build_mipmaps()
            t.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            t.anisotropy = 16.0
        t.repeat_x = t.repeat_y = repeat
        return t

    def _box(self):
        bv, bi = [], []
        faces = [((0, 0, 1), [(-.5, 0, .5), (.5, 0, .5), (.5, 1, .5), (-.5, 1, .5)]), ((0, 0, -1), [(.5, 0, -.5), (-.5, 0, -.5), (-.5, 1, -.5), (.5, 1, -.5)]),
                 ((1, 0, 0), [(.5, 0, .5), (.5, 0, -.5), (.5, 1, -.5), (.5, 1, .5)]), ((-1, 0, 0), [(-.5, 0, -.5), (-.5, 0, .5), (-.5, 1, .5), (-.5, 1, -.5)]),
                 ((0, 1, 0), [(-.5, 1, .5), (.5, 1, .5), (.5, 1, -.5), (-.5, 1, -.5)])]
        for n, q in faces:
            b0 = len(bv)
            for pt in q:
                bv.append((*pt, *n))
            bi += [b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3]
        return np.array(bv, "f4"), np.array(bi, "i4")

    def _cube(self):
        v, i = self._box()
        v = v.copy()
        v[:, 1] -= 0.5
        bottom = np.array([(-.5, -.5, -.5, 0, -1, 0), (.5, -.5, -.5, 0, -1, 0), (.5, -.5, .5, 0, -1, 0), (-.5, -.5, .5, 0, -1, 0)], "f4")
        b0 = len(v)
        return np.concatenate([v, bottom]).astype("f4"), np.concatenate([i, [b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3]]).astype("i4")

    def _tree_meshes(self):
        out = {}
        sv, sf = _icosphere(1)
        rng = np.random.default_rng(2)
        lump = sv * (1 + 0.13 * rng.standard_normal((len(sv), 1)).clip(-1.5, 1.5)).astype("f4")
        crown = np.c_[lump, sv, np.ones(len(sv))].astype("f4")
        trunk_v, trunk_i = [], []
        for k in range(6):
            a0, a1 = k / 6 * 2 * math.pi, (k + 1) / 6 * 2 * math.pi
            for (a, yy) in ((a0, 0), (a1, 0), (a1, 1), (a0, 1)):
                trunk_v.append((math.cos(a), yy, math.sin(a), math.cos(a), 0, math.sin(a), 0))
            b = k * 4
            trunk_i += [b, b + 2, b + 1, b, b + 3, b + 2]
        tv = np.array(trunk_v, "f4")
        verts = np.concatenate([tv, crown]).astype("f4")
        idx = np.concatenate([np.array(trunk_i, "i4"), sf.ravel() + len(tv)]).astype("i4")
        out["round"] = (verts, idx)
        # conifer: stacked cones
        cv, ci = list(map(tuple, tv)), list(trunk_i)
        for tier, (y0, y1, r) in enumerate(((-0.75, 0.25, 0.95), (-0.25, 0.75, 0.75), (0.25, 1.15, 0.5))):
            b0 = len(cv)
            for k in range(8):
                a = k / 8 * 2 * math.pi
                nx, nz = math.cos(a), math.sin(a)
                cv.append((r * nx, y0, r * nz, nx * 0.8, 0.6, nz * 0.8, 1))
            cv.append((0, y1, 0, 0, 1, 0, 1))
            for k in range(8):
                ci += [b0 + k, b0 + 8, b0 + (k + 1) % 8]
        out["cone"] = (np.array(cv, "f4"), np.array(ci, "i4"))
        return out

    # ---- town resources (one town at a time)
    def use_town(self, key):
        if self.town is not None and self.town["key"] == key:
            return self.town
        self.release_town()
        ctx = self.ctx
        T = Town(key)
        R = dict(key=key, T=T, tex={}, box={})
        nl = T.nlcd_local(L1_KM, L1_N)
        cls_mid = None
        if T.region:
            for lvl in ("reg", "mid"):
                if lvl in T.region["levels"]:
                    img, cls = region_image(key, lvl)
                    R["tex"][lvl] = self._tex(img, mip=True)
                    R["box"][lvl] = (0.0, 0.0, T.region["levels"][lvl]["size"])
                    if lvl == "mid":
                        cls_mid = cls
        img, size = T.ground(L1_KM, L1_N, False, nl[0] if nl else None)
        R["tex"]["l1"], R["box"]["l1"] = self._tex(img, mip=True), (0.0, 0.0, size)
        nl2 = T.nlcd_local(L2_KM, L2_N)
        img, size = T.ground(L2_KM, L2_N, True, nl2[0] if nl2 else None)
        R["tex"]["l2"], R["box"]["l2"] = self._tex(img, mip=True), (0.0, 0.0, size)
        crop = None
        if cls_mid is not None:
            lv = T.region["levels"]["mid"]
            frac = (TREE_KM / KM_PER_UNIT) / lv["size"]
            Nn = cls_mid.shape[0]
            a = (0.5 - frac / 2) * Nn
            crop = np.asarray(Image.fromarray(cls_mid).crop((int(a), int(a), int(a + frac * Nn), int(a + frac * Nn))))
        m = T.building_mesh()
        if m is not None:
            data = np.concatenate([m["p"], m["n"], m["uv"], m["info"], m["bc"]], 1).astype("f4")
            buf = ctx.buffer(data.tobytes())
            R["vao_bld"] = _vao(ctx, self.p_bld, [(buf, "3f 3f 2f 4f 3f", "p", "n", "wuv", "info", "bc")])
            R["vao_bld_d"] = _vao(ctx, self.p_bld_d, [(buf, "3f 3f 2f 4f 3f", "p", "n", "wuv", "info", "bc")])
            R["nbv"] = len(data)
        # bridges as decks (same building program, typ 1 flat, part 1 roof colour = concrete)
        br = T.bridges()
        if br:
            V = []
            for a, b, w in br:
                d = b - a
                L = np.linalg.norm(d)
                if L < 1e-9:
                    continue
                nn = np.array([-d[1], d[0]]) / L * (w / 2 + 1) / M_PER_UNIT
                lon, lat = T.lonlat((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                by = float(height_at(lon, lat))
                y0, y1 = by + 5.5 / M_PER_UNIT, by + 7.0 / M_PER_UNIT
                c4 = [a - nn, b - nn, b + nn, a + nn]
                top = [(c[0], y1, c[1]) for c in c4]
                bot = [(c[0], y0, c[1]) for c in c4]
                cen = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, by)
                for tri, nrm in (([top[0], top[2], top[1], top[0], top[3], top[2]], (0, 1, 0)),
                                 ([bot[0], bot[1], top[1], bot[0], top[1], top[0]], (nn[0], 0, nn[1])),
                                 ([bot[3], top[3], top[2], bot[3], top[2], bot[2]], (-nn[0], 0, -nn[1]))):
                    for p in tri:
                        V.append((*p, *(np.array(nrm) / (np.linalg.norm(nrm) + 1e-12)), 0, 0, 0.5, 2, 7, 1, *cen))
            if V:
                data = np.array(V, "f4")
                buf = ctx.buffer(data.tobytes())
                R["vao_br"] = _vao(ctx, self.p_bld, [(buf, "3f 3f 2f 4f 3f", "p", "n", "wuv", "info", "bc")])
                R["nbr"] = len(data)
        dz = ROOT / "data" / "region" / f"{key}_dem.npz"
        if dz.exists():
            d = np.load(dz)
            z = d["z"].astype("f4")
            R["dem"] = ctx.texture((z.shape[1], z.shape[0]), 1, np.ascontiguousarray(z).tobytes(), dtype="f4")
            R["demsize"] = float(d["km"]) / KM_PER_UNIT
        tr = T.trees(crop)
        if len(tr):
            R["trees"] = tr
            for kind, sel in (("round", tr[:, 5] < 0.5), ("cone", tr[:, 5] >= 0.5)):
                inst = tr[sel]
                if not len(inst):
                    continue
                v, i = self.tree_mesh[kind]
                ib = ctx.buffer(inst.tobytes())
                vb = ctx.buffer(v.tobytes())
                ibuf = ctx.buffer(i.tobytes())
                R[f"tree_{kind}"] = (_vao(ctx, self.p_tree, [(vb, "3f 3f 1f", "p", "n", "part"), (ib, "4f 4f/i", "ia", "ib")], ibuf),
                                     _vao(ctx, self.p_tree_d, [(vb, "3f 3f 1f", "p", "n", "part"), (ib, "4f 4f/i", "ia", "ib")], ibuf), len(inst))
        self.town = R
        print("town", key, "buildings verts", R.get("nbv", 0), "trees", len(tr), "bridges", R.get("nbr", 0), flush=True)
        return R

    def release_town(self):
        if self.town is None:
            return
        for t in self.town["tex"].values():
            t.release()
        if "dem" in self.town:
            self.town["dem"].release()
        for k, v in list(self.town.items()):
            if k.startswith("vao"):
                v.release()
            if k.startswith("tree_"):
                v[0].release()
                v[1].release()
        self.town = None

    def set_overlay(self, img: Image.Image):
        self.t_ovl.write(img.tobytes())
        self.t_ovl.build_mipmaps()

    def make_puffs(self, r_cloud, base, seed=4):
        rng = np.random.default_rng(seed)
        P = []
        for k in range(620):
            rr = r_cloud * math.sqrt(rng.uniform(0.0, 1.0))
            a = rng.uniform(0, 2 * math.pi)
            edge = rr / r_cloud
            s = r_cloud * rng.uniform(0.05, 0.11) * (1.2 - 0.5 * edge)
            y = base + s * 0.3 + rng.uniform(0, 1) * r_cloud * 0.10 * (1.0 - 0.6 * edge)
            P.append((math.cos(a) * rr, y, math.sin(a) * rr, s, k + rng.uniform(0.85, 1.05) % 1.0))
        for k in range(40):      # lowered rotating wall cloud around the funnel
            a = rng.uniform(0, 2 * math.pi)
            rr = r_cloud * rng.uniform(0.05, 0.25)
            s = r_cloud * rng.uniform(0.06, 0.09)
            P.append((math.cos(a) * rr, base - s * 0.35, math.sin(a) * rr, s, 700 + k + 0.8))
        return np.array(P, "f4")

    # ---- frame
    def render(self, cam, tm=0.0, dim=1.0, ovl_alpha=1.0, vortices=(), bars=(), local=None, hurricanes=(), **_):
        """local: dict(key, pa, pb (world x/z), hw, along, clen (world), storm=(x,z,r,s), funnel=dict(pos, H, r0, alpha), cloud=dict(r, base, alpha))"""
        ctx, mgl = self.ctx, self.mgl
        R = self.use_town(local["key"]) if local else None
        O = R["T"].O if R is not None else np.zeros(3)
        yscale = self.yscale(cam)
        cam = self.view_cam(cam)
        M0, eye = mvp64(cam, self.W, self.H)
        Ml, _ = mvp64(cam, self.W, self.H, O)
        eye_l = eye - O
        common = dict(tm=float(tm), camdist=float(cam.dist), mpu=float(M_PER_UNIT), noise=3, yscale=yscale)
        shadow_on = 0.0
        Ls = np.eye(4)
        if R is not None and cam.dist < 0.03:
            shadow_on = 1.0
            Ls = self._shadow_pass(cam, O, local, yscale)
        sw = local is not None and local.get("hw", 0) > 0
        if local:
            pa_l = (local["pa"][0] - O[0], local["pa"][1] - O[2])
            pb_l = (local["pb"][0] - O[0], local["pb"][1] - O[2])
            st = local.get("storm", (0, 0, 1, 0))
            storm_l = (st[0] - O[0], st[1] - O[2], st[2], st[3])
        else:
            pa_l = pb_l = (0.0, 0.0)
            storm_l = (0.0, 0.0, 1.0, 0.0)
        self.msaa.use()
        ctx.clear(*SKY_HOR)
        ctx.disable(mgl.DEPTH_TEST)
        ctx.disable(mgl.BLEND)
        self.p_sky["storm_sky"].value = float(min(0.85, local.get("storm", (0, 0, 1, 0))[3] * 1.2)) if local else 0.0
        self.vao_sky.render(mgl.TRIANGLE_STRIP)
        ctx.enable(mgl.DEPTH_TEST)
        # terrain
        self.t_nat.use(0)
        (self.t_outer or self.t_dummy).use(11)
        (self.t_state or self.t_dummy).use(13)
        self.t_ovl.use(5)
        self.t_noise.use(3)
        self.smap.use(6)
        (R["dem"] if R is not None and "dem" in R else self.t_dem0).use(12)
        surge_m = float(local.get("surge", {}).get("level", 0.0)) if local else 0.0
        if R is None or "dem" not in R:
            surge_m = 0.0
        common.update(dem=12, demsize=float(R.get("demsize", 1.0)) if R is not None else 1.0, surge=surge_m,
                      wind=float(local.get("wind", 0.0)) if local else 0.0, wdir=(0.8, -0.6))
        pt = self.p_ter
        pt["mvp"].write(_b(M0))
        lvon = [0.0, 0.0, 0.0, 0.0]
        boxes = [(0.0, 0.0, 1.0)] * 4
        for j, name in enumerate(("reg", "mid", "l1", "l2")):
            unit = 7 + j
            if R is not None and name in R["tex"]:
                R["tex"][name].use(unit)
                lvon[j] = 1.0
                boxes[j] = R["box"][name]
            else:
                self.t_dummy.use(unit)
            _set(pt, f"lv{j}", unit)
            _set(pt, f"box{j}", tuple(float(v) for v in boxes[j]))
        q = local.get("quake") if local else None
        if q:
            qk_l = (q["pos"][0] - O[0], q["pos"][1] - O[2])
            qr = (float(q["R"]), float(q["speed"]), float(q["t"]), float(q["alpha"]))
        else:
            qk_l, qr = (0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
        uni = dict(stt=13, sbox=(float(self.sbox[0]), float(self.sbox[1]), float(self.sbox[2])), son=1.0 if self.t_state is not None else 0.0,
                   qk=qk_l, qr=qr, nat=0, ovl=5, smap=6, outer=11, outer_on=1.0 if self.t_outer is not None else 0.0, lvon=tuple(lvon), dim=float(dim), ovl_alpha=float(ovl_alpha), eye=tuple(float(v) for v in eye),
                   O=tuple(float(v) for v in O), shadow_on=shadow_on, sbias=0.00008, stex=1.0 / self.SHADOW, storm=storm_l,
                   swath_on=1.0 if sw else 0.0, sa=tuple(float(v) for v in (local["pa"] if sw else (0, 0))), sb=tuple(float(v) for v in (local["pb"] if sw else (1, 0))),
                   shw=float(local["hw"]) if sw else 0.0, salong=float(local["along"]) if sw else 0.0, **common)
        for k, v in uni.items():
            _set(pt, k, v)
        pt["lmvp"].write(_b(Ls))
        self.vao_ter.render()
        # town geometry
        if R is not None and cam.dist < 0.08:
            bu = dict(qk=qk_l, qd=(float(q["R"]), float(q["amp"]), float(q["t"]), float(q["alpha"])) if q else (1.0, 0.0, 0.0, 0.0),
                      eye=tuple(float(v) for v in eye_l), shadow_on=shadow_on, sbias=0.00008, stex=1.0 / self.SHADOW, smap=6, storm=storm_l,
                      pa=pa_l, pb=pb_l, hw=float(local.get("hw", 0.0)), along=float(local.get("along", 0.0)), clen=float(local.get("clen", 1e-4)),
                      alpha=1.0, **common)
            for prog in (self.p_bld, self.p_tree):
                prog["mvp"].write(_b(Ml))
                prog["lmvp"].write(_b(Ls))
                for k, v in bu.items():
                    _set(prog, k, v)
            if "vao_bld" in R:
                R["vao_bld"].render(vertices=R["nbv"])
            if "vao_br" in R:
                R["vao_br"].render(vertices=R["nbr"])
            for kind in ("round", "cone"):
                if f"tree_{kind}" in R:
                    vao, _, nI = R[f"tree_{kind}"]
                    vao.render(instances=nI)
        self._borders(M0, cam, yscale)
        # bars
        if bars:
            for (x, y, z, w, h, col) in bars:
                self.p_bar["mvp"].write(_b(M0))
                self.p_bar["base"].value = (x, y * yscale, z)
                self.p_bar["size"].value = (w, h)
                self.p_bar["col"].value = tuple(c / 255 for c in col)
                self.vao_bar.render()
        # event effects (local frame when a town is active)
        ctx.enable(mgl.BLEND)
        if local and local.get("funnel"):
            f = local["funnel"]
            c = np.array([f["pos"][0], f["pos"][1] * yscale, f["pos"][2]]) - O
            if local.get("debris", 0) > 0.01:
                ctx.disable(mgl.BLEND)
                pd = self.p_deb
                pd["mvp"].write(_b(Ml))
                for k, v in dict(cen=tuple(c), r0=float(f["r0"]), Hh=float(f["H"]), tm=float(tm), alpha=1.0).items():
                    _set(pd, k, v)
                inst = self.deb_base.copy()
                inst[:, 4] = inst[:, 6] / M_PER_UNIT * (1.0 + 0.0 * inst[:, 4])
                inst[:, 6] = inst[:, 2] * 1.6
                self.deb_inst.write(inst.tobytes())
                self.vao_deb.render(instances=int(self.n_deb * min(1.0, local["debris"])))
                ctx.enable(mgl.BLEND)
            self._funnel(Ml, eye_l, c, f, tm)
            cl = local.get("cloud")
            if cl and cl["alpha"] > 0.01:
                self._cloud(Ml, eye_l, c, cl, tm)
        if local and local.get("dust") and local["dust"]["alpha"] > 0.01:
            d_ = local["dust"]
            c_ = np.array([d_["pos"][0], d_["pos"][1] * yscale, d_["pos"][2]]) - O
            self._dust(Ml, eye_l, c_, d_, tm)
        if local and local.get("cloud") and not local.get("funnel") and local["cloud"]["alpha"] > 0.01:
            cp = local.get("cloud_pos", (O[0], 0.0, O[2]))
            c = np.array([cp[0], cp[1] * yscale, cp[2]]) - O
            self._cloud(Ml, eye_l, c, local["cloud"], tm)
        if local and local.get("rain", 0) > 0.01:
            ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
            ctx.disable(mgl.DEPTH_TEST)
            for k, v in dict(res=(float(self.W), float(self.H)), tm=float(tm), amount=float(local["rain"]), slant=0.28).items():
                _set(self.p_rain, k, v)
            self.vao_rain.render(mgl.TRIANGLE_STRIP)
            ctx.enable(mgl.DEPTH_TEST)
        for (x, y, z, sz, al) in vortices:
            f = dict(H=sz, r0=sz * 0.10, alpha=al, dark=1.0, dusty=0.6, flare=0.0)
            c = np.array([x, y * yscale, z])
            self._funnel(M0, eye, c, f, tm)
        for hu in hurricanes:
            self._hurricane(M0, eye, hu, tm)
        ctx.disable(mgl.BLEND)
        ctx.copy_framebuffer(self.out, self.msaa)
        raw = self.out.read(components=3)
        return Image.frombytes("RGB", (self.W, self.H), raw).transpose(Image.FLIP_TOP_BOTTOM)

    @staticmethod
    def yscale(cam):
        """relief exaggeration fades out near the ground: true scale in town close-ups"""
        return float(np.interp(math.log(max(cam.dist, 1e-6)), [math.log(0.02), math.log(0.35)], [1.0 / core.EXAG, 1.0]))

    def view_cam(self, cam):
        """camera with its target height in the flattened frame (use this one for projecting labels)"""
        from .core import Cam
        c = Cam(cam.tx, cam.ty * self.yscale(cam), cam.tz, cam.dist, cam.pitch, cam.yaw, cam.shake)
        return c

    def _funnel(self, M, eye, c, f, tm):
        ctx, mgl = self.ctx, self.mgl
        ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
        self.msaa.depth_mask = False
        ctx.enable(mgl.CULL_FACE)
        ctx.front_face = "ccw"
        p = self.p_fun
        p["mvp"].write(_b(M))
        R = f["r0"] * 3.0
        for k, v in dict(tm=float(tm), eye=tuple(float(x) for x in eye), noise=3, cen=tuple(float(x) for x in c), R=float(R), Hh=float(f["H"]),
                         r0=float(f["r0"]), alpha=float(f["alpha"]), dark=float(f.get("dark", 0.85)), dusty=float(f.get("dusty", 1.0)),
                         flare=float(f.get("flare", 1.0))).items():
            _set(p, k, v)
        self.vao_fun.render()
        ctx.disable(mgl.CULL_FACE)
        self.msaa.depth_mask = True

    def _hurricane(self, M, eye, hu, tm):
        """hu: dict(pos=(x, y, z) world (y = cloud base), R, H, alpha, eye (fraction of R), seed)"""
        ctx, mgl = self.ctx, self.mgl
        ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
        self.msaa.depth_mask = False
        ctx.enable(mgl.CULL_FACE)
        ctx.cull_face = "front"
        ctx.disable(mgl.DEPTH_TEST)
        p = self.p_hurr
        p["mvp"].write(_b(M))
        for k, v in dict(tm=float(tm), eye=tuple(float(x) for x in eye), noise=3, cen=tuple(float(x) for x in hu["pos"]), R=float(hu["R"]),
                         Hh=float(hu["H"]), alpha=float(hu["alpha"]), spin=float(hu.get("spin", 0.35)), eyer=float(hu.get("eye", 0.07)),
                         seed=float(hu.get("seed", 0.0))).items():
            _set(p, k, v)
        self.vao_hurr.render()
        ctx.cull_face = "back"
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        self.msaa.depth_mask = True

    def _dust(self, M, eye, c, d, tm):
        """rising dust plumes over the damaged blocks of an earthquake"""
        ctx, mgl = self.ctx, self.mgl
        ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
        self.msaa.depth_mask = False
        ctx.enable(mgl.CULL_FACE)
        ctx.cull_face = "front"
        ctx.disable(mgl.DEPTH_TEST)
        p = self.p_dust
        p["mvp"].write(_b(M))
        for k, v in dict(tm=float(tm), eye=tuple(float(x) for x in eye), noise=3, cen=tuple(float(x) for x in c), R=float(d["R"]), Hh=float(d["H"]),
                         alpha=float(d["alpha"]), grow=float(d["grow"])).items():
            _set(p, k, v)
        self.vao_dust.render()
        ctx.cull_face = "back"
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        self.msaa.depth_mask = True

    def _cloud(self, M, eye, c, cl, tm):
        """volumetric rotating mother cloud: slab from base-0.3*thick to base+0.7*thick"""
        ctx, mgl = self.ctx, self.mgl
        thick = cl.get("thick", cl["r"] * 0.45)
        y0 = c[1] + cl["base"] - 0.30 * thick
        ctx.blend_func = mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA
        self.msaa.depth_mask = False
        ctx.enable(mgl.CULL_FACE)
        ctx.cull_face = "front"
        ctx.disable(mgl.DEPTH_TEST)
        p = self.p_cloudv
        p["mvp"].write(_b(M))
        for k, v in dict(tm=float(tm), eye=tuple(float(x) for x in eye), noise=3, cen=(float(c[0]), float(y0), float(c[2])), R=float(cl["r"]),
                         Hh=float(thick), alpha=float(cl["alpha"]), spin=float(cl.get("spin", 0.06))).items():
            _set(p, k, v)
        self.vao_cloudv.render()
        ctx.cull_face = "back"
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        self.msaa.depth_mask = True

    def _shadow_pass(self, cam, O, local, yscale):
        ctx, mgl = self.ctx, self.mgl
        R = self.town
        tgt = np.array([cam.tx, cam.ty * yscale, cam.tz]) - O
        ext = max(1500.0 / M_PER_UNIT, cam.dist * 1.6)
        L = SUN
        eye = tgt + L * ext * 3
        f = tgt - eye
        f /= np.linalg.norm(f)
        s = np.cross(f, [0, 1, 0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        V = np.eye(4)
        V[0, :3], V[1, :3], V[2, :3] = s, u, -f
        V[:3, 3] = -V[:3, :3] @ eye
        n_, f_ = ext * 1.0, ext * 5.0
        P = np.array([[1 / ext, 0, 0, 0], [0, 1 / ext, 0, 0], [0, 0, -2 / (f_ - n_), -(f_ + n_) / (f_ - n_)], [0, 0, 0, 1]])
        Ls = P @ V
        self.sfb.use()
        self.sfb.clear(depth=1.0)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.BLEND)
        qq = local.get("quake")
        bu = dict(qk=((qq["pos"][0] - O[0], qq["pos"][1] - O[2]) if qq else (0.0, 0.0)),
                  qd=((float(qq["R"]), float(qq["amp"]), float(qq["t"]), float(qq["alpha"])) if qq else (1.0, 0.0, 0.0, 0.0)),
                  yscale=yscale, pa=(local["pa"][0] - O[0], local["pa"][1] - O[2]), pb=(local["pb"][0] - O[0], local["pb"][1] - O[2]),
                  hw=float(local.get("hw", 0.0)), along=float(local.get("along", 0.0)), clen=float(local.get("clen", 1e-4)))
        for prog in (self.p_bld_d, self.p_tree_d):
            prog["mvp"].write(_b(Ls))
            for k, v in bu.items():
                _set(prog, k, v)
        if "vao_bld_d" in R:
            R["vao_bld_d"].render(vertices=R["nbv"])
        for kind in ("round", "cone"):
            if f"tree_{kind}" in R:
                _, vao_d, nI = R[f"tree_{kind}"]
                vao_d.render(instances=nI)
        return Ls
