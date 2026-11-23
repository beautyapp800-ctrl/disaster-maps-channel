"""Build the 'living map' textures and local detail patches (run on a GitHub runner, results are committed).

Outputs (all in the renderer's Albers texture space unless noted):
  data/world/sat_8192.jpg      NASA Blue Marble Next Generation (GIBS layer BlueMarble_NextGeneration, ~500 m), colour
  data/world/lights_4096.png   NASA Black Marble 2016 night lights (GIBS layer VIIRS_Black_Marble), grey
  data/world/water_4096.png    water mask (NOAA ETOPO1 below sea level + Natural Earth lakes)
  data/local/<key>.jpg         USGS NAIP orthoimagery (USGSImageryOnly, public domain) 32 x 18 km around the town, 4096x2304
  data/local/<key>.json        window bounds + town check
  data/local/<key>_bld.npz     OpenStreetMap building footprints (ODbL) as oriented boxes
Usage: python scripts/prep_world.py [world] [local] [buildings]
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TEXW", "4096")
from mapviz.core import albers  # noqa: E402
import mapviz.core as mc  # noqa: E402
from disasters import core  # noqa: E402
from disasters.tornado import Data  # noqa: E402

UA = "disaster-maps-channel/1.0 (+https://github.com/beautyapp800-ctrl/disaster-maps-channel)"
Image.MAX_IMAGE_PIXELS = None


def get(url, data=None, tries=6, timeout=180):
    delay = 10
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print("  retry", k + 1, url[:90], e, flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("download failed: " + url[:120])


def albers_inv(x, y):
    N, C, R0, L0 = mc._N, mc._C, mc._R0, mc._L0
    rho = np.sqrt(x * x + (R0 - y) ** 2)
    th = np.arctan2(x, R0 - y)
    lon = np.degrees(L0 + th / N)
    s = np.clip((C - (rho * N) ** 2) / (2 * N), -1, 1)
    return lon, np.degrees(np.arcsin(s))


def tex_lonlat(tw, th):
    """lon/lat of every texel centre of a tw x th texture in the renderer's Albers frame."""
    fr = core.FR
    s = fr.s * tw / core.TW
    u = (np.arange(tw) + 0.5)
    v = (np.arange(th) + 0.5)
    U, V = np.meshgrid(u, v)
    x = (U - tw / 2) / s + fr.cx
    y = fr.cy - (V - th / 2) / s
    return albers_inv(x, y)


def gibs_mosaic(layer, lon0, lat0, lon1, lat1, deg_per_px, chunk=2048):
    W = int(round((lon1 - lon0) / deg_per_px))
    H = int(round((lat1 - lat0) / deg_per_px))
    out = np.zeros((H, W, 3), np.uint8)
    for y0 in range(0, H, chunk):
        for x0 in range(0, W, chunk):
            w, h = min(chunk, W - x0), min(chunk, H - y0)
            a0, a1 = lon0 + x0 * deg_per_px, lon0 + (x0 + w) * deg_per_px
            b1, b0 = lat1 - y0 * deg_per_px, lat1 - (y0 + h) * deg_per_px
            q = urllib.parse.urlencode(dict(SERVICE="WMS", VERSION="1.3.0", REQUEST="GetMap", LAYERS=layer, CRS="EPSG:4326",
                                            BBOX=f"{b0},{a0},{b1},{a1}", WIDTH=w, HEIGHT=h, FORMAT="image/jpeg"))
            im = Image.open(io.BytesIO(get("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?" + q))).convert("RGB")
            out[y0:y0 + h, x0:x0 + w] = np.asarray(im)
            print(f"  {layer} chunk {x0},{y0}", flush=True)
    return out, (lon0, lat1, deg_per_px)


def resample(eq, geo, lon, lat):
    lon0, lat1, d = geo
    fx = (lon - lon0) / d - 0.5
    fy = (lat1 - lat) / d - 0.5
    H, W = eq.shape[:2]
    x0 = np.clip(np.floor(fx).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, H - 2)
    ax = np.clip(fx - x0, 0, 1)[..., None]
    ay = np.clip(fy - y0, 0, 1)[..., None]
    e = eq.astype(np.float32)
    v = (e[y0, x0] * (1 - ax) * (1 - ay) + e[y0, x0 + 1] * ax * (1 - ay) + e[y0 + 1, x0] * (1 - ax) * ay + e[y0 + 1, x0 + 1] * ax * ay)
    return np.clip(v, 0, 255).astype(np.uint8)


def world():
    out = ROOT / "data" / "world"
    out.mkdir(parents=True, exist_ok=True)
    box = (-134.0, 17.0, -58.0, 57.0)
    print("Blue Marble ...", flush=True)
    eq, geo = gibs_mosaic("BlueMarble_NextGeneration", *box, 0.0045)
    lon, lat = tex_lonlat(8192, 4608)
    Image.fromarray(resample(eq, geo, lon, lat)).save(out / "sat_8192.jpg", quality=90)
    del eq
    print("Black Marble ...", flush=True)
    eq, geo = gibs_mosaic("VIIRS_Black_Marble", *box, 0.0045)
    lon, lat = tex_lonlat(4096, 2304)
    Image.fromarray(resample(eq, geo, lon, lat)).convert("L").save(out / "lights_4096.png", optimize=True)
    del eq
    print("water mask ...", flush=True)
    la, lo, z = core.relief()
    fi = np.clip((lat - la[0]) / (la[1] - la[0]), 0, len(la) - 1.001)
    fj = np.clip((lon - lo[0]) / (lo[1] - lo[0]), 0, len(lo) - 1.001)
    zz = z[fi.astype(int), fj.astype(int)]
    water = Image.fromarray(((zz <= 0) * 255).astype(np.uint8))
    d = ImageDraw.Draw(water)
    import shapefile
    s = 4096 / core.TW
    lk = shapefile.Reader(str(ROOT / "data" / "ne" / "ne_50m_lakes"))
    for sr in lk.iterShapeRecords():
        pts = np.array(sr.shape.points)
        if len(pts) == 0 or pts[:, 0].max() < -134 or pts[:, 0].min() > -58 or pts[:, 1].min() > 57 or pts[:, 1].max() < 17:
            continue
        parts = list(sr.shape.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            x, y = core.FR.px(pts[a:b, 0], pts[a:b, 1])
            d.polygon(list(zip(x * s, y * s)), fill=255)
    water.save(out / "water_4096.png", optimize=True)


# ---------------------------------------------------------------- local patches
TOWNS = {  # town centre (lat, lon) used to frame the close-up; names verified on NWS pages (sources.json)
    "waco": (31.549, -97.147), "flint": (43.090, -83.700), "ruskin": (38.940, -94.530), "jackson": (32.250, -90.210),
    "xenia": (39.685, -83.930), "moore99": (35.339, -97.487), "greensburg": (37.603, -99.293), "hackleburg": (34.277, -87.828),
    "joplin": (37.084, -94.513), "moore13": (35.339, -97.487), "mayfield": (36.742, -88.637),
    # short zooms (owner 23.11), town centres; names checked on NWS pages (sources.json event_names)
    "dunlap1965": (41.637, -85.922), "lubbock1970": (33.584, -101.845), "wichitafalls1979": (33.896, -98.513),
    "plainfield1990": (41.627, -88.204), "jarrell1997": (30.824, -97.604), "jacksontn2003": (35.6145, -88.8139),
    "vilonia2014": (35.084, -92.208), "rollingfork2023": (32.9068, -90.8782),
}
WIN_KM = (32.0, 18.0)


def km_xy(lat0, lon0, lat, lon):
    return (lon - lon0) * 111.32 * math.cos(math.radians(lat0)), (lat - lat0) * 110.57


def window(key, D):
    e = [e for e in D.events if e["key"] == key][0]
    i = e["idx"]
    lat0, lon0 = TOWNS[key]
    ax, ay = km_xy(lat0, lon0, D.slat[i], D.slon[i])
    bx, by = km_xy(lat0, lon0, D.elat[i], D.elon[i])
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    f = 0.0 if L2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L2))
    px, py = ax + dx * f, ay + dy * f
    off = math.hypot(px, py)
    clat = lat0 + py / 110.57
    clon = lon0 + px / (111.32 * math.cos(math.radians(lat0)))
    hw, hh = WIN_KM[0] / 2, WIN_KM[1] / 2
    dlon = hw / (111.32 * math.cos(math.radians(clat)))
    dlat = hh / 110.57
    return dict(key=key, town=[lat0, lon0], path_offset_km=round(off, 2), center=[clat, clon], frac_center=f,
                bbox=[clon - dlon, clat - dlat, clon + dlon, clat + dlat])


def local(D):
    out = ROOT / "data" / "local"
    out.mkdir(parents=True, exist_ok=True)
    for key in TOWNS:
        w = window(key, D)
        print(key, "path passes", w["path_offset_km"], "km from town", flush=True)
        if w["path_offset_km"] > 8:
            print("  WARNING: town far from SPC path line")
        b = w["bbox"]
        q = urllib.parse.urlencode(dict(bbox=",".join(f"{v:.5f}" for v in b), bboxSR=4326, imageSR=4326, size="4096,2304", format="jpg", f="image"))
        img = Image.open(io.BytesIO(get("https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/export?" + q))).convert("RGB")
        img.save(out / f"{key}.jpg", quality=86)
        (out / f"{key}.json").write_text(json.dumps(w, indent=1))


OVERPASS = ["https://overpass.kumi.systems/api/interpreter", "https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]


def buildings(D):
    out = ROOT / "data" / "local"
    for key in TOWNS:
        fn = out / f"{key}_bld.npz"
        if fn.exists():
            continue
        w = json.loads((out / f"{key}.json").read_text())
        c = w["center"]
        dlat = 4.5 / 110.57
        dlon = 8.0 / (111.32 * math.cos(math.radians(c[0])))
        b = [c[1] - dlon, c[0] - dlat, c[1] + dlon, c[0] + dlat]   # buildings only in the central 16 x 9 km
        q = f'[out:json][timeout:120];(way["building"]({b[1]:.5f},{b[0]:.5f},{b[3]:.5f},{b[2]:.5f}););out geom;'
        js = None
        for ep in OVERPASS * 2:
            try:
                js = json.loads(get(ep, urllib.parse.urlencode({"data": q}).encode(), tries=2, timeout=200))
                break
            except Exception as e:  # noqa: BLE001
                print("  overpass", ep, e, flush=True)
                time.sleep(30)
        if js is None:
            print("  no buildings for", key)
            continue
        rows = []
        for el in js.get("elements", []):
            g = el.get("geometry")
            if not g or len(g) < 4:
                continue
            la = np.array([p["lat"] for p in g])
            lo = np.array([p["lon"] for p in g])
            c0, c1 = la.mean(), lo.mean()
            x = (lo - c1) * 111320 * math.cos(math.radians(c0))
            y = (la - c0) * 110570
            # oriented box from the longest edge
            dx, dy = np.diff(x), np.diff(y)
            k = int(np.argmax(dx * dx + dy * dy))
            ang = math.atan2(dy[k], dx[k])
            ca, sa = math.cos(-ang), math.sin(-ang)
            rx, ry = x * ca - y * sa, x * sa + y * ca
            wdt, dep = float(rx.max() - rx.min()), float(ry.max() - ry.min())
            t = el.get("tags", {})
            hgt = None
            try:
                hgt = float(str(t.get("height", "")).split()[0].replace("m", ""))
            except Exception:  # noqa: BLE001
                hgt = None
            if hgt is None:
                try:
                    hgt = float(t.get("building:levels")) * 3.2
                except (TypeError, ValueError):
                    hgt = 7.0 if t.get("building") in ("commercial", "retail", "industrial", "school", "warehouse") else 5.0
            rows.append((c0, c1, max(wdt, 3), max(dep, 3), ang, min(hgt, 120)))
        a = np.array(rows, "f8") if rows else np.zeros((0, 6))
        np.savez_compressed(fn, lat=a[:, 0], lon=a[:, 1], w=a[:, 2].astype("f4"), d=a[:, 3].astype("f4"), ang=a[:, 4].astype("f4"), h=a[:, 5].astype("f4"))
        print(key, "buildings", len(a), flush=True)
        time.sleep(20)


if __name__ == "__main__":
    what = set(sys.argv[1:]) or {"world", "local", "buildings"}
    D = Data()
    if "world" in what:
        world()
    if "local" in what:
        local(D)
    if "buildings" in what:
        try:
            buildings(D)
        except Exception as e:  # noqa: BLE001  (buildings are optional; the rest must still be committed)
            print("buildings step failed:", e)
