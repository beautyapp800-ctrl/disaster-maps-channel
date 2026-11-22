"""Data for the stylized 'miniature world' look (run on a GitHub runner; results are committed).

All sources are public domain or open:
  NLCD 2021 land cover (USGS/MRLC, public domain)            -> class rasters (palette is applied by the renderer)
  USGS 3DEP elevation, 'Hillshade Gray' rendering (public domain)
  OpenStreetMap (ODbL, attribution in every description)     -> roads, water, land use, parks, buildings, place names

Outputs
  data/world/nlcd_8192.png       NLCD class per texel of the national texture (Albers texture frame), 0 = no data
  data/world/hs_8192.jpg         3DEP hillshade in the same frame
  data/region/<key>_<lvl>.png    NLCD classes in a square window around the town, axis-aligned in world x/z
  data/region/<key>_<lvl>_hs.jpg hillshade for that window
  data/region/<key>.json         window centres and sizes
  data/osm/<key>.json.gz         compact OSM features (lon/lat rounded to 1e-6)
Usage: python scripts/prep_mini.py [nation] [region] [osm] [keys...]
"""
from __future__ import annotations

import gzip
import io
import json
import math
import os
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TEXW", "4096")
from disasters import core  # noqa: E402
from prep_world import TOWNS, albers_inv, get  # noqa: E402
from disasters.hurricane import EVENTS as H_EVENTS  # noqa: E402

HTOWNS = [e[0] for e in H_EVENTS]
H_LEVELS = {"mid": (64.0, 4096, 0.0006)}      # hurricane towns: mid level only, coarser source (keeps the PC download small)

Image.MAX_IMAGE_PIXELS = None
NLCD_WMS = "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2021_Land_Cover_L48/wms"
DEP = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
NLCD_RGB = {11: (70, 107, 159), 12: (209, 222, 248), 21: (222, 197, 197), 22: (217, 146, 130), 23: (235, 0, 0), 24: (171, 0, 0),
            31: (179, 172, 159), 41: (104, 171, 95), 42: (28, 95, 44), 43: (181, 197, 143), 52: (204, 184, 121), 71: (223, 223, 194),
            81: (220, 217, 57), 82: (171, 108, 40), 90: (184, 217, 235), 95: (108, 159, 184)}
LEVELS = {"reg": (300.0, 4096, 0.0026), "mid": (64.0, 4096, 0.0003)}     # window size km, texture size, source degrees per pixel


def nlcd_classes(rgb):
    keys = np.array(list(NLCD_RGB), np.uint8)
    pal = np.array(list(NLCD_RGB.values()), np.int32)
    a = rgb.reshape(-1, 3).astype(np.int32)
    out = np.zeros(len(a), np.uint8)
    for s in range(0, len(a), 1 << 21):
        d = ((a[s:s + (1 << 21), None, :] - pal[None]) ** 2).sum(-1)
        k = d.argmin(1)
        ok = d[np.arange(len(k)), k] < 1200
        out[s:s + len(k)] = np.where(ok, keys[k], 0)
    return out.reshape(rgb.shape[:2])


def wms_mosaic(lon0, lat0, lon1, lat1, deg, chunk=2048):
    W, H = int(round((lon1 - lon0) / deg)), int(round((lat1 - lat0) / deg))
    out = np.zeros((H, W), np.uint8)
    for y0 in range(0, H, chunk):
        for x0 in range(0, W, chunk):
            w, h = min(chunk, W - x0), min(chunk, H - y0)
            a0, a1 = lon0 + x0 * deg, lon0 + (x0 + w) * deg
            b1, b0 = lat1 - y0 * deg, lat1 - (y0 + h) * deg
            q = urllib.parse.urlencode(dict(SERVICE="WMS", VERSION="1.1.1", REQUEST="GetMap", LAYERS="NLCD_2021_Land_Cover_L48", STYLES="",
                                            SRS="EPSG:4326", BBOX=f"{a0},{b0},{a1},{b1}", WIDTH=w, HEIGHT=h, FORMAT="image/png"))
            for k in range(5):
                try:
                    im = Image.open(io.BytesIO(get(NLCD_WMS + "?" + q))).convert("RGB")
                    break
                except Exception as e:  # noqa: BLE001  (server sometimes returns an error page)
                    print("  nlcd retry", k + 1, e, flush=True)
                    time.sleep(20 * (k + 1))
            else:
                raise RuntimeError("NLCD chunk failed")
            out[y0:y0 + h, x0:x0 + w] = nlcd_classes(np.asarray(im))
            print(f"  nlcd {x0},{y0} of {W}x{H}", flush=True)
    return out, (lon0, lat1, deg)


def dep_mosaic(lon0, lat0, lon1, lat1, deg, chunk=4000):
    W, H = int(round((lon1 - lon0) / deg)), int(round((lat1 - lat0) / deg))
    out = np.full((H, W), 180, np.uint8)
    rr = urllib.parse.quote(json.dumps({"rasterFunction": "Hillshade Gray"}))
    for y0 in range(0, H, chunk):
        for x0 in range(0, W, chunk):
            w, h = min(chunk, W - x0), min(chunk, H - y0)
            a0, a1 = lon0 + x0 * deg, lon0 + (x0 + w) * deg
            b1, b0 = lat1 - y0 * deg, lat1 - (y0 + h) * deg
            url = f"{DEP}?bbox={a0},{b0},{a1},{b1}&bboxSR=4326&imageSR=4326&size={w},{h}&format=jpg&renderingRule={rr}&f=image"
            try:
                im = Image.open(io.BytesIO(get(url, tries=4))).convert("L").resize((w, h))
                out[y0:y0 + h, x0:x0 + w] = np.asarray(im)
            except Exception as e:  # noqa: BLE001  (hillshade is decoration; flat if missing)
                print("  hillshade chunk failed", e)
            print(f"  3dep {x0},{y0} of {W}x{H}", flush=True)
    return out, (lon0, lat1, deg)


def sample(eq, geo, lon, lat, nearest):
    lon0, lat1, d = geo
    fx = (lon - lon0) / d - 0.5
    fy = (lat1 - lat) / d - 0.5
    H, W = eq.shape[:2]
    if nearest:
        return eq[np.clip(np.round(fy).astype(int), 0, H - 1), np.clip(np.round(fx).astype(int), 0, W - 1)]
    x0 = np.clip(np.floor(fx).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, H - 2)
    ax, ay = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    e = eq.astype(np.float32)
    v = e[y0, x0] * (1 - ax) * (1 - ay) + e[y0, x0 + 1] * ax * (1 - ay) + e[y0 + 1, x0] * (1 - ax) * ay + e[y0 + 1, x0 + 1] * ax * ay
    return np.clip(v, 0, 255).astype(np.uint8)


def world_grid_lonlat(cx, cz, size_w, n):
    """lon/lat of texel centres of an n x n texture covering the world-space square [cx +- size/2] x [cz +- size/2]."""
    fr = core.FR
    g = (np.arange(n) + 0.5) / n - 0.5
    X, Z = np.meshgrid(cx + g * size_w, cz + g * size_w)
    u = (X / core.WORLD_W + 0.5) * core.TW
    v = (Z / core.WORLD_H + 0.5) * core.TH
    ax = (u - core.TW / 2) / fr.s + fr.cx
    ay = fr.cy - (v - core.TH / 2) / fr.s
    return albers_inv(ax, ay)


def nation():
    out = ROOT / "data" / "world"
    box = (-125.5, 23.5, -66.0, 50.0)
    from prep_world import tex_lonlat
    lon, lat = tex_lonlat(8192, 4608)
    cls, geo = wms_mosaic(*box, 0.009)
    Image.fromarray(sample(cls, geo, lon, lat, True)).save(out / "nlcd_8192.png", optimize=True)
    del cls
    hs, geo = dep_mosaic(*box, 0.009)
    Image.fromarray(sample(hs, geo, lon, lat, False)).save(out / "hs_8192.jpg", quality=88)


OUTER = (-0.3, 1.3)    # outer texture covers u, v in this range of the national texture frame


def outer():
    """NASA Blue Marble around the map frame, so the national view has land and ocean up to the horizon"""
    sys.path.insert(0, str(ROOT / "scripts"))
    from prep_world import gibs_mosaic, resample
    out = ROOT / "data" / "world"
    eq, geo = gibs_mosaic("BlueMarble_NextGeneration", -160.0, 0.0, -30.0, 75.0, 0.03)
    n_w, n_h = 4096, 2304
    fr = core.FR
    uu = OUTER[0] + (np.arange(n_w) + 0.5) / n_w * (OUTER[1] - OUTER[0])
    vv = OUTER[0] + (np.arange(n_h) + 0.5) / n_h * (OUTER[1] - OUTER[0])
    U, V = np.meshgrid(uu * core.TW, vv * core.TH)
    lon, lat = albers_inv((U - core.TW / 2) / fr.s + fr.cx, fr.cy - (V - core.TH / 2) / fr.s)
    Image.fromarray(resample(eq, geo, lon, lat)).save(out / "outer_4096.jpg", quality=88)


def dem(keys, km=8.0, n=512):
    """USGS 3DEP elevation (metres, float) on an n x n grid axis-aligned in world x/z around the town (for storm-surge flooding)"""
    out = ROOT / "data" / "region"
    out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        fn = out / f"{key}_dem.npz"
        if fn.exists():
            continue
        clat, clon = json.loads((ROOT / "data" / "local" / f"{key}.json").read_text())["center"]
        cx, cz = (float(v) for v in core.FR.world(clon, clat))
        lon, lat = world_grid_lonlat(cx, cz, km / core.KM_PER_UNIT, n)
        b = (lon.min() - 0.01, lat.min() - 0.01, lon.max() + 0.01, lat.max() + 0.01)
        W_ = H_ = 768
        url = (f"{DEP}?bbox={b[0]},{b[1]},{b[2]},{b[3]}&bboxSR=4326&imageSR=4326&size={W_},{H_}&format=tiff&pixelType=F32"
               f"&noDataInterpretation=esriNoDataMatchAny&interpolation=RSP_BilinearInterpolation&f=image")
        a = np.asarray(Image.open(io.BytesIO(get(url))), np.float32)
        a = np.nan_to_num(a, nan=-2.0)
        a = np.where((a < -3) | (a > 80), -2.0, a)       # NoData and its interpolated edges = open water (coastal towns only)
        from scipy.ndimage import median_filter
        a = median_filter(a, 5)
        geo = (b[0], b[3], (b[2] - b[0]) / W_)
        fx = (lon - b[0]) / ((b[2] - b[0]) / W_) - 0.5
        fy = (b[3] - lat) / ((b[3] - b[1]) / H_) - 0.5
        z = a[np.clip(np.round(fy).astype(int), 0, H_ - 1), np.clip(np.round(fx).astype(int), 0, W_ - 1)]
        np.savez_compressed(fn, z=z.astype(np.float16), km=km)
        print(key, "dem", float(z.min()), float(z.max()), flush=True)


def region(keys):
    out = ROOT / "data" / "region"
    out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        w = json.loads((ROOT / "data" / "local" / f"{key}.json").read_text())
        clat, clon = w["center"]
        cx, cz = (float(v) for v in core.FR.world(clon, clat))
        if (out / f"{key}.json").exists():
            continue
        meta = dict(key=key, center=[clat, clon], cx=cx, cz=cz, levels={})
        for lvl, (km, n, deg) in (H_LEVELS if key in HTOWNS else LEVELS).items():
            size_w = km / core.KM_PER_UNIT
            lon, lat = world_grid_lonlat(cx, cz, size_w, n)
            pad = 0.02
            b = (lon.min() - pad, lat.min() - pad, lon.max() + pad, lat.max() + pad)
            cls, geo = wms_mosaic(*b, deg)
            Image.fromarray(sample(cls, geo, lon, lat, True)).save(out / f"{key}_{lvl}.png", optimize=True)
            hs, geo = dep_mosaic(*b, deg)
            Image.fromarray(sample(hs, geo, lon, lat, False)).save(out / f"{key}_{lvl}_hs.jpg", quality=88)
            meta["levels"][lvl] = dict(km=km, size=size_w, n=n)
            print(key, lvl, "ok", flush=True)
        (out / f"{key}.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- OpenStreetMap
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
GROUND_KM, BLD_KM = 16.0, 10.0
HW = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|pedestrian"


def overpass(q):
    for ep in OVERPASS * 2:
        try:
            return json.loads(get(ep, urllib.parse.urlencode({"data": q}).encode(), tries=2, timeout=300))
        except Exception as e:  # noqa: BLE001
            print("  overpass", ep, e, flush=True)
            time.sleep(30)
    raise RuntimeError("overpass failed")


def bbox_km(clat, clon, km):
    dlat = km / 2 / 110.57
    dlon = km / 2 / (111.32 * math.cos(math.radians(clat)))
    return f"{clat - dlat:.5f},{clon - dlon:.5f},{clat + dlat:.5f},{clon + dlon:.5f}"


def rnd(g):
    return [round(v, 6) for p in g for v in (p["lon"], p["lat"])]


def stitch(ways):
    """join open member ways into closed rings"""
    rings, open_ = [], [list(w) for w in ways if len(w) >= 2]
    while open_:
        cur = open_.pop()
        changed = True
        while changed and cur[0] != cur[-1]:
            changed = False
            for k, w in enumerate(open_):
                if w[0] == cur[-1]:
                    cur += w[1:]
                elif w[-1] == cur[-1]:
                    cur += w[::-1][1:]
                elif w[-1] == cur[0]:
                    cur = w[:-1] + cur
                elif w[0] == cur[0]:
                    cur = w[::-1][:-1] + cur
                else:
                    continue
                open_.pop(k)
                changed = True
                break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append(cur)
    return rings


KEEP = ("highway", "lanes", "bridge", "layer", "oneway", "railway", "natural", "water", "waterway", "landuse", "leisure", "amenity",
        "aeroway", "building", "height", "building:levels", "roof:shape", "roof:colour", "building:colour", "place", "name", "tourism", "surface")


def osm(keys):
    out = ROOT / "data" / "osm"
    out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        fn = out / f"{key}.json.gz"
        if fn.exists():
            continue
        w = json.loads((ROOT / "data" / "local" / f"{key}.json").read_text())
        clat, clon = w["center"]
        g, b = bbox_km(clat, clon, GROUND_KM), bbox_km(clat, clon, BLD_KM)
        q1 = (f'[out:json][timeout:240];(way["highway"~"^({HW})$"]({g});way["railway"="rail"]({g});'
              f'way["natural"~"^(water|wood|scrub|grassland|beach|sand|wetland)$"]({g});relation["natural"="water"]({g});'
              f'way["waterway"~"^(river|stream|canal|riverbank)$"]({g});relation["waterway"="riverbank"]({g});'
              f'way["landuse"]({g});way["leisure"~"^(park|pitch|golf_course|stadium|playground|garden|track|sports_centre|nature_reserve)$"]({g});'
              f'way["amenity"~"^(parking|school|hospital|university|college)$"]({g});way["aeroway"~"^(runway|taxiway|apron)$"]({g}););out geom;')
        q2 = f'[out:json][timeout:240];(way["building"]({b}););out geom;'
        q3 = (f'[out:json][timeout:120];(node["place"~"^(city|town|village|suburb|neighbourhood|quarter|hamlet)$"]({g});'
              f'nwr["name"]["leisure"~"^(stadium|park|golf_course|sports_centre)$"]({b});nwr["name"]["amenity"~"^(hospital|university|college|school|townhall)$"]({b});'
              f'nwr["name"]["natural"="water"]({g});way["name"]["waterway"="river"]({g}););out center tags;')
        feats, blds, labels = [], [], []
        for el in overpass(q1).get("elements", []):
            t = {k: v for k, v in el.get("tags", {}).items() if k in KEEP}
            if el["type"] == "way" and el.get("geometry"):
                feats.append(dict(t=t, g=rnd(el["geometry"])))
            elif el["type"] == "relation":
                outer = [[(p["lon"], p["lat"]) for p in m["geometry"]] for m in el.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
                for r in stitch(outer):
                    feats.append(dict(t=t, g=[round(v, 6) for p in r for v in p]))
        time.sleep(10)
        for el in overpass(q2).get("elements", []):
            if el.get("geometry") and len(el["geometry"]) >= 4:
                t = {k: v for k, v in el.get("tags", {}).items() if k in KEEP}
                blds.append(dict(t=t, g=rnd(el["geometry"])))
        time.sleep(10)
        for el in overpass(q3).get("elements", []):
            c = el.get("center") or ({"lat": el["lat"], "lon": el["lon"]} if "lat" in el else None)
            tg = el.get("tags", {})
            if c and tg.get("name"):
                kind = tg.get("place") or tg.get("leisure") or tg.get("amenity") or tg.get("natural") or tg.get("waterway")
                labels.append(dict(name=tg["name"], kind=kind, lon=round(c["lon"], 6), lat=round(c["lat"], 6)))
        data = dict(key=key, center=[clat, clon], ground_km=GROUND_KM, bld_km=BLD_KM, feats=feats, buildings=blds, labels=labels,
                    source="(c) OpenStreetMap contributors, ODbL 1.0", fetched=time.strftime("%Y-%m-%d"))
        with gzip.open(fn, "wt", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        print(key, "features", len(feats), "buildings", len(blds), "labels", len(labels), flush=True)
        time.sleep(20)


if __name__ == "__main__":
    args = sys.argv[1:]
    what = {a for a in args if a in ("nation", "region", "osm", "outer", "dem")} or {"nation", "region", "osm"}
    keys = [a for a in args if a in TOWNS or a in HTOWNS] or list(TOWNS)
    if "hurricanes" in args:
        keys = HTOWNS
    if "osm" in what:
        try:
            osm(keys)
        except Exception as e:  # noqa: BLE001
            print("osm step failed:", e)
    if "region" in what:
        region(keys)
    if "nation" in what:
        nation()
    if "outer" in what:
        outer()
    if "dem" in what:
        dem(keys)
