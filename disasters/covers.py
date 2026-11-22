"""Three cover candidates (1280x720) for the tornado pilot + automatic choice.

Run: RES=1440 TEXW=4096 python -m disasters.covers out_dir
Score = text contrast at 320x180 (WCAG ratio vs the ring around the glyphs) + colourfulness + size of the big number + edge detail.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RES", "1440")
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import core
from .core import Cam, font, world3
from .renderer import H, W
from .tornado import fmt

YEL = (255, 214, 0)
WHITE = (255, 255, 255)
INK = (14, 10, 30)


def text_layer(items):
    """items: [(text, size, (x, y), fill)] drawn with a thick dark outline. Returns (rgba layer, fill-only mask)."""
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mask = Image.new("L", (W, H), 0)
    d, dm = ImageDraw.Draw(lay), ImageDraw.Draw(mask)
    boxes = []
    for text, size, xy, fill in items:
        f = font(size)
        d.text(xy, text, font=f, fill=fill, stroke_width=max(4, int(size * 0.085)), stroke_fill=INK, anchor="ls")
        dm.text(xy, text, font=f, fill=255, anchor="ls")
        boxes.append(d.textbbox(xy, text, font=f, anchor="ls"))
    return lay, mask, boxes


def shade_left(img, strength=0.78, width=0.62):
    g = np.linspace(1, 0, int(W * width)) ** 1.6 * strength
    a = np.zeros((H, W), "f4")
    a[:, : len(g)] = g[None, :]
    ov = Image.new("RGB", (W, H), (8, 6, 28))
    return Image.composite(ov, img, Image.fromarray((a * 255).astype("u1"), "L"))


def contrast_score(img, mask):
    """WCAG-style contrast between the glyph fill and the ring around the glyphs, measured at 320x180."""
    s = img.resize((320, 180), Image.LANCZOS)
    m = mask.resize((320, 180), Image.LANCZOS).point(lambda p: 255 if p > 128 else 0)
    ring = ImageChops_sub(m.filter(ImageFilter.MaxFilter(9)), m)
    lum = np.asarray(s.convert("L"), "f4") / 255
    ms, rs = np.asarray(m) > 0, np.asarray(ring) > 0
    if ms.sum() < 20 or rs.sum() < 20:
        return 0.0
    a, b = lum[ms].mean(), lum[rs].mean()
    hi, lo = max(a, b) + 0.05, min(a, b) + 0.05
    return hi / lo


def ImageChops_sub(a, b):
    from PIL import ImageChops
    return ImageChops.subtract(a, b)


def score(img, mask, boxes):
    s = img.resize((320, 180), Image.LANCZOS)
    hsv = np.asarray(s.convert("HSV"), "f4") / 255
    sat = float(hsv[..., 1].mean())
    g = np.asarray(s.convert("L"), "f4")
    edge = float((np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2)
    big = max((b[3] - b[1]) for b in boxes) / H
    con = contrast_score(img, mask)
    parts = dict(contrast=con, saturation=sat, big_height=big, edge=edge)
    sc = 0.45 * min(1.0, con / 7.0) + 0.25 * min(1.0, sat * 2.0) + 0.20 * min(1.0, big / 0.33) + 0.10 * min(1.0, edge / 9.0)
    return sc, parts


def build(out: Path):
    from .renderer import Renderer
    from .tornado import EV_HIT
    out.mkdir(parents=True, exist_ok=True)
    r = Renderer()
    D, TL, P, sc = r.D, r.TL, r.painter, r.scene
    ev = {e["key"]: e for e in D.events}
    k = H / 720.0
    total = D.n
    res = {}

    # A: oblique 3D satellite map with all 73,458 tracks + three real strong tornadoes as 3D vortices
    sc.set_overlay(P.final_comp())
    vs = []
    for key in ("joplin", "moore13", "hackleburg"):
        i = ev[key]["idx"]
        p = world3(D.slon[i], D.slat[i])
        vs.append((float(p[0]), float(p[1]), float(p[2]), 0.42, 1.0))
    img = sc.render(Cam(-0.55, 0.0, 0.15, 3.9, 50, -6), tm=3.1, vortices=vs)
    img = shade_left(img)
    lay, mask, boxes = text_layer([(fmt(total), 250 * k, (40 * k, 330 * k), YEL), ("TORNADOES", 118 * k, (44 * k, 450 * k), WHITE),
                                   ("EVERY ONE MAPPED, 1950-2025", 52 * k, (46 * k, 520 * k), WHITE)])
    img.paste(lay, (0, 0), lay)
    res["A_all_tornadoes"] = (img, mask, boxes)

    # B: the town close-up: Moore 2013 EF5 crossing real USGS imagery with 3D OSM buildings, debris and damage path
    e = ev["moore13"]
    t = e["t0"] + EV_HIT + 3.6
    ph = TL.phase_at(t)
    P.update(t, "main")
    sc.set_overlay(P.texture(t, "main"))
    cam = r.dir.cam(t)
    img = sc.render(cam, tm=t, ovl_alpha=0.2, vortices=r._vortices(t, ph, cam), local=r.local_params(t, ph, cam))
    img = shade_left(img, 0.82, 0.55)
    lay, mask, boxes = text_layer([(str(len(TL.f5)), 330 * k, (36 * k, 400 * k), YEL), ("F5 TORNADOES", 100 * k, (44 * k, 500 * k), WHITE),
                                   ("THE STRONGEST SINCE 1950", 46 * k, (46 * k, 560 * k), WHITE)])
    img.paste(lay, (0, 0), lay)
    res["B_f5_closeup"] = (img, mask, boxes)

    # C: heat map of all tornado paths over the satellite map
    sc.set_overlay(P.heat_texture(76.0))
    img = sc.render(Cam(0.0, 0.0, 0.05, 3.9, 80, 0), tm=0.0, clouds=0.4)
    img = shade_left(img, 0.7, 0.5)
    lay, mask, boxes = text_layer([("EVERY", 150 * k, (40 * k, 220 * k), WHITE), ("TORNADO", 170 * k, (40 * k, 372 * k), YEL),
                                   (fmt(total), 130 * k, (44 * k, 520 * k), WHITE), ("1950-2025", 64 * k, (46 * k, 600 * k), YEL)])
    img.paste(lay, (0, 0), lay)
    res["C_heat_map"] = (img, mask, boxes)

    scores = {}
    for name, (im, mask, boxes) in res.items():
        im1280 = im.resize((1280, 720), Image.LANCZOS)
        im1280.convert("RGB").save(out / f"cover_{name}.jpg", quality=92)
        s, parts = score(im, mask, boxes)
        scores[name] = dict(score=round(float(s), 4), **{a: round(float(b), 4) for a, b in parts.items()})
        im.resize((320, 180), Image.LANCZOS).save(out / f"cover_{name}_320x180.png")
    best = max(scores, key=lambda n: scores[n]["score"])
    scores["chosen"] = best
    (out / "covers.json").write_text(json.dumps(scores, indent=1))
    Image.open(out / f"cover_{best}.jpg").save(out / "cover_CHOSEN.jpg", quality=92)
    print(json.dumps(scores, indent=1))


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "out/covers"))
