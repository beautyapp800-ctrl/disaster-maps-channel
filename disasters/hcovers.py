"""Three cover candidates for the hurricane film + automatic choice (same scoring as the tornado covers).

Run: RES=1440 TEXW=4096 python -m disasters.hcovers out_dir
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RES", "1440")
from PIL import Image

from .core import Cam, world3
from .covers import WHITE, YEL, score, shade_left, text_layer
from .hurricane import EV_HIT, EV_RUN, fmt
from .renderer import H


def build(out: Path):
    from .hurricane_render import Renderer
    out.mkdir(parents=True, exist_ok=True)
    r = Renderer()
    D, TL, P, sc = r.D, r.TL, r.painter, r.scene
    k = H / 720.0
    res = {}
    # A: all 289 tracks + three real hurricanes as 3D storms off the coast
    sc.set_overlay(P.all_tracks())
    hu = []
    for key, f in (("katrina2005", 0.55), ("andrew1992", 0.5), ("ian2022", 0.55)):
        e = [x for x in D.events if x["key"] == key][0]
        lon, lat, kt = r.storm_pos(e["idx"], f)
        p = world3(lon, lat)
        hu.append(dict(pos=(float(p[0]), float(p[1]) + 0.002, float(p[2])), R=0.2, H=0.022, alpha=1.0, eye=0.06, seed=float(e["idx"] % 7), spin=0.5))
    img = sc.render(Cam(0.45, 0.0, 0.55, 3.0, 50, -6), tm=2.0, hurricanes=hu)
    img = shade_left(img)
    lay, mask, boxes = text_layer([(fmt(D.n), 250 * k, (40 * k, 330 * k), YEL), ("HURRICANES", 118 * k, (44 * k, 450 * k), WHITE),
                                   ("EVERY ONE THAT HIT AMERICA", 52 * k, (46 * k, 520 * k), WHITE)])
    img.paste(lay, (0, 0), lay)
    res["A_all_hurricanes"] = (img, mask, boxes)
    # B: flooded town close-up (Katrina)
    e = [x for x in D.events if x["key"] == "katrina2005"][0]
    t = e["t0"] + e["HIT"] + e["RUN"] * 0.55
    ph = TL.phase_at(t)
    P.update(t, "main")
    sc.set_overlay(P.texture(t, "main"))
    cam = r.dir.cam(t)
    img = sc.render(cam, tm=t, ovl_alpha=0.2, local=r.local_params(t, ph, cam))
    img = shade_left(img, 0.82, 0.55)
    n5 = int((D.cat == 5).sum())
    lay, mask, boxes = text_layer([(str(n5), 330 * k, (36 * k, 400 * k), YEL), ("CATEGORY 5 LANDFALLS", 86 * k, (44 * k, 500 * k), WHITE),
                                   ("SINCE 1851", 46 * k, (46 * k, 560 * k), WHITE)])
    img.paste(lay, (0, 0), lay)
    res["B_town_flood"] = (img, mask, boxes)
    # C: heat map
    sc.set_overlay(P.heat_texture(175.0))
    img = sc.render(Cam(0.3, 0.0, 0.45, 3.2, 80, 0), tm=0.0)
    img = shade_left(img, 0.7, 0.5)
    lay, mask, boxes = text_layer([("EVERY", 150 * k, (40 * k, 220 * k), WHITE), ("HURRICANE", 160 * k, (40 * k, 372 * k), YEL),
                                   (fmt(D.n), 130 * k, (44 * k, 520 * k), WHITE), ("1851-2025", 64 * k, (46 * k, 600 * k), YEL)])
    img.paste(lay, (0, 0), lay)
    res["C_heat_map"] = (img, mask, boxes)
    scores = {}
    for name, (im, mask, boxes) in res.items():
        im.resize((1280, 720), Image.LANCZOS).convert("RGB").save(out / f"cover_{name}.jpg", quality=92)
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
