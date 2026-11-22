"""Style test frames for the miniature-world look: python scripts/minitest.py out_dir W H [key] [shots...]
shots: city, nation, region, tornado, after"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from disasters import mini  # noqa: E402
from disasters.core import FR, Cam, height_at  # noqa: E402
from disasters.tornado import Data  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
W, H = int(sys.argv[2]), int(sys.argv[3])
key = sys.argv[4] if len(sys.argv) > 4 else "moore13"
shots = sys.argv[5:] or ["city", "nation", "tornado"]
D = Data()
e = [e for e in D.events if e["key"] == key][0]
i = e["idx"]
S = mini.MiniScene(W, H)
print("GL", S.renderer)
U = mini.M_PER_UNIT
pa = np.array(FR.world(D.slon[i], D.slat[i]), float)
pb = np.array(FR.world(D.elon[i], D.elat[i]), float)
T = mini.Town(key)
O = T.O
dirv = (pb - pa) / np.linalg.norm(pb - pa)
s_c = float(np.dot(np.array([O[0], O[2]]) - pa, dirv))
hw = max(60.0, D.wd[i] * 0.9144 / 2) / U


def pt(s):
    p = pa + dirv * s
    lon, lat = mini._inv_world(p[0], p[1])
    return np.array([p[0], float(height_at(lon, lat)), p[1]])


def local(along, funnel=True, cloud=1.0):
    fp = pt(along)
    d = dict(key=key, pa=tuple(pa), pb=tuple(pb), hw=hw, along=along, clen=hw * 0.9,
             storm=(fp[0], fp[2], 2600 / U, 0.55 * cloud))
    if funnel:
        d["funnel"] = dict(pos=tuple(fp), H=680 / U, r0=min(250, max(80, hw * U * 0.3)) / U, alpha=1.0, dark=0.85, dusty=1.0)
        d["debris"] = 1.0
        d["cloud"] = dict(r=2600 / U, base=660 / U, thick=600 / U, alpha=cloud)
    return d


yaw = math.degrees(math.atan2(-dirv[0], -dirv[1])) + 70
for sh in shots:
    t0 = time.time()
    if sh == "city":
        c = pt(s_c + 900 / U)
        cam = Cam(c[0], c[1], c[2], float(__import__("os").environ.get("CD", "800")) / U, float(__import__("os").environ.get("CP", "30")), yaw - 40)
        img = S.render(cam, tm=3.0, local=dict(key=key, pa=tuple(pa), pb=tuple(pb), hw=0.0, along=0.0, clen=1e-4))
    elif sh == "nation":
        img = S.render(Cam(0.0, 0.0, 0.15, 3.6, 54.0, 0.0), tm=0.0)
    elif sh == "region":
        c = pt(s_c)
        img = S.render(Cam(c[0], c[1], c[2], 0.25, 45.0, yaw), tm=0.0, local=dict(key=key, pa=tuple(pa), pb=tuple(pb), hw=0.0, along=0.0, clen=1e-4))
    elif sh == "tornado":
        al = s_c + 300 / U
        c = pt(al - 200 / U)
        cam = Cam(c[0], c[1] + 240 / U * 14, c[2], 2100 / U, 3.0, yaw)
        img = S.render(cam, tm=5.0, local=local(al))
    elif sh == "after":
        al = s_c + 2500 / U
        c = pt(s_c)
        cam = Cam(c[0], c[1], c[2], 2600 / U, 38.0, yaw + 20)
        img = S.render(cam, tm=9.0, local=local(al, funnel=False, cloud=0.0))
    img.save(out / f"{key}_{sh}.jpg", quality=92)
    print(sh, f"{time.time() - t0:.1f}s", flush=True)
