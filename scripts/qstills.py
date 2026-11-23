"""Earthquake still frames: python scripts/qstills.py out_dir t1 t2 ..."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from disasters.quake_render import Renderer, FPS, pixel_qa
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
r = Renderer()
print("total", round(r.TL.total, 1))
for ts in sorted(float(x) for x in sys.argv[2:]):
    t0 = time.time()
    img, meta = r.frame(int(ts * FPS))
    img.save(out / f"q{ts:07.2f}.jpg", quality=90)
    print(ts, meta["mode"], meta["count"], meta.get("maxmag"), meta.get("states"), f"{time.time()-t0:.2f}s", pixel_qa(img)[0])
