"""Render still frames at given video seconds (sorted ascending): python scripts/stills.py out_dir t1 t2 ..."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from disasters.renderer import Renderer, FPS, pixel_qa
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
r = Renderer()
print("total", r.TL.total, "captions", len(r.TL.captions), "phases", len(r.TL.phases), r.scene.renderer)
for ts in sorted(float(x) for x in sys.argv[2:]):
    t0 = time.time()
    img, meta = r.frame(int(ts * FPS))
    pr, oc = pixel_qa(img)
    img.save(out / f"s{ts:07.2f}.jpg", quality=90)
    print(ts, meta["mode"], meta["count"], meta["deaths"], f"{time.time()-t0:.2f}s", pr, oc)
