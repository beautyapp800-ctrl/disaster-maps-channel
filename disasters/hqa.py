"""Final QA of the hurricane film: counters recomputed from the raw HURDAT2 file (own parser), HUD layout, pixel checks.

Usage: python -m disasters.hqa <dir with qa_*.jsonl> <report.md>     (exit code 1 on any failure)
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

RAW = "data/raw/hurdat2.txt"


def landfalls():
    import os
    topic = os.environ.get("TOPIC") or "us"
    """(decimal year of first lower-48 hurricane landfall, Saffir-Simpson category of the strongest one) per storm"""
    import datetime as dt
    from .hurricane import category, in_lower48
    out, cur, lf = [], None, []

    def flush():
        if cur is not None and lf:
            pts = lf
            if topic == "florida":
                from .hurricane import landfall_state
                pts = [x for x in lf if landfall_state(x[2], x[3]) == "FL"]
            if not pts:
                return
            cat = max(category(x[1]) for x in pts)
            if topic == "major" and cat < 3:
                return
            out.append((min(x[0] for x in pts), cat))
    for line in open(RAW, encoding="utf-8"):
        p = [x.strip() for x in line.split(",")]
        if len(p[0]) == 8 and p[0].startswith("AL"):
            flush()
            cur, lf = p[0], []
            continue
        if len(p) < 7 or p[2] != "L" or p[3] != "HU":
            continue
        lat = float(p[4][:-1]) * (1 if p[4].endswith("N") else -1)
        lon = float(p[5][:-1]) * (-1 if p[5].endswith("W") else 1)
        if not in_lower48(lon, lat):
            continue
        d = dt.datetime(int(p[0][:4]), int(p[0][4:6]), int(p[0][6:8]), int(p[1][:2]), int(p[1][2:]))
        y0, y1 = dt.datetime(d.year, 1, 1), dt.datetime(d.year + 1, 1, 1)
        lf.append((d.year + (d - y0).total_seconds() / (y1 - y0).total_seconds(), int(p[6]), lon, lat))
    flush()
    a = np.array(sorted(out))
    return a[:, 0], a[:, 1].astype(int)


def intersect(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def main(d, report):
    dec, cat = landfalls()
    total = len(dec)
    frames = []
    for fn in sorted(glob.glob(f"{d}/**/qa_*.jsonl", recursive=True)):
        frames += [json.loads(x) for x in open(fn, encoding="utf-8")]
    frames.sort(key=lambda m: m["n"])
    fails, notes = [], []
    if not frames:
        fails.append("no frames in QA log")
    ns = [m["n"] for m in frames]
    if frames and ns != list(range(ns[0], ns[0] + len(ns))):
        fails.append("frame numbers are not contiguous")
    prev = -1
    sync = pix = 0
    for m in frames:
        c = m.get("count")
        if m["mode"] == "main" and c is not None:
            if c < prev:
                fails.append(f"frame {m['n']}: counter went down {prev} -> {c}")
            prev = c
            dt_ = m["dt"]
            lo = int((dec < dt_ - 0.01).sum())
            hi = int((dec <= dt_ + 0.01).sum())
            slack = 1 if m["kind"] == "event" else 0
            if not (lo - slack <= c <= hi + slack):
                fails.append(f"frame {m['n']}: {c} hurricanes shown, HURDAT2 gives {lo}-{hi} by {dt_:.3f}")
            cats = {int(k): v for k, v in m["cats"].items()}
            if sum(cats.values()) != c or m["major"] != cats[3] + cats[4] + cats[5]:
                fails.append(f"frame {m['n']}: legend {cats} does not add up to {c} / major {m['major']}")
            sync += 1
        if m["mode"] == "final" and c is not None and c != total:
            fails.append(f"frame {m['n']}: final count {c}, HURDAT2 gives {total}")
        ps = [p for p in m.get("panels", []) if p[2] > 0.05]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                if intersect(ps[i][1], ps[j][1]):
                    fails.append(f"frame {m['n']}: panels {ps[i][0]} and {ps[j][0]} overlap")
        if "pixel_problems" in m:
            pix += 1
            if m["pixel_problems"]:
                fails.append(f"frame {m['n']} ({m['t']}s): {', '.join(m['pixel_problems'])}")
    if prev != -1 and prev != total and frames and frames[-1]["mode"] != "main":
        notes.append(f"last main-phase count {prev} (total {total})")
    ok = not fails
    lines = [f"# Hurricane film QA: {'PASS' if ok else 'FAIL'}", "", f"- HURDAT2 lower-48 hurricane landfalls: {total}",
             f"- frames with counter checks: {sync}", f"- pixel checks (1 per second): {pix}", ""]
    lines += [f"- NOTE {x}" for x in notes] + [f"- FAIL {x}" for x in fails[:200]]
    open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
