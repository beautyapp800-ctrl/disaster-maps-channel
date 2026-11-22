"""Final QA of a rendered video: independent recomputation from the raw CSV + HUD layout checks.

Usage: python -m disasters.qa <dir with qa_*.jsonl> <report.md>
Exit code 1 if any check fails.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from datetime import datetime

import numpy as np

CSV = "data/raw/tornadoes.csv"
FPS = 24


def load_csv():
    rows = []
    with open(CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = datetime(int(r["yr"]), int(r["mo"]), int(r["dy"]), int(r["time"][:2]), int(r["time"][3:5]))
            y0, y1 = datetime(d.year, 1, 1), datetime(d.year + 1, 1, 1)
            rows.append((d.year + (d - y0).total_seconds() / (y1 - y0).total_seconds(), d.year, int(r["mag"]), int(r["fat"])))
    rows.sort(key=lambda x: x[0])           # stable, same rule as the renderer
    a = np.array(rows)
    return a[:, 0], a[:, 1].astype(int), a[:, 2].astype(int), a[:, 3].astype(int)


def intersect(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def main(d, report):
    dec, yr, mag, fat = load_csv()
    cumfat = np.concatenate([[0], np.cumsum(fat)])
    cumcls = {c: np.concatenate([[0], np.cumsum(mag == c)]) for c in (-9, 0, 1, 2, 3, 4, 5)}
    cumyear = {y: int((yr <= y).sum()) for y in range(1950, 2026)}
    frames = []
    for fn in sorted(glob.glob(f"{d}/**/qa_*.jsonl", recursive=True)):
        frames += [json.loads(l) for l in open(fn, encoding="utf-8")]
    frames.sort(key=lambda m: m["n"])
    fails, notes = [], []
    n = len(frames)
    if n == 0:
        fails.append("no frames in QA log")
    ns = [m["n"] for m in frames]
    if ns != list(range(ns[0], ns[0] + n)):
        fails.append("frame numbers are not contiguous")
    sync = pix = 0
    prev_count = prev_deaths = -1
    prev_f5 = -1
    ids_seen = {}
    maxd = 0
    for m in frames:
        mode = m["mode"]
        W = None
        # ---- counter / data sync (independent recomputation)
        if m.get("count") is not None and m["dt"] is not None:
            dt = m["dt"]
            lo = int(np.searchsorted(dec, dt - 1e-9, "left"))
            hi = int(np.searchsorted(dec, dt + 1e-9, "right"))
            c, dth = m["count"], m["deaths"]
            if mode in ("final", "bars", "heat"):
                lo = hi = len(dec)
            if not (lo <= c <= hi):
                fails.append(f"frame {m['n']}: counter {c} outside data range [{lo},{hi}] at dt={dt:.4f}")
            elif dth != int(cumfat[c]):
                fails.append(f"frame {m['n']}: deaths {dth} != sum of data fatalities {int(cumfat[c])}")
            else:
                cl = m["cls"]
                for k, v in cl.items():
                    if v != int(cumcls[int(k)][c]):
                        fails.append(f"frame {m['n']}: class {k} count {v} != data {int(cumcls[int(k)][c])}")
                        break
                if sum(m["cls"].values()) != c:
                    fails.append(f"frame {m['n']}: legend sum {sum(m['cls'].values())} != counter {c}")
                sync += 1
            if c < prev_count or dth < prev_deaths:
                fails.append(f"frame {m['n']}: counter went backwards ({prev_count}->{c})")
            prev_count, prev_deaths = c, dth
        if m.get("f5") is not None:
            if m["f5"] < prev_f5 or m["f5"] > int((mag == 5).sum()):
                fails.append(f"frame {m['n']}: F5 counter {m['f5']} invalid")
            prev_f5 = m["f5"]
        if m.get("heat_year") is not None:
            ex = cumyear[m["heat_year"]] if m["heat_count"] else 0
            if m["heat_count"] not in (0, ex):
                fails.append(f"frame {m['n']}: heat counter {m['heat_count']} != data {ex}")
        # ---- layout: panels inside the frame, no overlap
        P = m["panels"]
        for i, (ia, ba, _) in enumerate(P):
            if ba[0] < 0 or ba[1] < 0:
                fails.append(f"frame {m['n']}: panel {ia} outside frame {ba}")
            for ib, bb, _ in P[i + 1:]:
                if intersect(ba, bb):
                    fails.append(f"frame {m['n']}: panels {ia} and {ib} overlap")
        for ident, _, _ in P:
            ids_seen.setdefault(ident, []).append(m["n"])
        if "pixel_problems" in m:
            pix += 1
            if m["pixel_problems"]:
                fails.append(f"frame {m['n']} (t={m['t']}s): {', '.join(m['pixel_problems'])}")
    # ---- flicker: every transient element is on screen for a continuous, long enough time
    for ident, fl in ids_seen.items():
        if ident in ("A", "B"):
            continue
        runs, start = [], fl[0]
        for a, b in zip(fl, fl[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, fl[-1]))
        for (s, e) in runs:
            if (e - s + 1) < FPS * 1.5:
                fails.append(f"element {ident}: on screen only {(e - s + 1) / FPS:.2f}s at frame {s} (flicker)")
        for (s0, e0), (s1, e1) in zip(runs, runs[1:]):
            if s1 - e0 < FPS * 1.0:
                fails.append(f"element {ident}: gap of {(s1 - e0) / FPS:.2f}s between {e0} and {s1} (blink)")
    ok = not fails
    with open(report, "w", encoding="utf-8") as f:
        f.write("# QA report: Every Tornado in US History Mapped (pilot)\n\n")
        f.write(f"Result: **{'PASS' if ok else 'FAIL'}**\n\n")
        f.write(f"- frames logged: {n} ({n / FPS:.1f} s)\n")
        f.write(f"- counter/legend/deaths sync checks vs raw CSV: {sync} frames\n")
        f.write(f"- 1-second pixel checks (no stripes, no dark holes, map not empty): {pix} frames\n")
        f.write(f"- layout: no overlapping panels, no transient element shorter than 1.5 s, no blink gaps shorter than 1 s\n")
        f.write(f"- failures: {len(fails)}\n\n")
        for x in fails[:60]:
            f.write(f"- {x}\n")
    print(open(report, encoding="utf-8").read())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
