"""Final QA of the earthquake film: counters recomputed from the raw USGS CSV (own parser), HUD layout, pixel checks, label overlaps.

Usage: python -m disasters.qqa <dir with qa_*.jsonl> <report.md>     (exit code 1 on any failure)
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from datetime import datetime

import numpy as np

from .quake import CSV, in_california

FPS = 24


def load():
    rows = []
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if r["type"] != "earthquake" or not r["mag"]:
            continue
        lat, lon = float(r["latitude"]), float(r["longitude"])
        if not in_california(lon, lat):
            continue
        d = datetime.strptime(r["time"][:19], "%Y-%m-%dT%H:%M:%S")
        y0, y1 = datetime(d.year, 1, 1), datetime(d.year + 1, 1, 1)
        rows.append((d.year + (d - y0).total_seconds() / (y1 - y0).total_seconds(), d.year, float(r["mag"])))
    rows.sort(key=lambda x: x[0])
    a = np.array(rows)
    return a[:, 0], a[:, 1].astype(int), a[:, 2]


def intersect(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def main(d, report):
    dec, yr, mag = load()
    total = len(dec)
    cummax = np.maximum.accumulate(mag)
    cumcls = {c: np.concatenate([[0], np.cumsum((mag >= lo) & (mag < hi))]) for c, lo, hi in ((4, 4, 5), (5, 5, 6), (6, 6, 7), (7, 7, 99))}
    cumyear = {y: int((yr <= y).sum()) for y in range(1900, 2026)}
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
    sync = pix = 0
    prev = -1
    ids_seen = {}
    for m in frames:
        c = m.get("count")
        if m["mode"] in ("main", "final") and c is not None and m.get("dt") is not None or m["mode"] == "final":
            dt_ = m.get("dt")
            if m["mode"] == "final":
                lo = hi = total
            else:
                lo = int(np.searchsorted(dec, dt_ - 1e-9, "left"))
                hi = int(np.searchsorted(dec, dt_ + 1e-9, "right"))
            if not (lo <= c <= hi):
                fails.append(f"frame {m['n']}: {c} earthquakes shown, the USGS data give {lo}-{hi}")
            elif m["mode"] == "main":
                if abs(m["maxmag"] - round(float(cummax[c - 1]), 2)) > 0.011 and c:
                    fails.append(f"frame {m['n']}: largest {m['maxmag']} != data {cummax[c - 1]}")
                for k, v in m["cls"].items():
                    if v != int(cumcls[int(k)][c]):
                        fails.append(f"frame {m['n']}: class {k} count {v} != data {int(cumcls[int(k)][c])}")
                        break
                if sum(m["cls"].values()) != c:
                    fails.append(f"frame {m['n']}: legend sum != counter")
                sync += 1
            if c < prev:
                fails.append(f"frame {m['n']}: counter went down {prev} -> {c}")
            prev = c
        if m.get("heat_year") is not None:
            ex = cumyear[m["heat_year"]] if m["heat_count"] else 0
            if m["heat_count"] not in (0, ex):
                fails.append(f"frame {m['n']}: heat counter {m['heat_count']} != data {ex}")
        P = m.get("panels", [])
        for i, (ia, ba, _) in enumerate(P):
            if ba[0] < 0 or ba[1] < 0:
                fails.append(f"frame {m['n']}: panel {ia} outside frame")
            for ib, bb, _ in P[i + 1:]:
                if intersect(ba, bb):
                    fails.append(f"frame {m['n']}: panels {ia} and {ib} overlap")
        for ident, _, _ in P:
            ids_seen.setdefault(ident, []).append(m["n"])
        if "pixel_problems" in m:
            pix += 1
            if m["pixel_problems"]:
                fails.append(f"frame {m['n']} ({m['t']}s): {', '.join(m['pixel_problems'])}")
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
    lines = [f"# Earthquake film QA: {'PASS' if ok else 'FAIL'}", "", f"- USGS earthquakes in the film (California, M4+): {total}",
             f"- counter / legend / largest-so-far checks vs the raw USGS CSV: {sync} frames", f"- 1-second pixel checks: {pix} frames",
             "- layout: no overlapping panels, no transient element shorter than 1.5 s, no blink gaps shorter than 1 s", ""]
    lines += [f"- NOTE {x}" for x in notes] + [f"- FAIL {x}" for x in fails[:200]]
    open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
