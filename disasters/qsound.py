"""Procedural sound for the earthquake film (noise + sine only: no third-party audio, no licence or Content ID risk).

earth hum  - very low, grows with the number of earthquakes appearing per second
thumps     - a soft sub-bass thump for every M5+ earthquake that appears on the map (size by magnitude)
close-ups  - a rolling rumble that builds for a second, holds while the ground shakes, then dies away; rattling debris on top
Usage: python -m disasters.qsound out.wav
"""
from __future__ import annotations

import sys
import wave

import numpy as np

from .quake import Data, Timeline
from .sound import SR, shaped_noise, smooth_env


def build(path):
    D = Data()
    TL = Timeline(D)
    n = int(TL.total * SR)
    rng = np.random.default_rng(1900)
    t = np.arange(n) / SR
    hits = np.zeros(n, np.float32)
    idx = np.clip((TL.ta * SR).astype(int), 0, n - 1)
    np.add.at(hits, idx[TL.ta > 1e-6], 1.0)
    act = smooth_env(hits, 1.5) * SR
    act = np.clip(act / np.percentile(act[act > 0], 95), 0, 1.2)
    hum_l = shaped_noise(n, rng, lo=25, hi=220, slope=1.0)
    hum_r = shaped_noise(n, rng, lo=25, hi=220, slope=1.0)
    amp = (0.05 + 0.16 * act) * (0.7 + 0.3 * np.sin(2 * np.pi * t / 11.0))
    L, R = hum_l * amp, hum_r * amp
    # thumps for M5+
    thump = np.zeros(n, np.float32)
    for i in np.where(D.mag >= 5.0)[0]:
        a = int(TL.ta[i] * SR)
        dur = int((0.5 + 0.35 * (D.mag[i] - 5.0)) * SR)
        if a + dur >= n:
            continue
        tt = np.arange(dur) / SR
        f = 48.0 - 6.0 * (D.mag[i] - 5.0)
        thump[a:a + dur] += (np.sin(2 * np.pi * f * tt) * np.exp(-tt * (5.0 - 0.8 * (D.mag[i] - 5.0))) * (0.16 + 0.30 * (D.mag[i] - 4.5) / 3.5)).astype(np.float32)
    L += thump
    R += thump * 0.97
    # close-ups
    rumble_env = np.zeros(n, np.float32)
    for e in D.events:
        t0, HIT, RUN = e["t0"], e["HIT"], e["RUN"]
        xs = [t0 + HIT - 0.4, t0 + HIT + 1.0, t0 + HIT + RUN - 1.5, t0 + HIT + RUN + 2.5]
        ys = [0.0, 1.0 if not e["short"] else 0.8, 0.8, 0.0]
        i0, i1 = int(xs[0] * SR), int(xs[-1] * SR)
        rumble_env[i0:i1] = np.maximum(rumble_env[i0:i1], np.interp(t[i0:i1], xs, ys) * (0.55 + 0.45 * min(1.0, (e["mag"] - 5.5) / 2.0)))
    rum = shaped_noise(n, rng, lo=18, hi=130, slope=1.0)
    mid = shaped_noise(n, rng, lo=120, hi=700, slope=0.6)
    rattle_src = shaped_noise(n, rng, lo=900, hi=5500)
    imp = np.zeros(n, np.float32)
    k = rng.poisson(46 * (rumble_env > 0.1).sum() / SR)
    cand = np.where(rumble_env > 0.1)[0]
    if len(cand):
        pos = rng.choice(cand, size=k)
        imp[pos] = rng.uniform(0.2, 1.0, size=k)
    kern = np.exp(-np.arange(int(0.03 * SR)) / (0.007 * SR)).astype(np.float32)
    rattle = np.convolve(imp, kern)[:n] * rattle_src
    e2 = rumble_env * rumble_env
    L += rum * e2 * 1.1 + mid * e2 * 0.35 + rattle * rumble_env * 0.5
    R += rum * e2 * 1.05 + mid * e2 * 0.35 + rattle * rumble_env * 0.5
    fade = np.clip(t / 2.0, 0, 1) * np.clip((TL.total - t) / 4.0, 0, 1)
    st = np.stack([L * fade, R * fade], 1)
    st /= np.abs(st).max() + 1e-9
    st *= 0.89
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())
    print("sfx", path, f"{TL.total:.1f}s")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "sfx.wav")
