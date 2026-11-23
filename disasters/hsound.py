"""Procedural sound for the hurricane film (noise + sine only: no third-party audio, no Content ID risk).

wind bed   - gusty filtered noise, grows with the number of hurricanes on screen
storm      - in the town close-ups: howling wind, heavy rain hiss and a low surge rumble that build up and fade
Usage: python -m disasters.hsound out.wav
"""
from __future__ import annotations

import sys
import wave

import numpy as np

from .hurricane import EV_HIT, EV_RUN, Data, Timeline
from .sound import SR, shaped_noise, smooth_env


def build(path):
    D = Data()
    TL = Timeline(D)
    n = int(TL.total * SR)
    rng = np.random.default_rng(1851)
    t = np.arange(n) / SR
    act = np.zeros(n, np.float32)
    for a, d in zip(TL.ta, TL.anim):
        i0, i1 = int(a * SR), int(min(TL.total, a + d) * SR)
        if i1 > i0 > 0:
            act[i0:i1] += 1.0
    act = np.clip(smooth_env(act, 1.5) / 3.0, 0, 1)
    wl = shaped_noise(n, rng, lo=50, hi=1500, slope=1.2)
    wr = shaped_noise(n, rng, lo=50, hi=1500, slope=1.2)
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * t / 8.0 + 0.8 * np.sin(2 * np.pi * t / 21.0))
    amp = (0.09 + 0.25 * act) * gust
    L, R = wl * amp, wr * amp
    # close-ups
    storm = np.zeros(n, np.float32)
    for e in D.events:
        t0 = e["t0"]
        HIT, RUN = e["HIT"], e["RUN"]
        pts = [(t0 + HIT - min(2.5, HIT), 0.0), (t0 + HIT, 0.7), (t0 + HIT + RUN * 0.6, 1.0), (t0 + HIT + RUN + 1.5, 0.0)]
        xs, ys = zip(*pts)
        i0, i1 = int(xs[0] * SR), int(xs[-1] * SR)
        storm[i0:i1] = np.maximum(storm[i0:i1], np.interp(t[i0:i1], xs, ys))
    howl_src = shaped_noise(n, rng, lo=300, hi=2600)
    f = 420 + 180 * np.sin(2 * np.pi * t * 0.23) + 90 * np.sin(2 * np.pi * t * 0.61)
    ph = np.cumsum(2 * np.pi * f / SR)
    howl = howl_src * 0.5 + np.sin(ph) * 0.18 * (0.6 + 0.4 * np.sin(2 * np.pi * t * 0.9))
    rain = shaped_noise(n, rng, lo=2500, hi=9000)
    surge = shaped_noise(n, rng, lo=20, hi=180, slope=1.0)
    s = storm * storm
    L += (howl * 0.55 + rain * 0.35 + surge * 0.6) * s
    R += (howl * 0.5 + rain * 0.38 + surge * 0.58) * s
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
