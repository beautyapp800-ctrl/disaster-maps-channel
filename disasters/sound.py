"""Procedural sound design for the tornado video (generated here from noise and sine waves: no third-party audio,
so no licence or Content ID risk for the effects).

Layers, all driven by the same timeline as the picture:
  wind bed  - soft filtered noise, grows with the number of tornadoes appearing per second (quiet between events)
  vortex    - low rumble + swirling whistle for every F4/F5 path run; much louder in the town close-ups
  debris    - short crackles of flying debris while a famous tornado crosses a town
Usage: python -m disasters.sound out.wav
"""
from __future__ import annotations

import sys
import wave
from bisect import bisect_left

import numpy as np

from .tornado import EV_HIT, EV_RUN, Data, Timeline

SR = 24000


def shaped_noise(n, rng, lo=None, hi=None, slope=0.0):
    """white noise shaped in the frequency domain: band [lo, hi] Hz and 1/f^slope tilt."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    g = np.ones_like(f)
    if slope:
        g /= np.maximum(f, 20.0) ** (slope / 2)
    if lo:
        g *= 1.0 / (1.0 + (lo / np.maximum(f, 1e-3)) ** 4)
    if hi:
        g *= 1.0 / (1.0 + (f / hi) ** 4)
    y = np.fft.irfft(X * g, n)
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def env_from_points(n, pts):
    """piecewise-linear envelope from (seconds, value) points."""
    t = np.arange(n) / SR
    xs, ys = zip(*sorted(pts))
    return np.interp(t, xs, ys).astype(np.float32)


def smooth_env(x, sec):
    k = max(1, int(sec * SR))
    c = np.cumsum(np.concatenate([[0.0], x]))
    y = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, y[0]), y]).astype(np.float32)


def build(path):
    D = Data()
    TL = Timeline(D)
    n = int(TL.total * SR)
    rng = np.random.default_rng(1950)
    t_axis = np.arange(n) / SR
    # ---- activity: tornado appearances per second (smoothed), 0..1
    hits = np.zeros(n, np.float32)
    idx = np.clip((TL.ta * SR).astype(int), 0, n - 1)
    np.add.at(hits, idx[TL.ta > 1e-6], 1.0)
    act = smooth_env(hits, 1.2) * SR
    act = np.clip(act / np.percentile(act[act > 0], 95), 0, 1.2)
    # ---- wind bed (stereo-decorrelated)
    wl = shaped_noise(n, rng, lo=60, hi=1800, slope=1.2)
    wr = shaped_noise(n, rng, lo=60, hi=1800, slope=1.2)
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * t_axis / 9.0 + 0.7 * np.sin(2 * np.pi * t_axis / 23.0))
    wind_amp = (0.10 + 0.32 * act) * gust
    L = wl * wind_amp
    R = wr * wind_amp
    # ---- vortices
    rumble = shaped_noise(n, rng, lo=25, hi=160, slope=1.0)
    deb_src = shaped_noise(n, rng, lo=900, hi=6000)
    vort_env = np.zeros(n, np.float32)
    whistle = np.zeros(n, np.float32)
    debris = np.zeros(n, np.float32)
    evs = {e["idx"]: e for e in D.events}
    for i in np.where(TL.anim > 0)[0]:
        is_ev = i in evs
        t0 = TL.ta[i]
        dur = EV_RUN if is_ev else TL.anim[i]
        a0, a1 = int(max(0, t0 - 0.4) * SR), int(min(TL.total, t0 + dur + (1.6 if is_ev else 0.8)) * SR)
        if a1 <= a0:
            continue
        loud = 1.0 if is_ev else (0.28 if D.mag[i] == 5 else 0.18)
        tt = (np.arange(a0, a1) / SR) - t0
        e = np.clip((tt + 0.4) / 0.9, 0, 1) * np.clip(1 - (tt - dur) / (1.6 if is_ev else 0.8), 0, 1)
        e = (e * e * (3 - 2 * e)).astype(np.float32) * loud
        vort_env[a0:a1] = np.maximum(vort_env[a0:a1], e)
        f = 520 + 260 * np.sin(2 * np.pi * tt * (0.6 if is_ev else 1.1)) + 90 * rng.standard_normal(len(tt)).cumsum() / np.sqrt(np.arange(1, len(tt) + 1))
        ph = np.cumsum(2 * np.pi * np.clip(f, 200, 1400) / SR)
        am = 0.6 + 0.4 * np.sin(2 * np.pi * tt * 7.0)
        whistle[a0:a1] += (np.sin(ph) * am * e * 0.22).astype(np.float32)
        if is_ev:
            rate = 38.0
            k = rng.poisson(rate * (a1 - a0) / SR)
            pos = rng.integers(a0, a1, size=k)
            imp = np.zeros(a1 - a0, np.float32)
            imp[pos - a0] = rng.uniform(0.3, 1.0, size=k)
            kern = np.exp(-np.arange(int(0.025 * SR)) / (0.006 * SR)).astype(np.float32)
            burst = np.convolve(imp, kern)[: a1 - a0] * deb_src[a0:a1]
            debris[a0:a1] += burst * e * 0.55
    vortex = rumble * vort_env * 0.9 + whistle
    L += vortex + debris * 0.9
    R += vortex * 0.95 + debris
    # ---- gentle fades
    fade = np.clip(t_axis / 2.0, 0, 1) * np.clip((TL.total - t_axis) / 4.0, 0, 1)
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
