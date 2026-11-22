"""Render benchmark on a GitHub runner: 1080p vs 4K.

Measures per-frame cost of the three kinds of frames the channel needs:
  2d  - flat bright map, all tornado tracks of a year drawn on an accumulating layer + glow + counter text
        (the America Over Time style, numpy + Pillow)
  3d  - tilted 3D relief (NOAA ETOPO1 via ERDDAP) with the 2D map draped on it, lit, MSAA,
        orbiting camera + a 3D vortex funnel (moderngl, headless EGL / Mesa llvmpipe = CPU)
  enc - x264 encode of the rendered frames (preset medium, CRF 20), fps + bitrate
Writes bench_<res>.json. Real data: NOAA SPC tornadoes 1950-2025, NOAA ETOPO1.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mapviz.core import albers, states  # noqa: E402

RES = os.environ.get("RES", "1080")
W, H = (1920, 1080) if RES == "1080" else (3840, 2160)
N2D = int(os.environ.get("N2D", "120"))
N3D = int(os.environ.get("N3D", "120"))
FONT = str(ROOT / "assets" / "fonts" / "Anton-Regular.ttf")
OUT = ROOT / "bench" / "out"
OUT.mkdir(parents=True, exist_ok=True)
F_COL = {0: (120, 200, 90), 1: (255, 225, 40), 2: (255, 150, 0), 3: (240, 60, 30), 4: (205, 0, 90), 5: (130, 0, 170)}


def tornadoes():
    rows = []
    with open(ROOT / "data" / "raw" / "tornadoes.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                m = int(r["mag"]); y = int(r["yr"])
                sl, so, el, eo = map(float, (r["slat"], r["slon"], r["elat"], r["elon"]))
            except ValueError:
                continue
            if m < 0 or sl == 0:
                continue
            if el == 0:
                el, eo = sl, so
            rows.append((y, m, sl, so, el, eo))
    return rows


class Fr:
    def __init__(self, w, h):
        xs, ys = albers([-125, -66, -125, -66, -96], [24, 24, 50, 50, 50])
        self.cx, self.cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        self.s = min(w / (max(xs) - min(xs)), h / (max(ys) - min(ys))) * 0.98
        self.w, self.h = w, h

    def xy(self, x, y):
        return (np.asarray(x) - self.cx) * self.s + self.w / 2, self.h / 2 - (np.asarray(y) - self.cy) * self.s

    def ll(self, lon, lat):
        return self.xy(*albers(lon, lat))


def base_map(w, h):
    fr = Fr(w, h)
    img = Image.new("RGB", (w, h), (64, 170, 235))
    d = ImageDraw.Draw(img)
    for rings in states().values():
        for r in rings:
            x, y = fr.xy(r[:, 0], r[:, 1])
            d.polygon(list(zip(x, y)), fill=(250, 240, 214))
    lw = max(2, w // 900)
    for rings in states().values():
        for r in rings:
            x, y = fr.xy(r[:, 0], r[:, 1])
            p = list(zip(x, y))
            d.line(p + p[:1], fill=(40, 40, 60), width=lw)
    return fr, img


def bench_2d(rows):
    fr, base = base_map(W, H)
    k = W / 1920
    acc = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    da = ImageDraw.Draw(acc)
    font = ImageFont.truetype(FONT, int(110 * k))
    small = ImageFont.truetype(FONT, int(46 * k))
    by_year = {}
    for r in rows:
        by_year.setdefault(r[0], []).append(r)
    years = sorted(by_year)
    t0 = time.time()
    count = 0
    for i in range(N2D):
        yr = years[min(len(years) - 1, int(i / N2D * len(years) * 2) % len(years))]
        # each frame adds 1/12 of a year's tracks (a year spans ~12 frames = 0.5 s at 24 fps)
        chunk = by_year[yr][(i % 12) * len(by_year[yr]) // 12:((i % 12) + 1) * len(by_year[yr]) // 12]
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dg = ImageDraw.Draw(glow)
        for _, m, sl, so, el, eo in chunk:
            x, y = fr.ll([so, eo], [sl, el])
            seg = [(x[0], y[0]), (x[1], y[1])]
            w = max(1, int((1 + m) * k))
            da.line(seg, fill=F_COL[m] + (255,), width=w)
            dg.line(seg, fill=(255, 255, 200, 255), width=w * 5)
        count += len(chunk)
        frame = base.copy()
        frame.paste(acc, (0, 0), acc)
        g = glow.filter(ImageFilter.GaussianBlur(6 * k))
        frame.paste(g, (0, 0), g)
        d = ImageDraw.Draw(frame)
        d.text((W * 0.04, H * 0.05), str(yr), font=font, fill=(255, 255, 255), stroke_width=int(8 * k), stroke_fill=(15, 15, 30))
        d.text((W * 0.04, H * 0.18), f"{count:,} TORNADOES", font=small, fill=(255, 214, 0), stroke_width=int(5 * k), stroke_fill=(15, 15, 30))
        a = np.asarray(frame)
        if i == N2D // 2:
            frame.save(OUT / f"sample2d_{RES}.jpg", quality=90)
        _ = a.tobytes()
    return (time.time() - t0) / N2D


def etopo():
    """CONUS relief, NOAA ETOPO1 (ERDDAP etopo180, stride 3 -> 3 arc-min; production: ETOPO 2022 / USGS 3DEP)."""
    p = ROOT / "data" / "raw" / "etopo.csv"
    if not p.exists():
        url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.csv?"
               "altitude%5B(23.5):3:(50.5)%5D%5B(-126):3:(-65)%5D")
        subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(p), url], check=True)
    lat, lon, z = np.loadtxt(p, delimiter=",", skiprows=2, unpack=True)
    la, lo = np.unique(lat), np.unique(lon)
    grid = np.full((len(la), len(lo)), np.nan)
    grid[np.searchsorted(la, lat), np.searchsorted(lo, lon)] = z
    return la, lo, np.nan_to_num(grid)


def bench_3d(rows):
    import moderngl

    ctx = moderngl.create_standalone_context(backend="egl")
    info = ctx.info.get("GL_RENDERER", "?")
    la, lo, z = etopo()
    LO, LA = np.meshgrid(lo, la)
    x, y = albers(LO, LA)
    # texture: the bright 2D map with all F3+ tracks (draped on the relief)
    tw, th = (4096, 2304) if RES == "2160" else (2048, 1152)
    fr, tex_img = base_map(tw, th)
    d = ImageDraw.Draw(tex_img)
    for _, m, sl, so, el, eo in rows:
        if m >= 3:
            px, py = fr.ll([so, eo], [sl, el])
            d.line([(px[0], py[0]), (px[1], py[1])], fill=F_COL[m], width=max(1, m - 1))
    u, v = fr.xy(x, y)
    u, v = u / tw, 1 - v / th
    sx = (x - fr.cx) * 1e-6 * 1.6
    sy = (y - fr.cy) * 1e-6 * 1.6
    sz = np.where(z > 0, z, z * 0.15) * 1e-4 * 0.9   # vertical exaggeration ~ x35
    nr, nc = z.shape
    verts = np.dstack([sx, sz, -sy, u, v]).astype("f4").reshape(-1, 5)
    idx = []
    i = np.arange(nr - 1)[:, None] * nc + np.arange(nc - 1)[None, :]
    idx = np.stack([i, i + 1, i + nc, i + 1, i + nc + 1, i + nc], -1).astype("i4").ravel()
    prog = ctx.program(
        vertex_shader="""#version 330
        uniform mat4 mvp; in vec3 p; in vec2 t; out vec2 vt; out vec3 vp;
        void main(){ gl_Position = mvp*vec4(p,1.0); vt=t; vp=p; }""",
        fragment_shader="""#version 330
        uniform sampler2D tex; in vec2 vt; in vec3 vp; out vec4 c;
        void main(){ vec3 n = normalize(cross(dFdx(vp), dFdy(vp)));
          float l = clamp(dot(n, normalize(vec3(-0.4,0.9,0.3))),0.0,1.0)*0.55+0.55;
          c = vec4(texture(tex, vt).rgb*l, 1.0); }""")
    vbo = ctx.buffer(verts.tobytes())
    ibo = ctx.buffer(idx.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 2f", "p", "t")], ibo)
    tex = ctx.texture(tex_img.size, 3, tex_img.tobytes())
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    # vortex funnel (cone of rotating rings) - simple geometry, coloured
    vprog = ctx.program(
        vertex_shader="""#version 330
        uniform mat4 mvp; in vec3 p; in vec3 col; out vec3 vc;
        void main(){ gl_Position = mvp*vec4(p,1.0); vc=col; }""",
        fragment_shader="""#version 330
        in vec3 vc; out vec4 c; void main(){ c = vec4(vc,1.0); }""")
    msaa = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), samples=4)], depth_attachment=ctx.depth_renderbuffer((W, H), samples=4))
    outfb = ctx.simple_framebuffer((W, H))
    ctx.enable(moderngl.DEPTH_TEST)

    def persp(fov, asp, n, f):
        t = 1 / math.tan(math.radians(fov) / 2)
        return np.array([[t / asp, 0, 0, 0], [0, t, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]], "f4")

    def look(eye, at):
        eye, at = np.array(eye, "f4"), np.array(at, "f4")
        f = at - eye; f /= np.linalg.norm(f)
        s = np.cross(f, [0, 1, 0]); s /= np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.eye(4, dtype="f4")
        m[0, :3], m[1, :3], m[2, :3] = s, u, -f
        m[:3, 3] = -m[:3, :3] @ eye
        return m

    P = persp(40, W / H, 0.05, 50)
    # Joplin 2011 (SPC: 37.06 N, 94.57 W) - vortex position
    jx, jy = albers(-94.57, 37.06)
    vx, vz = (jx - fr.cx) * 1.6e-6, -(jy - fr.cy) * 1.6e-6
    times = []
    sample_at = N3D // 2
    for i in range(N3D):
        t0 = time.time()
        a = i / N3D * 0.8
        eye = (vx + 1.6 * math.sin(a), 1.1, vz + 1.6 * math.cos(a))
        mvp = (P @ look(eye, (vx, 0, vz))).T.copy()
        msaa.use(); ctx.clear(0.25, 0.67, 0.92)
        prog["mvp"].write(mvp.tobytes()); tex.use(0)
        vao.render()
        # funnel: 40 rings x 48 points as triangle strips
        pts, cols = [], []
        for r in range(40):
            hgt = r / 40 * 0.45
            rad = 0.008 + (r / 40) ** 1.6 * 0.09
            for j in range(48):
                ang = j / 48 * 2 * math.pi + i * 0.35 + r * 0.25
                for dh, dr in ((0, 0), (0.45 / 40, 0.0)):
                    pts.append((vx + rad * math.cos(ang), hgt + dh, vz + rad * math.sin(ang)))
                    g = 0.55 + 0.4 * math.sin(ang * 3)
                    cols.append((g, g, g * 1.05))
        vb = ctx.buffer(np.hstack([np.array(pts, "f4"), np.array(cols, "f4")]).tobytes())
        va = ctx.vertex_array(vprog, [(vb, "3f 3f", "p", "col")])
        vprog["mvp"].write(mvp.tobytes())
        va.render(moderngl.TRIANGLE_STRIP)
        va.release(); vb.release()
        ctx.copy_framebuffer(outfb, msaa)
        raw = outfb.read(components=3)
        if i == sample_at:
            Image.frombytes("RGB", (W, H), raw).transpose(Image.FLIP_TOP_BOTTOM).save(OUT / f"sample3d_{RES}.jpg", quality=90)
        times.append(time.time() - t0)
    return float(np.mean(times[3:])), info, int(len(idx) / 3)


def bench_encode():
    src = Image.open(OUT / f"sample3d_{RES}.jpg").convert("RGB")
    a = np.asarray(src)
    n = 240
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(OUT / f"enc_{RES}.mp4")]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(np.roll(a, i * 6, axis=1).tobytes())   # panning = realistic motion for the encoder
    p.stdin.close(); p.wait()
    dt = time.time() - t0
    mb = (OUT / f"enc_{RES}.mp4").stat().st_size / 1e6
    return n / dt, mb / (n / 24) * 60


if __name__ == "__main__":
    rows = tornadoes()
    res = {"res": f"{W}x{H}", "cpus": os.cpu_count()}
    res["s_per_frame_2d"] = bench_2d(rows)
    try:
        res["s_per_frame_3d"], res["gl_renderer"], res["triangles"] = bench_3d(rows)
    except Exception as e:  # report, do not hide
        res["error_3d"] = repr(e)
    if (OUT / f"sample3d_{RES}.jpg").exists():
        res["encode_fps"], res["mb_per_min"] = bench_encode()
    print(json.dumps(res, indent=1))
    (OUT / f"bench_{RES}.json").write_text(json.dumps(res, indent=1))
