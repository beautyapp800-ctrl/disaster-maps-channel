"""Shared pieces for the disaster-map renderer: map texture, relief mesh, OpenGL scene.

Reuses the America Over Time engine (mapviz.core: Albers projection + Census state shapes).
Real data only: US Census state borders, Natural Earth neighbours/lakes, NOAA ETOPO1 relief.
"""
from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapefile
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mapviz.core import albers, states  # noqa: E402

FONT_PATH = str(ROOT / "assets" / "fonts" / "Anton-Regular.ttf")
TW = int(os.environ.get("TEXW", "4096"))
TH = TW * 9 // 16
WORLD_W = 4.0
WORLD_H = WORLD_W * TH / TW
KM_PER_UNIT = 4600.0 / WORLD_W
EXAG = float(os.environ.get("EXAG", "14"))

OCEAN = (64, 170, 235)
HAZE = OCEAN
LAND = (250, 240, 214)
NEUTRAL = (222, 216, 196)
BORDER = (40, 40, 60)

# F / EF scale colours: bright and saturated, readable on cream land
F_COL = {-9: (120, 120, 130), 0: (20, 175, 95), 1: (255, 170, 0), 2: (255, 85, 0),
         3: (235, 0, 45), 4: (195, 0, 130), 5: (120, 0, 200)}
F_W = {-9: 2, 0: 2, 1: 2, 2: 3, 3: 3, 4: 5, 5: 7}


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, int(size))


# ------------------------------------------------------------------ map <-> texture
class Fr:
    """lon/lat -> texture pixel (Albers), lower 48 fitted into the texture with a small margin."""

    def __init__(self):
        xs, ys = albers([-125, -66.5, -125, -66.5, -96], [24.3, 24.3, 49.6, 49.6, 49.6])
        self.cx, self.cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        self.s = min(TW / (max(xs) - min(xs)), TH / (max(ys) - min(ys))) * 0.985

    def xy(self, x, y):
        return (np.asarray(x) - self.cx) * self.s + TW / 2, TH / 2 - (np.asarray(y) - self.cy) * self.s

    def px(self, lon, lat):
        return self.xy(*albers(lon, lat))

    def world(self, lon, lat):
        """lon/lat -> (x, z) world units (x east, z south)."""
        u, v = self.px(lon, lat)
        return (np.asarray(u) / TW - 0.5) * WORLD_W, (np.asarray(v) / TH - 0.5) * WORLD_H


FR = Fr()


@lru_cache(None)
def base_map() -> Image.Image:
    """Static bright map: ocean, neutral Canada/Mexico, cream states, dark borders, lakes."""
    img = Image.new("RGB", (TW, TH), OCEAN)
    d = ImageDraw.Draw(img)
    lw = max(1, round(TW / 1400))
    r = shapefile.Reader(str(ROOT / "data" / "ne" / "ne_50m_admin_0_countries"))
    for sr in r.iterShapeRecords():
        if sr.record["ADM0_A3"] not in ("CAN", "MEX", "CUB", "BHS"):
            continue
        pts = np.array(sr.shape.points)
        parts = list(sr.shape.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ring = pts[a:b]
            if ring[:, 1].min() > 75 or ring[:, 0].max() < -140 or ring[:, 0].min() > -40:
                continue
            x, y = FR.px(ring[:, 0], ring[:, 1])
            p = list(zip(x, y))
            d.polygon(p, fill=NEUTRAL)
            d.line(p + p[:1], fill=(120, 120, 130), width=lw)
    for rings in states().values():
        for rg in rings:
            x, y = FR.xy(rg[:, 0], rg[:, 1])
            d.polygon(list(zip(x, y)), fill=LAND)
    lk = shapefile.Reader(str(ROOT / "data" / "ne" / "ne_50m_lakes"))
    for sr in lk.iterShapeRecords():
        pts = np.array(sr.shape.points)
        if len(pts) == 0 or pts[:, 0].max() < -126 or pts[:, 0].min() > -64 or pts[:, 1].min() > 52 or pts[:, 1].max() < 23:
            continue
        parts = list(sr.shape.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            x, y = FR.px(pts[a:b, 0], pts[a:b, 1])
            d.polygon(list(zip(x, y)), fill=OCEAN)
    for rings in states().values():
        for rg in rings:
            x, y = FR.xy(rg[:, 0], rg[:, 1])
            p = list(zip(x, y))
            d.line(p + p[:1], fill=BORDER, width=lw)
    return img


# ------------------------------------------------------------------ relief
@lru_cache(None)
def relief():
    d = np.load(ROOT / "data" / "relief" / "etopo1_na_4min.npz")
    la, lo, z = d["lat"], d["lon"], d["z"].astype(np.float32)
    k = np.ones(3) / 3
    for _ in range(2):  # smooth the 3 arc-minute grid
        z = np.apply_along_axis(lambda a: np.convolve(np.pad(a, 1, mode="edge"), k, "valid"), 0, z)
        z = np.apply_along_axis(lambda a: np.convolve(np.pad(a, 1, mode="edge"), k, "valid"), 1, z)
    return la, lo, z


def height_world(z):
    """metres -> world units (land exaggerated, sea floor almost flat)."""
    zz = np.where(z > 0, z, z * 0.06)
    return zz / 1000.0 / KM_PER_UNIT * EXAG


def height_at(lon, lat):
    la, lo, z = relief()
    fi = np.clip((np.asarray(lat) - la[0]) / (la[1] - la[0]), 0, len(la) - 1.001)
    fj = np.clip((np.asarray(lon) - lo[0]) / (lo[1] - lo[0]), 0, len(lo) - 1.001)
    i, j = fi.astype(int), fj.astype(int)
    a, b = fi - i, fj - j
    v = (z[i, j] * (1 - a) * (1 - b) + z[i + 1, j] * a * (1 - b) + z[i, j + 1] * (1 - a) * b + z[i + 1, j + 1] * a * b)
    return height_world(v)


def world3(lon, lat, lift=0.0):
    x, z = FR.world(lon, lat)
    return np.stack([x, height_at(lon, lat) + lift, z], -1)


# ------------------------------------------------------------------ camera maths
def persp(fov, asp, n, f):
    t = 1 / math.tan(math.radians(fov) / 2)
    return np.array([[t / asp, 0, 0, 0], [0, t, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]], "f4")


def look(eye, at):
    eye, at = np.array(eye, "f4"), np.array(at, "f4")
    f = at - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, [0, 1, 0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype="f4")
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class Cam:
    FOV = 30.0

    def __init__(self, tx=0.0, ty=0.0, tz=0.0, dist=4.3, pitch=60.0, yaw=0.0, shake=(0.0, 0.0, 0.0)):
        self.tx, self.ty, self.tz, self.dist, self.pitch, self.yaw, self.shake = tx, ty, tz, dist, pitch, yaw, shake

    def mvp(self, W, H):
        p, y = math.radians(min(self.pitch, 89.0)), math.radians(self.yaw)
        tgt = np.array([self.tx, self.ty, self.tz], "f4") + np.array(self.shake, "f4")
        eye = tgt + self.dist * np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)], "f4")
        return persp(self.FOV, W / H, 0.05, 40) @ look(eye, tgt)

    def project(self, pts, W, H):
        """world points (N,3) -> pixel coords (N,2), depth ok mask."""
        m = self.mvp(W, H)
        p = np.asarray(pts, "f4").reshape(-1, 3)
        h = np.c_[p, np.ones(len(p), "f4")] @ m.T
        w = h[:, 3:4]
        ndc = h[:, :3] / np.where(np.abs(w) < 1e-6, 1e-6, w)
        return np.c_[(ndc[:, 0] * 0.5 + 0.5) * W, (1 - (ndc[:, 1] * 0.5 + 0.5)) * H], (w[:, 0] > 0)


def lerp_cam(a: Cam, b: Cam, w: float) -> Cam:
    L = lambda x, y: x + (y - x) * w  # noqa: E731
    dy = (b.yaw - a.yaw + 180) % 360 - 180
    return Cam(L(a.tx, b.tx), L(a.ty, b.ty), L(a.tz, b.tz), math.exp(L(math.log(a.dist), math.log(b.dist))),
               L(a.pitch, b.pitch), a.yaw + dy * w)


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ OpenGL scene
VS_TERRAIN = """#version 330
uniform mat4 mvp; in vec3 p; in vec2 t; in vec3 n; out vec2 vt; out vec3 vn;
void main(){ gl_Position = mvp*vec4(p,1.0); vt=t; vn=n; }"""
FS_TERRAIN = """#version 330
uniform sampler2D tex; uniform float dim; in vec2 vt; in vec3 vn; out vec4 c;
void main(){ vec3 n = normalize(vn);
  float l = clamp(dot(n, normalize(vec3(-0.5,0.8,0.35))),0.0,1.0)*0.55+0.52;
  float e = min(min(vt.x, 1.0-vt.x), min(vt.y, 1.0-vt.y));
  if (e < 0.0) discard;
  float f = smoothstep(0.0, 0.045, e);
  vec3 col = texture(tex, clamp(vt, 0.0, 1.0)).rgb*l*dim;
  c = vec4(mix(vec3(0.251,0.667,0.922), col, f), 1.0); }"""
VS_FUNNEL = """#version 330
uniform mat4 mvp; uniform vec3 off; uniform float sc; uniform float tm; in vec3 p; in vec2 ah; out vec2 vah;
void main(){ float h=p.y; vec3 q = vec3(p.x*(1.0+0.15*sin(tm*9.0+h*7.0)), p.y, p.z);
  q.x += sin(tm*3.0 + h*4.0)*0.06*h; q.z += cos(tm*2.6 + h*3.5)*0.05*h;
  gl_Position = mvp*vec4(off + q*sc, 1.0); vah=ah; }"""
FS_FUNNEL = """#version 330
uniform float tm; uniform float alpha; in vec2 vah; out vec4 c;
void main(){ float a=vah.x, h=vah.y;
  float band = 0.5+0.5*sin(a*5.0 + h*26.0 - tm*16.0);
  float rim = 0.65+0.35*max(0.0, cos(a-0.9));
  vec3 col = mix(vec3(0.16,0.16,0.19), vec3(0.86,0.86,0.9), band*0.75+0.15*h) * rim;
  c = vec4(col, alpha); }"""
VS_DUST = """#version 330
uniform mat4 mvp; uniform vec3 off; uniform float sc; in vec3 p; out float vr;
void main(){ vr=length(p.xz); gl_Position = mvp*vec4(off + p*sc, 1.0); }"""
FS_DUST = """#version 330
uniform float alpha; in float vr; out vec4 c;
void main(){ float a = clamp(1.0-vr, 0.0, 1.0); c = vec4(0.45,0.33,0.22, alpha*a*0.45); }"""
VS_BAR = """#version 330
uniform mat4 mvp; uniform vec3 base; uniform vec2 size; in vec3 p; in vec3 n; out vec3 vn; out float vh;
void main(){ vec3 w = vec3(base.x + p.x*size.x, base.y + p.y*size.y, base.z + p.z*size.x);
  gl_Position = mvp*vec4(w,1.0); vn=n; vh=p.y; }"""
FS_BAR = """#version 330
uniform vec3 col; in vec3 vn; in float vh; out vec4 c;
void main(){ vec3 n=normalize(vn); float l = clamp(dot(n, normalize(vec3(-0.5,0.8,0.35))),0.0,1.0)*0.6+0.45;
  c = vec4(col*l*(0.85+0.15*vh), 1.0); }"""


class Scene:
    def __init__(self, W: int, H: int, msaa: int = 4):
        import moderngl
        self.mgl = moderngl
        try:
            self.ctx = moderngl.create_standalone_context(backend="egl")
        except Exception:
            self.ctx = moderngl.create_standalone_context()
        ctx = self.ctx
        self.W, self.H = W, H
        self.renderer = ctx.info.get("GL_RENDERER", "?")
        la, lo, z = relief()
        LO, LA = np.meshgrid(lo, la)
        x, zz = FR.world(LO, LA)
        y = height_world(z)
        pos = np.dstack([x, y, zz]).astype("f4")
        di = np.gradient(pos, axis=0)
        dj = np.gradient(pos, axis=1)
        nrm = np.cross(dj, di)
        nrm /= np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-9
        nrm[nrm[..., 1] < 0] *= -1
        u, v = (x / WORLD_W + 0.5), (zz / WORLD_H + 0.5)
        nr, nc = z.shape
        verts = np.dstack([pos, u, v, nrm]).astype("f4").reshape(-1, 8)
        inside = (u > -0.03) & (u < 1.03) & (v > -0.03) & (v < 1.03)
        i = np.arange(nr - 1)[:, None] * nc + np.arange(nc - 1)[None, :]
        keep = inside[:-1, :-1] | inside[1:, :-1] | inside[:-1, 1:] | inside[1:, 1:]
        i = i[keep]
        idx = np.stack([i, i + 1, i + nc, i + 1, i + nc + 1, i + nc], -1).astype("i4").ravel()
        self.p_ter = ctx.program(vertex_shader=VS_TERRAIN, fragment_shader=FS_TERRAIN)
        self.vao_ter = ctx.vertex_array(self.p_ter, [(ctx.buffer(verts.tobytes()), "3f 2f 3f", "p", "t", "n")], ctx.buffer(idx.tobytes()))
        self.tex = ctx.texture((TW, TH), 3, b"\x00" * (TW * TH * 3))
        self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.tex.anisotropy = 8.0
        self.tex.repeat_x = self.tex.repeat_y = False
        # funnel (unit height cone of rings) + dust disc
        R, S = 36, 40
        fv = []
        for r in range(R + 1):
            h = r / R
            rad = 0.02 + 0.2 * h ** 1.8
            for s in range(S + 1):
                a = s / S * 2 * math.pi
                fv.append((rad * math.cos(a), h, rad * math.sin(a), a, h))
        fi = []
        for r in range(R):
            for s in range(S):
                a = r * (S + 1) + s
                fi += [a, a + 1, a + S + 1, a + 1, a + S + 2, a + S + 1]
        self.p_fun = ctx.program(vertex_shader=VS_FUNNEL, fragment_shader=FS_FUNNEL)
        fv = np.array(fv, "f4")
        self.vao_fun = ctx.vertex_array(self.p_fun, [(ctx.buffer(fv.tobytes()), "3f 2f", "p", "ah")], ctx.buffer(np.array(fi, "i4").tobytes()))
        dv, di_ = [(0.0, 0.0, 0.0)], []
        for s in range(41):
            a = s / 40 * 2 * math.pi
            dv.append((math.cos(a), 0.0, math.sin(a)))
        for s in range(40):
            di_ += [0, s + 1, s + 2]
        self.p_dust = ctx.program(vertex_shader=VS_DUST, fragment_shader=FS_DUST)
        self.vao_dust = ctx.vertex_array(self.p_dust, [(ctx.buffer(np.array(dv, "f4").tobytes()), "3f", "p")], ctx.buffer(np.array(di_, "i4").tobytes()))
        # bar (unit box with normals)
        bv, bi = [], []
        faces = [((0, 0, 1), [(-.5, 0, .5), (.5, 0, .5), (.5, 1, .5), (-.5, 1, .5)]), ((0, 0, -1), [(.5, 0, -.5), (-.5, 0, -.5), (-.5, 1, -.5), (.5, 1, -.5)]),
                 ((1, 0, 0), [(.5, 0, .5), (.5, 0, -.5), (.5, 1, -.5), (.5, 1, .5)]), ((-1, 0, 0), [(-.5, 0, -.5), (-.5, 0, .5), (-.5, 1, .5), (-.5, 1, -.5)]),
                 ((0, 1, 0), [(-.5, 1, .5), (.5, 1, .5), (.5, 1, -.5), (-.5, 1, -.5)])]
        for n, q in faces:
            b0 = len(bv)
            for pt in q:
                bv.append((*pt, *n))
            bi += [b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3]
        self.p_bar = ctx.program(vertex_shader=VS_BAR, fragment_shader=FS_BAR)
        self.vao_bar = ctx.vertex_array(self.p_bar, [(ctx.buffer(np.array(bv, "f4").tobytes()), "3f 3f", "p", "n")], ctx.buffer(np.array(bi, "i4").tobytes()))
        self.msaa = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), samples=msaa)], depth_attachment=ctx.depth_renderbuffer((W, H), samples=msaa))
        self.out = ctx.simple_framebuffer((W, H))

    def set_texture(self, img: Image.Image):
        self.tex.write(img.tobytes())
        self.tex.build_mipmaps()

    def render(self, cam: Cam, tm: float = 0.0, dim: float = 1.0, vortices=(), bars=()) -> Image.Image:
        """vortices: [(x,y,z, size, alpha)], bars: [(x,y,z, w, h, (r,g,b))] in world units."""
        ctx, mgl = self.ctx, self.mgl
        mvp = cam.mvp(self.W, self.H).T.copy().tobytes()
        self.msaa.use()
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.BLEND)
        ctx.clear(HAZE[0] / 255, HAZE[1] / 255, HAZE[2] / 255)
        self.p_ter["mvp"].write(mvp)
        self.p_ter["dim"].value = dim
        self.tex.use(0)
        self.vao_ter.render()
        for (x, y, z, w, h, col) in bars:
            self.p_bar["mvp"].write(mvp)
            self.p_bar["base"].value = (x, y, z)
            self.p_bar["size"].value = (w, h)
            self.p_bar["col"].value = tuple(c / 255 for c in col)
            self.vao_bar.render()
        if vortices:
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            for (x, y, z, sz, al) in vortices:
                self.p_dust["mvp"].write(mvp)
                self.p_dust["off"].value = (x, y + 0.002, z)
                self.p_dust["sc"].value = sz * 0.42
                self.p_dust["alpha"].value = al
                self.msaa.depth_mask = False
                self.vao_dust.render()
                self.msaa.depth_mask = True
                self.p_fun["mvp"].write(mvp)
                self.p_fun["off"].value = (x, y, z)
                self.p_fun["sc"].value = sz
                self.p_fun["tm"].value = tm
                self.p_fun["alpha"].value = al
                self.vao_fun.render()
        ctx.copy_framebuffer(self.out, self.msaa)
        raw = self.out.read(components=3)
        return Image.frombytes("RGB", (self.W, self.H), raw).transpose(Image.FLIP_TOP_BOTTOM)
