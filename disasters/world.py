"""'Living map' OpenGL scene: real satellite colour on 3D relief, moving clouds and haze, live water,
night lights, local close-ups (USGS NAIP + OSM 3D buildings) and event effects (funnel, debris, damage swath).

All rendering runs on the CPU (Mesa llvmpipe) on GitHub runners.
Data: NASA Blue Marble NG + Black Marble (GIBS), NOAA ETOPO1, Natural Earth lakes, USGS NAIP, OpenStreetMap buildings.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from . import core
from .core import FR, KM_PER_UNIT, TH, TW, WORLD_H, WORLD_W, height_at, height_world, relief

ROOT = core.ROOT
SKY_TOP = (0.33, 0.60, 0.92)
SKY_HOR = (0.80, 0.89, 0.97)
SUN = np.array([-0.45, 0.80, 0.40]) / np.linalg.norm([-0.45, 0.80, 0.40])
M_PER_UNIT = KM_PER_UNIT * 1000.0

COMMON = """
uniform float tm; uniform vec3 eye; uniform float camdist; uniform vec2 wind;
uniform sampler2D noise;
const vec3 SUN = vec3(-0.4719, 0.8389, 0.4195);
const vec3 HAZE = vec3(0.80, 0.89, 0.97);
float cloudAt(vec2 p){
  float a = texture(noise, p*0.33 + wind*tm).r;
  float b = texture(noise, p*1.05 + wind*tm*1.7 + vec2(0.37,0.11)).g;
  return smoothstep(0.66, 0.90, a*0.68 + b*0.32);
}
vec3 fogged(vec3 col, vec3 p){
  float d = length(p - eye);
  float near = camdist < 0.2 ? 0.55 : 0.95;
  float far = camdist < 0.2 ? 1.6 : 2.8;
  float f = smoothstep(camdist*near, camdist*far, d) * (camdist < 0.2 ? 0.85 : 0.6);
  return mix(col, HAZE, f);
}
"""

VS_TER = """#version 330
uniform mat4 mvp; in vec3 p; in vec2 t; in vec3 n; out vec2 vt; out vec3 vn; out vec3 vp;
void main(){ gl_Position = mvp*vec4(p,1.0); vt=t; vn=n; vp=p; }"""
FS_TER = """#version 330
""" + COMMON + """
uniform sampler2D sat; uniform sampler2D aux; uniform sampler2D ovl;
uniform float dim; uniform float ovl_alpha; uniform float lights_amt; uniform float cloud_amt;
in vec2 vt; in vec3 vn; in vec3 vp; out vec4 c;
void main(){
  float e = min(min(vt.x, 1.0-vt.x), min(vt.y, 1.0-vt.y));
  if (e < 0.0) discard;
  vec3 n = normalize(vn);
  vec2 uv = clamp(vt, 0.0, 1.0);
  vec3 a = texture(aux, uv).rgb;
  vec3 col;
  if (a.r > 0.5) {
    vec2 q = vp.xz*1.1;
    float h1 = texture(noise, q + vec2(tm*0.0015, tm*0.001)).r;
    float h2 = texture(noise, q*2.3 + vec2(-tm*0.0012, tm*0.0016)).g;
    vec3 wn = normalize(vec3((h1-0.5)*0.35, 1.0, (h2-0.5)*0.35));
    vec3 v = normalize(eye - vp);
    float spec = pow(max(dot(reflect(-SUN, wn), v), 0.0), 40.0);
    col = mix(vec3(0.06,0.36,0.72), vec3(0.20,0.66,0.86), a.g);
    col += spec*0.25;
    float foam = smoothstep(0.55, 0.95, a.g) * (0.5+0.5*sin(tm*1.6 - a.g*30.0 + h1*7.0));
    col = mix(col, vec3(0.97), foam*0.45);
  } else {
    float l = clamp(dot(n, SUN), 0.0, 1.0)*0.62 + 0.50;
    col = texture(sat, uv).rgb * l;
  }
  col += vec3(1.0,0.78,0.45) * pow(a.b, 1.6) * lights_amt;
  float cs = cloudAt(vp.xz - SUN.xz*(0.05/SUN.y));
  col *= 1.0 - 0.32*cs*cloud_amt;
  vec4 o = texture(ovl, uv);
  col = mix(col, o.rgb, o.a*ovl_alpha);
  col *= dim;
  col = fogged(col, vp);
  col = mix(HAZE, col, smoothstep(0.0, 0.04, e));
  c = vec4(col, 1.0);
}"""

VS_SKY = """#version 330
in vec2 p; out float vy; void main(){ gl_Position = vec4(p, 0.9999, 1.0); vy = p.y*0.5+0.5; }"""
FS_SKY = """#version 330
in float vy; out vec4 c;
void main(){ c = vec4(mix(vec3(0.80,0.89,0.97), vec3(0.33,0.60,0.92), smoothstep(0.35, 1.0, vy)), 1.0); }"""

VS_CLOUD = """#version 330
uniform mat4 mvp; in vec2 p; out vec3 vp;
void main(){ vp = vec3(p.x, 0.055, p.y); gl_Position = mvp*vec4(vp,1.0); }"""
FS_CLOUD = """#version 330
""" + COMMON + """
uniform float cloud_amt; in vec3 vp; out vec4 c;
void main(){
  float k = cloudAt(vp.xz);
  float rim = cloudAt(vp.xz + SUN.xz*0.01);
  vec3 col = mix(vec3(0.82,0.85,0.90), vec3(1.0), clamp(1.0 - (rim-k)*3.0, 0.0, 1.0));
  c = vec4(fogged(col, vp), k*0.7*cloud_amt);
}"""

VS_PATCH = """#version 330
uniform mat4 mvp; in vec3 p; in vec2 t; out vec2 vt; out vec3 vp;
void main(){ gl_Position = mvp*vec4(p,1.0); vt=t; vp=p; }"""
FS_PATCH = """#version 330
""" + COMMON + """
uniform sampler2D img; uniform float alpha;
uniform vec2 sa; uniform vec2 sb; uniform float hw; uniform float along;   // damage swath in patch uv (aspect-corrected)
uniform float aspect;
in vec2 vt; in vec3 vp; out vec4 c;
void main(){
  vec3 col = texture(img, vt).rgb;
  col = pow(col, vec3(0.85)); float lum = dot(col, vec3(0.3,0.59,0.11)); col = mix(vec3(lum), col, 1.25);
  vec2 P = vec2(vt.x*aspect, vt.y); vec2 A = vec2(sa.x*aspect, sa.y); vec2 B = vec2(sb.x*aspect, sb.y);
  vec2 ab = B - A; float L = max(length(ab), 1e-6); vec2 dir = ab / L;
  float s = dot(P - A, dir); float d = abs(dot(P - A, vec2(-dir.y, dir.x)));
  float rough = texture(noise, vt*9.0).r;
  float inside = (1.0 - smoothstep(hw*0.75, hw*(1.0+0.35*rough), d)) * step(0.0, s) * (1.0 - smoothstep(along - 0.004, along, s));
  col = mix(col, vec3(0.43,0.35,0.27) + (rough-0.5)*0.15, inside*0.72);
  float e = min(min(vt.x, 1.0-vt.x), min(vt.y, 1.0-vt.y));
  c = vec4(fogged(col, vp), alpha*smoothstep(0.0, 0.10, e));
}"""

VS_BLD = """#version 330
uniform mat4 mvp; uniform float hscale; uniform float base_y; uniform vec2 pa; uniform vec2 pb; uniform float hw; uniform float along;
in vec3 p; in vec3 n; in vec4 ia; in vec2 ib;   // ia = (x, z, w, d) world; ib = (angle, height)
out vec3 vn; out float vdead; out float vh; out vec3 vp; flat out float vid;
void main(){
  vec2 ab = pb - pa; float L = max(length(ab), 1e-9); vec2 dir = ab/L;
  vec2 rel = ia.xy - pa; float s = dot(rel, dir); float d = abs(dot(rel, vec2(-dir.y, dir.x)));
  float dead = step(d, hw) * step(0.0, s) * step(s, along);
  float h = ib.y * hscale * mix(1.0, 0.12, dead);
  float ca = cos(ib.x), sn = sin(ib.x);
  vec2 q = vec2(p.x*ia.z, p.z*ia.w);
  vec2 r = vec2(q.x*ca - q.y*sn, q.x*sn + q.y*ca);
  vec3 w = vec3(ia.x + r.x, base_y + p.y*h, ia.y + r.y);
  vp = w; gl_Position = mvp*vec4(w, 1.0);
  vn = vec3(n.x*ca - n.z*sn, n.y, n.x*sn + n.z*ca); vdead = dead; vh = p.y; vid = float(gl_InstanceID);
}"""
FS_BLD = """#version 330
""" + COMMON + """
uniform float alpha;
in vec3 vn; in float vdead; in float vh; in vec3 vp; flat in float vid; out vec4 c;
void main(){
  float hsh = fract(sin(vid*12.9898)*43758.5453);
  vec3 roof = mix(vec3(0.78,0.76,0.72), vec3(0.62,0.58,0.55), hsh);
  vec3 wall = mix(vec3(0.92,0.89,0.84), vec3(0.80,0.74,0.66), hsh);
  vec3 col = vn.y > 0.5 ? roof : wall;
  col = mix(col, vec3(0.42,0.34,0.27), vdead*0.85);
  float l = clamp(dot(normalize(vn), SUN), 0.0, 1.0)*0.55 + 0.5;
  c = vec4(fogged(col*l, vp), alpha);
}"""

VS_FUN = """#version 330
uniform mat4 mvp; uniform vec3 off; uniform float sc; uniform float tm; in vec3 p; in vec2 ah; out vec2 vah; out vec3 vp;
void main(){ float h=p.y; vec3 q = vec3(p.x*(1.0+0.15*sin(tm*9.0+h*7.0)), p.y, p.z*(1.0+0.15*sin(tm*9.0+h*7.0)));
  q.x += sin(tm*3.0 + h*4.0)*0.06*h; q.z += cos(tm*2.6 + h*3.5)*0.05*h;
  vp = off + q*sc; gl_Position = mvp*vec4(vp, 1.0); vah=ah; }"""
FS_FUN = """#version 330
uniform float tm; uniform float alpha; in vec2 vah; in vec3 vp; out vec4 c;
void main(){ float a=vah.x, h=vah.y;
  float band = 0.5+0.5*sin(a*5.0 + h*26.0 - tm*16.0);
  float rim = 0.62+0.38*max(0.0, cos(a-0.9));
  vec3 col = mix(vec3(0.42,0.42,0.46), vec3(0.88,0.88,0.92), band*0.5+0.3*h) * rim;
  col = mix(col, vec3(0.45,0.37,0.30), (1.0-smoothstep(0.0, 0.25, h))*0.6);
  c = vec4(col, alpha*(0.62+0.3*h)); }"""
VS_DISC = """#version 330
uniform mat4 mvp; uniform vec3 off; uniform float sc; in vec3 p; out float vr;
void main(){ vr=length(p.xz); gl_Position = mvp*vec4(off + p*sc, 1.0); }"""
FS_DISC = """#version 330
uniform float alpha; uniform vec3 col; in float vr; out vec4 c;
void main(){ float a = clamp(1.0-vr, 0.0, 1.0); c = vec4(col, alpha*a*a); }"""
VS_DEB = """#version 330
uniform mat4 mvp; uniform vec3 off; uniform float sc; uniform float tm; uniform float psize; in float id; out float vk;
float h1(float x){ return fract(sin(x*78.233)*43758.5453); }
void main(){
  float r = 0.05 + 0.32*pow(h1(id), 0.7);
  float hh = 0.42*pow(h1(id+1.7), 1.8);
  float w = 2.0 + 5.0*h1(id+3.1);
  float ph = 6.2831*h1(id+5.3) + tm*w;
  vec3 q = vec3(cos(ph)*r, hh, sin(ph)*r);
  gl_Position = mvp*vec4(off + q*sc, 1.0);
  gl_PointSize = psize*(0.6 + 1.2*h1(id+9.1));
  vk = h1(id+11.0);
}"""
FS_DEB = """#version 330
uniform float alpha; in float vk; out vec4 c;
void main(){ vec2 d = gl_PointCoord - 0.5; if (dot(d,d) > 0.25) discard;
  c = vec4(mix(vec3(0.30,0.24,0.18), vec3(0.62,0.58,0.52), vk), alpha*0.9); }"""
VS_BAR = core.VS_BAR
FS_BAR = core.FS_BAR


# ------------------------------------------------------------------ static data textures
@lru_cache(None)
def sat_image(w, h):
    p = ROOT / "data" / "world" / "sat_8192.jpg"
    im = Image.open(p).convert("RGB").resize((w, h), Image.LANCZOS)
    a = np.asarray(im).astype("f4") / 255.0
    a = np.power(a, 0.72)
    lum = a @ np.array([0.30, 0.59, 0.11], "f4")
    a = lum[..., None] + (a - lum[..., None]) * 1.5
    a = (a - 0.5) * 1.06 + 0.53
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype("u1"))


@lru_cache(None)
def aux_image(w, h):
    water = Image.open(ROOT / "data" / "world" / "water_4096.png").convert("L").resize((w, h), Image.BILINEAR)
    shore = water.filter(ImageFilter.GaussianBlur(max(2, w / 600)))
    lights = Image.open(ROOT / "data" / "world" / "lights_4096.png").convert("L").resize((w, h), Image.BILINEAR)
    lv = np.clip((np.asarray(lights, "f4") - 45.0) / 170.0, 0, 1)
    return Image.merge("RGB", (water.point(lambda v: 255 if v > 127 else 0), shore, Image.fromarray((lv * 255).astype("u1"))))


@lru_cache(None)
def noise_image(n=1024, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(2):
        f = np.fft.fft2(rng.standard_normal((n, n)))
        fx = np.fft.fftfreq(n)[:, None]
        fy = np.fft.fftfreq(n)[None, :]
        amp = 1.0 / np.maximum(np.hypot(fx, fy), 1.0 / n) ** (1.7 + 0.2 * k)
        a = np.real(np.fft.ifft2(f * amp))
        a = (a - a.min()) / (a.max() - a.min())
        out.append((a * 255).astype("u1"))
    return Image.merge("RGB", (Image.fromarray(out[0]), Image.fromarray(out[1]), Image.fromarray(out[0])))


def have_world():
    return (ROOT / "data" / "world" / "sat_8192.jpg").exists()


# ------------------------------------------------------------------ local close-ups
class Local:
    def __init__(self, key):
        d = ROOT / "data" / "local"
        self.key = key
        self.meta = json.loads((d / f"{key}.json").read_text())
        self.img = Image.open(d / f"{key}.jpg").convert("RGB")
        b = self.meta["bbox"]
        self.lon0, self.lat0, self.lon1, self.lat1 = b
        bf = d / f"{key}_bld.npz"
        self.bld = np.load(bf) if bf.exists() else None

    def uv(self, lon, lat):
        return (np.asarray(lon) - self.lon0) / (self.lon1 - self.lon0), (self.lat1 - np.asarray(lat)) / (self.lat1 - self.lat0)

    def clip_fracs(self, slon, slat, elon, elat):
        """parameter range [f0,f1] of the straight SPC path inside the window (Liang-Barsky)."""
        x0, y0 = self.uv(slon, slat)
        x1, y1 = self.uv(elon, elat)
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0 - 0.04), (dx, 0.96 - x0), (-dy, y0 - 0.04), (dy, 0.96 - y0)):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
        return (t0, t1) if t1 > t0 else None


# ------------------------------------------------------------------ scene
def _set(prog, name, value):
    if name in prog:
        prog[name].value = value


class LivingScene:
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
        # terrain mesh (same as core.Scene)
        la, lo, z = relief()
        LO, LA = np.meshgrid(lo, la)
        x, zz = FR.world(LO, LA)
        y = height_world(z)
        pos = np.dstack([x, y, zz]).astype("f4")
        nrm = np.cross(np.gradient(pos, axis=1), np.gradient(pos, axis=0))
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
        self.p_ter = ctx.program(vertex_shader=VS_TER, fragment_shader=FS_TER)
        self.vao_ter = ctx.vertex_array(self.p_ter, [(ctx.buffer(verts.tobytes()), "3f 2f 3f", "p", "t", "n")], ctx.buffer(idx.tobytes()))
        # textures
        sw, sh = min(8192, TW * 2), min(4608, TH * 2)
        self.t_sat = self._tex(sat_image(sw, sh), mip=True)
        self.t_aux = self._tex(aux_image(TW, TH), mip=True)
        self.t_noise = self._tex(noise_image(), mip=True, repeat=True)
        self.t_ovl = ctx.texture((TW, TH), 4, b"\x00" * (TW * TH * 4))
        self.t_ovl.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.t_ovl.repeat_x = self.t_ovl.repeat_y = False
        # sky + cloud plane
        self.p_sky = ctx.program(vertex_shader=VS_SKY, fragment_shader=FS_SKY)
        self.vao_sky = ctx.vertex_array(self.p_sky, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes()), "2f", "p")])
        self.p_cloud = ctx.program(vertex_shader=VS_CLOUD, fragment_shader=FS_CLOUD)
        hw_, hh_ = WORLD_W * 0.55, WORLD_H * 0.6
        self.vao_cloud = ctx.vertex_array(self.p_cloud, [(ctx.buffer(np.array([-hw_, -hh_, hw_, -hh_, -hw_, hh_, hw_, hh_], "f4").tobytes()), "2f", "p")])
        # funnel, discs, debris, bars
        R, S = 36, 40
        fv, fi = [], []
        for r in range(R + 1):
            h = r / R
            rad = 0.02 + 0.2 * h ** 1.8
            for s in range(S + 1):
                a = s / S * 2 * math.pi
                fv.append((rad * math.cos(a), h, rad * math.sin(a), a, h))
        for r in range(R):
            for s in range(S):
                a = r * (S + 1) + s
                fi += [a, a + 1, a + S + 1, a + 1, a + S + 2, a + S + 1]
        self.p_fun = ctx.program(vertex_shader=VS_FUN, fragment_shader=FS_FUN)
        self.vao_fun = ctx.vertex_array(self.p_fun, [(ctx.buffer(np.array(fv, "f4").tobytes()), "3f 2f", "p", "ah")], ctx.buffer(np.array(fi, "i4").tobytes()))
        dv, di_ = [(0.0, 0.0, 0.0)], []
        for s in range(41):
            a = s / 40 * 2 * math.pi
            dv.append((math.cos(a), 0.0, math.sin(a)))
        for s in range(40):
            di_ += [0, s + 1, s + 2]
        self.p_disc = ctx.program(vertex_shader=VS_DISC, fragment_shader=FS_DISC)
        self.vao_disc = ctx.vertex_array(self.p_disc, [(ctx.buffer(np.array(dv, "f4").tobytes()), "3f", "p")], ctx.buffer(np.array(di_, "i4").tobytes()))
        self.p_deb = ctx.program(vertex_shader=VS_DEB, fragment_shader=FS_DEB)
        self.vao_deb = ctx.vertex_array(self.p_deb, [(ctx.buffer(np.arange(1400, dtype="f4").tobytes()), "1f", "id")])
        self.p_bar = ctx.program(vertex_shader=VS_BAR, fragment_shader=FS_BAR)
        self.box_v, self.box_i = self._box()
        self.vao_bar = ctx.vertex_array(self.p_bar, [(ctx.buffer(self.box_v.tobytes()), "3f 3f", "p", "n")], ctx.buffer(self.box_i.tobytes()))
        self.p_patch = ctx.program(vertex_shader=VS_PATCH, fragment_shader=FS_PATCH)
        self.p_bld = ctx.program(vertex_shader=VS_BLD, fragment_shader=FS_BLD)
        self.local = None
        self.msaa = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), samples=msaa)], depth_attachment=ctx.depth_renderbuffer((W, H), samples=msaa))
        self.out = ctx.simple_framebuffer((W, H))

    def _tex(self, img, mip=False, repeat=False):
        mgl = self.mgl
        t = self.ctx.texture(img.size, len(img.getbands()), img.tobytes())
        if mip:
            t.build_mipmaps()
            t.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            t.anisotropy = 8.0
        t.repeat_x = t.repeat_y = repeat
        return t

    def _box(self):
        bv, bi = [], []
        faces = [((0, 0, 1), [(-.5, 0, .5), (.5, 0, .5), (.5, 1, .5), (-.5, 1, .5)]), ((0, 0, -1), [(.5, 0, -.5), (-.5, 0, -.5), (-.5, 1, -.5), (.5, 1, -.5)]),
                 ((1, 0, 0), [(.5, 0, .5), (.5, 0, -.5), (.5, 1, -.5), (.5, 1, .5)]), ((-1, 0, 0), [(-.5, 0, -.5), (-.5, 0, .5), (-.5, 1, .5), (-.5, 1, -.5)]),
                 ((0, 1, 0), [(-.5, 1, .5), (.5, 1, .5), (.5, 1, -.5), (-.5, 1, -.5)])]
        for n, q in faces:
            b0 = len(bv)
            for pt in q:
                bv.append((*pt, *n))
            bi += [b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3]
        return np.array(bv, "f4"), np.array(bi, "i4")

    # -- local close-up resources (one at a time)
    def use_local(self, key):
        if self.local is not None and self.local["key"] == key:
            return self.local
        if self.local is not None:
            for k in ("tex", "vao_patch", "vao_bld"):
                if self.local.get(k) is not None:
                    self.local[k].release()
        L = Local(key)
        ctx = self.ctx
        n, m = 96, 54
        uu, vv = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m))
        lon = L.lon0 + uu * (L.lon1 - L.lon0)
        lat = L.lat1 - vv * (L.lat1 - L.lat0)
        x, z = FR.world(lon, lat)
        y = height_at(lon, lat) + 2e-6
        verts = np.dstack([x, y, z, uu, vv]).astype("f4").reshape(-1, 5)
        i = np.arange(m - 1)[:, None] * n + np.arange(n - 1)[None, :]
        idx = np.stack([i, i + 1, i + n, i + 1, i + n + 1, i + n], -1).astype("i4").ravel()
        vao_patch = ctx.vertex_array(self.p_patch, [(ctx.buffer(verts.tobytes()), "3f 2f", "p", "t")], ctx.buffer(idx.tobytes()))
        tex = self._tex(L.img, mip=True)
        vao_bld, nb = None, 0
        if L.bld is not None and len(L.bld["lat"]):
            b = L.bld
            bx, bz = FR.world(b["lon"], b["lat"])
            # rotation: building angle is measured in local east/north metres; world z points south
            inst = np.stack([bx, bz, b["w"] / M_PER_UNIT, b["d"] / M_PER_UNIT], 1).astype("f4")
            inst2 = np.stack([-b["ang"], b["h"] / M_PER_UNIT], 1).astype("f4")
            base = height_at(b["lon"], b["lat"]).astype("f4")
            self.bld_base = float(np.median(base))
            data = np.concatenate([inst, inst2], 1).astype("f4")
            vao_bld = ctx.vertex_array(self.p_bld, [(ctx.buffer(self.box_v.tobytes()), "3f 3f", "p", "n"),
                                                    (ctx.buffer(data.tobytes()), "4f 2f/i", "ia", "ib")], ctx.buffer(self.box_i.tobytes()))
            nb = len(data)
        self.local = dict(key=key, L=L, tex=tex, vao_patch=vao_patch, vao_bld=vao_bld, nb=nb)
        return self.local

    def set_overlay(self, img: Image.Image):
        self.t_ovl.write(img.tobytes())
        self.t_ovl.build_mipmaps()

    def render(self, cam, tm=0.0, dim=1.0, ovl_alpha=1.0, vortices=(), bars=(), local=None, lights=0.25, clouds=1.0):
        """local: dict(key, alpha, swath=(a_uv, b_uv, hw_uv, along_uv), bpath=(pa_xz, pb_xz, hw_world, along_world), bscale)"""
        ctx, mgl = self.ctx, self.mgl
        M = cam.mvp(self.W, self.H)
        mvp = M.T.copy().tobytes()
        p, yw = math.radians(min(cam.pitch, 89.0)), math.radians(cam.yaw)
        tgt = np.array([cam.tx, cam.ty, cam.tz]) + np.array(cam.shake)
        eye = tgt + cam.dist * np.array([math.sin(yw) * math.cos(p), math.sin(p), math.cos(yw) * math.cos(p)])
        common = dict(tm=float(tm), eye=tuple(float(v) for v in eye), camdist=float(cam.dist), wind=(0.0021, 0.0009), noise=3)
        self.msaa.use()
        ctx.clear(0.8, 0.89, 0.97)
        # sky
        ctx.disable(mgl.DEPTH_TEST)
        self.vao_sky.render(mgl.TRIANGLE_STRIP)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.BLEND)
        self.t_sat.use(0)
        self.t_aux.use(1)
        self.t_ovl.use(2)
        self.t_noise.use(3)
        pt = self.p_ter
        pt["mvp"].write(mvp)
        for k, v in dict(sat=0, aux=1, ovl=2, dim=float(dim), ovl_alpha=float(ovl_alpha), lights_amt=float(lights), cloud_amt=float(clouds), **common).items():
            _set(pt, k, v)
        self.vao_ter.render()
        # local close-up patch + buildings
        if local is not None and local.get("alpha", 0) > 0.01:
            R = self.use_local(local["key"])
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            ctx.depth_func = "<="
            R["tex"].use(4)
            pp = self.p_patch
            pp["mvp"].write(mvp)
            a_uv, b_uv, hw_uv, al_uv = local["swath"]
            asp = (R["L"].lon1 - R["L"].lon0) * math.cos(math.radians((R["L"].lat0 + R["L"].lat1) / 2)) / (R["L"].lat1 - R["L"].lat0)
            for k, v in dict(img=4, alpha=float(local["alpha"]), sa=tuple(a_uv), sb=tuple(b_uv), hw=float(hw_uv), along=float(al_uv), aspect=float(asp), **common).items():
                _set(pp, k, v)
            self.msaa.depth_mask = False
            ctx.disable(mgl.DEPTH_TEST)
            R["vao_patch"].render()
            ctx.enable(mgl.DEPTH_TEST)
            self.msaa.depth_mask = True
            ctx.depth_func = "<"
            if R["vao_bld"] is not None:
                pb = self.p_bld
                pb["mvp"].write(mvp)
                pa_, pb_, hw_w, al_w = local["bpath"]
                for k, v in dict(hscale=float(local.get("bscale", 2.5)), pa=tuple(pa_), pb=tuple(pb_), hw=float(hw_w), along=float(al_w),
                                 alpha=float(local["alpha"]), base_y=float(self.bld_base), **common).items():
                    _set(pb, k, v)
                ctx.disable(mgl.BLEND)
                R["vao_bld"].render(instances=R["nb"])
        # bars
        if bars:
            ctx.disable(mgl.BLEND)
            for (x, y, z, w, h, col) in bars:
                self.p_bar["mvp"].write(mvp)
                self.p_bar["base"].value = (x, y, z)
                self.p_bar["size"].value = (w, h)
                self.p_bar["col"].value = tuple(c / 255 for c in col)
                self.vao_bar.render()
        # clouds
        if clouds > 0.01:
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            self.msaa.depth_mask = False
            pc = self.p_cloud
            pc["mvp"].write(mvp)
            for k, v in dict(cloud_amt=float(clouds) * float(np.clip((cam.dist - 0.25) / 0.6, 0, 1)), **common).items():
                _set(pc, k, v)
            self.vao_cloud.render(mgl.TRIANGLE_STRIP)
            self.msaa.depth_mask = True
        # vortices: dust disc, debris, funnel, wall cloud
        if vortices:
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            ctx.enable(mgl.PROGRAM_POINT_SIZE)
            for (x, y, z, sz, al) in vortices:
                self.msaa.depth_mask = False
                for off_y, scale, colr, a in ((0.002 * sz, 0.55, (0.45, 0.36, 0.27), 0.55), (sz * 1.02, 1.1, (0.38, 0.39, 0.43), 0.75)):
                    self.p_disc["mvp"].write(mvp)
                    self.p_disc["off"].value = (x, y + off_y, z)
                    self.p_disc["sc"].value = sz * scale
                    self.p_disc["alpha"].value = al * a
                    self.p_disc["col"].value = colr
                    self.vao_disc.render()
                self.msaa.depth_mask = True
                for k, v in dict(off=(x, y, z), sc=sz, tm=float(tm), alpha=float(al), psize=float(max(1.5, 3.2 * self.H / 1080))).items():
                    _set(self.p_deb, k, v)
                self.p_deb["mvp"].write(mvp)
                self.vao_deb.render(mgl.POINTS)
                self.p_fun["mvp"].write(mvp)
                self.p_fun["off"].value = (x, y, z)
                self.p_fun["sc"].value = sz
                _set(self.p_fun, "tm", float(tm))
                self.p_fun["alpha"].value = al
                self.vao_fun.render()
        ctx.copy_framebuffer(self.out, self.msaa)
        raw = self.out.read(components=3)
        return Image.frombytes("RGB", (self.W, self.H), raw).transpose(Image.FLIP_TOP_BOTTOM)
