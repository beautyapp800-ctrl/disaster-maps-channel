"""Assemble the pilot: concat segments, mix CC0 music at -14 LUFS, verify, QA, preview, contact sheet.

python scripts/assemble.py <seg_dir> <out_dir> <expected_seconds> <expected_height>
"""
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

seg_dir, out_dir, exp_s, exp_h = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)


def sh(*a, **k):
    return subprocess.run(a, check=True, **k)


segs = sorted(glob.glob(str(seg_dir / "**" / "seg_*.mp4"), recursive=True))
print("segments:", len(segs))
lst = out_dir / "list.txt"
lst.write_text("".join(f"file '{Path(s).resolve().as_posix()}'\n" for s in segs))
video = out_dir / "video.mp4"
sh("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video))
D = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]).decode())
print("video seconds", D)

# ---- audio: 4 cinematic CC0 tracks (crossfaded) under procedural sound effects, then two-pass loudnorm to -14 LUFS
import urllib.request
tracks = json.load(open("data/music/tracks.json", encoding="utf-8"))
mdir = out_dir / "music"
mdir.mkdir(exist_ok=True)
files = []
for t in tracks["tracks"]:
    fn = mdir / (t["id"] + ".mp3")
    try:
        req = urllib.request.Request(t["url"], headers={"User-Agent": "disaster-maps-channel/1.0 (+https://github.com/beautyapp800-ctrl/disaster-maps-channel)"})
        fn.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        files.append(str(fn))
    except Exception as e:  # noqa: BLE001
        print("music download failed", t["id"], e)
if len(files) < 3:
    print("falling back to the two repo tracks")
    files = ["assets/music/oga-first-light-particles.mp3", "assets/music/oga-november-snow.mp3"] * 2
sfx = out_dir / "sfx.wav"
import os
HUR = os.environ.get("ENGINE") == "hurricane"
sh(sys.executable, "-m", "disasters.hsound" if HUR else "disasters.sound", str(sfx))
mix = out_dir / "mix.wav"
ins = []
for f in files:
    ins += ["-i", f]
chain = "[0:a]"
fc = []
for k in range(1, len(files)):
    fc.append(f"{chain}[{k}:a]acrossfade=d=6[x{k}]")
    chain = f"[x{k}]"
s_in = len(files)
fc.append(f"{chain}atrim=0:{D:.3f},afade=t=in:d=3,afade=t=out:st={D - 6:.3f}:d=6,aresample=48000,loudnorm=I=-22:TP=-4:LRA=11[m]")
fc.append(f"[{s_in}:a]aresample=48000,volume=0.85[s]")
fc.append("[m][s]amix=inputs=2:normalize=0[o]")
sh("ffmpeg", "-y", "-loglevel", "error", *ins, "-i", str(sfx), "-filter_complex", ";".join(fc), "-map", "[o]", "-ac", "2", str(mix))
r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(mix), "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"], capture_output=True, text=True)
j = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S).group(0))
print("pass1", j)
ln = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}"
      f":measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,alimiter=limit=0.708:level=disabled")
norm = out_dir / "mix_norm.wav"
sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(mix), "-af", ln, "-ar", "48000", str(norm))
final = out_dir / "final.mp4"
sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(norm), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
   "-shortest", "-movflags", "+faststart", str(final))

# ---- verify the delivered file
fails = []
r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(final), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
tail = r.stderr[r.stderr.rfind("Summary:"):]
I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", tail).group(1))
TP = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail).group(1))
probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate,duration", "-of", "json", str(final)]))
v = [s for s in probe["streams"] if s.get("width")][0]
print("loudness", I, "peak", TP, v)
if abs(I + 14.0) > 1.0:
    fails.append(f"loudness {I} LUFS is not -14 +-1")
if TP > -1.0:
    fails.append(f"true peak {TP} dBFS above -1")
if v["height"] != exp_h or v["r_frame_rate"] != "24/1":
    fails.append(f"video is {v['width']}x{v['height']} {v['r_frame_rate']}")
if abs(D - exp_s) > 0.6:
    fails.append(f"duration {D:.2f}s, expected {exp_s:.2f}s")

# ---- frame QA (pre-encode, 1 s grid) + data sync, independent of the renderer
rc = subprocess.run([sys.executable, "-m", "disasters.hqa" if HUR else "disasters.qa", str(seg_dir), str(out_dir / "qa_report.md")]).returncode
if rc:
    fails.append("frame QA failed (see qa_report.md)")

# ---- contact sheet + preview
th = sorted(glob.glob(str(seg_dir / "**" / "thumbs" / "t*.jpg"), recursive=True), key=lambda f: int(re.search(r"t(\d+)\.jpg", f).group(1)))
ims = [Image.open(f) for f in th]
if ims:
    cols = 5
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 640, rows * 360))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * 640, (i // cols) * 360))
    sheet.resize((cols * 400, rows * 225)).save(out_dir / "contact_sheet.jpg", quality=85)
sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(final), "-vf", "scale=854:480", "-c:v", "libx264", "-preset", "medium", "-crf", "29", "-c:a", "aac", "-b:a", "96k",
   "-movflags", "+faststart", str(out_dir / "preview_480p.mp4"))
(out_dir / "verify.json").write_text(json.dumps(dict(loudness_lufs=I, true_peak_dbfs=TP, duration_s=D, video=v, size_mb=round(final.stat().st_size / 1e6, 1), fails=fails), indent=1))
print(json.dumps(dict(loudness=I, peak=TP, dur=D, mb=round(final.stat().st_size / 1e6), fails=fails)))
sys.exit(1 if fails else 0)
