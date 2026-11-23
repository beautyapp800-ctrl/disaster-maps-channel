"""Content ID test of the 4 CC0 music tracks (owner rule): a private test upload on the channel BEFORE the first publication.

    python -m automation.contentid run      # build a 3-4 min test video with 50 s of every track, upload it PRIVATE, watch it, record the verdict
    python -m automation.contentid status   # print state/contentid.json

The YouTube Data API does not expose copyright claims directly; the signals it does expose are checked:
  status.uploadStatus == rejected / rejectionReason (claim, copyright), contentDetails.licensedContent (claimed by a partner),
  contentDetails.regionRestriction (blocked by a claim), processing failure.
A track is "clear" only when the test video was fully processed, stayed clean for at least 30 minutes after processing and none of
those signals appeared. The owner can double-check in YouTube Studio > Content > the test video > Copyright.
The test video is deleted at the end. Result: state/contentid.json (read by automation/publish.py as a gate).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation import youtube  # noqa: E402

OUT = ROOT / "out" / "contentid"
STATE = ROOT / "state" / "contentid.json"
UA = "disaster-maps-channel/1.0 (+https://github.com/beautyapp800-ctrl/disaster-maps-channel)"
CLEAN_MIN = 30
MAX_MIN = 150


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_video(tracks):
    OUT.mkdir(parents=True, exist_ok=True)
    segs = []
    for t in tracks:
        mp3 = OUT / (t["id"] + ".mp3")
        if not mp3.exists():
            req = urllib.request.Request(t["url"], headers={"User-Agent": UA})
            mp3.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        seg = OUT / (t["id"] + "_seg.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "20", "-i", str(mp3), "-t", "50", "-ar", "44100", "-ac", "2", str(seg)], check=True)
        segs.append(seg)
    lst = OUT / "list.txt"
    lst.write_text("".join(f"file '{s.resolve().as_posix()}'\n" for s in segs))
    mp4 = OUT / "contentid_test.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x101830:s=640x360:r=1", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-shortest", "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(mp4)], check=True)
    return mp4


def signals(v):
    st, cd, pd = v.get("status", {}), v.get("contentDetails", {}), v.get("processingDetails", {})
    bad = []
    if st.get("uploadStatus") in ("rejected", "failed"):
        bad.append("uploadStatus " + str(st.get("uploadStatus")) + " " + str(st.get("rejectionReason", "")) + str(st.get("failureReason", "")))
    if st.get("rejectionReason"):
        bad.append("rejectionReason " + str(st["rejectionReason"]))
    if cd.get("licensedContent"):
        bad.append("licensedContent (claimed by a content owner)")
    if cd.get("regionRestriction"):
        bad.append("regionRestriction " + json.dumps(cd["regionRestriction"]))
    return bad, st.get("uploadStatus"), pd.get("processingStatus")


def run():
    tracks = json.loads((ROOT / "data" / "music" / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    yt = youtube.service()
    ch = youtube.assert_channel(yt, os.environ.get("YT_CHANNEL_ID", ""))
    print("channel ok:", ch["id"], ch["snippet"]["title"])
    videos = youtube.all_uploads(yt, [])
    if any(v["title"].startswith("Content ID test") for v in videos):
        print("a Content ID test video is already on the channel: not uploading another")
        return 3
    mp4 = build_video(tracks)
    from googleapiclient.http import MediaFileUpload
    lines = [f"{t['title']} by {t['creator']}: " + t["page"].replace("https://", "") for t in tracks]
    desc = "Private audio test of four CC0 tracks (OpenGameArt).\n" + "\n".join(lines)
    body = {"snippet": {"title": "Content ID test (private)", "description": desc, "categoryId": "10"},
            "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": False}}
    req = yt.videos().insert(part="snippet,status", body=body, media_body=MediaFileUpload(str(mp4), chunksize=4 * 1024 * 1024, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    vid = resp["id"]
    print("test video uploaded (private):", vid)
    rec = {"video_id": vid, "uploaded_at": now(), "tracks": {t["id"]: {"verdict": "pending"} for t in tracks}, "status": "pending"}
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    t0, processed_at, verdict, notes = time.time(), None, "pending", []
    while time.time() - t0 < MAX_MIN * 60:
        time.sleep(300)
        v = yt.videos().list(part="status,contentDetails,processingDetails", id=vid).execute()["items"]
        if not v:
            notes.append("test video disappeared (removed by YouTube?)")
            verdict = "claimed"
            break
        bad, up, proc = signals(v[0])
        print(now(), "upload=" + str(up), "processing=" + str(proc), "signals=" + str(bad), flush=True)
        if bad:
            verdict, notes = "claimed", bad
            break
        if up == "processed" and processed_at is None:
            processed_at = time.time()
        if processed_at and time.time() - processed_at >= CLEAN_MIN * 60:
            verdict = "clear"
            break
    rec["finished_at"] = now()
    rec["status"] = verdict
    rec["notes"] = notes
    for t in tracks:
        rec["tracks"][t["id"]] = {"verdict": verdict}
    STATE.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    if verdict != "pending":
        try:
            yt.videos().delete(id=vid).execute()
            rec["test_video_deleted"] = True
            STATE.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print("note: test video not deleted:", str(e)[:120])
    print("verdict:", verdict, notes)
    return 0 if verdict == "clear" else 2


if __name__ == "__main__":
    if sys.argv[1:] == ["status"]:
        print(STATE.read_text(encoding="utf-8") if STATE.exists() else "no test yet")
        sys.exit(0)
    sys.exit(run())
