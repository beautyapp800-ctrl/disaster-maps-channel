"""Upload the next reserve video(s) of Storm Atlas USA as PRIVATE videos with publishAt (Wed and Sat 20:00 Kyiv).

Run by .github/workflows/upload.yml (daily). Safe by design:
  - no token                      -> nothing happens (exit 0)
  - music Content ID test not clear (state/contentid.json) -> nothing is uploaded (exit 0)
  - wrong channel / empty YT_CHANNEL_ID / channel list not read in full / rate limit (Doubt) -> nothing uploaded, exit 3
  - the episode is already on the channel (marker "[ref: stormatlasusa/<slug>]" or same title) -> never uploaded again
  - one video per slot, slots are Wed/Sat 20:00 Europe/Kyiv, at least 30 min ahead
  - the state file (state/episodes.json) is committed right after every upload; if that fails the run stops
Reserve = DRAFT release `reserve-<slug>` (title starts with RESERVE), made by pilot.yml: final.mp4, cover.jpg, meta.json.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation import youtube  # noqa: E402
from disasters import meta as vmeta  # noqa: E402

BUFFER = int(os.environ.get("SA_BUFFER", "2"))          # videos kept scheduled ahead
MAX_PER_RUN = 2
STATE = ROOT / "state" / "episodes.json"
CONTENTID = ROOT / "state" / "contentid.json"
QUEUE = ROOT / "data" / "topic_queue.json"
DRY = os.environ.get("DRY_RUN", "") == "1"


def sh(*a, check=True):
    return subprocess.run(a, capture_output=True, text=True, check=check, cwd=ROOT).stdout


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []


def persist(state, msg):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if DRY:
        return
    sh("git", "config", "user.name", "atlas-bot")
    sh("git", "config", "user.email", "atlas-bot@users.noreply.github.com")
    sh("git", "add", "state/episodes.json")
    sh("git", "commit", "-m", msg)
    for _ in range(3):
        r = subprocess.run(["git", "push"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            return
        sh("git", "pull", "--rebase", "--autostash", check=False)
    raise RuntimeError("state commit could not be pushed: stop (a duplicate upload would be possible)")


def reserve_releases():
    out = json.loads(sh("gh", "release", "list", "--limit", "100", "--json", "tagName,name,isDraft") or "[]")
    return {r["tagName"][len("reserve-"):]: r for r in out if r["tagName"].startswith("reserve-") and r["name"].startswith("RESERVE")}


def _state_ahead(state):
    now = datetime.now(timezone.utc)
    return sum(1 for e in state if e.get("publish_at") and datetime.fromisoformat(e["publish_at"].replace("Z", "+00:00")) > now)


def music_clear():
    """owner rule: every music track went through a test private upload and a Content ID check before the first publication"""
    if not CONTENTID.exists():
        return False, "no Content ID test yet (state/contentid.json missing)"
    c = json.loads(CONTENTID.read_text(encoding="utf-8"))
    tracks = json.loads((ROOT / "data" / "music" / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    bad = [t["id"] for t in tracks if c.get("tracks", {}).get(t["id"], {}).get("verdict") != "clear"]
    return (not bad), ("tracks not clear: " + ", ".join(bad)) if bad else "all clear"


def candidates():
    q = json.loads(QUEUE.read_text(encoding="utf-8"))["topics"]
    return [t for t in q if not t["engine"].startswith("engine_pending")]


def main():
    if not os.environ.get("YT_TOKEN_JSON", "").strip():
        print("no YT_TOKEN_JSON: nothing to do (the token is not stored yet)")
        return 0
    ok, why = music_clear()
    if not ok:
        print(f"music Content ID gate: {why} - nothing uploaded")
        return 0
    yt = youtube.service()
    try:
        ch = youtube.assert_channel(yt, os.environ.get("YT_CHANNEL_ID", ""))
        print(f"channel ok: {ch['id']} {ch['snippet']['title']}")
        state = load_state()
        known = [e["youtube_id"] for e in state if e.get("youtube_id")]
        videos = youtube.all_uploads(yt, known)
    except youtube.Doubt as e:
        print(f"DOUBT - nothing uploaded: {e}")
        return 3
    print(f"the channel has {len(videos)} videos; scheduled ahead: {youtube.scheduled_ahead(videos)}")
    done = 0
    rel = reserve_releases()
    sim_taken = []
    for t in candidates():
        ahead = max(youtube.scheduled_ahead(videos), _state_ahead(state)) + len(sim_taken)
        if ahead >= BUFFER or done >= MAX_PER_RUN:
            break
        slug = t["slug"]
        if slug not in rel:
            continue
        work = ROOT / "out" / "upload" / slug
        work.mkdir(parents=True, exist_ok=True)
        sh("gh", "release", "download", f"reserve-{slug}", "-p", "meta.json", "-p", "cover.jpg", "-D", str(work), "--clobber")
        m = json.loads((work / "meta.json").read_text(encoding="utf-8"))
        pr = vmeta.problems(m)
        if pr or f"[ref: {vmeta.CHANNEL_REF}/{slug}]" not in m["description"]:
            print(f"{slug}: metadata problems {pr}: skipped")
            continue
        title, desc, tags = m["title"], m["description"], m["tags"]
        ex = youtube.find_existing(videos, slug, title)
        if ex:
            print(f"{slug}: already on the channel as {ex['id']} - not uploading again")
            if not any(e.get("youtube_id") == ex["id"] for e in state):
                state.append(dict(slug=slug, title=title, status="scheduled", youtube_id=ex["id"], publish_at=ex["publish_at"]))
                persist(state, f"state: {slug} found on the channel ({ex['id']})")
            continue
        taken = list(youtube.channel_dates(videos)) + [e["publish_at"] for e in state if e.get("publish_at")] + sim_taken
        slot = youtube.next_slot(taken)
        print(f"{slug}: slot {slot.isoformat()}")
        if DRY:
            print("DRY RUN: would upload", title, "\n", desc[:600])
            sim_taken.append(slot.strftime("%Y-%m-%dT%H:%M:%SZ"))
            done += 1
            continue
        sh("gh", "release", "download", f"reserve-{slug}", "-p", "final.mp4", "-D", str(work), "--clobber")
        vid = youtube.upload(yt, work / "final.mp4", title, desc, tags, slot)
        state.append(dict(slug=slug, title=title, status="scheduled", youtube_id=vid, publish_at=slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          uploaded_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        persist(state, f"state: {slug} uploaded {vid}, publishAt {slot.isoformat()}")
        try:
            youtube.set_thumbnail(yt, vid, work / "cover.jpg")
        except Exception as e:  # noqa: BLE001  (custom thumbnails need a phone-verified channel; never lose the state of an uploaded video)
            print("note: custom thumbnail skipped:", str(e)[:120])
        sh("gh", "release", "edit", f"reserve-{slug}", "--title", f"UPLOADED {vid}: {slug}", check=False)
        videos = youtube.all_uploads(yt, [e["youtube_id"] for e in state if e.get("youtube_id")])
        done += 1
        print(f"{slug}: uploaded {vid}")
    print(f"uploaded {done} video(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
