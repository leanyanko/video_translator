#!/usr/bin/env python
"""Re-dub: turn a video + final Russian segmentation into a finished
voice-over, reusing speakers/references from a previous run.

Model: orchestrator only (RUAccent + F5-TTS via the stage scripts).

Usage:
  # in place — after editing segments_ru.json inside an existing run:
  venv/bin/python scripts/redub.py work/<id>

  # standalone — a video plus a segments_ru.json, voices from a prior run:
  venv/bin/python scripts/redub.py <video.mp4> <segments_ru.json> \
      --refs work/<previous-id>/refs_f5 [--out work/<new-name>]

Only segments whose text, speaker, or stress marks changed since the last
redub are re-synthesized (a manifest tracks content hashes), so editing a
few lines costs minutes, not hours. Produces video_ru.mp4 and
video_ru_h264.mp4, and exports transcripts.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "venv" / "bin" / "python"
PY_F5 = ROOT / "venv-f5" / "bin" / "python"
SCRIPTS = ROOT / "scripts"


def run(title: str, cmd: list) -> None:
    print(f"=== {title}")
    subprocess.check_call([str(c) for c in cmd], cwd=ROOT)


def setup_workdir(args: list) -> Path:
    if len(args) == 1 and Path(args[0]).is_dir():
        work = Path(args[0])
        for req in ("video.mp4", "segments_ru.json", "refs_f5"):
            if not (work / req).exists():
                sys.exit(f"{work} lacks {req}")
        return work

    video, seg_ru = Path(args[0]), Path(args[1])
    refs = Path(args[args.index("--refs") + 1]) if "--refs" in args else None
    out = Path(args[args.index("--out") + 1]) if "--out" in args else ROOT / "work" / f"redub_{video.stem}"
    if refs is None or not refs.is_dir():
        sys.exit("--refs <dir with SPEAKER_*.wav/.txt from a previous run> is required")

    out.mkdir(parents=True, exist_ok=True)
    if not (out / "video.mp4").exists():
        (out / "video.mp4").symlink_to(video.resolve())
    shutil.copyfile(seg_ru, out / "segments_ru.json")
    if not (out / "refs_f5").exists():
        shutil.copytree(refs, out / "refs_f5")
    return out


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    work = setup_workdir(args)
    print(f"work dir: {work}")

    if not (work / "audio.wav").exists():
        run("extract audio", ["ffmpeg", "-y", "-v", "quiet",
                              "-i", work / "video.mp4",
                              "-vn", "-ac", "1", "-ar", "44100",
                              work / "audio.wav"])

    segments = json.loads((work / "segments_ru.json").read_text())["segments"]
    segments = [s for s in segments if s.get("type") != "orig"]
    missing_refs = {
        s["speaker"] for s in segments
        if not (work / "refs_f5" / f"{s['speaker']}.wav").exists()
    }
    if missing_refs:
        sys.exit(f"refs_f5 lacks voices for: {sorted(missing_refs)} — "
                 "copy them from the previous run or rerun make_refs.py")

    run("stress marks", [PY_F5, SCRIPTS / "accent_texts.py", work])
    accents = json.loads((work / "accents.json").read_text())

    # invalidate clips whose content changed since the last redub
    tts = work / "tts_f5"
    tts.mkdir(exist_ok=True)
    manifest_path = tts / "manifest.json"
    # no manifest yet = first redub over this dir: existing clips are
    # taken as the baseline (only missing ones get synthesized)
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    new, stale = {}, []
    for s in segments:
        sid = str(s["id"])
        text = accents.get(sid, s["text_ru"])
        h = hashlib.sha1(f"{s['speaker']}|{text}".encode()).hexdigest()
        new[sid] = h
        wav = tts / f"seg_{s['id']:04d}.wav"
        if old is not None and wav.exists() and old.get(sid) != h:
            wav.unlink()
            stale.append(s["id"])
    if stale:
        print(f"=== {len(stale)} changed segments to re-synthesize: "
              f"{stale[:12]}{'...' if len(stale) > 12 else ''}")

    run("synthesize (missing/changed only)",
        [PY_F5, SCRIPTS / "synthesize_f5.py", work])
    manifest_path.write_text(json.dumps(new, indent=1))

    run("assemble", [PY, SCRIPTS / "assemble.py", work,
                     "--tts-dir", "tts_f5", "--out", "video_ru.mp4"])
    run("H.264 encode", ["ffmpeg", "-y", "-v", "error",
                         "-i", work / "video_ru.mp4",
                         "-c:v", "h264_videotoolbox", "-b:v", "6000k",
                         "-c:a", "copy", "-movflags", "+faststart",
                         work / "video_ru_h264.mp4"])
    run("export transcripts", [PY, SCRIPTS / "export_transcripts.py", work])
    print(f"\nDONE: {work}/video_ru_h264.mp4")


if __name__ == "__main__":
    main()
