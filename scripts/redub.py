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

Only segments whose text, speaker, stress marks or reference clip changed
since the last synthesis are re-synthesized: synthesize_f5.py owns
tts_f5/manifest.json (sha1 of speaker|accented text|ref hash per clip)
and re-does every clip whose hash no longer matches, so editing a few
lines costs minutes, not hours. This script never writes the manifest
itself — the doctor (check_pipeline.py) and run_pipeline.py verify
clips against the same manifest, so there must be exactly one scheme.

Pre-manifest dirs migrate automatically and safely: accent_texts.py
keeps the previous stress file as accents.prev.json (the record of what
existing clips were SUNG from), and synthesize_f5.py bridges a missing
or old-scheme manifest from it (scripts/manifest_bridge.py) — so a
two-line edit re-synthesizes two clips, never the whole dir, and an
edited line can never be passed off as fresh. --adopt-existing remains
only for hand-copied clips with no record; --seed N is forwarded.

Then runs run_pipeline's shared tail (d10): QC, re-roll of flagged clips,
non-speech restore with the speech check, the doctor (aborts on
failures), assemble with the music stem, H.264 — and exports transcripts.
A re-voiced line passes every gate a fresh run does.
"""
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
    if Path(args[0]).is_dir():
        # in place (first positional is a work dir); any further args are
        # flags forwarded to synthesize_f5.py
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

    # Invalidation is synthesize_f5.py's job: it compares every existing
    # clip against tts_f5/manifest.json (speaker|accented text|ref hash)
    # and re-synthesizes the ones that no longer match. Keeping a second
    # hash scheme here used to overwrite that manifest after each redub,
    # which made the next synthesize/doctor run see EVERY clip as stale.
    synth_flags = []
    for flag in ("--adopt-existing",):
        if flag in args:
            synth_flags.append(flag)
    if "--seed" in args:
        synth_flags += ["--seed", args[args.index("--seed") + 1]]
    run("synthesize (missing/changed only, per tts_f5/manifest.json)",
        [PY_F5, SCRIPTS / "synthesize_f5.py", work] + synth_flags)

    # the SAME tail as run_pipeline (d10): QC, re-roll, non-speech restore
    # with the speech check, doctor, assemble with the music stem, H.264 —
    # a re-voiced line passes every gate a fresh run does
    sys.path.insert(0, str(SCRIPTS))
    from run_pipeline import tail
    tail(work, multitrack=(work / "tracks.json").exists(),
         reroll="--no-reroll" not in args)
    run("export transcripts", [PY, SCRIPTS / "export_transcripts.py", work])


if __name__ == "__main__":
    main()
