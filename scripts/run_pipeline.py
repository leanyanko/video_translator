#!/usr/bin/env python
"""One-command dubbing pipeline: give it a YouTube link, it does the rest.

Model: orchestrator only — each stage's model is listed in that script.

Usage:
  venv/bin/python scripts/run_pipeline.py "<youtube-url>" [--limit-secs N]
  venv/bin/python scripts/run_pipeline.py --work work/<id> [--limit-secs N]
      (skip download; operate on an existing work dir)

Runs every stage that has not produced its output yet (idempotent and
resumable — rerun the same command after an interruption and it picks up
where it stopped):

  1 download            → work/<id>/video.mp4 + audio.wav
  2 transcribe (words)  → words.json
  3 diarize             → diarization.json
  4 segment_sentences   → segments.json          [sentence-boundary cuts]
  * TRANSLATION         → segments_ru.json       [NOT automated: an LLM or
      human translates per RU_INSTRUCTIONS.md §«Правила перевода», applying
      chunks with scripts/apply_translation.py — then RERUN this command]
  5 accent_texts        → accents.json
  6 make_refs           → refs_f5/
  7 synthesize_f5       → tts_f5/seg_*.wav       (the long stage)
  8 assemble            → video_ru.mp4
  9 h264 encode         → video_ru_h264.mp4      (QuickTime-compatible)

--limit-secs N processes only the first N seconds (preview mode).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "venv" / "bin" / "python"
PY_F5 = ROOT / "venv-f5" / "bin" / "python"
SCRIPTS = ROOT / "scripts"


def run(title: str, cmd: list) -> None:
    print(f"\n=== {title}")
    subprocess.check_call([str(c) for c in cmd], cwd=ROOT)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    url = sys.argv[1]
    limit = (
        ["--limit-secs", sys.argv[sys.argv.index("--limit-secs") + 1]]
        if "--limit-secs" in sys.argv
        else []
    )

    if url == "--work":
        work = Path(sys.argv[2])
        if not (work / "audio.wav").exists():
            sys.exit(f"no audio.wav in {work}")
    else:
        # 1. download (idempotent: download.py skips existing files)
        out = subprocess.check_output([str(PY), str(SCRIPTS / "download.py"), url],
                                      cwd=ROOT, text=True)
        print(out.strip())
        work = Path(out.strip().splitlines()[-1].split("WORKDIR=", 1)[1])
    print(f"work dir: {work}")

    if not (work / "words.json").exists():
        run("2/9 transcribe (word timestamps)",
            [PY, SCRIPTS / "transcribe.py", work] + limit)
    if not (work / "diarization.json").exists():
        run("3/9 diarize", [PY, SCRIPTS / "diarize.py", work])
    if not (work / "segments.json").exists():
        run("4/9 segment into sentences",
            [PY, SCRIPTS / "segment_sentences.py", work] + limit)

    # translation gate: everything below needs a complete segments_ru.json
    n_segments = len(json.loads((work / "segments.json").read_text())["segments"])
    ru_path = work / "segments_ru.json"
    n_ru = (
        sum("text_ru" in s for s in json.loads(ru_path.read_text())["segments"])
        if ru_path.exists()
        else 0
    )
    if n_ru < n_segments:
        print(f"\n=== STOP: translation needed ({n_ru}/{n_segments} segments done)")
        print("Translate per RU_INSTRUCTIONS.md («Правила перевода»: ≤13 chars/sec")
        print("of slot, stress-mark homographs like в д+уше / ст+оит), apply chunks:")
        print(f"  venv/bin/python scripts/apply_translation.py {work} <chunk.json>")
        print("then RERUN this same command to continue.")
        sys.exit(2)

    if not (work / "accents.json").exists():
        run("5/9 stress marks (RUAccent, separate process)",
            [PY_F5, SCRIPTS / "accent_texts.py", work])
    if not (work / "refs_f5").is_dir():
        run("6/9 voice references",
            [PY, SCRIPTS / "make_refs.py", work, "--max-secs", "11.5",
             "--outdir", "refs_f5"])

    done = len(list((work / "tts_f5").glob("seg_*.wav"))) if (work / "tts_f5").is_dir() else 0
    if done < n_segments:
        print(f"\n=== 7/9 synthesize ({n_segments - done} segments to go; "
              f"resumable, ~19s each)")
        run("synthesize_f5", [PY_F5, SCRIPTS / "synthesize_f5.py", work])

    run("8/9 assemble", [PY, SCRIPTS / "assemble.py", work,
                         "--tts-dir", "tts_f5", "--out", "video_ru.mp4"])
    run("9/9 H.264 encode", ["ffmpeg", "-y", "-v", "error",
                             "-i", work / "video_ru.mp4",
                             "-c:v", "h264_videotoolbox", "-b:v", "6000k",
                             "-c:a", "copy", "-movflags", "+faststart",
                             work / "video_ru_h264.mp4"])
    print(f"\nDONE: {work}/video_ru_h264.mp4 (watch) and video_ru.mp4 (upload)")


if __name__ == "__main__":
    main()
