#!/usr/bin/env python
"""One-command dubbing pipeline — BOTH modes, with consistency gates.

Model: orchestrator only — each stage's model is listed in that script.

Usage:
  venv/bin/python scripts/run_pipeline.py "<youtube-url>" [--limit-secs N]
  venv/bin/python scripts/run_pipeline.py --work work/<id> [--limit-secs N]
      (skip download; operate on an existing work dir)

The mode is chosen by the work dir itself:
  MULTITRACK — work/<id>/tracks/ + tracks.json exist (the editor's
      per-speaker stems, RU_INSTRUCTIONS §3б). No diarization: speakers
      come from the stems, cross-talk from timeline geometry.
  MIX — otherwise (YouTube audio only): diarize + sentence segmentation.

Runs every stage that has not produced its output yet (idempotent and
resumable — rerun the same command after an interruption and it picks
up where it stopped). scripts/check_pipeline.py (the doctor) runs
before synthesis and before assembly and ABORTS the run on failures:
stale stress marks, dirty reference clips or stale synthesized audio
can therefore not reach the final video silently. The doctor can always
be run by hand, too.

MIX stages:         download → diarize → transcribe → segment_sentences
MULTITRACK stages:  download → clean_stems (mute other voices bleeding
                    into each mic) → transcribe_tracks → segment_tracks

Shared tail (both modes):
  * TRANSLATION gate → segments_ru.json   [NOT automated: an LLM or a
      human translates per RU_INSTRUCTIONS.md §«Правила перевода» and
      applies chunks with scripts/apply_translation.py — then RERUN
      this command. scripts/carry_over.py pre-fills from an older run.]
  accent_texts → accents.json             (stress marks; venv-f5)
  make_refs    → refs_f5/                 (--tracks in multitrack mode)
  DOCTOR — abort on failures
  synthesize_f5 → tts_f5/ + manifest.json (long; via synth_supervisor.sh,
      which restarts after the random MPS deaths; edited texts/accents/
      refs re-synthesize automatically via the manifest)
  qc_tts        → qc_tts.json             (back-transcription audit;
      flagged clips are reported, fix with synthesize_f5 --only ... )
  harvest_laughs (multitrack only)        (clean laughs off the stems)
  DOCTOR — abort on failures
  assemble      → video_ru.mp4            (--music <stem> in multitrack)
  h264 encode   → video_ru_h264.mp4       (QuickTime-compatible)

--limit-secs N processes only the first N seconds (preview, mix mode).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "venv" / "bin" / "python"
PY_F5 = ROOT / "venv-f5" / "bin" / "python"
SCRIPTS = ROOT / "scripts"


import time

TIMINGS = []  # (stage, seconds) for this invocation


def run(title: str, cmd: list) -> None:
    print(f"\n=== {title}")
    t0 = time.time()
    subprocess.check_call([str(c) for c in cmd], cwd=ROOT)
    dt = time.time() - t0
    TIMINGS.append((title, dt))
    print(f"    ({dt:.0f}s)")


def save_timings(work) -> None:
    """Append this invocation's stage timings to work/<id>/timings.json —
    the baseline benchmark needs wall-clock per stage, and the pipeline
    is re-entered several times (translation gate), so entries
    accumulate per invocation."""
    path = work / "timings.json"
    data = json.loads(path.read_text()) if path.exists() else []
    import datetime
    data.append({
        "invoked": datetime.datetime.now().isoformat(timespec="seconds"),
        "stages": [{"stage": s, "secs": round(d, 1)} for s, d in TIMINGS],
        "total_secs": round(sum(d for _, d in TIMINGS), 1),
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    if TIMINGS:
        print("\nstage timings (this invocation):")
        for s, d in TIMINGS:
            print(f"  {d/60:7.1f} min  {s}")
        print(f"  {sum(d for _, d in TIMINGS)/60:7.1f} min  TOTAL "
              f"(cumulative log: {path})")


def doctor(work: Path, when: str) -> None:
    print(f"\n=== doctor ({when})")
    r = subprocess.run([str(PY), str(SCRIPTS / "check_pipeline.py"), str(work)],
                       cwd=ROOT)
    if r.returncode != 0:
        save_timings(work)
        sys.exit(f"ABORT: the doctor found failures {when} — fix them and "
                 "rerun this same command (details above).")


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
        if not (work / "audio.wav").exists() and not (work / "tracks.json").exists():
            sys.exit(f"no audio.wav and no tracks.json in {work}")
    else:
        # 1. download (idempotent: download.py skips existing files)
        out = subprocess.check_output([str(PY), str(SCRIPTS / "download.py"), url],
                                      cwd=ROOT, text=True)
        print(out.strip())
        work = Path(out.strip().splitlines()[-1].split("WORKDIR=", 1)[1])

    multitrack = (work / "tracks.json").exists()
    print(f"work dir: {work}  mode: {'MULTITRACK' if multitrack else 'MIX'}")

    if multitrack:
        # stems: mute other voices bleeding into each mic, then swap the
        # cleaned set in (originals are kept as tracks_raw/)
        if not (work / "tracks_raw.json").exists():
            run("clean stems (mute cross-talk bleed)",
                [PY, SCRIPTS / "clean_stems.py", work])
            (work / "tracks").rename(work / "tracks_raw")
            (work / "tracks.json").rename(work / "tracks_raw.json")
            (work / "tracks_clean").rename(work / "tracks")
            (work / "tracks_clean.json").rename(work / "tracks.json")
            # clean_stems links the music stem by its ORIGINAL path, which
            # the rename above just moved — re-point the dangling symlink
            meta = json.loads((work / "tracks.json").read_text())
            m = meta.get("music")
            if m:
                p = work / "tracks" / m
                if p.is_symlink() and not p.exists():
                    p.unlink()
                    p.symlink_to((work / "tracks_raw" / m).resolve())
            print("tracks/ now holds the cleaned stems (originals in tracks_raw/)")
        if not (work / "words_tracks.json").exists():
            run("transcribe each stem (word timestamps)",
                [PY, SCRIPTS / "transcribe_tracks.py", work, "--language", "en"])
        if not (work / "segments.json").exists():
            run("segment per speaker (sentences + cross-talk geometry)",
                [PY, SCRIPTS / "segment_tracks.py", work])
    else:
        if not (work / "diarization.json").exists():
            run("diarize", [PY, SCRIPTS / "diarize.py", work])
        if not (work / "words.json").exists():
            # after diarize: chunked transcription snaps cuts to pauses
            run("transcribe (word timestamps, chunked)",
                [PY, SCRIPTS / "transcribe.py", work] + limit)
        if not (work / "segments.json").exists():
            run("segment into sentences",
                [PY, SCRIPTS / "segment_sentences.py", work] + limit)

    # translation gate: everything below needs a complete segments_ru.json
    base = json.loads((work / "segments.json").read_text())["segments"]
    n_speech = sum(1 for s in base if s.get("type") != "orig")
    ru_path = work / "segments_ru.json"
    n_ru = (
        sum(1 for s in json.loads(ru_path.read_text())["segments"]
            if s.get("type") != "orig" and s.get("text_ru"))
        if ru_path.exists()
        else 0
    )
    if n_ru < n_speech:
        print(f"\n=== STOP: translation needed ({n_ru}/{n_speech} segments done)")
        print("Translate per RU_INSTRUCTIONS.md («Правила перевода»: budget")
        print("≤13 chars per second of the segment's own span), apply chunks:")
        print(f"  venv/bin/python scripts/apply_translation.py {work} <chunk.json>")
        print("(scripts/carry_over.py pre-fills matches from an earlier run.)")
        print("Then RERUN this same command to continue.")
        save_timings(work)
        sys.exit(2)

    # accents must be re-run whenever the translation changed; the cheap
    # way to know is the same staleness rule the doctor uses, so just
    # re-run accent_texts when the doctor would flag it
    need_accents = not (work / "accents.json").exists()
    if not need_accents:
        import re
        sys.path.insert(0, str(SCRIPTS))
        from stress_rules import force_name_stress
        acc = json.loads((work / "accents.json").read_text())
        from manifest_bridge import norm_text
        norm = lambda t: norm_text(t, respell=force_name_stress)  # noqa: E731
        need_accents = any(
            s.get("type") != "orig" and s.get("text_ru")
            and norm(acc.get(str(s["id"]))) != norm(s["text_ru"])
            for s in json.loads(ru_path.read_text())["segments"])
    if need_accents:
        run("stress marks (RUAccent + dictionaries, separate process)",
            [PY_F5, SCRIPTS / "accent_texts.py", work])
    if not (work / "refs_f5").is_dir():
        run("voice references" + (" (from solo stems)" if multitrack else ""),
            [PY, SCRIPTS / "make_refs.py", work, "--max-secs", "11.5",
             "--outdir", "refs_f5"] + (["--tracks"] if multitrack else []))

    doctor(work, "before synthesis")

    print(f"\n=== synthesize (long: hours; resumable, auto-restarted; "
          f"manifest re-does edited segments)")
    run("synthesize_f5 via supervisor",
        ["zsh", SCRIPTS / "synth_supervisor.sh", work])

    run("qc: back-transcribe every clip",
        [PY, SCRIPTS / "qc_tts.py", work])

    if multitrack:
        sys.path.insert(0, str(SCRIPTS))
        from cli_args import opt as _opt
        restore = _opt("--restore-sounds", "laughs")
        mode_file = work / "laughs" / ".mode"
        prev_mode = mode_file.read_text().strip() if mode_file.exists() else None
        if restore != "none" and prev_mode != restore:
            run(f"restore non-speech sounds off the raw stems ({restore})",
                [PY, SCRIPTS / "harvest_laughs.py", work,
                 "--restore-sounds", restore])
            mode_file.parent.mkdir(exist_ok=True)
            mode_file.write_text(restore)

    doctor(work, "before assembly")

    music = []
    if multitrack:
        meta = json.loads((work / "tracks.json").read_text())
        if meta.get("music"):
            music = ["--music", str(work / "tracks" / meta["music"])]
    run("assemble", [PY, SCRIPTS / "assemble.py", work,
                     "--tts-dir", "tts_f5", "--out", "video_ru.mp4"] + music)
    run("H.264 encode", ["ffmpeg", "-y", "-v", "error",
                         "-i", work / "video_ru.mp4",
                         "-c:v", "h264_videotoolbox", "-b:v", "6000k",
                         "-c:a", "copy", "-movflags", "+faststart",
                         work / "video_ru_h264.mp4"])
    save_timings(work)
    print(f"\nDONE: {work}/video_ru_h264.mp4 (watch) and video_ru.mp4 (upload)")
    print("Flagged clips from the qc step (if any) are listed in qc_tts.json —")
    print("fix with synthesize_f5.py --only <ids> --seed 7 and rerun this command.")


if __name__ == "__main__":
    main()
