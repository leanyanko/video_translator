#!/usr/bin/env python
"""Transcribe each SOLO speaker stem (multitrack mode) with word timestamps.

Model: mlx-community/whisper-large-v3-turbo (MLX build, auto-downloaded).

Usage: venv/bin/python scripts/transcribe_tracks.py work/<id>
           [--language en] [--chunk-secs 1200]

An ALTERNATIVE to transcribe.py + diarize.py when the editor's per-track
stems exist (work/<id>/tracks/ + tracks.json, see RU_INSTRUCTIONS §3б):
the speaker is known from the track, so no diarization is needed, and
words hidden under cross-talk in the mix are read clean off the solo
stem. The mix flow stays available for episodes without stems.

A solo stem is mostly silence, so only its voiced islands are
transcribed (streamed RMS scan; islands are voiced regions merged across
gaps ≤ ISLAND_GAP and capped at --chunk-secs — whisper degrades on very
long passes). Writes words_tracks.json (words carry "speaker") and
segments_tracks_raw.json (whisper's raw sentences, for review —
check it for hallucinations on breath/noise islands).
"""
import json
import subprocess
import sys
from pathlib import Path

import mlx_whisper
import numpy as np

MODEL = "mlx-community/whisper-large-v3-turbo"
FRAME = 0.25          # energy-scan frame, sec
RMS_SILENCE = 0.0056  # ~-45 dBFS: below = the stem is silent here
ISLAND_GAP = 8.0      # sec; voiced regions closer than this share an island
PAD = 0.4             # sec of context kept around an island
MIN_ISLAND = 0.5      # sec; shorter islands are breath/clicks, skipped

work = Path(sys.argv[1])
chunk_secs = float(sys.argv[sys.argv.index("--chunk-secs") + 1]) if "--chunk-secs" in sys.argv else 1200.0
language = sys.argv[sys.argv.index("--language") + 1] if "--language" in sys.argv else None

tracks_meta = json.loads((work / "tracks.json").read_text())
tmp = work / "tmp"
tmp.mkdir(exist_ok=True)


def voiced_islands(path: Path) -> list[tuple[float, float]]:
    """Streamed RMS scan → padded voiced islands, each ≤ chunk_secs."""
    proc = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", "16000", "-",
    ], stdout=subprocess.PIPE)
    frame_n = int(FRAME * 16000)
    regions, t, cur = [], 0.0, None
    buf = b""
    while True:
        chunk = proc.stdout.read(frame_n * 4 * 64)
        if not chunk:
            break
        buf += chunk
        n = len(buf) // (frame_n * 4)
        if not n:
            continue
        data = np.frombuffer(buf[: n * frame_n * 4], dtype=np.float32)
        buf = buf[n * frame_n * 4:]
        rms = np.sqrt((data.reshape(n, frame_n) ** 2).mean(axis=1))
        for r in rms:
            if r > RMS_SILENCE:
                if cur is None:
                    cur = t
            elif cur is not None:
                regions.append((cur, t))
                cur = None
            t += FRAME
    if cur is not None:
        regions.append((cur, t))
    proc.wait()
    regions = [(a, b) for a, b in regions if b - a >= MIN_ISLAND]
    # merge into islands; a new island also starts when the current one
    # would exceed chunk_secs (the break then falls in a real pause)
    islands = []
    for a, b in regions:
        if islands and a - islands[-1][1] <= ISLAND_GAP \
                and b - islands[-1][0] <= chunk_secs:
            islands[-1][1] = b
        else:
            islands.append([a, b])
    return [(max(0.0, a - PAD), b + PAD) for a, b in islands]


all_words, raw_segments = [], []
for spk, fname in tracks_meta["speakers"].items():
    track = work / "tracks" / fname
    islands = voiced_islands(track)
    total = sum(b - a for a, b in islands)
    print(f"{spk} ({fname}): {len(islands)} islands, {total / 60:.1f} min",
          flush=True)
    # detect the language once, on the longest island of the first track
    if language is None and islands:
        pass  # mlx_whisper detects per call; the first call below decides
    for a, b in sorted(islands, key=lambda ab: ab[0]):
        clip = tmp / "track_island.wav"
        subprocess.check_call([
            "ffmpeg", "-y", "-v", "quiet", "-ss", f"{a:.3f}",
            "-t", f"{b - a:.3f}", "-i", str(track),
            "-ac", "1", "-ar", "16000", str(clip)])
        r = mlx_whisper.transcribe(
            str(clip), path_or_hf_repo=MODEL, word_timestamps=True,
            **({"language": language} if language else {}))
        language = language or r.get("language")
        for s in r["segments"]:
            if s["text"].strip():
                raw_segments.append({
                    "start": round(s["start"] + a, 3),
                    "end": round(s["end"] + a, 3),
                    "speaker": spk, "text": s["text"].strip()})
            for w in s.get("words", []):
                all_words.append({
                    "start": round(w["start"] + a, 3),
                    "end": round(w["end"] + a, 3),
                    "word": w["word"], "speaker": spk})
        clip.unlink(missing_ok=True)
    print(f"  {sum(1 for w in all_words if w['speaker'] == spk)} words",
          flush=True)

all_words.sort(key=lambda w: (w["start"], w["end"]))
raw_segments.sort(key=lambda s: (s["start"], s["end"]))
(work / "words_tracks.json").write_text(
    json.dumps({"language": language, "words": all_words}, ensure_ascii=False))
(work / "segments_tracks_raw.json").write_text(
    json.dumps({"language": language, "segments": raw_segments},
               ensure_ascii=False, indent=1))
print(f"language={language} words={len(all_words)} "
      f"raw_segments={len(raw_segments)}")
