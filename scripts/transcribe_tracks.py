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

from cli_args import opt

import mlx_whisper

from audio_utils import RMS_SILENCE, SCAN_FRAME as FRAME, merge_regions, scan_voiced_regions

MODEL = "mlx-community/whisper-large-v3-turbo"
ISLAND_GAP = 8.0      # sec; voiced regions closer than this share an island
PAD = 0.4             # sec of context kept around an island
MIN_ISLAND = 0.5      # sec; shorter islands are breath/clicks, skipped

work = Path(sys.argv[1])
chunk_secs = opt("--chunk-secs", 1200.0, float)
language = opt("--language", None)

tracks_meta = json.loads((work / "tracks.json").read_text())
tmp = work / "tmp"
tmp.mkdir(exist_ok=True)


def voiced_islands(path: Path) -> list[tuple[float, float]]:
    """Streamed RMS scan → padded voiced islands, each ≤ chunk_secs."""
    regions = [(a, b) for a, b in scan_voiced_regions(path, FRAME, RMS_SILENCE)
               if b - a >= MIN_ISLAND]
    # merge into islands; a new island also starts when the current one
    # would exceed chunk_secs (the break then falls in a real pause)
    islands = merge_regions(regions, ISLAND_GAP, chunk_secs)
    return [(max(0.0, a - PAD), b + PAD) for a, b in islands]


# The collapse detector, restart loop and stretched-word splice live in
# whisper_guard.py (shared with transcribe.py — the mix flow used to run
# unguarded, TODO #12); measurements and thresholds are documented there.
from whisper_guard import guarded_pass


def run_whisper(track: Path, a: float, b: float):
    """Transcribe [a, b) of the track → (segments, words), timeline-offset."""
    clip = tmp / "track_island.wav"
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "quiet", "-ss", f"{a:.3f}",
        "-t", f"{b - a:.3f}", "-i", str(track),
        "-ac", "1", "-ar", "16000", str(clip)])
    global language
    r = mlx_whisper.transcribe(
        str(clip), path_or_hf_repo=MODEL, word_timestamps=True,
        **({"language": language} if language else {}))
    clip.unlink(missing_ok=True)
    language = language or r.get("language")
    segs, ws = [], []
    for s in r["segments"]:
        if s["text"].strip():
            segs.append({"start": round(s["start"] + a, 3),
                         "end": round(s["end"] + a, 3), "text": s["text"].strip(),
                         "avg_logprob": round(s["avg_logprob"], 3)})
        for w in s.get("words", []):
            # probability: whisper's confidence — low on crosstalk/bleed
            # garbage, used by segment_tracks.py to drop or flag it
            ws.append({"start": round(w["start"] + a, 3),
                       "end": round(w["end"] + a, 3), "word": w["word"],
                       "prob": round(w["probability"], 3)})
    return segs, ws


all_words, raw_segments = [], []
for spk, fname in tracks_meta["speakers"].items():
    track = work / "tracks" / fname
    islands = voiced_islands(track)
    total = sum(b - a for a, b in islands)
    print(f"{spk} ({fname}): {len(islands)} islands, {total / 60:.1f} min",
          flush=True)
    for a, b in sorted(islands, key=lambda ab: ab[0]):
        segs, ws = guarded_pass(lambda x, y: run_whisper(track, x, y), a, b)
        raw_segments += [{**s, "speaker": spk} for s in segs]
        all_words += [{**w, "speaker": spk} for w in ws]
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
