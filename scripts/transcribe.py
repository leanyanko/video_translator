#!/usr/bin/env python
"""Transcribe work/<id>/audio.wav with mlx-whisper (word timestamps).

Model: mlx-community/whisper-large-v3-turbo (Whisper, MLX build for
Apple Silicon; auto-downloaded from HF on first run).

Usage: venv/bin/python scripts/transcribe.py work/<id>
           [--limit-secs N] [--chunk-secs 1200]

Long audio is transcribed in CHUNKS (default 20 min): whisper's decoding
degrades on very long single passes (observed: punctuation disappears
after ~90 min, which breaks sentence segmentation downstream). Chunk
boundaries are snapped to speech pauses so no word is ever cut:
diarization gaps when diarization.json exists (run diarize.py first for
best cuts), otherwise the quietest moment near the nominal boundary.
Writes words.json (+ segments_raw.json for the legacy flow).
"""
import json
import subprocess
import sys
from pathlib import Path

import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"
SEARCH = 60.0  # seconds around the nominal boundary to look for a pause

work = Path(sys.argv[1])
limit_secs = float(sys.argv[sys.argv.index("--limit-secs") + 1]) if "--limit-secs" in sys.argv else None
chunk_secs = float(sys.argv[sys.argv.index("--chunk-secs") + 1]) if "--chunk-secs" in sys.argv else 1200.0

audio = work / "audio.wav"
probe = subprocess.check_output(
    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
     "-of", "csv=p=0", str(audio)], text=True)
total = float(probe.strip())
if limit_secs:
    total = min(total, limit_secs)


def pause_near(t: float) -> float:
    """Snap a nominal boundary to the middle of a nearby speech pause."""
    diar = work / "diarization.json"
    if diar.exists():
        turns = sorted(json.loads(diar.read_text()), key=lambda x: x["start"])
        best, best_d = None, None
        prev = 0.0
        for turn in turns:
            if turn["start"] - prev >= 0.3:  # a gap
                mid = (prev + turn["start"]) / 2
                d = abs(mid - t)
                if d <= SEARCH and (best_d is None or d < best_d):
                    best, best_d = mid, d
            prev = max(prev, turn["end"])
        if best is not None:
            return best
    # fallback: quietest 300ms window within ±SEARCH
    import numpy as np
    import soundfile as sf
    with sf.SoundFile(audio) as f:
        sr = f.samplerate
        lo = max(0.0, t - SEARCH)
        f.seek(int(lo * sr))
        data = f.read(int(2 * SEARCH * sr))
        if data.ndim > 1:
            data = data.mean(axis=1)
    hop = int(0.3 * sr)
    n = len(data) // hop
    rms = ((data[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) ** 0.5
    return lo + (int(rms.argmin()) + 0.5) * hop / sr


# chunk boundaries snapped to pauses
bounds = [0.0]
t = chunk_secs
while t < total - SEARCH:
    bounds.append(round(pause_near(t), 3))
    t = bounds[-1] + chunk_secs
bounds.append(total)

language = None
words, segments = [], []
for a, b in zip(bounds, bounds[1:]):
    clip = work / f"chunk_{int(a)}.wav"
    subprocess.check_call(["ffmpeg", "-y", "-v", "quiet", "-ss", str(a),
                           "-t", str(b - a), "-i", str(audio), str(clip)])
    r = mlx_whisper.transcribe(str(clip), path_or_hf_repo=MODEL, word_timestamps=True)
    language = language or r.get("language")
    for s in r["segments"]:
        if s["text"].strip():
            segments.append({"start": round(s["start"] + a, 3),
                             "end": round(s["end"] + a, 3),
                             "text": s["text"].strip()})
        for w in s.get("words", []):
            words.append({"start": round(w["start"] + a, 3),
                          "end": round(w["end"] + a, 3), "word": w["word"]})
    clip.unlink()
    enders = sum(1 for s_ in segments if s_["start"] >= a
                 and s_["text"].rstrip().endswith((".", "!", "?", "…")))
    print(f"chunk {a:.0f}-{b:.0f}s: {len(words)} words so far", flush=True)

(work / "segments_raw.json").write_text(
    json.dumps({"language": language, "segments": segments}, ensure_ascii=False, indent=1))
(work / "words.json").write_text(
    json.dumps({"language": language, "words": words}, ensure_ascii=False))
print(f"language={language} chunks={len(bounds)-1} segments={len(segments)} words={len(words)}")
