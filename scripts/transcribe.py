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
Each chunk also runs under whisper_guard.guarded_pass (TODO #12): a
collapse that starts INSIDE a chunk used to survive to the chunk's end
unnoticed — now the pass restarts at the last sentence end before it,
and stretched words (skipped speech) are re-transcribed and spliced,
the same protection the multitrack flow already had.
Writes words.json (+ segments_raw.json for the legacy flow).
"""
import json
import subprocess
import sys
from pathlib import Path

from cli_args import opt

import mlx_whisper

from audio_utils import probe_duration

MODEL = "mlx-community/whisper-large-v3-turbo"
SEARCH = 60.0  # seconds around the nominal boundary to look for a pause

work = Path(sys.argv[1])
limit_secs = opt("--limit-secs", None, float)
chunk_secs = opt("--chunk-secs", 1200.0, float)

audio = work / "audio.wav"
total = probe_duration(audio)
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

from whisper_guard import guarded_pass  # noqa: E402

tmp = work / "tmp"
tmp.mkdir(exist_ok=True)
language = None
words, segments = [], []


def run_whisper(a: float, b: float):
    """Transcribe [a, b) of the mix → (segments, words), timeline-offset."""
    clip = tmp / "mix_chunk.wav"
    subprocess.check_call(["ffmpeg", "-y", "-v", "quiet", "-ss", f"{a:.3f}",
                           "-t", f"{b - a:.3f}", "-i", str(audio),
                           "-ac", "1", "-ar", "16000", str(clip)])
    global language
    r = mlx_whisper.transcribe(str(clip), path_or_hf_repo=MODEL,
                               word_timestamps=True)
    clip.unlink(missing_ok=True)
    language = language or r.get("language")
    segs, ws = [], []
    for s in r["segments"]:
        if s["text"].strip():
            segs.append({"start": round(s["start"] + a, 3),
                         "end": round(s["end"] + a, 3),
                         "text": s["text"].strip()})
        for w in s.get("words", []):
            ws.append({"start": round(w["start"] + a, 3),
                       "end": round(w["end"] + a, 3), "word": w["word"]})
    return segs, ws


for a, b in zip(bounds, bounds[1:]):
    segs, ws = guarded_pass(run_whisper, a, b)
    segments += segs
    words += ws
    print(f"chunk {a:.0f}-{b:.0f}s: {len(words)} words so far", flush=True)

(work / "segments_raw.json").write_text(
    json.dumps({"language": language, "segments": segments}, ensure_ascii=False, indent=1))
(work / "words.json").write_text(
    json.dumps({"language": language, "words": words}, ensure_ascii=False))
print(f"language={language} chunks={len(bounds)-1} segments={len(segments)} words={len(words)}")
