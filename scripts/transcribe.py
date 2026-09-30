#!/usr/bin/env python
"""Transcribe work/<id>/audio.wav with mlx-whisper (segment timestamps).

Model: mlx-community/whisper-large-v3-turbo (Whisper, MLX build for
Apple Silicon; auto-downloaded from HF on first run).

Usage: venv/bin/python scripts/transcribe.py work/<id>
Writes segments_raw.json (unmerged; speaker assignment and merging
happen in diarize.py).
"""
import json
import sys
from pathlib import Path

import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"

work = Path(sys.argv[1])
limit_secs = float(sys.argv[sys.argv.index("--limit-secs") + 1]) if "--limit-secs" in sys.argv else None

audio = work / "audio.wav"
if limit_secs:
    import subprocess
    clip = work / f"audio_first{int(limit_secs)}s.wav"
    if not clip.exists():
        subprocess.check_call(["ffmpeg", "-y", "-v", "quiet", "-i", str(audio),
                               "-t", str(limit_secs), "-c", "copy", str(clip)])
    audio = clip

result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=MODEL, word_timestamps=True)

segments = [
    {"start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"].strip()}
    for s in result["segments"]
    if s["text"].strip()
]
words = [
    {"start": w["start"], "end": w["end"], "word": w["word"]}
    for s in result["segments"]
    for w in s.get("words", [])
]
out = {"language": result.get("language"), "segments": segments}
(work / "segments_raw.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
(work / "words.json").write_text(
    json.dumps({"language": result.get("language"), "words": words}, ensure_ascii=False)
)
print(f"language={out['language']} segments={len(segments)} words={len(words)}")
