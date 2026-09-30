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
result = mlx_whisper.transcribe(str(work / "audio.wav"), path_or_hf_repo=MODEL)

segments = [
    {"start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"].strip()}
    for s in result["segments"]
    if s["text"].strip()
]
out = {"language": result.get("language"), "segments": segments}
(work / "segments_raw.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"language={out['language']} segments={len(segments)}")
