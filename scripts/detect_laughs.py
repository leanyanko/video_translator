#!/usr/bin/env python
"""Detect laughter/applause in the ORIGINAL audio and mark it for
restoration in the dub.

Model: PANNs Cnn14 (AudioSet tagging, CPU; checkpoint auto-downloads to
~/panns_data on first run).

Usage: venv/bin/python scripts/detect_laughs.py work/<id> [--threshold 0.15]

Candidate windows are the GAPS between diarization turns (nobody is
speaking there, so restoring original audio cannot collide with dubbed
speech — and diarization gaps are clean, unlike whisper's ragged edges).
Each window is tagged; windows where laughter/applause/giggle score above
the threshold become segments in segments_ru.json:

    {"id": 9001, "type": "orig", "start": ..., "end": ...,
     "label": "Laughter", "score": 0.42, "auto": true}

assemble.py pastes original audio for every type:"orig" segment (manual
entries welcome too — add one by hand for a jingle or an interjection).
Re-running replaces previous auto entries; hand-added ones (no "auto")
are kept. Laughs that overlap speech are NOT restorable this way.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from panns_inference import AudioTagging

MIN_GAP = 0.5    # sec; shorter gaps aren't worth restoring
PAD = 0.08       # sec; keep a hair away from neighbouring speech
TARGET_LABELS = ("Laughter", "Giggle", "Chuckle, chortle", "Snicker",
                 "Belly laugh", "Applause", "Cheering")

work = Path(sys.argv[1])
threshold = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 0.15

turns = json.loads((work / "diarization.json").read_text())
turns.sort(key=lambda t: t["start"])

audio, sr = sf.read(work / "audio.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
total = len(audio) / sr

ru = json.loads((work / "segments_ru.json").read_text())
dubbed_end = max((s["end"] for s in ru["segments"]), default=total)

gaps, prev = [], 0.0
for t in turns:
    if t["start"] - prev >= MIN_GAP:
        gaps.append((prev, t["start"]))
    prev = max(prev, t["end"])
if total - prev >= MIN_GAP:
    gaps.append((prev, total))
# only within the dubbed range (for partial/preview runs)
gaps = [(a, b) for a, b in gaps if a < dubbed_end]
print(f"{len(gaps)} silent-in-diarization gaps to classify")

tagger = AudioTagging(checkpoint_path=None, device="cpu")
labels = tagger.labels
idx = [i for i, l in enumerate(labels) if l in TARGET_LABELS]

found = []
for a, b in gaps:
    lo, hi = a + PAD, b - PAD
    if hi - lo < 0.3:
        continue
    clip = audio[int(lo * sr): int(hi * sr)]
    # PANNs expects 32 kHz
    import librosa
    clip32 = librosa.resample(clip, orig_sr=sr, target_sr=32000)
    scores = tagger.inference(clip32[None, :])[0][0]
    best_i = max(idx, key=lambda i: scores[i])
    if scores[best_i] >= threshold:
        found.append({"start": round(lo, 3), "end": round(hi, 3),
                      "label": labels[best_i], "score": round(float(scores[best_i]), 3)})
        print(f"  {lo:8.1f}-{hi:8.1f}  {labels[best_i]:12s} {scores[best_i]:.2f}")

# merge into segments_ru.json: drop previous auto entries, keep manual
segs = [s for s in ru["segments"] if not (s.get("type") == "orig" and s.get("auto"))]
next_id = 9001
for f in found:
    segs.append({"id": next_id, "type": "orig", "auto": True, **f})
    next_id += 1
ru["segments"] = sorted(segs, key=lambda s: s["start"])
(work / "segments_ru.json").write_text(json.dumps(ru, ensure_ascii=False, indent=1))
print(f"{len(found)} orig-audio segments written to segments_ru.json")
