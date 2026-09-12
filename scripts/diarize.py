#!/usr/bin/env python
"""Speaker diarization + speaker-aware segment merging.

Usage: venv/bin/python scripts/diarize.py work/<id> [max_speakers]
Runs pyannote/speaker-diarization-3.1 on audio.wav, tags each whisper
segment with the speaker who overlaps it most, then merges adjacent
same-speaker segments. Writes:
  diarization.json  - raw speaker turns
  segments.json     - final segments with "speaker" field
Needs HF access to pyannote/speaker-diarization-3.1 and
pyannote/segmentation-3.0 (accept conditions on their HF pages).
"""
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline

MAX_MERGED_SECS = 12.0
MAX_GAP_SECS = 0.4

work = Path(sys.argv[1])
max_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None

wav, sr = sf.read(work / "audio.wav", dtype="float32")
waveform = torch.from_numpy(wav).unsqueeze(0)  # (channel, time), mono input

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipeline.to(torch.device(device))
print(f"diarizing on {device}...")

kwargs = {"max_speakers": max_speakers} if max_speakers else {}
diarization = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
# pyannote 4.x wraps the Annotation in a DiarizeOutput
if hasattr(diarization, "speaker_diarization"):
    diarization = diarization.speaker_diarization

turns = [
    {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": spk}
    for turn, _, spk in diarization.itertracks(yield_label=True)
]
(work / "diarization.json").write_text(json.dumps(turns, indent=1))
speakers = sorted({t["speaker"] for t in turns})
print(f"{len(turns)} turns, {len(speakers)} speakers: {speakers}")

# assign each whisper segment the speaker with max time overlap
raw = json.loads((work / "segments_raw.json").read_text())


def best_speaker(start, end):
    overlap = {}
    for t in turns:
        ov = min(end, t["end"]) - max(start, t["start"])
        if ov > 0:
            overlap[t["speaker"]] = overlap.get(t["speaker"], 0) + ov
    return max(overlap, key=overlap.get) if overlap else speakers[0]


tagged = [
    {**s, "speaker": best_speaker(s["start"], s["end"])} for s in raw["segments"]
]

# merge adjacent segments only within the same speaker
merged = []
for s in tagged:
    if (
        merged
        and s["speaker"] == merged[-1]["speaker"]
        and s["start"] - merged[-1]["end"] <= MAX_GAP_SECS
        and s["end"] - merged[-1]["start"] <= MAX_MERGED_SECS
    ):
        merged[-1]["end"] = s["end"]
        merged[-1]["text"] += " " + s["text"]
    else:
        merged.append(dict(s))

segments = [{"id": i, **s} for i, s in enumerate(merged)]
out = {"language": raw.get("language"), "speakers": speakers, "segments": segments}
(work / "segments.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

counts = {spk: sum(1 for s in segments if s["speaker"] == spk) for spk in speakers}
print(f"final segments={len(segments)} per-speaker counts={counts}")
