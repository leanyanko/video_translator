#!/usr/bin/env python
"""Tier A overlap marking: restore true simultaneity of interruptions.

Model: none (arithmetic over diarization.json vs segments.json).

Usage: venv/bin/python scripts/mark_overlaps.py work/<id> [--min-lead 0.4]

For every segment, finds the diarization turn of the SAME speaker that
covers the segment's start. If that turn begins ≥ min-lead seconds
earlier than whisper's segment start AND the lead interval lies inside a
DIFFERENT speaker's segment, the line is an interruption that whisper
sequentialized. The script then:
  - moves the segment start back to the turn's true start,
  - marks it  "overlap": true, "pan": +0.35, "gain_db": -2,
  - marks the interrupted segment "pan": -0.35
    (budgets are span-based, so its translation is already full).
Prints every marked pair for review. Idempotent.
"""
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
min_lead = float(sys.argv[sys.argv.index("--min-lead") + 1]) if "--min-lead" in sys.argv else 0.4
MAX_LEAD = 6.0  # longer "leads" are usually diarization glitches

turns = sorted(json.loads((work / "diarization.json").read_text()), key=lambda t: t["start"])
data = json.loads((work / "segments.json").read_text())
segments = data["segments"]

marked = 0
for i, seg in enumerate(segments):
    if seg.get("type") == "orig" or seg.get("overlap"):
        continue
    # the same-speaker turn that covers (or starts just before) this segment
    cand = [t for t in turns
            if t["speaker"] == seg["speaker"]
            and t["start"] < seg["start"] + 0.2
            and t["end"] > seg["start"] + 0.2]
    if not cand:
        continue
    turn = max(cand, key=lambda t: t["start"])
    lead = seg["start"] - turn["start"]
    if not (min_lead <= lead <= MAX_LEAD):
        continue
    # the lead must fall inside a different speaker's segment
    prev = next((p for p in segments
                 if p is not seg and p.get("type") != "orig"
                 and p["speaker"] != seg["speaker"]
                 and p["start"] < turn["start"] < p["end"]), None)
    if prev is None:
        continue
    seg["orig_start"] = seg["start"]
    seg["start"] = round(turn["start"], 3)
    seg["overlap"] = True
    seg["pan"] = 0.35
    seg["gain_db"] = -2
    prev["pan"] = -0.35
    marked += 1
    print(f"  OVERLAP {seg['id']:4d} {seg['speaker'][-2:]} true start "
          f"{seg['start']:8.1f} (was {seg['orig_start']:.1f}, lead {lead:.1f}s) "
          f"over seg {prev['id']} | {seg['text'][:50]}")

data["segments"] = segments
(work / "segments.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
print(f"{marked} interruptions marked as true overlaps")
