#!/usr/bin/env python
"""Carry translations and synthesized clips over from an earlier run.

Model: none (matching over segments files + file copies).

Usage: venv/bin/python scripts/carry_over.py work/<new> work/<old>
           [--out carry.json]

After re-segmenting (new ids), most segments are still identical to the
earlier run's. A segment matches when speaker and the English words are
the same (ignoring punctuation and case) and start/end are within ±TOL;
then:
  - its old text_ru goes into <out> (default work/<new>/carry.json),
    a translation map for apply_translation.py;
  - the old run's tts_f5/seg_<old id>.wav is copied to the new run's
    tts_f5/seg_<new id>.wav, so synthesize_f5.py (resumable: it skips
    existing files) only synthesizes what is new or changed.
Stress marks are not copied — rerun accent_texts.py on the new run.
"""
import json
import re
import shutil
import sys
from pathlib import Path

new, old = Path(sys.argv[1]), Path(sys.argv[2])
out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else new / "carry.json"

old_segs = [s for s in json.loads((old / "segments_ru.json").read_text())["segments"]
            if s.get("type") != "orig" and s.get("text_ru")]
TOL = 0.3  # sec; re-transcription (e.g. from cleaned stems) shifts word edges a bit


def words(t: str) -> str:
    """Text without punctuation/case: whisper re-runs differ in those."""
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", t.lower()).split())


index = {}
for s in old_segs:
    index.setdefault((s["speaker"], words(s["text"])), []).append(s)

new_segs = [s for s in json.loads((new / "segments.json").read_text())["segments"]
            if s.get("type") != "orig"]
carry, copied = {}, 0
(new / "tts_f5").mkdir(exist_ok=True)
for s in new_segs:
    for o in index.get((s["speaker"], words(s["text"])), ()):
        if abs(o["start"] - s["start"]) <= TOL and abs(o["end"] - s["end"]) <= TOL:
            carry[str(s["id"])] = o["text_ru"]
            src = old / "tts_f5" / f"seg_{o['id']:04d}.wav"
            dst = new / "tts_f5" / f"seg_{s['id']:04d}.wav"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
            break
out.write_text(json.dumps(carry, ensure_ascii=False, indent=0))
print(f"{len(carry)}/{len(new_segs)} segments carried over ({copied} clips copied) -> {out}; "
      f"{len(new_segs) - len(carry)} need translation + synthesis")
