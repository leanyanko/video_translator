#!/usr/bin/env python
"""Pre-compute RUAccent stress marks for all translated segments.

Usage: venv-f5/bin/python scripts/accent_texts.py work/<id>
Writes accents.json ({segment id: stressed text}). Run BEFORE
synthesize_f5.py — RUAccent and F5 must not share a process.
"""
import json
import sys
from pathlib import Path

from ruaccent import RUAccent

work = Path(sys.argv[1])
segments = json.loads((work / "segments_ru.json").read_text())["segments"]

accent = RUAccent()
accent.load(omograph_model_size="turbo", use_dictionary=True)

out, failed = {}, []
for s in segments:
    try:
        out[str(s["id"])] = accent.process_all(s["text_ru"])
    except Exception as e:
        failed.append((s["id"], str(e)[:80]))
(work / "accents.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"{len(out)} segments stress-marked, {len(failed)} failed")
for sid, err in failed:
    print(f"  seg {sid}: {err}")
