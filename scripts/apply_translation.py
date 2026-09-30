#!/usr/bin/env python
"""Apply a translation map to a video's segments and save transcripts.

Usage: venv/bin/python scripts/apply_translation.py work/<id> <translations.json>

<translations.json> maps segment id → Russian text, e.g.
  {"0": "Когда делаешь гайки...", "1": "..."}
Merges into any existing segments_ru.json (so translation can be applied
in chunks as it is produced), preserves hand-added split segments whose
ids aren't in segments.json, writes segments_ru.json, and immediately
exports work/<id>/transcripts/ — transcripts are saved the moment the
translation lands, not as a separate scrape afterwards.
"""
import json
import sys
from pathlib import Path

from export_transcripts import export_work

work = Path(sys.argv[1])
translations = {
    str(k): v for k, v in json.loads(Path(sys.argv[2]).read_text()).items()
}

base = json.loads((work / "segments.json").read_text())
base_ids = {str(s["id"]) for s in base["segments"]}

unknown = set(translations) - base_ids
if unknown:
    sys.exit(f"ERROR: translation ids not in segments.json: {sorted(unknown)[:10]}")

ru_path = work / "segments_ru.json"
existing = {}
extra = []  # segments added by hand (splits) that aren't in segments.json
if ru_path.exists():
    for s in json.loads(ru_path.read_text())["segments"]:
        if str(s["id"]) not in base_ids:
            extra.append(s)
        elif "text_ru" in s:
            existing[str(s["id"])] = s["text_ru"]

existing.update(translations)

segments = [
    {**s, "text_ru": existing[str(s["id"])]}
    for s in base["segments"]
    if str(s["id"]) in existing
]
segments = sorted(segments + extra, key=lambda s: s["start"])
ru_path.write_text(
    json.dumps(
        {"language": "ru", "speakers": base.get("speakers"), "segments": segments},
        ensure_ascii=False,
        indent=1,
    )
)

missing = len(base_ids) - len(existing)
print(f"segments_ru.json: {len(segments)} segments"
      + (f", {missing} still untranslated" if missing else " (complete)"))
export_work(work)
