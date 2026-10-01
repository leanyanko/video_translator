#!/usr/bin/env python
"""Pre-compute RUAccent stress marks for all translated segments.

Model: RUAccent (Den4ikAI), omograph model "turbo" + dictionary;
auto-downloaded on first run. Known bug: ONNX input mismatch fails on
~9% of segments — those fall back to unstressed text.

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

def reapply_translator_marks(marked_src: str, accented: str) -> str:
    """Translator-placed stress marks are law: any word carrying '+' in
    text_ru overrides whatever RUAccent chose for that word."""
    import re

    overrides = {}
    for token in marked_src.split():
        core = token.strip(".,!?…;:—()«»\"'")
        if "+" in core:
            overrides[core.replace("+", "").lower()] = core
    if not overrides:
        return accented
    def sub(m):
        return overrides.get(m.group(0).replace("+", "").lower(), m.group(0))
    return re.sub(r"[\w+ёЁ-]+", sub, accented)


from stress_rules import resolve as resolve_lists  # noqa: E402

out, failed = {}, []
for s in segments:
    plain = s["text_ru"].replace("+", "")  # RUAccent gets unmarked input
    try:
        accented = accent.process_all(plain)
    except Exception as e:
        failed.append((s["id"], str(e)[:80]))
        accented = plain  # list/translator marks still applied below
    # priority (low → high): RUAccent < name/homograph lists < translator
    listed = resolve_lists(plain, s.get("text"))
    accented = reapply_translator_marks(listed, accented)
    out[str(s["id"])] = reapply_translator_marks(s["text_ru"], accented)
(work / "accents.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"{len(out)} segments stress-marked, {len(failed)} failed")
for sid, err in failed:
    print(f"  seg {sid}: {err}")
