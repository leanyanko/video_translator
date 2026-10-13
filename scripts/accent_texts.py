#!/usr/bin/env python
"""Pre-compute RUAccent stress marks for all translated segments.

Model: RUAccent (Den4ikAI), omograph model "turbo" + dictionary;
auto-downloaded on first run. The library's turbo omograph model used
to fail on ~9% of segments ("Required inputs (['token_type_ids']) are
missing"): its ONNX graph requires token_type_ids but the wrapper's
odd-hypotheses code path doesn't pass it. Patched below by feeding
zeros for any declared-but-missing token_type_ids (correct for a
single-segment BERT pair encode); those sentences now get full marks
instead of falling back to unstressed text.

Usage: venv-f5/bin/python scripts/accent_texts.py work/<id>
Writes accents.json ({segment id: stressed text}). Run BEFORE
synthesize_f5.py — RUAccent and F5 must not share a process.
"""
import json
import sys
from pathlib import Path

import numpy as np
from onnxruntime import InferenceSession

_orig_run = InferenceSession.run

def _run_with_token_type_ids(self, output_names, input_feed, *args, **kwargs):
    try:
        need = {i.name for i in self.get_inputs()}
        if ("token_type_ids" in need and "token_type_ids" not in input_feed
                and "input_ids" in input_feed):
            input_feed = {**input_feed,
                          "token_type_ids": np.zeros_like(input_feed["input_ids"])}
    except Exception:
        pass
    return _orig_run(self, output_names, input_feed, *args, **kwargs)

InferenceSession.run = _run_with_token_type_ids

from ruaccent import RUAccent  # noqa: E402

work = Path(sys.argv[1])
segments = json.loads((work / "segments_ru.json").read_text())["segments"]
segments = [s for s in segments if s.get("type") != "orig"]  # orig = no text to mark

accent = RUAccent()
accent.load(omograph_model_size="turbo", use_dictionary=True)

def reapply_translator_marks(marked_src: str, accented: str) -> str:
    """Translator-placed stress marks are law: any word carrying '+' in
    text_ru overrides whatever RUAccent chose for that word."""
    import re

    # keys are ё-normalized on both sides: RUAccent ёfies as it marks
    # («самое» → «само+ё», «еще» → «ещё»), which used to make the word
    # miss its override and silently drop the translator's mark (s6)
    key = lambda w: w.replace("+", "").replace("ё", "е").replace("Ё", "Е").lower()  # noqa: E731
    overrides = {}
    for token in marked_src.split():
        core = token.strip(".,!?…;:—()«»\"'")
        if "+" in core:
            overrides[key(core)] = core
    if not overrides:
        return accented
    def sub(m):
        return overrides.get(key(m.group(0)), m.group(0))
    return re.sub(r"[\w+ёЁ-]+", sub, accented)


from stress_rules import force_name_stress, apply_homographs  # noqa: E402

out, failed = {}, []
for s in segments:
    plain = s["text_ru"].replace("+", "")  # RUAccent gets unmarked input
    try:
        accented = accent.process_all(plain)
    except Exception as e:
        failed.append((s["id"], str(e)[:80]))
        accented = plain  # list/translator marks still applied below
    # priority (low → high): RUAccent < name/homograph lists < translator.
    # The names dictionary is FORCED over RUAccent's own marks (it also
    # respells: гельбарт → Г+ельберт); homographs only fill words whose
    # English cue is present, keyed on the plain text, applied as
    # overrides so RUAccent keeps the rest.
    accented = force_name_stress(accented)
    homographed = apply_homographs(plain, s.get("text"))
    accented = reapply_translator_marks(homographed, accented)
    out[str(s["id"])] = reapply_translator_marks(s["text_ru"], accented)
# keep the record of what already-synthesized clips were SUNG from:
# rotate the old file to accents.prev.json — but only when the output
# actually changed (a no-op rerun must not bury the record under a
# fresh mtime; see manifest_bridge.rotate_accents)
from manifest_bridge import rotate_accents  # noqa: E402
rotate_accents(work, out)
(work / "accents.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"{len(out)} segments stress-marked, {len(failed)} failed")
for sid, err in failed:
    print(f"  seg {sid}: {err}")
