#!/usr/bin/env python
"""Apply a translation map to a video's segments and save transcripts.

Model: none locally — the translations themselves are produced by an LLM
(or a human) following the timing-budget and stress-mark rules in
RU_INSTRUCTIONS.md, and passed in as a JSON chunk.

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

# words the TTS provably mispronounces regardless of stress marks —
# the translation must rephrase around them (data/forbidden_words.json)
forbidden_path = Path(__file__).resolve().parent.parent / "data" / "forbidden_words.json"
if forbidden_path.exists():
    forbidden = json.loads(forbidden_path.read_text())
    forbidden.pop("_comment", None)
    for sid, text in translations.items():
        for stem, advice in forbidden.items():
            if stem in text.lower():
                print(f"WARNING seg {sid}: '{stem}' — {advice}")

base = json.loads((work / "segments.json").read_text())
base_ids = {str(s["id"]) for s in base["segments"]}

unknown = set(translations) - base_ids
if unknown:
    sys.exit(f"ERROR: translation ids not in segments.json: {sorted(unknown)[:10]}")

ru_path = work / "segments_ru.json"
prior = {}  # full prior segment dicts — they may carry hand edits
extra = []  # segments added by hand (splits) that aren't in segments.json
if ru_path.exists():
    for s in json.loads(ru_path.read_text())["segments"]:
        if str(s["id"]) not in base_ids:
            extra.append(s)
        else:
            prior[str(s["id"])] = s

segments = []
for s in base["segments"]:
    sid = str(s["id"])
    # keep the prior dict verbatim (hand edits to end/text survive);
    # only fall back to the base segment for ids seen for the first time
    seg = dict(prior.get(sid, s))
    if sid in translations:
        seg["text_ru"] = translations[sid]
    # type:"orig" (nonverbal stem cuts from segment_tracks.py) carry no
    # text to translate but must reach assemble.py all the same
    if "text_ru" in seg or seg.get("type") == "orig":
        segments.append(seg)
translated_ids = {str(s["id"]) for s in segments}
segments = sorted(segments + extra, key=lambda s: s["start"])
# timing budget: the Russian must FILL the original speech (≈13 chars per
# second of end−start), not just fit under it — a squeezed line ends while
# the speaker's mouth still moves. Short lines (<1.5 s) are exempt.
CPS = 13.0
for seg in segments:
    sid = str(seg["id"])
    if sid not in translations or seg.get("type") == "orig":
        continue
    dur = seg["end"] - seg["start"]
    n = len(seg["text_ru"].replace("+", ""))
    if dur >= 1.5 and n < 0.8 * CPS * dur:
        print(f"SHORT  seg {sid}: {n} chars, budget {CPS * dur:.0f} "
              f"({n / (CPS * dur):.0%}) — translate fuller, keep false starts")
    elif n > 1.15 * CPS * dur + 10:
        print(f"LONG   seg {sid}: {n} chars, budget {CPS * dur:.0f} "
              f"({n / (CPS * dur):.0%})")

ru_path.write_text(
    json.dumps(
        {"language": "ru", "speakers": base.get("speakers"), "segments": segments},
        ensure_ascii=False,
        indent=1,
    )
)

missing = len(base_ids - translated_ids)
print(f"segments_ru.json: {len(segments)} segments"
      + (f", {missing} still untranslated" if missing else " (complete)"))
export_work(work)
