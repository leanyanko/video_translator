#!/usr/bin/env python
"""Restore lost punctuation in whisper run-ons, keeping word timestamps.

Model: none locally — the punctuation itself is produced by an LLM (or a
human) from the exported text, like translations (apply_translation.py).

Usage:
  venv/bin/python scripts/repunctuate_words.py work/<id> export runons.json
      [--words words_tracks.json] [--min-secs 18]
  venv/bin/python scripts/repunctuate_words.py work/<id> apply fixed.json
      [--words words_tracks.json]

Whisper sometimes drops punctuation for minutes at a time. The sentence
segmenters then have no sentence ends to cut at, so run-ons longer than
MAX_SENT_SECS get split at their longest pause — often mid-sentence,
which makes the halves untranslatable on their own.

export: finds every per-speaker run of words spanning more than
--min-secs without a sentence-final mark and writes
  [{"key": "SPEAKER_00@886.01", "speaker": ..., "start": ..., "end": ...,
    "text": "<the run's words, as whisper wrote them>"}, ...]
apply: takes {"<key>": "<same text with punctuation/capitals restored>"}.
Only punctuation, capitalization and spacing may differ — the letters
and digits must match exactly, or that run is rejected. Punctuation is
attached to the word it follows, so every timestamp is kept. The words
file is backed up to <name>.bak.json before the first write.

Then re-run the segmenter (segment_tracks.py / segment_sentences.py).
"""
import json
import sys
from pathlib import Path

from cli_args import opt

from segmenting import GAP_BREAK, SENT_END  # same rules as the segmenters

work = Path(sys.argv[1])
mode, arg = sys.argv[2], Path(sys.argv[3])
words_name = opt("--words", "words_tracks.json")
min_secs = opt("--min-secs", 18.0, float)

words_path = work / words_name
data = json.loads(words_path.read_text())
words = data["words"]


def runs():
    """Yield (key, [indices into words]) of unpunctuated run-ons."""
    by_spk = {}
    for i, w in enumerate(words):
        by_spk.setdefault(w.get("speaker"), []).append(i)
    for spk, idx in by_spk.items():
        cur = []
        for i in idx + [None]:
            end_run = i is None or (cur and words[i]["start"] - words[cur[-1]]["end"] > GAP_BREAK)
            if end_run:
                if cur and words[cur[-1]]["end"] - words[cur[0]]["start"] > min_secs:
                    yield f"{spk}@{words[cur[0]]['start']:.2f}", cur
                cur = []
            if i is None:
                break
            cur.append(i)
            if words[i]["word"].strip().endswith(SENT_END):
                cur = []


def alnum(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


if mode == "export":
    out = []
    for key, idx in runs():
        out.append({"key": key, "speaker": words[idx[0]].get("speaker"),
                    "start": words[idx[0]]["start"], "end": words[idx[-1]]["end"],
                    "text": "".join(words[i]["word"] for i in idx).strip()})
    arg.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{len(out)} run-ons ({sum(r['end'] - r['start'] for r in out) / 60:.1f} min) -> {arg}")

elif mode == "apply":
    fixed = json.loads(arg.read_text())
    found = dict(runs())
    applied = 0
    for key, text in fixed.items():
        idx = found.get(key)
        if idx is None:
            print(f"SKIP {key}: no such run-on (already punctuated?)")
            continue
        orig = "".join(words[i]["word"] for i in idx)
        if alnum(orig) != alnum(text):
            print(f"REJECT {key}: letters/digits differ from whisper's words")
            continue
        new_words, pos = [], 0
        for i in idx:
            n = len(alnum(words[i]["word"]))
            if n == 0:  # pure-punctuation token: leave as is
                new_words.append(words[i]["word"])
                continue
            while pos < len(text) and text[pos].isspace():
                pos += 1
            start, seen = pos, 0
            while seen < n:
                if text[pos].isalnum():
                    seen += 1
                pos += 1
            # trailing punctuation belongs to this word
            while pos < len(text) and not text[pos].isspace() and not text[pos].isalnum():
                pos += 1
            lead = " " if words[i]["word"].startswith(" ") else ""
            new_words.append(lead + text[start:pos])
        for i, w in zip(idx, new_words):
            words[i]["word"] = w
        applied += 1
    backup = words_path.with_suffix(".bak.json")
    if applied and not backup.exists():
        backup.write_text(words_path.read_text())
    if applied:
        words_path.write_text(json.dumps(data, ensure_ascii=False))
    print(f"applied {applied}/{len(fixed)} run-ons to {words_path.name}")

else:
    sys.exit("mode must be export or apply")
