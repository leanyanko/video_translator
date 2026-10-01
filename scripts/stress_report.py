#!/usr/bin/env python
"""Stress QA report: find words that need dictionary attention.

Model: none (reads accents.json + RUAccent's bundled homograph list).

Usage: venv-f5/bin/python scripts/stress_report.py work/<id>

Reports, per episode:
  1. NAME CANDIDATES — capitalized words that ended up with NO stress
     mark after all three layers (RUAccent 3.2M-form dictionary, our
     data/ lists, translator marks). These are exactly the names every
     dictionary is missing; printed as ready-to-paste entries for
     data/stress_names.json.
  2. HOMOGRAPHS — occurrences of words from RUAccent's 19.7k homograph
     list, with the stress that was chosen and the English original, for
     targeted review. Recurring offenders belong in data/homographs.json
     with English cues.
"""
import gzip
import inspect
import json
import re
import sys
from collections import Counter
from pathlib import Path

import ruaccent

work = Path(sys.argv[1])
accents = json.loads((work / "accents.json").read_text())
segments = {str(s["id"]): s for s in json.loads((work / "segments_ru.json").read_text())["segments"]}

pkg = Path(inspect.getfile(ruaccent)).parent
omo_path = next(p for p in pkg.rglob("omographs.json*") if p.stat().st_size > 1000)
opener = gzip.open if omo_path.suffix == ".gz" else open
OMOGRAPHS = set(json.load(opener(omo_path, "rt")))

WORD = re.compile(r"[А-ЯЁа-яё+-]{3,}")

# a real name never occurs lowercase; sentence-initial common words do
seen_lower = set()
for text in accents.values():
    for token in WORD.findall(text):
        plain = token.replace("+", "")
        if plain[:1].islower():
            seen_lower.add(plain.lower())

names = Counter()
homo_hits = []
for sid, text in accents.items():
    seg = segments.get(sid, {})
    for token in WORD.findall(text):
        plain = token.replace("+", "")
        if ("+" not in token and plain[:1].isupper()
                and plain.lower() not in seen_lower):
            names[plain.lower()] += 1
        if plain.lower() in OMOGRAPHS:
            homo_hits.append((sid, token, seg.get("text", "")[:70]))

print("== 1. NAME CANDIDATES (unmarked capitalized words) ==")
print("   paste the right stress into data/stress_names.json:")
for word, n in names.most_common():
    if n >= 1:
        print(f'  "{word}": "{word}",   // seen {n}x — add + before stressed vowel')

print(f"\n== 2. HOMOGRAPH OCCURRENCES ({len(homo_hits)}) ==")
for sid, token, en in homo_hits:
    print(f"  seg {sid:>4}: {token:20s} | EN: {en}")
