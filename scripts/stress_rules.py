#!/usr/bin/env python
"""Automatic stress resolution from consultable data files.

Model: none (pure text rules; used by accent_texts.py).

Two sources, applied to the Russian text of every segment:
  data/stress_names.json  - foreign names / rare words with known stress;
      stems, so all inflected forms are covered (Ванкувере, Ванкувером...)
  data/homographs.json    - identically spelled words whose stress depends
      on meaning; disambiguated by cue words in the segment's ORIGINAL
      (English) text. No cue -> the word is left untouched (RUAccent
      decides), so the lists can only improve, never degrade.

Priority in accent_texts.py: translator '+' marks > these lists > RUAccent.
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
WORD_RE = re.compile(r"[\w+ёЁ-]+")


def _load(name: str) -> dict:
    d = json.loads((DATA / name).read_text())
    d.pop("_comment", None)
    return d


NAMES = _load("stress_names.json")
# longest stems first so e.g. "ванкувер" beats a hypothetical "ван"
NAME_STEMS = sorted(NAMES, key=len, reverse=True)
HOMOGRAPHS = _load("homographs.json")


def _restress(original_token: str, marked_stem: str, stem_len: int) -> str:
    """Apply a marked stem to a token, preserving its suffix and case."""
    suffix = original_token[stem_len:]
    if original_token[:1].islower():
        marked_stem = marked_stem[:1].lower() + marked_stem[1:]
    return marked_stem + suffix


def apply_name_stress(text_ru: str) -> str:
    def sub(m):
        token = m.group(0)
        if "+" in token:  # already marked (translator or earlier rule)
            return token
        low = token.lower()
        for stem in NAME_STEMS:
            if low.startswith(stem):
                return _restress(token, NAMES[stem], len(stem))
        return token

    return WORD_RE.sub(sub, text_ru)


def force_name_stress(accented_text: str) -> str:
    """Apply the names dictionary OVER already-accented text.

    apply_name_stress skips words that carry a '+' — correct for
    translator marks, but RUAccent marks everything, so on RUAccent
    output the dictionary silently lost every fight (seen: RUAccent's
    «Гельб+арта» surviving over the dictionary's «Г+ельберта»;
    respelled stems never matched at all because the word in the text
    keeps the original spelling). Here the token's own marks are
    stripped before stem matching and a dictionary hit always wins.
    Translator marks are reapplied AFTER this in accent_texts.py, so
    they still have the last word.
    """
    def sub(m):
        token = m.group(0)
        plain = token.replace("+", "")
        low = plain.lower()
        for stem in NAME_STEMS:
            if low.startswith(stem):
                return _restress(plain, NAMES[stem], len(stem))
        return token

    return WORD_RE.sub(sub, accented_text)


def apply_homographs(text_ru: str, text_en: str | None) -> str:
    en = (text_en or "").lower()

    def sub(m):
        token = m.group(0)
        if "+" in token:
            return token
        variants = HOMOGRAPHS.get(token.lower())
        if not variants:
            return token
        for v in variants:
            if any(cue in en for cue in v["en"] if cue):
                marked = v["marked"]
                return marked[:1].upper() + marked[1:] if token[:1].isupper() else marked
        return token  # no cue in the original -> let RUAccent decide

    return WORD_RE.sub(sub, text_ru)


def resolve(text_ru: str, text_en: str | None) -> str:
    return apply_homographs(apply_name_stress(text_ru), text_en)


def _mark_key(word: str) -> str:
    """Lookup key for a (possibly marked) word: no '+', ё folded to е,
    lowercase. RUAccent ёfies as it marks («самое» → «само+ё», «еще» →
    «ещё»), so without the fold a marked word misses its own override
    (s6)."""
    return word.replace("+", "").replace("ё", "е").replace("Ё", "Е").lower()


def reapply_translator_marks(marked_src: str, accented: str) -> str:
    """Stress marks placed upstream are law: any word carrying '+' in
    `marked_src` (translator text, or the dictionary pass) replaces
    RUAccent's choice for that word in `accented`."""
    overrides = {}
    for token in marked_src.split():
        core = token.strip(".,!?…;:—()«»\"'")
        if "+" in core:
            overrides[_mark_key(core)] = core
    if not overrides:
        return accented
    return re.sub(r"[\w+ёЁ-]+",
                  lambda m: overrides.get(_mark_key(m.group(0)), m.group(0)),
                  accented)
