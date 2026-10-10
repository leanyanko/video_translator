#!/usr/bin/env python
"""Text comparison for the TTS back-transcription QC — pure functions.

Model: none. Used by qc_tts.py; unit-testable without Whisper.

The QC must compare what Whisper HEARD against what was actually SUNG:
the accents.json text, where the dictionary deliberately respells names
(«Гельбарта»→«Гельберта»). Comparing against text_ru flagged every such
correct clip MISMATCH forever (TODO #6) — and any "re-sing the flagged"
automation would re-roll correct clips in a loop. The plain text_ru is
kept as a second variant (best of the two counts), so a dir with no
accents still QCs.

Numerals: Whisper often writes digits («в 1960 году») while the script
spells words («в тысяча девятьсот шестидесятом году»). words() expands
digit tokens into Russian cardinal words so «20» ↔ «двадцать» matches
exactly; an inflected form still differs by a suffix but costs one word
of similarity instead of a guaranteed token mismatch.
"""
import re
from difflib import SequenceMatcher

_UNITS = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь",
          "восемь", "девять"]
_UNITS_F = ["", "одна", "две"] + _UNITS[3:]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать",
          "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
          "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят",
         "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
             "шестьсот", "семьсот", "восемьсот", "девятьсот"]


def _plural(k: int, forms: tuple) -> str:
    k %= 100
    if 11 <= k <= 14:
        return forms[2]
    k %= 10
    if k == 1:
        return forms[0]
    if 2 <= k <= 4:
        return forms[1]
    return forms[2]


def _triple(k: int, fem: bool = False) -> list:
    out = []
    h, r = divmod(k, 100)
    if h:
        out.append(_HUNDREDS[h])
    if 10 <= r < 20:
        out.append(_TEENS[r - 10])
    else:
        t, u = divmod(r, 10)
        if t:
            out.append(_TENS[t])
        if u:
            out.append((_UNITS_F if fem else _UNITS)[u])
    return out


def ru_num_words(n: int) -> list:
    """Russian cardinal (nominative) for a non-negative integer, as a
    word list: 1960 → [тысяча, девятьсот, шестьдесят]."""
    if n == 0:
        return ["ноль"]
    out = []
    for div, forms, fem in (
            (10**9, ("миллиард", "миллиарда", "миллиардов"), False),
            (10**6, ("миллион", "миллиона", "миллионов"), False),
            (10**3, ("тысяча", "тысячи", "тысяч"), True)):
        k, n = divmod(n, div)
        if k:
            words = _triple(k, fem)
            # «одна тысяча X» is spoken «тысяча X» — drop the bare one
            if not (div == 10**3 and words == ["одна"]):
                out += words
            out.append(_plural(k, forms))
    out += _triple(n)
    return out


def words(t: str) -> list:
    """Comparison word list: stress marks stripped, ё→е, lowercase,
    digit runs expanded into Russian number words."""
    t = (t or "").replace("+", "").lower().replace("ё", "е")
    out = []
    for tok in re.findall(r"[а-яa-z]+|\d+", t):
        if tok.isdigit():
            out += ru_num_words(int(tok))
        else:
            out.append(tok)
    return out


def script_sim(heard: str, sung: str, plain: str):
    """(similarity, heard_words, sung_words) — similarity is the best of
    heard-vs-sung and heard-vs-plain; sung_words sizes the EXTRA check
    (that is the text the clip was synthesized from)."""
    heard_w, sung_w = words(heard), words(sung)
    sim = SequenceMatcher(None, sung_w, heard_w).ratio() if sung_w else 0.0
    plain_w = words(plain)
    if plain_w and plain_w != sung_w:
        sim = max(sim, SequenceMatcher(None, plain_w, heard_w).ratio())
    return sim, heard_w, sung_w
