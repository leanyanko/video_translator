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

# similarity thresholds — one home for qc_tts (flags) and reroll_qc (re-sing),
# so the two cannot drift (s24). SIM_REROLL: on traps 10-13 every bad
# rendering of line 9 scored 0.59-0.80, every good one 1.00 (s25 class).
SIM_MIN = 0.75      # below: MISMATCH flag
SIM_REROLL = 0.9    # below (or any flag): re-sing up the ladder

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


# Latin letters as Russian says them: whisper writes «CD», the script
# spells «си-ди» (the way the translator must write acronyms for F5)
_LATIN_LETTER = {"a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф",
                 "g": "джи", "h": "эйч", "i": "ай", "j": "джей", "k": "кей",
                 "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью",
                 "r": "ар", "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл ю",
                 "x": "экс", "y": "уай", "z": "зед"}
# function words whisper drops on a one-word clip («А дальше?» → «дальше»)
_SHORT_STOP = {"а", "в", "и", "о", "у", "с", "к", "но", "то", "же", "ну", "вот", "так"}
SHORT_SCRIPT_WORDS = 3


def words(t: str, raw: str | None = None) -> list:
    """Comparison word list: stress marks stripped, ё→е, lowercase,
    digit runs expanded into Russian number words, short Latin acronyms
    spelled letter by letter («CD» → «си ди»; `raw` is the original-case
    text, needed to tell an acronym from a Latin word)."""
    src = raw if raw is not None else (t or "")
    t = (t or "").replace("+", "").lower().replace("ё", "е")
    acronyms = {m.lower() for m in re.findall(r"\b[A-Z]{2,5}\b", src)}
    out = []
    for tok in re.findall(r"[а-яa-z]+|\d+", t):
        if tok.isdigit():
            out += ru_num_words(int(tok))
        elif tok in acronyms:
            out += " ".join(_LATIN_LETTER.get(c, c) for c in tok).split()
        else:
            out.append(tok)
    return out


def stem(w: str) -> str:
    """Crude Russian stem: a word of 6+ letters minus its last 3 — enough
    to pair inflections («крайние/крайний», «компенсируют/компенсирует»);
    shorter words are compared whole. QC leniency only: used as the
    'best of' alongside the exact match, so it can never lower a score."""
    return w[:-3] if len(w) >= 6 and re.fullmatch(r"[а-я]+", w) else w


def _ratio(a: list, b: list) -> float:
    if not a:
        return 0.0
    exact = SequenceMatcher(None, a, b).ratio()
    stemmed = SequenceMatcher(None, [stem(x) for x in a], [stem(x) for x in b]).ratio()
    return max(exact, stemmed)


def script_sim(heard: str, sung: str, plain: str):
    """(similarity, heard_words, sung_words) — similarity is the best of
    heard-vs-sung and heard-vs-plain, each the best of exact and
    stem-level word matching; on a script of <= SHORT_SCRIPT_WORDS words
    the function words whisper drops on tiny clips are ignored on both
    sides. sung_words sizes the EXTRA check (that is the text the clip
    was synthesized from)."""
    heard_w, sung_w = words(heard, heard), words(sung, sung)
    plain_w = words(plain, plain)
    variants = [sung_w] + ([plain_w] if plain_w and plain_w != sung_w else [])
    sim = max((_ratio(v, heard_w) for v in variants), default=0.0)
    if sung_w and len(sung_w) <= SHORT_SCRIPT_WORDS:
        hs = [w for w in heard_w if w not in _SHORT_STOP]
        for v in variants:
            vs = [w for w in v if w not in _SHORT_STOP]
            if vs:
                sim = max(sim, _ratio(vs, hs))
    return sim, heard_w, sung_w
