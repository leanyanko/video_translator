#!/usr/bin/env python
"""Does a voice reference's TEXT match its AUDIO? (tracker s7)

Model: whisper, passed in as a callback — the rule itself is pure and
unit-tested; make_refs.py and check_pipeline.py call it.

F5 conditions on the reference text and audio as one aligned pair. Any
mismatch between them — a trailing word the audio does not contain, an
opening «yeah, yeah» the text omits — makes the model "finish" the
reference at the start of every generation: long lines open with the
reference's own words (QC LEAD), short lines come out silent. Measured
on baseline20 (10-13): one such reference gave 7 of 16 QC flags and
three silent 2–6-letter clips; the same speaker's verified reference
gave 0. So a reference is accepted only when its text matches the
back-transcription of its own audio to REF_TEXT_MATCH_MIN, and the text
file is WRITTEN from that transcription. Record: docs/pacing.md,
«Reference quality».
"""
import difflib
import re

REF_TEXT_MATCH_MIN = 0.98   # word-level similarity text vs back-transcription:
                            # a trailing word the audio lacks is 1 of ~40 ->
                            # 0.975, so 0.95 would miss the case that started
                            # this (s25: revisit with the e1v6 candidate log)
REF_TAIL_WARN = 0.5         # sec of audio after the last word: a warning
                            # (the rule is text match; the tail is advice)
SLOW_REF_SHARE = 0.7        # a reference slower than this share of the other
                            # references' median tempo clones a drawl (s27):
                            # relative to the panel, as CLAUDE.md §1 asks


def norm_words(text: str) -> list:
    return re.sub(r"[^\w' ]+", " ", text.lower()).replace("'", "").split()


def word_match(text: str, heard: str) -> float:
    """0..1 similarity of the two word sequences (punctuation and case
    ignored). 1.0 = the reference text says exactly what the audio says."""
    a, b = norm_words(text), norm_words(heard)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def check_reference(text: str, heard: str, last_word_end: float,
                    audio_secs: float) -> dict:
    """Verdict for one reference: match (text vs heard), tail (seconds
    of audio after whisper's last word), ok (match passes)."""
    match = word_match(text, heard)
    tail = max(0.0, audio_secs - last_word_end)
    return {"match": round(match, 3), "tail": round(tail, 2),
            "chars_per_sec": round(len(heard.strip()) / audio_secs, 1) if audio_secs else 0.0,
            "ok": match >= REF_TEXT_MATCH_MIN, "heard": heard.strip()}


def slow_references(rates: dict, floor: float) -> list:
    """Speakers whose reference tempo (chars/s) is under SLOW_REF_SHARE of
    the median of the OTHER speakers' references — 10 vs 16–22 fires, a
    uniformly slow panel does not. With one speaker only the absolute
    `floor` applies."""
    out = []
    for spk, r in rates.items():
        others = [v for k, v in rates.items() if k != spk]
        if others:
            others.sort()
            n = len(others)
            med = others[n // 2] if n % 2 else (others[n // 2 - 1] + others[n // 2]) / 2
            if r < SLOW_REF_SHARE * med:
                out.append(spk)
        elif r < floor:
            out.append(spk)
    return out


def whisper_transcribe(path, language=None):
    """(text, last_word_end_sec) via mlx-whisper; the only model call."""
    import mlx_whisper  # lazy: make_refs and the doctor import this module
    r = mlx_whisper.transcribe(
        str(path), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        word_timestamps=True, **({"language": language} if language else {}))
    words = [w for s in r["segments"] for w in s.get("words", [])]
    last = words[-1]["end"] if words else 0.0
    return r["text"].strip(), float(last)
