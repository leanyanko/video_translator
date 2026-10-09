#!/usr/bin/env python
"""Sentence building and merging shared by both segmenters.

Model: none (arithmetic over whisper words). Imported by
segment_sentences.py (mix flow) and segment_tracks.py (multitrack flow),
which used to carry private copies whose constants had started to drift.

A "word" is {"start", "end", "word", ...}; a sentence is a list of them.
"""
SENT_END = (".", "!", "?", "…")
TARGET_SECS = 12.0   # merge short same-speaker sentences up to this
MAX_PAUSE = 1.0      # never merge across a pause longer than this
GAP_BREAK = 2.5      # sec; a silence this long always ends a sentence
                     # (segment_tracks run-on guard; repunctuate_words
                     # ends its run-on search at the same silence)


def words_text(ws: list) -> str:
    return "".join(w["word"] for w in ws).strip()


def build_sentences(words: list, gap_break: float | None = None) -> list[list]:
    """Cut at sentence-final punctuation; with `gap_break`, a silence
    longer than it also ends the sentence (run-on guard)."""
    sentences, cur = [], []
    for w in words:
        if gap_break is not None and cur and w["start"] - cur[-1]["end"] > gap_break:
            sentences.append(cur)
            cur = []
        cur.append(w)
        if w["word"].strip().endswith(SENT_END):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    return sentences


def split_long(ws: list, max_secs: float,
               clause_marks=(",", ";", ":", "—")) -> list[list]:
    """Split a sentence longer than `max_secs`: at the largest pause after
    a clause mark (translatable on its own); at the largest pause at all
    only when the sentence has no clause marks."""
    if ws[-1]["end"] - ws[0]["start"] <= max_secs or len(ws) < 2:
        return [ws]
    gaps = [(ws[i + 1]["start"] - ws[i]["end"], i) for i in range(len(ws) - 1)]
    clause = [g for g in gaps if ws[g[1]]["word"].strip().endswith(clause_marks)]
    _, cut = max(clause or gaps)
    return (split_long(ws[: cut + 1], max_secs, clause_marks)
            + split_long(ws[cut + 1:], max_secs, clause_marks))


def merge_short(items: list, target_secs: float = TARGET_SECS,
                max_pause: float = MAX_PAUSE) -> list:
    """Merge consecutive same-speaker sentence items ({start, end,
    speaker, text}) while the gap stays ≤ max_pause and the merged span
    ≤ target_secs. Returns new dicts; input is not modified."""
    merged = []
    for s in items:
        if (merged and s["speaker"] == merged[-1]["speaker"]
                and s["start"] - merged[-1]["end"] <= max_pause
                and s["end"] - merged[-1]["start"] <= target_secs):
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] += " " + s["text"]
        else:
            merged.append(dict(s))
    return merged
