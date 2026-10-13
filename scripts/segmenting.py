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
MAX_SENT_SECS = 25.0  # longer sentences are split (whisper sometimes
                      # loses punctuation, and very long lines are hard
                      # to dub): at the largest pause after a clause
                      # mark; at the largest pause at all only when the
                      # sentence has no clause marks — that cut is
                      # flagged (split_long_flagged, TODO #12)


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


def split_long_flagged(ws: list, max_secs: float,
                       clause_marks=(",", ";", ":", "—")) -> list[tuple]:
    """Split a sentence longer than `max_secs`: at the largest pause after
    a clause mark (translatable on its own); at the largest pause at all
    only when the sentence has no clause marks. Returns (piece, flag)
    pairs; flag=True marks the piece whose END is such a last-resort
    mid-sentence cut — the half is untranslatable on its own, and the
    translator / doctor must see that instead of hoping (TODO #12)."""
    if ws[-1]["end"] - ws[0]["start"] <= max_secs or len(ws) < 2:
        return [(ws, False)]
    gaps = [(ws[i + 1]["start"] - ws[i]["end"], i) for i in range(len(ws) - 1)]
    clause = [g for g in gaps if ws[g[1]]["word"].strip().endswith(clause_marks)]
    _, cut = max(clause or gaps)
    left = split_long_flagged(ws[: cut + 1], max_secs, clause_marks)
    right = split_long_flagged(ws[cut + 1:], max_secs, clause_marks)
    if not clause:  # this very cut is mid-sentence: flag the piece ending at it
        left[-1] = (left[-1][0], True)
    return left + right


def split_long(ws: list, max_secs: float,
               clause_marks=(",", ";", ":", "—")) -> list[list]:
    return [p for p, _ in split_long_flagged(ws, max_secs, clause_marks)]


def runon_cut_ids(segments: list) -> tuple[list, int]:
    """(ids of segments cut mid-sentence inside run-ons, count of cuts
    legitimately explained by an interruption). A segment should end at
    a sentence end; a cut is fine when the speaker really stopped —
    another voice fills the gap. A cut inside a run-on means whisper
    lost punctuation there (shared by segment_tracks's report and the
    doctor's gate, TODO #12)."""
    runons, interrupted = [], 0
    for spk in sorted({s["speaker"] for s in segments}):
        own = [s for s in segments
               if s["speaker"] == spk and s.get("type") != "orig"]
        for a, b in zip(own, own[1:]):
            if a["text"].rstrip().endswith(SENT_END + ('"',)):
                continue
            if any(o["speaker"] != spk and o["start"] < b["start"]
                   and o["end"] > a["end"] for o in segments):
                interrupted += 1
            elif b["start"] - a["end"] <= GAP_BREAK:
                runons.append(a["id"])
    return runons, interrupted


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
            # the mid-sentence flag describes the END of a piece — after
            # a merge it is the TAIL's end that counts
            if s.get("cut"):
                merged[-1]["cut"] = s["cut"]
            else:
                merged[-1].pop("cut", None)
        else:
            merged.append(dict(s))
    return merged
