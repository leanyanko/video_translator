#!/usr/bin/env python
"""Clip duration planning for F5 (tracker d5; story in docs/pacing.md).

Model: none (pure arithmetic; synthesize_f5.py applies the result via
F5's fix_duration).

Rule: a dubbed line lasts as long as the ORIGINAL line (lip sync),
bounded to a plausible tempo — never faster than PACE_MAX (F5 is never
asked to articulate above it; a long translation overruns into the
slot and assemble's atempo takes the rest), never slower than PACE_MIN.
The plan is an exact fix_duration per piece: F5 budgets by UTF-8 BYTES
(Cyrillic = 2/char, so it gave every Russian line ~1.9x) and sings any
piece under 10 bytes at speed 0.3 — fix_duration bypasses both. Each
piece is over-asked by SILENCE_PAD (F5 paints silence around every
call), trimmed back (audio_utils.speech_bounds) and rejoined with
PIECE_GAP. Constants are hypotheses from the traps clip until they hold
on baseline20 and a full episode (s25).
"""
import re

PACE_MIN = 9.0        # chars/s; slower than this is a drawl, the line is shortened
PACE_MAX = 13.0       # chars/s; faster than this swallows words, the line overruns.
                      # The SAME number is the translator's budget
                      # (apply_translation, check_pipeline import it): if the
                      # translation is budgeted at 13 ch/s, F5 must sing 13.
                      # 15 was tried (traps v7-v9): endings swallowed by ear.
SLOT_FILL = 0.95      # share of the slot a line may occupy
SILENCE_PAD = 0.5     # sec of lead+tail silence F5 paints per call (0.46-0.67
                      # on single-piece traps clips); trimmed away after
PIECE_MIN_SECS = 0.4  # speech floor per piece: a 3-letter piece at tempo is
                      # ~0.25 s and comes out clipped; F5's own answer was speed 0.3
PIECE_GAP = 0.35      # sec of air between pieces (sentence ends): 0.15 made
                      # the last syllable sound clipped by ear (traps 1:07, 10-13)
PACE_VERSION = "pace7"  # manifest salt: clips sung under older pacing are stale


def pace_chars(text: str) -> int:
    """Characters that cost speaking time: stress marks dropped,
    whitespace collapsed."""
    return len(re.sub(r"\s+", " ", text.replace("+", "")).strip())


def plan_clip(chars: int, span_secs: float, slot_secs: float) -> tuple:
    """(wanted_secs, rate) for a line of `chars` whose original lasted
    `span_secs` and whose slot (until the next line) is `slot_secs`.
    The original span, bounded to [chars/PACE_MAX, chars/PACE_MIN];
    when that does not fit SLOT_FILL of the slot, shortened towards it
    but never below chars/PACE_MAX — past that the line overruns and
    assemble's atempo (<= MAX_TEMPO) takes it from there, so the whole
    chain never asks F5 to articulate above PACE_MAX (review 10-13,
    s30). `rate` is the resulting chars/s, for the log and the doctor."""
    fastest = chars / PACE_MAX
    slowest = chars / PACE_MIN
    wanted = min(max(span_secs, fastest), slowest)
    budget = slot_secs * SLOT_FILL
    if wanted > budget:
        wanted = max(budget, fastest)
    return wanted, (chars / wanted if wanted > 0 else 0.0)


def allocate_pieces(pieces: list, wanted: float) -> list:
    """Seconds of F5 output to ask for per piece: the piece's share of
    `wanted` (by pace_chars, at least PIECE_MIN_SECS of speech) plus
    SILENCE_PAD for the silence F5 paints around each call. The caller
    trims that silence off again, so the speech that remains ≈ the
    share."""
    weights = [pace_chars(p) for p in pieces]
    total = sum(weights) or 1
    return [max(wanted * w / total, PIECE_MIN_SECS) + SILENCE_PAD
            for w in weights]
