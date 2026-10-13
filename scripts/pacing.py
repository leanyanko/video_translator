#!/usr/bin/env python
"""Clip duration planning for F5 — the speaker's own tempo, squeezed
only to fit the slot (tracker d5).

Model: none (pure arithmetic; synthesize_f5.py applies the result via
F5's fix_duration).

Why this exists. F5 sets the length of what it generates by a formula,
not by listening to itself:

    duration = ref_sec + ref_sec / len(ref_text.utf8) * len(gen_text.utf8) / speed

Bytes, not characters. Our references are English (1 byte/char), the
lines are Russian (2 bytes/char, 1.92 with stress marks), so every
line was budgeted ~1.9x the time the speaker would actually need —
measured on baseline20: clips came out at 0.53x the reference's own
chars/sec for all three speakers. The old pacing (`est = chars / 12.5`,
speed 1.0–1.3) did not know this; it fought the phantom with speed 1.3
and assemble fought the rest with atempo 1.35, and lines still ran
over. A second rule in the same F5 function sings any piece under 10
bytes (five Cyrillic letters: «Да.», «Нет.») at speed 0.3. And a slow
reference (Misha: 10 chars/s vs Dan 16, Mike 22) multiplies on top.

The rule here: a dubbed line lasts as long as the ORIGINAL line — that
is how long the speaker's lips move, and the old pipeline matched it
only by accident (x1.9 bytes / 1.3 speed / 1.35 atempo ≈ x1.08). The
span is bounded to a plausible Russian tempo: never faster than
PACE_MAX (a long translation overruns into the slot instead, squeezed
at most MAX_SQUEEZE beyond that) and never slower than PACE_MIN (a
hesitant original does not make the dub drawl). A first version paced
lines by the speaker's REFERENCE tempo instead; it lost lip sync (Dan
20–30 % shorter than his lips), kept Misha slow (his reference is
slow) and swallowed words. The result goes to F5 as an exact duration
(fix_duration = processed_ref_sec + wanted + SILENCE_PAD), which also
bypasses the 10-byte rule. F5 paints ~0.5 s of silence around EVERY
call (lead and tail, imitating the reference's air) and crams the
words into the rest; a line is sung in sentence-sized pieces, one call
each, so the waste is per piece, not per line — a 3-piece line lost
~1 s of speaking time and swallowed its last word (traps 2:36). So
each piece gets SILENCE_PAD on top of its share, synthesize_f5 trims
each piece's own silence away (audio_utils.speech_bounds, the rule
assemble trims by) and joins the pieces with a short natural gap; the
speech then lasts the plan to within trim error — no
measure-and-resynthesize loop is needed. A small per-piece floor
keeps «Да.» from being clipped.
"""
import re

PACE_MIN = 9.0        # chars/s; slower than this is a drawl, the line is shortened
PACE_MAX = 15.0       # chars/s; faster than this swallows words, the line overruns
MAX_SQUEEZE = 1.3     # how far past PACE_MAX a line may go to fit its slot
SLOT_FILL = 0.95      # share of the slot a line may occupy
SILENCE_PAD = 0.5     # sec of lead+tail silence F5 paints per call (0.46-0.67
                      # on single-piece traps clips); trimmed away after
PIECE_MIN_SECS = 0.4  # speech floor per piece: a 3-letter piece at tempo is
                      # ~0.25 s and comes out clipped; F5's own answer was speed 0.3
PIECE_GAP = 0.15      # sec of air between pieces (they split at sentence ends)
PACE_VERSION = "pace4"  # manifest salt: clips sung under older pacing are stale


def pace_chars(text: str) -> int:
    """Characters that cost speaking time: stress marks dropped,
    whitespace collapsed."""
    return len(re.sub(r"\s+", " ", text.replace("+", "")).strip())


def plan_clip(chars: int, span_secs: float, slot_secs: float) -> tuple:
    """(wanted_secs, rate) for a line of `chars` whose original lasted
    `span_secs` and whose slot (until the next line) is `slot_secs`.
    The original span, bounded to [chars/PACE_MAX, chars/PACE_MIN];
    when that does not fit SLOT_FILL of the slot, as fast as needed but
    never beyond PACE_MAX * MAX_SQUEEZE — past that the line overruns
    (assemble's atempo takes it from there). `rate` is the resulting
    chars/s, for the log and the doctor."""
    fastest = chars / PACE_MAX
    slowest = chars / PACE_MIN
    wanted = min(max(span_secs, fastest), slowest)
    budget = slot_secs * SLOT_FILL
    if wanted > budget:
        wanted = max(budget, fastest / MAX_SQUEEZE)
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
