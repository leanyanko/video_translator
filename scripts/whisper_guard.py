#!/usr/bin/env python
"""Whisper long-pass degradation guard — shared by BOTH transcribers.

Model: none here (pure word arithmetic); the callers run whisper.

Whisper conditions each 30 s window on its own previous output. On a
long pass it can derail and STAY derailed until the pass ends:
punctuation vanishes for 15+ minutes, "words" get stretched over many
seconds of speech it silently skipped, a phrase can repeat in a loop.
Measured on a 123-min solo stem (Dan, Gelbart episode): fixed ~30-min
chunks left 14.9 min unpunctuated; restarting at the last sentence end
before the collapse left 0 min and recovered +351 words. The guard
used to live only in transcribe_tracks.py; the mix flow had chunking
alone, and NOTHING measured the damage after the fact (TODO #12).

Three tools:
  guarded_pass(run, a, b)   transcribe [a, b) with adaptive restart at
      the last sentence end before a punctuation collapse, then splice
      re-transcriptions over stretched words. `run(a, b)` is the
      caller's whisper invocation returning (segments, words) with
      ABSOLUTE times.
  degraded_windows(words)   the doctor's measurement: per 5-minute
      window, punctuation share / hallucination tempo / stretched-word
      count. Pure, unit-tested.
  hallucination_loops(words)  the loop whisper punctuates correctly —
      the same 3-gram repeating back to back, invisible to the
      punctuation-share rule.
"""
PUNCT = (".", ",", "!", "?", "…")
SENT = (".", "!", "?", "…")
PUNCT_WIN = 80        # words per sliding check window
PUNCT_MIN = 0.04      # min share of words carrying punctuation per window
MIN_ADVANCE = 60.0    # sec; a restart must move forward at least this much
STRETCH_SECS = 2.0    # a "word" longer than this hides skipped speech
SPLICE_REACH = 30.0   # sec; how far to look for the sentence ends around it

# doctor gate (degraded_windows)
WIN_SECS = 300.0      # measurement window
WIN_MIN_WORDS = 80    # fewer words = mostly silence, window skipped
TEMPO_MAX = 260.0     # words/min above this = dense hallucination
STRETCH_FAIL = 3      # stretched words per window that mean skipped speech

import re


def first_poor(ws: list):
    """Index of the first word of the first poorly punctuated window."""
    if len(ws) < PUNCT_WIN:
        return None
    marks = [w["word"].strip().endswith(PUNCT) for w in ws]
    run = sum(marks[:PUNCT_WIN])
    if run / PUNCT_WIN < PUNCT_MIN:
        return 0
    for i in range(PUNCT_WIN, len(marks)):
        run += marks[i] - marks[i - PUNCT_WIN]
        if run / PUNCT_WIN < PUNCT_MIN:
            return i - PUNCT_WIN + 1
    return None


def fix_stretched(run, ws: list, log=print) -> list:
    """Re-transcribe the sentences around stretched words; splice if better."""
    regions = []
    for i, w in enumerate(ws):
        if w["end"] - w["start"] <= STRETCH_SECS:
            continue
        lo = i
        while lo > 0 and w["start"] - ws[lo - 1]["start"] < SPLICE_REACH:
            if ws[lo - 1]["word"].strip().endswith(SENT):
                break
            lo -= 1
        hi = i
        while hi + 1 < len(ws) and ws[hi + 1]["end"] - w["end"] < SPLICE_REACH:
            if ws[hi]["word"].strip().endswith(SENT) and hi > i:
                break
            hi += 1
        if regions and lo <= regions[-1][1] + 1:
            regions[-1][1] = max(regions[-1][1], hi)
        else:
            regions.append([lo, hi])
    fixed = 0
    for lo, hi in reversed(regions):
        x = max(ws[lo]["start"] - 0.15, ws[lo - 1]["end"] if lo else 0.0)
        y = min(ws[hi]["end"] + 0.15, ws[hi + 1]["start"] if hi + 1 < len(ws) else 1e9)
        _, new = run(x, y)
        new = [w for w in new if x - 0.05 <= w["start"] < y]
        old_n = hi - lo + 1
        old_stretch = sum(w["end"] - w["start"] > STRETCH_SECS for w in ws[lo: hi + 1])
        new_stretch = sum(w["end"] - w["start"] > STRETCH_SECS for w in new)
        if len(new) > old_n and new_stretch < old_stretch:
            ws[lo: hi + 1] = new
            fixed += 1
            log(f"    skipped speech {x:7.1f}-{y:7.1f}: {old_n} -> {len(new)} words",
                flush=True)
    if regions and fixed < len(regions):
        log(f"    {len(regions) - fixed} stretched-word region(s) left as is",
            flush=True)
    return ws


def guarded_pass(run, a: float, b: float, log=print):
    """Transcribe [a, b) via `run`, restarting a fresh pass at the last
    sentence end before every punctuation collapse; then splice over
    stretched words. Returns (segments, words), absolute times."""
    segs_all, ws_all, t = [], [], a
    while t < b - 0.5:
        segs, ws = run(t, b)
        bad = first_poor(ws)
        cut = None
        if bad is not None:
            for j in range(bad - 1, -1, -1):
                if ws[j]["word"].strip().endswith(SENT):
                    cut = j
                    break
            if cut is None or ws[cut]["end"] - t < MIN_ADVANCE:
                cut = bad - 1  # no sentence end: restart at the collapse
            if cut < 0 or ws[cut]["end"] - t < MIN_ADVANCE:
                log(f"    {t:7.1f}-{b:7.1f}: punctuation collapses right "
                    "away — STILL POOR, run repunctuate_words.py", flush=True)
                cut = None
        if cut is None:
            segs_all += segs
            ws_all += ws
            break
        nxt = (ws[cut]["end"] + ws[cut + 1]["start"]) / 2
        segs_all += [s for s in segs if s["start"] < nxt]
        ws_all += ws[: cut + 1]
        log(f"    punctuation collapsed at {ws[bad]['start']:7.1f}: "
            f"restart from sentence end {nxt:7.1f}", flush=True)
        t = nxt
    return segs_all, fix_stretched(run, ws_all, log=log)


def degraded_windows(words: list, win_secs: float = WIN_SECS) -> list:
    """[{t0, t1, reason}] — fixed win_secs windows of the word stream
    where whisper's output looks degraded: punctuation share < PUNCT_MIN
    (the validated collapse signal), words/min > TEMPO_MAX (dense
    hallucination), or >= STRETCH_FAIL stretched words (skipped speech).
    A window with fewer than WIN_MIN_WORDS is skipped: it is mostly
    silence between islands, and a slow-tempo rule there would flag
    every long pause — which is why the review's "< 40 words/min" check
    is deliberately NOT implemented; the stretched-word count measures
    skipped speech directly, without that false positive.

    Severity, calibrated on shipped runs (e1v6/e1v5): collapsed
    punctuation and hallucination tempo are "fail" — those transcripts
    are unusable downstream; a cluster of stretched words alone is
    "warn" — on e1v6 it matched slow, pausing speech whose segments
    translated and dubbed fine (fix_stretched already spliced what a
    re-transcription could improve)."""
    out = []
    if not words:
        return out
    t0 = words[0]["start"]
    buckets = {}
    for w in words:
        buckets.setdefault(int((w["start"] - t0) // win_secs), []).append(w)
    for k in sorted(buckets):
        ws = buckets[k]
        if len(ws) < WIN_MIN_WORDS:
            continue
        reasons, severity = [], "warn"
        share = sum(w["word"].strip().endswith(PUNCT) for w in ws) / len(ws)
        if share < PUNCT_MIN:
            reasons.append(f"punctuation {share:.1%}")
            severity = "fail"
        tempo = len(ws) / (win_secs / 60.0)
        if tempo > TEMPO_MAX:
            reasons.append(f"{tempo:.0f} words/min")
            severity = "fail"
        stretched = sum(w["end"] - w["start"] > STRETCH_SECS for w in ws)
        if stretched >= STRETCH_FAIL:
            reasons.append(f"{stretched} stretched words (possible "
                           "skipped speech)")
        if reasons:
            out.append({"t0": t0 + k * win_secs, "t1": t0 + (k + 1) * win_secs,
                        "reason": "; ".join(reasons), "severity": severity})
    return out


# Words people genuinely repeat in threes: backchannel, fillers,
# thanks/laughs. An n-gram made ONLY of these is real speech, not a
# loop — on live runs the raw detector's reports were half such
# («yeah yeah yeah», «thanks thanks thanks», «hahahaha»; review 10-12).
# repeat_audit.py keeps similar sets, but it is a script (runs on
# import), so the loop detector carries its own.
FILLER_LOOP = {
    "yeah", "yes", "no", "okay", "ok", "right", "sure", "yep", "exactly",
    "mm", "mmm", "hmm", "mmhmm", "uhhuh", "uh", "um", "oh", "wow", "so",
    "and", "i", "dont", "know", "thanks", "thank", "you", "ha", "wait",
}


def _filler_tok(t: str) -> bool:
    return t in FILLER_LOOP or re.fullmatch(r"(ha|ah|he)+", t) is not None


def hallucination_loops(words: list, n: int = 3, repeats: int = 3) -> list:
    """[(start_sec, phrase)] where the same n-gram repeats `repeats`
    times back to back — the loop that keeps its punctuation and slips
    past the share rule. An n-gram of nothing but filler words is real
    speech (people do say «yeah yeah yeah» three times) and is skipped."""
    toks = [re.sub(r"[^\w]+", "", w["word"].lower()) for w in words]
    out, i, need = [], 0, n * repeats
    while i + need <= len(toks):
        g = toks[i: i + n]
        if (g[0] and not all(_filler_tok(t) for t in g)
                and all(toks[i + j * n: i + (j + 1) * n] == g
                        for j in range(1, repeats))):
            # one report per burst: nearby repeats of the same phrase
            # collapse into the first hit
            if not (out and out[-1][1] == " ".join(g)
                    and words[i]["start"] - out[-1][0] < 60.0):
                out.append((words[i]["start"], " ".join(g)))
            i += need
        else:
            i += 1
    return out
