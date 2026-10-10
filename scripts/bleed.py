#!/usr/bin/env python
"""Per-mic-pair bleed calibration, shared by the non-speech restore
(harvest_laughs.py) and the low-confidence paste decision
(segment_tracks.py).

Model: none (medians over RMS frames). A fixed cross-stem ratio either
drops quiet real sounds or keeps loud bleed; physics is per mic pair:
over spans where ONLY speaker j talks, median(RMS_mic_i / RMS_stem_j)
IS mic i's bleed coefficient for j.
"""
import numpy as np

# ── per-mic-pair bleed calibration (reviewer 10-10) ───────────────────
# A fixed cross-stem ratio drops a quiet laugh over loud speech (the
# 6-7× zone). Physics is per pair: over spans where ONLY speaker j
# talks, median(RMS_mic_i / RMS_stem_j) IS mic i's bleed coefficient
# for j. A candidate on mic i is his own sound when it exceeds
# OWN_OVER_BLEED× what j's bleed alone would produce.
OWN_OVER_BLEED = 3.0


def solo_spans(speech: dict, spk: str, min_secs: float = 1.0):
    """Spans where spk talks and nobody else does."""
    others = sorted(sp for o, ss in speech.items() if o != spk for sp in ss)
    out = []
    for a, b in speech.get(spk, []):
        cur = a
        for sa, sb in others:
            if sb <= cur or sa >= b:
                continue
            if sa > cur:
                out.append((cur, sa))
            cur = max(cur, sb)
        if cur < b:
            out.append((cur, b))
    return [(a, b) for a, b in out if b - a >= min_secs]


def pair_coeff(rms_i, rms_j, spans, frame: float):
    """median(rms_i/rms_j) over frames of `spans` where j is audible."""
    ratios = []
    for a, b in spans:
        lo, hi = int(a / frame), int(b / frame)
        ri, rj = rms_i[lo:hi], rms_j[lo:hi]
        m = min(len(ri), len(rj))
        if not m:
            continue
        ri, rj = ri[:m], rj[:m]
        keep = rj > 0.02
        ratios.extend((ri[keep] / rj[keep]).tolist())
    return float(np.median(ratios)) if len(ratios) >= 8 else None


def bleed_expected(coeffs: dict, spk: str, others_rms: dict) -> float | None:
    """Max bleed any other speaker's current loudness would explain;
    None when some audible pair is uncalibrated."""
    worst = 0.0
    for o, rj in others_rms.items():
        c = coeffs.get((spk, o))
        if c is None:
            if rj > 0.02:
                return None
            continue
        worst = max(worst, c * rj)
    return worst




def calibrate(rms_by_spk: dict, speech: dict, frame: float) -> dict:
    """{(mic_i, speaker_j): coeff} for every pair with enough solo-j."""
    coeffs = {}
    for i in rms_by_spk:
        for j in rms_by_spk:
            if i == j:
                continue
            c = pair_coeff(rms_by_spk[i], rms_by_spk[j],
                           solo_spans(speech, j), frame)
            if c is not None:
                coeffs[(i, j)] = c
    return coeffs
