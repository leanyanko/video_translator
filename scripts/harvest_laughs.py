#!/usr/bin/env python
"""Restore the speakers' NON-SPEECH sounds (laughs, hums, sighs) from
the raw stems — by construction, not by detection.

Model: none for the keep/drop decision; PANNs Cnn14 only LABELS what
was kept (Laughter/Applause/… for the transcripts), loaded lazily and
skippable with --no-label.

Usage: venv/bin/python scripts/harvest_laughs.py work/<id>
           [--dry-run] [--debug] [--no-label] [--speakers SPEAKER_03]
       (legacy --threshold is accepted; it only tunes the LABEL
        confidence, never whether a sound is kept)

The previous version asked a classifier what is a laugh and lost both
laughs of the traps baseline: breathy laughter scores ~0.05, a chuckle
sat exactly on the 0.12 threshold, and clean_stems had muted them
anyway (its voice profile is built from speech). This version uses an
invariant instead:

    On a speaker's own RAW stem, every voiced island NOT covered by his
    dubbed speech segments IS his non-speech vocalization. His speech
    spans are known word-exactly; everything else of his voice belongs
    in the dub as-is.

Keep/drop chain per speaker:
  1. voiced islands of the RAW stem (shared scan; raw because cleaning
     mutes exactly these sounds);
  2. minus his speech spans from segments_ru.json (±MARGIN), minus
     spans already pasted as orig by other stages (no duplicates);
  3. anti-bleed: dropped when some OTHER raw stem is ≥ BLEED_RATIO×
     louder over the span — someone else's voice in his mic. A real
     laugh OVER someone's speech survives: his mic is close, the ratio
     stays small. Measured anchors (work/exp_traps_v3): Misha's laugh
     own 0.054 vs Dan 0.2 → ratio ~4, kept; Dan-bleed on Misha's stem
     own 0.017 vs Dan 0.3 → ratio ~18, dropped;
  4. the rest is cut to laughs/<spk>_<t>.wav and appended to
     segments_ru.json as {"type": "orig", "auto_track": true, id 9500+}
     — re-runs replace only their own entries, manual ones survive.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from audio_utils import (SCAN_FRAME, SCAN_SR, decode, frame_rms,
                         merge_regions, scan_voiced_regions)
from cli_args import flag, opt

MARGIN = 0.25        # sec around dubbed speech that still counts as speech
MIN_SOUND = 0.4      # sec; shorter leftovers are breath clicks
ISLAND_GAP = 0.4     # sec; bursts this close are one sound
BLEED_RATIO = 6.0    # another stem this much louder = bleed, not him
ABS_FLOOR = 0.012    # absolute RMS floor for a keepable sound
PAD = 0.08           # sec of air kept around the cut
LABEL_SET = ("Laughter", "Giggle", "Chuckle, chortle", "Snicker",
             "Belly laugh", "Applause", "Sigh", "Cough")


def subtract_speech(islands, speech_spans, margin=MARGIN, min_secs=MIN_SOUND):
    """Parts of voiced islands not covered by (padded) speech spans."""
    spans = sorted((a - margin, b + margin) for a, b in speech_spans)
    out = []
    for a, b in islands:
        cur = a
        for sa, sb in spans:
            if sb <= cur or sa >= b:
                continue
            if sa > cur:
                out.append((cur, min(sa, b)))
            cur = max(cur, sb)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return [(a, b) for a, b in out if b - a >= min_secs]


def bleed_dropped(own_rms: float, others_max: float,
                  ratio: float = BLEED_RATIO) -> bool:
    """Fallback bleed test when a mic pair has no calibration:
    someone else's voice in this mic (their stem dwarfs this one)."""
    return others_max >= ratio * max(own_rms, 1e-6)


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
    import numpy as _np
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
    return float(_np.median(ratios)) if len(ratios) >= 8 else None


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


# ── laugh discrimination (mode --restore-sounds laughs) ───────────────
# Laughter pulses: "ha-ha-ha" is 2+ energy bursts over >=0.7 s, while a
# hum/sigh/grunt is one flat burst. PANNs alone missed a breathy laugh
# at score ~0.05, so the verdict is an OR of two independent voices:
# the acoustic signature and a LOW-threshold PANNs label.
LAUGH_MIN_SECS = 0.7
LAUGH_MIN_BURSTS = 2
LAUGH_PANNS_THR = 0.05
# burst counting needs to RESOLVE 4-6 Hz laugh pulses: 30 ms envelope
# (the 0.25 s scan frames sit at Nyquist and smear "ha-ha-ha" into one
# lump — reviewer catch 10-10)
BURST_FRAME_SECS = 0.03


def envelope_bursts(env) -> int:
    """Count energy bursts in an RMS envelope (frames): local maxima
    above half the peak, separated by a dip below 60 % of the smaller
    neighbour peak."""
    env = [float(x) for x in env]
    if not env:
        return 0
    peak = max(env)
    if peak <= 0:
        return 0
    bursts, in_burst = 0, False
    hi, lo = 0.5 * peak, 0.3 * peak
    for x in env:
        if not in_burst and x >= hi:
            bursts += 1
            in_burst = True
        elif in_burst and x <= lo:
            in_burst = False
    return bursts


def is_laugh(dur_secs: float, bursts: int, panns_laugh_score: float) -> bool:
    acoustic = dur_secs >= LAUGH_MIN_SECS and bursts >= LAUGH_MIN_BURSTS
    labeled = panns_laugh_score >= LAUGH_PANNS_THR
    return acoustic or labeled


def main() -> None:
    work = Path(sys.argv[1])
    dry_run = flag("--dry-run")
    mode = opt("--restore-sounds", "laughs")
    if mode not in ("laughs", "all", "none"):
        sys.exit("--restore-sounds must be laughs|all|none")
    if mode == "none":
        print("non-speech restore disabled (--restore-sounds none)")
        return
    debug = flag("--debug")
    label_on = not flag("--no-label")
    label_thr = opt("--threshold", 0.1, float)
    only_spk = opt("--speakers", None, lambda v: set(v.split(",")))

    # RAW stems when they exist — clean_stems mutes exactly these sounds
    from audio_utils import raw_stem_source
    speakers, tracks_dir = raw_stem_source(work)

    ru = json.loads((work / "segments_ru.json").read_text())
    speech, origs = {}, {}
    for s in ru["segments"]:
        if s.get("auto_track"):
            continue  # ours from a previous run: replaced wholesale
        dst = origs if s.get("type") == "orig" else speech
        dst.setdefault(s.get("speaker"), []).append((s["start"], s["end"]))

    # all stems are decoded even under --speakers: the anti-bleed check
    # needs every OTHER stem's loudness
    audios = {spk: decode(tracks_dir / f, SCAN_SR)
              for spk, f in speakers.items()}
    frame_n = int(SCAN_FRAME * SCAN_SR)
    rms = {spk: frame_rms(a, frame_n) for spk, a in audios.items()}

    def span_rms(spk, a, b):
        r = rms[spk][int(a / SCAN_FRAME): max(int(b / SCAN_FRAME),
                                              int(a / SCAN_FRAME) + 1)]
        return float(r.mean()) if len(r) else 0.0

    coeffs = {}
    for i in speakers:
        for j in speakers:
            if i == j:
                continue
            c = pair_coeff(rms[i], rms[j], solo_spans(speech, j), SCAN_FRAME)
            if c is not None:
                coeffs[(i, j)] = c
    if coeffs:
        print("bleed coefficients: " + ", ".join(
            f"{i[-2:]}<-{j[-2:]}={c:.3f}" for (i, j), c in sorted(coeffs.items())),
            flush=True)

    found = []
    for spk, fname in speakers.items():
        if only_spk and spk not in only_spk:
            continue
        islands = merge_regions(
            scan_voiced_regions(tracks_dir / fname), ISLAND_GAP)
        # speech spans AND existing orig pastes both count as covered
        covered = speech.get(spk, []) + origs.get(spk, [])
        cands = subtract_speech(islands, covered)
        print(f"{spk}: {len(islands)} voiced islands -> "
              f"{len(cands)} non-speech candidates", flush=True)
        for a, b in cands:
            own = span_rms(spk, a, b)
            others_rms = {o: span_rms(o, a, b) for o in audios if o != spk}
            others = max(others_rms.values(), default=0.0)
            expected = bleed_expected(coeffs, spk, others_rms)
            verdict = None
            if own < ABS_FLOOR:
                verdict = "too quiet"
            elif expected is not None:
                if own <= OWN_OVER_BLEED * expected:
                    verdict = (f"bleed (own {own:.3f} <= {OWN_OVER_BLEED}x"
                               f"expected {expected:.3f})")
            elif bleed_dropped(own, others):
                verdict = f"bleed (others {others:.3f} vs own {own:.3f})"
            if debug or verdict is None:
                print(f"  {spk[-2:]} {a:7.1f}-{b:7.1f} own={own:.3f} "
                      f"others={others:.3f} -> {verdict or 'KEEP'}", flush=True)
            if verdict:
                continue
            found.append({"speaker": spk,
                          "start": round(max(0.0, a - PAD), 3),
                          "end": round(b + PAD, 3), "label": "nonspeech"})

    LAUGH_FAMILY = ("Laughter", "Giggle", "Chuckle, chortle", "Snicker",
                    "Belly laugh")
    if (label_on or mode == "laughs") and found:
        from panns_inference import AudioTagging  # lazy: labels only
        import librosa
        tagger = AudioTagging(checkpoint_path=None, device="cpu")
        names = tagger.labels
        want = [i for i, l in enumerate(names) if l in LABEL_SET]
        for f in found:
            seg16 = audios[f["speaker"]][int(f["start"] * SCAN_SR):
                                         int(f["end"] * SCAN_SR)]
            clip32 = librosa.resample(np.asarray(seg16, dtype=np.float32),
                                      orig_sr=SCAN_SR, target_sr=32000)
            scores = tagger.inference(clip32[None, :])[0][0]
            best = max(want, key=lambda i: scores[i])
            if float(scores[best]) >= label_thr:
                f["label"] = names[best]
                f["score"] = round(float(scores[best]), 3)
            f["_panns_laugh"] = max(float(scores[i]) for i in want
                                    if names[i] in LAUGH_FAMILY)

    if mode == "laughs":
        kept = []
        burst_n = int(BURST_FRAME_SECS * SCAN_SR)
        for f in found:
            a, b = f["start"], f["end"]
            seg = audios[f["speaker"]][int(a * SCAN_SR): int(b * SCAN_SR)]
            env = frame_rms(seg, burst_n)
            bursts = envelope_bursts(env)
            panns = f.pop("_panns_laugh", 0.0)
            keep = is_laugh(b - a, bursts, panns)
            print(f"  {'LAUGH' if keep else 'skip '} {f['speaker'][-2:]} "
                  f"{a:7.1f}-{b:7.1f} dur={b-a:.1f}s bursts={bursts} "
                  f"panns={panns:.2f} {f['label']}", flush=True)
            if keep:
                kept.append(f)
        print(f"mode=laughs: kept {len(kept)} of {len(found)} sounds "
              "(hums/sighs/grunts between words are NOT pasted)")
        found = kept
    else:
        for f in found:
            f.pop("_panns_laugh", None)

    if dry_run:
        print(f"DRY RUN: {len(found)} non-speech sounds found, nothing written")
        return

    laughs_dir = work / "laughs"
    laughs_dir.mkdir(exist_ok=True)
    segs = [s for s in ru["segments"] if not s.get("auto_track")]
    next_id = 9500
    for f in sorted(found, key=lambda x: x["start"]):
        wav = laughs_dir / f"{f['speaker']}_{int(f['start'] * 10):06d}.wav"
        subprocess.check_call([
            "ffmpeg", "-y", "-v", "error", "-ss", f"{f['start']:.3f}",
            "-t", f"{f['end'] - f['start']:.3f}",
            "-i", str(tracks_dir / speakers[f["speaker"]]),
            "-ar", "44100", "-ac", "1", str(wav)])
        segs.append({"id": next_id, "type": "orig", "auto_track": True,
                     "file": f"laughs/{wav.name}", **f})
        next_id += 1
    ru["segments"] = sorted(segs, key=lambda s: s["start"])
    (work / "segments_ru.json").write_text(
        json.dumps(ru, ensure_ascii=False, indent=1))
    print(f"{len(found)} non-speech segments written to segments_ru.json")


if __name__ == "__main__":
    main()
