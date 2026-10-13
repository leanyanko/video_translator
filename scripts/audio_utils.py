#!/usr/bin/env python
"""Shared audio helpers: ffmpeg decode, duration, frame RMS, streamed
voiced-region scan, and the reference-clip audition rule.

Model: none. Imported by clean_stems, harvest_laughs, segment_tracks,
transcribe_tracks, make_refs, check_pipeline, assemble, transcribe.

Every function here used to exist as a private copy in two or more of
those scripts. Thresholds that MUST agree across stages (the stem
silence floor, the audition rule the doctor re-checks) now live here.
"""
import subprocess
from pathlib import Path

import numpy as np

# multitrack tools: a 0.25 s frame below this RMS (~-45 dBFS) is "the
# stem is silent here" — transcribe_tracks, harvest_laughs, clean_stems
RMS_SILENCE = 0.0056
SCAN_FRAME = 0.25
SCAN_SR = 16000
VOICED_REL = 0.0316  # -30 dB of a clip's own p99 RMS: 'voiced' for the
                     # pre-flight (synthesize pace.json) and the trim decay floor

# reference audition (make_refs picks, check_pipeline re-verifies):
# 50 ms frames; voiced = rms above max(absolute floor, 10 % of the clip's
# own p95) — quiet mic noise above the floor is still junk next to the voice
AUDITION_FRAME = 0.05
AUDITION_FLOOR = 0.01  # ~-40 dBFS


def probe_duration(path) -> float:
    """Container duration in seconds (ffprobe)."""
    out = subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path)], text=True)
    return float(out.strip())


def decode(path, sr: int = SCAN_SR, start: float | None = None,
           dur: float | None = None, end: float | None = None) -> np.ndarray:
    """Decode (a window of) any audio file to mono float32 at `sr`.

    start/dur/end are input-side seeks (fast; the same -ss/-t/-to form
    the callers always used), so a 3-second window of a 2-hour stem does
    not decode the whole stem.
    """
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    elif end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)


def speech_bounds(data: np.ndarray, sr: int, floor: float = 0.01,
                  rel: float = 0.1, margin: float = 0.05,
                  decay_rel: float = 0.01, tail_max: float = 0.35,
                  head_rel: float = 0.05, head_max: float = 0.1,
                  tail_margin: float | None = None):
    """(lo, hi) sample bounds of the speech inside a TTS clip, or None
    when it is all silence. Shared by synthesize_f5 (per piece) and
    assemble (per clip). Two thresholds: speech is FOUND where RMS over
    20 ms frames exceeds max(floor, rel × the clip's own 99th
    percentile) — quiet painted lead/tail breath does not count as
    onset; then the bounds are EXTENDED outward while the signal is
    still decaying: the tail while above decay_rel × p99 (−40 dB, at
    most tail_max), the head while above head_rel × p99 (−26 dB, at
    most head_max — stricter, so a painted breath still does not delay
    the line). Unvoiced word endings («-ть», «-с») sit 20–30 dB under
    the vowels and last longer than `margin` — the single-threshold cut
    clipped them at −30…−15 dB on the traps clip (tracker d5, 10-13).
    `margin` seconds of air stay on both sides."""
    frame = sr // 50
    n = len(data) // frame
    if n == 0:
        return None
    rms = np.sqrt((data[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    p99 = float(np.percentile(rms, 99))
    thr = max(floor, rel * p99)
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return None
    low = max(floor * 0.3, decay_rel * p99)
    head_low = max(floor * 0.3, head_rel * p99)
    a, b = int(voiced[0]), int(voiced[-1])
    while (a > 0 and (voiced[0] - a) * frame < head_max * sr
           and rms[a - 1] > head_low):
        a -= 1
    while (b + 1 < n and (b - voiced[-1]) * frame < tail_max * sr
           and rms[b + 1] > low):
        b += 1
    lo = max(0, a * frame - int(margin * sr))
    tail = margin if tail_margin is None else tail_margin
    hi = min(len(data), (b + 1) * frame + int(tail * sr))
    return lo, hi


def join_trimmed(waves: list, sr: int, gap_secs: float) -> np.ndarray:
    """Concatenate TTS pieces with each piece's painted lead/tail silence
    cut (speech_bounds) and `gap_secs` of air between them, so the
    speech that remains ≈ the sum of the per-piece plans (pacing)."""
    out = []
    for w in waves:
        w = np.asarray(w, dtype=np.float32)
        # a longer tail margin: the release of a final consonant needs
        # air after it or the syllable sounds swallowed (ear, 10-13)
        b = speech_bounds(w, sr, tail_margin=0.12)
        if b is not None:
            w = w[b[0]: b[1]]
        if out:
            out.append(np.zeros(int(gap_secs * sr), dtype=np.float32))
        out.append(w)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def frame_rms(audio: np.ndarray, frame_n: int) -> np.ndarray:
    """RMS per non-overlapping frame of `frame_n` samples (tail dropped)."""
    n = len(audio) // frame_n
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((audio[: n * frame_n].reshape(n, frame_n) ** 2).mean(axis=1))


def scan_voiced_regions(path, frame: float = SCAN_FRAME,
                        threshold: float = RMS_SILENCE,
                        sr: int = SCAN_SR) -> list[tuple[float, float]]:
    """Streamed RMS scan of a whole file → [(start, end)] of consecutive
    frames above `threshold`. Streams through ffmpeg so hours-long solo
    stems never sit in memory. Callers merge/filter the raw regions."""
    proc = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-",
    ], stdout=subprocess.PIPE)
    frame_n = int(frame * sr)
    regions, t, cur = [], 0.0, None
    buf = b""
    while True:
        chunk = proc.stdout.read(frame_n * 4 * 64)
        if not chunk:
            break
        buf += chunk
        n = len(buf) // (frame_n * 4)
        if not n:
            continue
        data = np.frombuffer(buf[: n * frame_n * 4], dtype=np.float32)
        buf = buf[n * frame_n * 4:]
        for r in frame_rms(data, frame_n):
            if r > threshold:
                if cur is None:
                    cur = t
            elif cur is not None:
                regions.append((cur, t))
                cur = None
            t += frame
    if cur is not None:
        regions.append((cur, t))
    proc.wait()
    return regions


def merge_regions(regions, gap: float,
                  max_span: float | None = None) -> list[tuple[float, float]]:
    """Merge regions closer than `gap`; with `max_span`, a new region also
    starts when merging would exceed it (the break then falls in a pause)."""
    out: list[list[float]] = []
    for a, b in regions:
        if (out and a - out[-1][1] <= gap
                and (max_span is None or b - out[-1][0] <= max_span)):
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def audition(audio: np.ndarray, sr: int = SCAN_SR,
             frame: float = AUDITION_FRAME,
             floor: float = AUDITION_FLOOR) -> tuple[float, float, float]:
    """(head_silence, tail_silence, voiced_share) of a decoded clip.

    A reference whose audio opens with silence or mumble makes F5 paint
    the same junk before every line (e1v5: 2.85 s head → LEAD on every
    clip), so make_refs auditions candidates with this and the doctor
    re-checks the chosen ref with the same rule.
    """
    dur = len(audio) / sr
    rms = frame_rms(audio, int(frame * sr))
    n = len(rms)
    if n == 0:
        return dur, dur, 0.0
    voiced = rms > max(floor, 0.1 * float(np.percentile(rms, 95)))
    if not voiced.any():
        return dur, dur, 0.0
    first = int(np.argmax(voiced))
    last = n - 1 - int(np.argmax(voiced[::-1]))
    return first * frame, (n - 1 - last) * frame, float(voiced.mean())


def raw_stem_source(work):
    """(speakers map, stems dir) preferring the RAW stems.

    Original-audio pastes (laughs, backchannels, low-confidence lines)
    must be cut from tracks_raw/: clean_stems.py builds voice profiles
    from SPEECH, so it can mute exactly these owner sounds on the
    cleaned copies — a paste from tracks/ may then be silence dressed
    up as restored original (review 10-09 finding #7).
    """
    import json
    from pathlib import Path
    work = Path(work)
    if (work / "tracks_raw.json").exists():
        meta = json.loads((work / "tracks_raw.json").read_text())
        return meta["speakers"], work / "tracks_raw"
    meta = json.loads((work / "tracks.json").read_text())
    return meta["speakers"], work / "tracks"


def lone_sfx_gain(bed_rms, dub_loud, reach: int = 3,
                  poke_ratio: float = 2.0, max_run: int = 15,
                  duck: float = 0.01, noise_floor: float = 1e-3):
    """Per-frame gain that ducks LONE FOLEY in the music bed.

    Foley's distinguishing property is not that it is quiet — an
    editor's underscore sits at −25..−35 dBFS too — but that it POKES
    OUT: louder than the bed's own level in the nearest speech-covered
    frames on either side (review 10-10: a level-only gate pumped the
    music by 40 dB in every dub pause). Rules per frame:
      lone   = no dub speech within ±reach frames;
      ref    = bed level at the nearest NON-lone frame each side (max);
      foley  = lone and bed > poke_ratio × max(ref, noise_floor)
               and the lone run is ≤ max_run frames (longer = music).
    Returns float32 gains (1.0 or duck); caller ramps the steps.
    """
    import numpy as np
    n = len(bed_rms)
    bed_rms = np.asarray(bed_rms, dtype=np.float32)
    near_speech = np.zeros(n, bool)
    for d in range(-reach, reach + 1):
        lo, hi = max(0, d), n + min(0, d)
        near_speech[lo:hi] |= np.asarray(dub_loud, bool)[max(0, -d): n - max(0, d)]
    lone = ~near_speech
    # reference bed level: nearest frame UNDER ACTUAL SPEECH each side.
    # Anchoring on merely speech-ADJACENT frames lets a foley burst that
    # straddles the speech boundary become its own reference and pass.
    anchors = np.asarray(dub_loud, bool)
    # running reference starts at +inf: BEFORE the first dub line (and
    # after the last) there is no anchor, and a 0-init would collapse
    # the reference to the noise floor — ducking e.g. a 1 s intro
    # stinger at the episode edge (review 10-10)
    ref_l = np.full(n, np.inf, np.float32)
    cur = np.inf
    for i in range(n):
        if anchors[i]:
            cur = bed_rms[i]
        ref_l[i] = cur
    ref_r = np.full(n, np.inf, np.float32)
    cur = np.inf
    for i in range(n - 1, -1, -1):
        if anchors[i]:
            cur = bed_rms[i]
        ref_r[i] = cur
    # MIN of the two sides: a continuous bed is symmetric (min = its
    # level, passes), while foley straddling one speech edge poisons
    # only ITS side's anchor — the opposite side stays honest and low.
    ref = np.minimum(ref_l, ref_r)
    # judge only sounds BRACKETED by speech: a frame with no anchor on
    # one side (before the first line / after the last) gets no verdict
    # — an intro stinger is not foley
    both_anchored = np.isfinite(ref_l) & np.isfinite(ref_r)
    pokes = lone & both_anchored & (bed_rms > noise_floor) \
            & (bed_rms > poke_ratio * np.maximum(ref, noise_floor))
    # cap run length: a long "transient" is music, not foley
    gain = np.ones(n, np.float32)
    i = 0
    while i < n:
        if not pokes[i]:
            i += 1
            continue
        # the run that must stay short is the POKE run itself — walking
        # the whole audible lone stretch made the cap swallow every
        # pause with underscore playing (review 10-10: gate was a no-op
        # for foley over music)
        j = i
        while j < n and pokes[j]:
            j += 1
        if (j - i) <= max_run:
            gain[i:j] = duck
        i = j
    return gain


def smooth_gain(gain_frames, frame_len: int, ramp_len: int):
    """Expand per-frame gains to per-sample and ramp the steps with a
    moving average (hard 1.0→0.01 steps click on entry/exit)."""
    import numpy as np
    g = np.repeat(np.asarray(gain_frames, np.float32), frame_len)
    if ramp_len <= 1:
        return g
    c = np.cumsum(np.concatenate(([0.0], g)))
    out = np.empty_like(g)
    half = ramp_len // 2
    idx_hi = np.minimum(np.arange(len(g)) + half + 1, len(g))
    idx_lo = np.maximum(np.arange(len(g)) - half, 0)
    out = (c[idx_hi] - c[idx_lo]) / (idx_hi - idx_lo)
    return out.astype(np.float32)
