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
