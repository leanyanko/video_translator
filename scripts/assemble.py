#!/usr/bin/env python
"""Lay the synthesized Russian segments onto the video's timeline and mux.

Model: none (ffprobe/ffmpeg + pydub audio arithmetic).

Usage: venv/bin/python scripts/assemble.py work/<id>
Each clip is placed at its segment's start time. If a clip is longer than
the slot before the next segment, it is sped up with ffmpeg atempo
(capped at MAX_TEMPO so it still sounds natural; beyond that it simply
overflows into the following pause). Produces dubbed.wav and video_ru.mp4.
"""
import json
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment

MAX_TEMPO = 1.35  # max speed-up before we allow overflow instead
SR = 44100

work = Path(sys.argv[1])
tts_name = sys.argv[sys.argv.index("--tts-dir") + 1] if "--tts-dir" in sys.argv else "tts"
out_name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "video_ru.mp4"
segments = json.loads((work / "segments_ru.json").read_text())["segments"]
tts_dir = work / tts_name
tmp = work / "tmp"
tmp.mkdir(exist_ok=True)


def probe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ], text=True)
    return float(out.strip())


video_dur = probe_duration(work / "video.mp4")
canvas = AudioSegment.silent(duration=int(video_dur * 1000), frame_rate=SR)

import numpy as np
import soundfile as sf_

TRIM_THRESHOLD = 0.01  # amplitude fraction (~-40 dBFS): below = silence
TRIM_MARGIN = 0.05  # keep 50 ms of natural air around the speech


def trim_silence(path: Path, out_path: Path) -> Path | None:
    """Cut leading/trailing silence so speech starts at sample ~0.

    The TTS model paints silence around the speech (imitating the
    reference's pauses); alignment must use where speech actually starts.
    Returns None for an all-silent clip.
    """
    data, sr = sf_.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    voiced = np.where(np.abs(data) > TRIM_THRESHOLD)[0]
    if len(voiced) == 0:
        return None
    lo = max(0, voiced[0] - int(TRIM_MARGIN * sr))
    hi = min(len(data), voiced[-1] + int(TRIM_MARGIN * sr))
    sf_.write(out_path, data[lo:hi], sr)
    return out_path


placed = []  # (start_sec, end_sec) of every dubbed clip, for gap-filling
cursor = 0.0  # end time of the previously placed clip
for i, seg in enumerate(segments):
    raw_path = tts_dir / f"seg_{seg['id']:04d}.wav"
    if not raw_path.exists():
        print(f"WARNING: missing {raw_path.name}, skipping")
        continue
    clip_path = trim_silence(raw_path, tmp / f"trim_{raw_path.name}")
    if clip_path is None:
        print(f"WARNING: {raw_path.name} is silent, skipping")
        continue

    # lip sync beats gapless audio: every line starts at its speaker's
    # own timestamp, tolerating at most 1s delay from an overrunning
    # predecessor (the overlap tail is quiet crosstalk, lag is worse)
    place_at = max(seg["start"], min(cursor, seg["start"] + 1.0))
    next_start = segments[i + 1]["start"] if i + 1 < len(segments) else video_dur
    slot = max(next_start - place_at, 0.5)

    dur = probe_duration(clip_path)
    tempo = min(max(dur / slot, 1.0), MAX_TEMPO)
    if tempo > 1.02:
        # distinct name: clip_path may itself live in tmp/ (trimmed),
        # and ffmpeg must never write the file it is reading
        fitted = tmp / f"fit_{raw_path.name}"
        subprocess.check_call([
            "ffmpeg", "-y", "-v", "quiet", "-i", str(clip_path),
            "-filter:a", f"atempo={tempo:.4f}", "-ar", str(SR), str(fitted),
        ])
        clip_path, dur = fitted, dur / tempo

    clip = AudioSegment.from_wav(clip_path).set_frame_rate(SR).set_channels(1)
    canvas = canvas.overlay(clip, position=int(place_at * 1000))
    placed.append((place_at, place_at + dur))
    cursor = place_at + dur

# optionally fill gaps between dubbed lines with the ORIGINAL audio.
# Off by default: whisper's segment edges leak English word tails and
# breaths into the gaps, which sounds worse than silence. (A future
# laughter-only variant would need a laughter detector to pick regions.)
if "--fill-gaps" in sys.argv:
    MIN_GAP = 0.4  # sec; ignore hairline gaps
    FADE_MS = 120
    original = AudioSegment.from_wav(work / "audio.wav").set_frame_rate(SR).set_channels(1)
    placed.sort()
    gaps, prev_end = [], 0.0
    for s, e in placed:
        if s - prev_end >= MIN_GAP:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if video_dur - prev_end >= MIN_GAP:
        gaps.append((prev_end, video_dur))
    for gs, ge in gaps:
        piece = original[int(gs * 1000) : int(ge * 1000)]
        fade = min(FADE_MS, len(piece) // 2)
        canvas = canvas.overlay(piece.fade_in(fade).fade_out(fade), position=int(gs * 1000))
    print(f"gap-filled {len(gaps)} regions with original audio "
          f"({sum(ge-gs for gs, ge in gaps):.0f}s total)")

dubbed = work / "dubbed.wav"
canvas.export(dubbed, format="wav")

out_video = work / out_name
subprocess.check_call([
    "ffmpeg", "-y", "-i", str(work / "video.mp4"), "-i", str(dubbed),
    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    "-shortest", str(out_video),
])
print(f"DONE: {out_video}")
