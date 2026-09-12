#!/usr/bin/env python
"""Lay the synthesized Russian segments onto the video's timeline and mux.

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
segments = json.loads((work / "segments_ru.json").read_text())["segments"]
tts_dir = work / "tts"
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

cursor = 0.0  # end time of the previously placed clip
for i, seg in enumerate(segments):
    clip_path = tts_dir / f"seg_{seg['id']:04d}.wav"
    if not clip_path.exists():
        print(f"WARNING: missing {clip_path.name}, skipping")
        continue

    # never overlap the previous clip
    place_at = max(seg["start"], cursor)
    next_start = segments[i + 1]["start"] if i + 1 < len(segments) else video_dur
    slot = max(next_start - place_at, 0.5)

    dur = probe_duration(clip_path)
    tempo = min(max(dur / slot, 1.0), MAX_TEMPO)
    if tempo > 1.02:
        fitted = tmp / clip_path.name
        subprocess.check_call([
            "ffmpeg", "-y", "-v", "quiet", "-i", str(clip_path),
            "-filter:a", f"atempo={tempo:.4f}", "-ar", str(SR), str(fitted),
        ])
        clip_path, dur = fitted, dur / tempo

    clip = AudioSegment.from_wav(clip_path).set_frame_rate(SR).set_channels(1)
    canvas = canvas.overlay(clip, position=int(place_at * 1000))
    cursor = place_at + dur

dubbed = work / "dubbed.wav"
canvas.export(dubbed, format="wav")

out_video = work / "video_ru.mp4"
subprocess.check_call([
    "ffmpeg", "-y", "-i", str(work / "video.mp4"), "-i", str(dubbed),
    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    "-shortest", str(out_video),
])
print(f"DONE: {out_video}")
