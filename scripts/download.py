#!/usr/bin/env python
"""Download a YouTube video and extract its audio track.

Usage: venv/bin/python scripts/download.py <youtube-url>
Creates work/<video-id>/{video.mp4, audio.wav}
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YTDLP = ROOT / "venv" / "bin" / "yt-dlp"

url = sys.argv[1]
vid = subprocess.check_output([YTDLP, "--get-id", url], text=True).strip()
work = ROOT / "work" / vid
work.mkdir(parents=True, exist_ok=True)

video = work / "video.mp4"
if not video.exists():
    subprocess.check_call([
        YTDLP,
        "-f", "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format", "mp4",
        "-o", str(video),
        url,
    ])

audio = work / "audio.wav"
if not audio.exists():
    subprocess.check_call([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "44100", str(audio),
    ])

print(f"WORKDIR={work}")
