#!/usr/bin/env python
"""Extract a voice-clone reference clip for EVERY speaker.

Model: none (ffmpeg slicing over segments.json timings).

Usage: venv/bin/python scripts/make_refs.py work/<id>
For each speaker, picks their longest run of consecutive segments with
tight gaps (continuous, uninterrupted speech) and extracts
refs/<SPEAKER>.wav + refs/<SPEAKER>.txt.
"""
import json
import subprocess
import sys
from pathlib import Path

MIN_SECS, MAX_SECS = 6.0, 20.0
MAX_GAP = 0.5

work = Path(sys.argv[1])
if "--max-secs" in sys.argv:
    MAX_SECS = float(sys.argv[sys.argv.index("--max-secs") + 1])
    MIN_SECS = min(MIN_SECS, MAX_SECS / 2)
refs_name = (
    sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else "refs"
)
data = json.loads((work / "segments.json").read_text())
refs_dir = work / refs_name
refs_dir.mkdir(exist_ok=True)

for spk in data["speakers"]:
    segs = [s for s in data["segments"] if s["speaker"] == spk]
    best = None  # (duration, start, end, texts)
    i = 0
    while i < len(segs):
        j = i
        while (
            j + 1 < len(segs)
            and segs[j + 1]["start"] - segs[j]["end"] <= MAX_GAP
            and segs[j + 1]["end"] - segs[i]["start"] <= MAX_SECS
        ):
            j += 1
        dur = segs[j]["end"] - segs[i]["start"]
        # audio and text must cover exactly the same span — never clip
        # audio mid-segment, skip runs that exceed the cap instead
        if dur <= MAX_SECS and (best is None or dur > best[0]):
            best = (dur, segs[i]["start"], segs[j]["end"],
                    [s["text"] for s in segs[i : j + 1]])
        i = j + 1

    dur, start, end, texts = best
    if dur < MIN_SECS:
        print(f"WARNING: {spk} best clip is only {dur:.1f}s — cloning may be weak")
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "quiet", "-i", str(work / "audio.wav"),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-ac", "1", "-ar", "44100", str(refs_dir / f"{spk}.wav"),
    ])
    (refs_dir / f"{spk}.txt").write_text(" ".join(texts))
    print(f"{spk}: {start:.1f}s-{end:.1f}s ({dur:.1f}s)")
