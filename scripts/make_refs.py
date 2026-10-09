#!/usr/bin/env python
"""Extract a voice-clone reference clip for EVERY speaker.

Model: none (ffmpeg slicing over segments.json timings).

Usage: venv/bin/python scripts/make_refs.py work/<id> [--tracks]
           [--speaker SPEAKER_XX]   (rebuild just one speaker's ref;
           the tts manifest then re-synthesizes only that speaker)
For each speaker, picks their longest run of consecutive segments with
tight gaps (continuous, uninterrupted speech) and extracts
refs/<SPEAKER>.wav + refs/<SPEAKER>.txt.

--tracks (multitrack mode): cut each clip from that speaker's SOLO stem
(work/<id>/tracks + tracks.json) instead of the mix — the reference is
then free of music and other voices, which clones noticeably cleaner.
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

track_of = {}
if "--tracks" in sys.argv:
    tracks_meta = json.loads((work / "tracks.json").read_text())
    track_of = {spk: work / "tracks" / f
                for spk, f in tracks_meta["speakers"].items()}

# A ref whose AUDIO does not match its text head-on ruins every clip:
# F5 imitates the reference pattern, so a ref that opens with 2-3 s of
# silence or someone else's mumble makes the model paint the same junk
# before every line (seen in e1v5: Dan's ref had a 2.85 s silent head —
# every clip got a LEAD artifact and the dub felt delayed). Candidates
# are therefore AUDITIONED, longest first, and the first one that passes
# the audio checks wins.
import numpy as np

MAX_HEAD_SIL = 0.3   # sec of silence allowed before the first word
MAX_TAIL_SIL = 0.6   # sec allowed after the last word
MIN_VOICED = 0.55    # min share of voiced frames inside the clip
SIL_RMS = 0.01       # ~-40 dBFS on 50 ms frames


def audition(source, start, end):
    """(head_sil, tail_sil, voiced_share) of the span, by 50 ms RMS."""
    out = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(source), "-f", "f32le", "-ac", "1", "-ar", "16000", "-",
    ], capture_output=True, check=True)
    a = np.frombuffer(out.stdout, dtype=np.float32)
    n = len(a) // 800
    if n == 0:
        return end - start, end - start, 0.0
    rms = np.sqrt((a[: n * 800].reshape(n, 800) ** 2).mean(axis=1))
    # silence threshold relative to THIS clip's speech level: quiet mic
    # noise above an absolute floor is still junk compared to the voice
    voiced = rms > max(SIL_RMS, 0.1 * float(np.percentile(rms, 95)))
    if not voiced.any():
        return end - start, end - start, 0.0
    first, last = int(np.argmax(voiced)), n - 1 - int(np.argmax(voiced[::-1]))
    return first * 0.05, (n - 1 - last) * 0.05, float(voiced.mean())


only_spk = sys.argv[sys.argv.index("--speaker") + 1] if "--speaker" in sys.argv else None

for spk in data["speakers"]:
    if only_spk and spk != only_spk:
        continue
    segs = [s for s in data["segments"] if s["speaker"] == spk
            and s.get("type") != "orig"]
    runs = []  # (duration, start, end, texts)
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
        if dur <= MAX_SECS:
            runs.append((dur, segs[i]["start"], segs[j]["end"],
                         [s["text"] for s in segs[i: j + 1]]))
        i = j + 1

    source = track_of.get(spk, work / "audio.wav")
    runs.sort(key=lambda r: r[0], reverse=True)
    candidates = [r for r in runs if r[0] >= MIN_SECS] or runs
    chosen = None
    for dur, start, end, texts in candidates:
        head, tail, voiced = audition(source, start, end)
        if head <= MAX_HEAD_SIL and tail <= MAX_TAIL_SIL and voiced >= MIN_VOICED:
            chosen = (dur, start, end, texts, head, tail, voiced)
            break
        print(f"  {spk}: rejected {start:.1f}-{end:.1f}s "
              f"(head {head:.2f}s, tail {tail:.2f}s, voiced {voiced:.0%})")
    if chosen is None:
        dur, start, end, texts = runs[0]
        head, tail, voiced = audition(source, start, end)
        chosen = (dur, start, end, texts, head, tail, voiced)
        print(f"WARNING: {spk}: no candidate passed the audio checks — "
              f"using the longest anyway (head {head:.2f}s, voiced {voiced:.0%})")
    dur, start, end, texts, head, tail, voiced = chosen
    if dur < MIN_SECS:
        print(f"WARNING: {spk} best clip is only {dur:.1f}s — cloning may be weak")
    # pad the cut past the last word's decay: cutting exactly at the
    # word-end timestamp clips the final consonant, and F5 then imitates
    # the abrupt ending (seen: English ref tail leaking into Russian
    # clips). 0.15 s into the following pause is inaudible and safe —
    # the audition above already rejected runs with noisy tails.
    # ...but never past F5's ~11.6 s ref limit
    end_padded = end + max(0.0, min(0.15, 11.6 - dur))
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "quiet", "-i", str(source),
        "-ss", f"{start:.3f}", "-to", f"{end_padded:.3f}",
        "-ac", "1", "-ar", "44100", str(refs_dir / f"{spk}.wav"),
    ])
    (refs_dir / f"{spk}.txt").write_text(" ".join(texts))
    print(f"{spk}: {start:.1f}s-{end:.1f}s ({dur:.1f}s, "
          f"head {head:.2f}s, voiced {voiced:.0%})")
