#!/usr/bin/env python
"""Harvest CLEAN laughs from the editor's per-speaker stems (multitrack mode).

Model: PANNs Cnn14 (AudioSet tagging, CPU; checkpoint in ~/panns_data).

Usage: venv/bin/python scripts/harvest_laughs.py work/<id> [--threshold 0.12]
           [--speech-max 0.5] [--dry-run]

Requires work/<id>/tracks.json:

    {"speakers": {"SPEAKER_00": "… Dan Audio Only.mp3", ...},
     "music": "… Music and SFX Audio Only.mp3"}

with the files in work/<id>/tracks/. Unlike detect_laughs.py (mix mode,
gaps only), a solo stem holds ONE voice, so a laugh found anywhere on it
is clean — it may be overlaid even while another speaker's dub is
talking, like a real listener laughing over speech. Only collisions with
the SAME speaker's dubbed lines are rejected (his dub would fight his
own laugh).

Found laughs are cropped to work/<id>/laughs/<speaker>_<start>.wav and
appended to segments_ru.json as

    {"id": 95xx, "type": "orig", "auto_track": true, "speaker": ...,
     "file": "laughs/....wav", "start": ..., "end": ..., "label": ..., "score": ...}

assemble.py pastes every type:"orig" segment with file= as-is (stems are
already at the editor's levels). Re-running replaces previous auto_track
entries; manual and detect_laughs entries are untouched.
"""
import json
import subprocess
import sys
from pathlib import Path

from cli_args import flag, opt

import numpy as np

TARGET_LABELS = ("Laughter", "Giggle", "Chuckle, chortle", "Snicker",
                 "Belly laugh")
from audio_utils import (RMS_SILENCE, SCAN_FRAME as FRAME, decode,  # noqa: E402
                         merge_regions, scan_voiced_regions)

SR = 32000          # PANNs native rate
WIN, HOP = 4.0, 2.0  # classification window / hop, sec
MIN_REGION = 0.4     # sec; shorter voiced blips aren't classified
MERGE_GAP = 0.6      # sec; voiced regions closer than this are merged
EDGE_PAD = 0.12      # sec of air kept around the harvested laugh
COLLIDE_PAD = 0.15   # sec; min clearance from the same speaker's dub

work = Path(sys.argv[1])
threshold = opt("--threshold", 0.12, float)
speech_max = opt("--speech-max", 0.5, float)
dry_run = flag("--dry-run")

# laughs are cut from the RAW stems when they exist: clean_stems.py
# builds each speaker's profile from their SPEECH, so it can mute the
# owner's own LAUGHTER as "foreign" (seen: a loud laugh at full level
# on the raw stem, silence on the cleaned one). Bleed is no problem
# here — PANNs only accepts laugh-scored, speech-poor windows anyway.
if (work / "tracks_raw.json").exists():
    tracks_meta = json.loads((work / "tracks_raw.json").read_text())
    TRACKS_DIR = work / "tracks_raw"
else:
    tracks_meta = json.loads((work / "tracks.json").read_text())
    TRACKS_DIR = work / "tracks"
ru = json.loads((work / "segments_ru.json").read_text())
# same-speaker dubbed spans (the collision filter); orig segments excluded
dub_spans = {}
# same-speaker orig clips already on the timeline (segment_tracks.py
# nonverbal cuts, manual entries): a laugh found there is a duplicate
orig_spans = {}
for s in ru["segments"]:
    if s.get("auto_track"):
        continue
    spans = orig_spans if s.get("type") == "orig" else dub_spans
    spans.setdefault(s.get("speaker"), []).append((s["start"], s["end"]))


def voiced_regions(path: Path) -> list[tuple[float, float]]:
    """Streamed RMS scan of the whole stem (solo tracks are mostly
    silence): merge near regions, drop blips."""
    merged = merge_regions(scan_voiced_regions(path, FRAME, RMS_SILENCE), MERGE_GAP)
    return [(a, b) for a, b in merged if b - a >= MIN_REGION]


def collides(spk: str, a: float, b: float, spans=dub_spans) -> bool:
    for s, e in spans.get(spk, ()):
        if a < e + COLLIDE_PAD and b > s - COLLIDE_PAD:
            return True
    return False


from panns_inference import AudioTagging  # slow import; after arg errors

tagger = AudioTagging(checkpoint_path=None, device="cpu")
labels = tagger.labels
laugh_idx = [i for i, l in enumerate(labels) if l in TARGET_LABELS]
speech_i = labels.index("Speech")

laughs_dir = work / "laughs"
laughs_dir.mkdir(exist_ok=True)
found = []

debug = flag("--debug")
only_spk = opt("--speakers", None, lambda v: set(v.split(",")))

for spk, fname in tracks_meta["speakers"].items():
    if only_spk and spk not in only_spk:
        continue
    track = TRACKS_DIR / fname
    regions = voiced_regions(track)
    print(f"{spk} ({fname}): {len(regions)} voiced regions, "
          f"{sum(b - a for a, b in regions) / 60:.1f} min", flush=True)
    # a short voiced region is classified WHOLE (a laugh between lines is
    # its own energy island on a solo stem — a fixed window grid would
    # dilute its score with silence/speech); long regions get windows
    windows = []
    for a, b in regions:
        if b - a <= WIN * 1.5:
            windows.append((a, b))
        else:
            t = a
            while t < b:
                windows.append((t, min(t + WIN, b)))
                if t + WIN >= b:
                    break
                t += HOP
    hits = []  # (start, end, label, laugh_score)
    BATCH = 16
    for i in range(0, len(windows), BATCH):
        batch = windows[i: i + BATCH]
        maxlen = max(int((b - a) * SR) for a, b in batch)
        clips = np.zeros((len(batch), max(maxlen, SR)), dtype=np.float32)
        for j, (a, b) in enumerate(batch):
            c = decode(track, SR, start=a, dur=b - a)
            clips[j, : len(c)] = c[: clips.shape[1]]
        scores = tagger.inference(clips)[0]
        for j, (a, b) in enumerate(batch):
            best_i = max(laugh_idx, key=lambda k: scores[j][k])
            ls, ss = float(scores[j][best_i]), float(scores[j][speech_i])
            if debug and ls >= 0.05:
                print(f"  dbg {a:8.1f}-{b:8.1f} laugh={ls:.2f} "
                      f"speech={ss:.2f} {labels[best_i]}", flush=True)
            if ls >= threshold and ss <= speech_max:
                hits.append([a, b, labels[best_i], ls])
    # merge overlapping hit windows, keep the best label/score
    hits.sort()
    spans = []
    for a, b, lab, sc in hits:
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
            if sc > spans[-1][3]:
                spans[-1][2], spans[-1][3] = lab, sc
        else:
            spans.append([a, b, lab, sc])
    for a, b, lab, sc in spans:
        # trim the span to actual energy (the window grid is coarse)
        c = decode(track, 16000, start=a, dur=b - a)
        voiced = np.where(np.abs(c) > RMS_SILENCE)[0]
        if len(voiced) == 0:
            continue
        lo = max(a, a + voiced[0] / 16000 - EDGE_PAD)
        hi = min(b, a + voiced[-1] / 16000 + EDGE_PAD)
        if hi - lo < 0.3:
            continue
        if collides(spk, lo, hi):
            print(f"  {lo:8.1f}-{hi:8.1f}  {lab:16s} {sc:.2f}  SKIP (own dub)")
            continue
        if collides(spk, lo, hi, orig_spans):
            print(f"  {lo:8.1f}-{hi:8.1f}  {lab:16s} {sc:.2f}  SKIP (already orig)")
            continue
        wav = laughs_dir / f"{spk}_{int(lo):05d}.wav"
        if not dry_run:
            subprocess.check_call([
                "ffmpeg", "-y", "-v", "error", "-ss", f"{lo:.3f}",
                "-t", f"{hi - lo:.3f}", "-i", str(track),
                "-ar", "44100", "-ac", "1", str(wav)])
        found.append({"speaker": spk, "file": f"laughs/{wav.name}",
                      "start": round(lo, 3), "end": round(hi, 3),
                      "label": lab, "score": round(sc, 3)})
        print(f"  {lo:8.1f}-{hi:8.1f}  {lab:16s} {sc:.2f}  -> {wav.name}")

if dry_run:
    print(f"DRY RUN: {len(found)} laughs found, nothing written")
    sys.exit(0)

segs = [s for s in ru["segments"] if not s.get("auto_track")]
next_id = 9500
for f in sorted(found, key=lambda f: f["start"]):
    segs.append({"id": next_id, "type": "orig", "auto_track": True, **f})
    next_id += 1
ru["segments"] = sorted(segs, key=lambda s: s["start"])
(work / "segments_ru.json").write_text(json.dumps(ru, ensure_ascii=False, indent=1))
print(f"{len(found)} track-laugh segments written to segments_ru.json")
