#!/usr/bin/env python
"""Transcribe each SOLO speaker stem (multitrack mode) with word timestamps.

Model: mlx-community/whisper-large-v3-turbo (MLX build, auto-downloaded).

Usage: venv/bin/python scripts/transcribe_tracks.py work/<id>
           [--language en] [--chunk-secs 1200]

An ALTERNATIVE to transcribe.py + diarize.py when the editor's per-track
stems exist (work/<id>/tracks/ + tracks.json, see RU_INSTRUCTIONS §3б):
the speaker is known from the track, so no diarization is needed, and
words hidden under cross-talk in the mix are read clean off the solo
stem. The mix flow stays available for episodes without stems.

A solo stem is mostly silence, so only its voiced islands are
transcribed (streamed RMS scan; islands are voiced regions merged across
gaps ≤ ISLAND_GAP and capped at --chunk-secs — whisper degrades on very
long passes). Writes words_tracks.json (words carry "speaker") and
segments_tracks_raw.json (whisper's raw sentences, for review —
check it for hallucinations on breath/noise islands).
"""
import json
import subprocess
import sys
from pathlib import Path

import mlx_whisper
import numpy as np

MODEL = "mlx-community/whisper-large-v3-turbo"
FRAME = 0.25          # energy-scan frame, sec
RMS_SILENCE = 0.0056  # ~-45 dBFS: below = the stem is silent here
ISLAND_GAP = 8.0      # sec; voiced regions closer than this share an island
PAD = 0.4             # sec of context kept around an island
MIN_ISLAND = 0.5      # sec; shorter islands are breath/clicks, skipped

work = Path(sys.argv[1])
chunk_secs = float(sys.argv[sys.argv.index("--chunk-secs") + 1]) if "--chunk-secs" in sys.argv else 1200.0
language = sys.argv[sys.argv.index("--language") + 1] if "--language" in sys.argv else None

tracks_meta = json.loads((work / "tracks.json").read_text())
tmp = work / "tmp"
tmp.mkdir(exist_ok=True)


def voiced_islands(path: Path) -> list[tuple[float, float]]:
    """Streamed RMS scan → padded voiced islands, each ≤ chunk_secs."""
    proc = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", "16000", "-",
    ], stdout=subprocess.PIPE)
    frame_n = int(FRAME * 16000)
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
        rms = np.sqrt((data.reshape(n, frame_n) ** 2).mean(axis=1))
        for r in rms:
            if r > RMS_SILENCE:
                if cur is None:
                    cur = t
            elif cur is not None:
                regions.append((cur, t))
                cur = None
            t += FRAME
    if cur is not None:
        regions.append((cur, t))
    proc.wait()
    regions = [(a, b) for a, b in regions if b - a >= MIN_ISLAND]
    # merge into islands; a new island also starts when the current one
    # would exceed chunk_secs (the break then falls in a real pause)
    islands = []
    for a, b in regions:
        if islands and a - islands[-1][1] <= ISLAND_GAP \
                and b - islands[-1][0] <= chunk_secs:
            islands[-1][1] = b
        else:
            islands.append([a, b])
    return [(max(0.0, a - PAD), b + PAD) for a, b in islands]


# Whisper conditions each 30 s window on its own previous output. On a
# long monologue island it can derail and STAY derailed until the pass
# ends: punctuation vanishes for 15+ min, and words get stretched over
# many seconds of speech it silently skipped. Measured on a 123-min solo
# stem (Dan, Gelbart episode): fixed ~30-min chunks (14.9 min still
# unpunctuated), unconditioned / 5-min retries (12.2 min) barely helped;
# restarting at the last sentence end before the collapse brought it to
# 0 min and recovered +351 words. So:
# 1. adaptive restart — when an 80-word window drops below 4 %
#    punctuation, keep everything up to the last sentence end before it
#    and start a fresh pass from there;
# 2. skipped-speech guard — any word lasting > STRETCH_SECS marks speech
#    whisper skipped; its sentence(s) are re-transcribed as a short clip
#    and spliced in when that yields more words.
PUNCT = (".", ",", "!", "?", "…")
SENT = (".", "!", "?", "…")
PUNCT_WIN = 80        # words per sliding check window
PUNCT_MIN = 0.04      # min share of words carrying punctuation per window
MIN_ADVANCE = 60.0    # sec; a restart must move forward at least this much
STRETCH_SECS = 2.0    # a "word" longer than this hides skipped speech
SPLICE_REACH = 30.0   # sec; how far to look for the sentence ends around it


def first_poor(ws: list):
    """Index of the first word of the first poorly punctuated window."""
    if len(ws) < PUNCT_WIN:
        return None
    marks = [w["word"].strip().endswith(PUNCT) for w in ws]
    run = sum(marks[:PUNCT_WIN])
    if run / PUNCT_WIN < PUNCT_MIN:
        return 0
    for i in range(PUNCT_WIN, len(marks)):
        run += marks[i] - marks[i - PUNCT_WIN]
        if run / PUNCT_WIN < PUNCT_MIN:
            return i - PUNCT_WIN + 1
    return None


def run_whisper(track: Path, a: float, b: float):
    """Transcribe [a, b) of the track → (segments, words), timeline-offset."""
    clip = tmp / "track_island.wav"
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "quiet", "-ss", f"{a:.3f}",
        "-t", f"{b - a:.3f}", "-i", str(track),
        "-ac", "1", "-ar", "16000", str(clip)])
    global language
    r = mlx_whisper.transcribe(
        str(clip), path_or_hf_repo=MODEL, word_timestamps=True,
        **({"language": language} if language else {}))
    clip.unlink(missing_ok=True)
    language = language or r.get("language")
    segs, ws = [], []
    for s in r["segments"]:
        if s["text"].strip():
            segs.append({"start": round(s["start"] + a, 3),
                         "end": round(s["end"] + a, 3), "text": s["text"].strip(),
                         "avg_logprob": round(s["avg_logprob"], 3)})
        for w in s.get("words", []):
            # probability: whisper's confidence — low on crosstalk/bleed
            # garbage, used by segment_tracks.py to drop or flag it
            ws.append({"start": round(w["start"] + a, 3),
                       "end": round(w["end"] + a, 3), "word": w["word"],
                       "prob": round(w["probability"], 3)})
    return segs, ws


def fix_stretched(track: Path, ws: list) -> list:
    """Re-transcribe the sentences around stretched words; splice if better."""
    regions = []
    for i, w in enumerate(ws):
        if w["end"] - w["start"] <= STRETCH_SECS:
            continue
        lo = i
        while lo > 0 and w["start"] - ws[lo - 1]["start"] < SPLICE_REACH:
            if ws[lo - 1]["word"].strip().endswith(SENT):
                break
            lo -= 1
        hi = i
        while hi + 1 < len(ws) and ws[hi + 1]["end"] - w["end"] < SPLICE_REACH:
            if ws[hi]["word"].strip().endswith(SENT) and hi > i:
                break
            hi += 1
        if regions and lo <= regions[-1][1] + 1:
            regions[-1][1] = max(regions[-1][1], hi)
        else:
            regions.append([lo, hi])
    fixed = 0
    for lo, hi in reversed(regions):
        x = max(ws[lo]["start"] - 0.15, ws[lo - 1]["end"] if lo else 0.0)
        y = min(ws[hi]["end"] + 0.15, ws[hi + 1]["start"] if hi + 1 < len(ws) else 1e9)
        _, new = run_whisper(track, x, y)
        new = [w for w in new if x - 0.05 <= w["start"] < y]
        old_n = hi - lo + 1
        old_stretch = sum(w["end"] - w["start"] > STRETCH_SECS for w in ws[lo: hi + 1])
        new_stretch = sum(w["end"] - w["start"] > STRETCH_SECS for w in new)
        if len(new) > old_n and new_stretch < old_stretch:
            ws[lo: hi + 1] = new
            fixed += 1
            print(f"    skipped speech {x:7.1f}-{y:7.1f}: {old_n} -> {len(new)} words",
                  flush=True)
    if regions and fixed < len(regions):
        print(f"    {len(regions) - fixed} stretched-word region(s) left as is",
              flush=True)
    return ws


def transcribe_island(track: Path, a: float, b: float):
    segs_all, ws_all, t = [], [], a
    while t < b - 0.5:
        segs, ws = run_whisper(track, t, b)
        bad = first_poor(ws)
        cut = None
        if bad is not None:
            for j in range(bad - 1, -1, -1):
                if ws[j]["word"].strip().endswith(SENT):
                    cut = j
                    break
            if cut is None or ws[cut]["end"] - t < MIN_ADVANCE:
                cut = bad - 1  # no sentence end: restart at the collapse
            if cut < 0 or ws[cut]["end"] - t < MIN_ADVANCE:
                print(f"    {t:7.1f}-{b:7.1f}: punctuation collapses right "
                      "away — STILL POOR, run repunctuate_words.py", flush=True)
                cut = None
        if cut is None:
            segs_all += segs
            ws_all += ws
            break
        nxt = (ws[cut]["end"] + ws[cut + 1]["start"]) / 2
        segs_all += [s for s in segs if s["start"] < nxt]
        ws_all += ws[: cut + 1]
        print(f"    punctuation collapsed at {ws[bad]['start']:7.1f}: "
              f"restart from sentence end {nxt:7.1f}", flush=True)
        t = nxt
    return segs_all, fix_stretched(track, ws_all)


all_words, raw_segments = [], []
for spk, fname in tracks_meta["speakers"].items():
    track = work / "tracks" / fname
    islands = voiced_islands(track)
    total = sum(b - a for a, b in islands)
    print(f"{spk} ({fname}): {len(islands)} islands, {total / 60:.1f} min",
          flush=True)
    for a, b in sorted(islands, key=lambda ab: ab[0]):
        segs, ws = transcribe_island(track, a, b)
        raw_segments += [{**s, "speaker": spk} for s in segs]
        all_words += [{**w, "speaker": spk} for w in ws]
    print(f"  {sum(1 for w in all_words if w['speaker'] == spk)} words",
          flush=True)

all_words.sort(key=lambda w: (w["start"], w["end"]))
raw_segments.sort(key=lambda s: (s["start"], s["end"]))
(work / "words_tracks.json").write_text(
    json.dumps({"language": language, "words": all_words}, ensure_ascii=False))
(work / "segments_tracks_raw.json").write_text(
    json.dumps({"language": language, "segments": raw_segments},
               ensure_ascii=False, indent=1))
print(f"language={language} words={len(all_words)} "
      f"raw_segments={len(raw_segments)}")
