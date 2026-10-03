#!/usr/bin/env python
"""Sentence segmentation from PER-TRACK words (multitrack mode).

Model: none (arithmetic over words_tracks.json from transcribe_tracks.py).

Usage: venv/bin/python scripts/segment_tracks.py work/<id>
           [--min-overlap 0.3] [--out segments.json]

An ALTERNATIVE to segment_sentences.py (mix flow, which stays available):
every word already carries its speaker, so there is no diarization and
no majority voting — each speaker's stream is segmented independently
with the same rules (cut at sentence-final punctuation, merge short
same-speaker sentences up to TARGET_SECS, never across pauses >
MAX_PAUSE). Cross-talk needs no Tier A/B recovery: a segment that starts
while another speaker's segment is still running is marked concurrent
with the same flags mark_overlaps.py uses ("overlap": true, pan +0.35,
gain −2; the interrupted line gets pan −0.35) — assemble.py overlays it
at its own start. Intrusions shorter than --min-overlap are treated as
sequential crosstalk, not concurrency.

Writes segments.json (same schema as segment_sentences.py; use --out
segments_tracks.json to compare against an existing mix-flow run
without touching it) and exports the EN transcript unless --out is set.
"""
import json
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

TARGET_SECS = 12.0
MAX_PAUSE = 1.0
SENT_END = (".", "!", "?", "…")

# cross-track housekeeping (none of this exists in the mix flow):
ECHO_DT = 0.7        # sec; same text within this start offset = mic bleed
ECHO_SIM = 0.8       # text similarity to call two segments the same phrase
# non-verbal vocalizations are language-independent: pasted from the solo
# stem as type:"orig" instead of being translated and synthesized
NONVERBAL = re.compile(r"^[-\s.,!?…]*(?:(?:m+[-\s]?h?m+|h+m+|uh[-\s]?huh|mhm+|ha(?:[-\s]?ha)+)[-.!?,…\s]*)+$", re.I)
ECHO_FRAG_WORDS = 0.7   # share of words found in neighbours' speech
ECHO_FRAG_RMS = 0.5     # of the speaker's median segment RMS
# classic whisper hallucinations on breath islands — dropped when quiet
HALLUCINATIONS = {"thank you.", "thanks for watching.", "thank you for watching.",
                  "subscribe!", "thanks."}
QUIET_FRACTION = 0.25  # of the speaker's median segment RMS

work = Path(sys.argv[1])
min_overlap = float(sys.argv[sys.argv.index("--min-overlap") + 1]) if "--min-overlap" in sys.argv else 0.3
out_name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "segments.json"

data = json.loads((work / "words_tracks.json").read_text())
words, language = data["words"], data.get("language")
tracks_meta = json.loads((work / "tracks.json").read_text())

# ── word-level bleed filter ─────────────────────────────────────────
# a solo stem still carries the OTHER speakers quietly (mic bleed), and
# whisper garbles that into junk words. Real speech into one's own mic
# is an order of magnitude louder, so words far below the speaker's own
# loudness (p95) are bleed and are dropped BEFORE sentence building.
BLEED_FRACTION = 0.12   # of the speaker's p95 word RMS
BLEED_FLOOR = 0.015     # absolute floor for very quiet speakers

kept = []
for spk in sorted({w["speaker"] for w in words}):
    path = work / "tracks" / tracks_meta["speakers"][spk]
    proc = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    ws = [w for w in words if w["speaker"] == spk]
    rms = []
    for w in ws:
        span = audio[int(w["start"] * 16000): int(w["end"] * 16000)]
        rms.append(float(np.sqrt((span ** 2).mean())) if len(span) else 0.0)
    thr = max(BLEED_FLOOR, BLEED_FRACTION * float(np.percentile(rms, 95)))
    good = [w for w, r in zip(ws, rms) if r >= thr]
    print(f"{spk}: bleed filter thr={thr:.4f} dropped "
          f"{len(ws) - len(good)}/{len(ws)} words")
    kept.extend(good)
words = sorted(kept, key=lambda w: (w["start"], w["end"]))
del kept

GAP_BREAK = 2.5      # sec; a silence this long always ends the sentence
MAX_SENT_SECS = 18.0  # whisper sometimes loses punctuation for minutes —
                      # split such run-ons at their largest internal pauses


def split_long(ws: list) -> list[list]:
    if ws[-1]["end"] - ws[0]["start"] <= MAX_SENT_SECS or len(ws) < 2:
        return [ws]
    gaps = [(ws[i + 1]["start"] - ws[i]["end"], i) for i in range(len(ws) - 1)]
    _, cut = max(gaps)
    return split_long(ws[: cut + 1]) + split_long(ws[cut + 1:])


# per-speaker sentence building (same rules as segment_sentences.py,
# plus the run-on guards above)
segments = []
dropped_orphan = 0
for spk in sorted({w["speaker"] for w in words}):
    stream = [w for w in words if w["speaker"] == spk]
    sentences, cur = [], []
    for w in stream:
        if cur and w["start"] - cur[-1]["end"] > GAP_BREAK:
            sentences.append(cur)
            cur = []
        cur.append(w)
        if w["word"].strip().endswith(SENT_END):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    sentences = [p for ws in sentences for p in split_long(ws)]
    # stray bleed words split off by the gap rule come out as tiny
    # fragments with no sentence-final punctuation — not translatable
    kept_sents = []
    for ws in sentences:
        text = "".join(w["word"] for w in ws).strip()
        if (len(ws) <= 3 and not text.rstrip().endswith(SENT_END)
                and not NONVERBAL.match(text)):
            dropped_orphan += 1
            continue
        kept_sents.append(ws)
    sentences = kept_sents
    merged = []
    for ws in sentences:
        text = "".join(w["word"] for w in ws).strip()
        if not text:
            continue
        s = {"start": round(ws[0]["start"], 3), "end": round(ws[-1]["end"], 3),
             "speaker": spk, "text": text}
        if (merged and s["start"] - merged[-1]["end"] <= MAX_PAUSE
                and s["end"] - merged[-1]["start"] <= TARGET_SECS):
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] += " " + s["text"]
        else:
            merged.append(s)
    segments.extend(merged)

# ── cross-track housekeeping ────────────────────────────────────────
def track_rms(spk: str, segs: list) -> None:
    """Attach each segment's RMS measured on its OWN speaker's stem."""
    path = work / "tracks" / tracks_meta["speakers"][spk]
    proc = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    for s in segs:
        span = audio[int(s["start"] * 16000): int(s["end"] * 16000)]
        s["_rms"] = float(np.sqrt((span ** 2).mean())) if len(span) else 0.0

for spk in {s["speaker"] for s in segments}:
    track_rms(spk, [s for s in segments if s["speaker"] == spk])

print(f"{dropped_orphan} orphan fragments (gap-split bleed strays) dropped")
segments.sort(key=lambda s: (s["start"], s["end"]))

# 1. mic-bleed echo dedupe: the same phrase heard on two stems at the
#    same moment was said once — keep the louder reading, drop the bleed
dropped_echo = []
for i, a in enumerate(segments):
    if a.get("_drop"):
        continue
    for b in segments[i + 1:]:
        if b["start"] - a["start"] > ECHO_DT:
            break
        if (b.get("_drop") or b["speaker"] == a["speaker"]):
            continue
        if SequenceMatcher(None, a["text"].lower(), b["text"].lower()).ratio() >= ECHO_SIM:
            loser = a if a["_rms"] < b["_rms"] else b
            loser["_drop"] = True
            dropped_echo.append(loser)

# 1b. echo FRAGMENTS: bleed loud enough to pass the word filter comes
#     out as a quiet collage of the neighbours' words — if most of a
#     segment's words occur in other speakers' speech nearby and the
#     segment is much quieter than its speaker's norm, it is bleed too
med = {spk: float(np.median([s["_rms"] for s in segments
                             if s["speaker"] == spk and not s.get("_drop")]))
       for spk in {s["speaker"] for s in segments}}
WORD = re.compile(r"[a-zа-яё]{3,}", re.I)
dropped_frag = []
for s in segments:
    if s.get("_drop") or s["_rms"] >= ECHO_FRAG_RMS * med[s["speaker"]]:
        continue
    own = WORD.findall(s["text"].lower())
    if len(own) < 3:
        continue
    neigh = " ".join(o["text"].lower() for o in segments
                     if o is not s and not o.get("_drop")
                     and o["speaker"] != s["speaker"]
                     and o["start"] < s["end"] + 4 and o["end"] > s["start"] - 4)
    found = sum(1 for w in own if w in neigh)
    if found / len(own) >= ECHO_FRAG_WORDS:
        s["_drop"] = True
        dropped_frag.append(s)

# 2. whisper hallucinations on breath islands: known phrases, quiet on
#    the stem relative to that speaker's own median loudness
dropped_halluc = []
for s in segments:
    if (not s.get("_drop")
            and s["text"].strip().lower() in HALLUCINATIONS
            and s["_rms"] < QUIET_FRACTION * med[s["speaker"]]):
        s["_drop"] = True
        dropped_halluc.append(s)

segments = [s for s in segments if not s.get("_drop")]

# 3. non-verbal vocalizations (mm-hmm, hmm, laughter-as-text): paste the
#    original sound from the solo stem instead of synthesizing it
nonverbal_dir = work / "nonverbal"
n_nonverbal = 0
for s in segments:
    if not NONVERBAL.match(s["text"].strip()):
        continue
    nonverbal_dir.mkdir(exist_ok=True)
    wav = nonverbal_dir / f"{s['speaker']}_{int(s['start'] * 10):06d}.wav"
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{s['start']:.3f}",
        "-t", f"{s['end'] - s['start']:.3f}",
        "-i", str(work / "tracks" / tracks_meta["speakers"][s["speaker"]]),
        "-ar", "44100", "-ac", "1", str(wav)])
    s["type"] = "orig"
    s["file"] = f"nonverbal/{wav.name}"
    s["label"] = "backchannel"
    n_nonverbal += 1

for s in segments:
    s.pop("_rms", None)

print(f"housekeeping: {len(dropped_echo)} mic-bleed echoes + "
      f"{len(dropped_frag)} echo fragments dropped, "
      f"{len(dropped_halluc)} hallucinations dropped, "
      f"{n_nonverbal} non-verbal segments pasted from stems")

# concurrency marking: a segment whose start lies ≥ min_overlap inside
# ANY other speaker's still-running segment is overlaid, not sequential
n_overlap = 0
for i, seg in enumerate(segments):
    if seg.get("type") == "orig":
        continue  # pasted stem audio needs no overlay math
    prev = None
    for p in segments[:i]:
        if (p["speaker"] != seg["speaker"] and not p.get("overlap")
                and p.get("type") != "orig"
                and p["start"] < seg["start"]
                and min(p["end"], seg["end"]) - seg["start"] >= min_overlap):
            if prev is None or p["start"] > prev["start"]:
                prev = p
    if prev is not None:
        seg["overlap"] = True
        seg["pan"] = 0.35
        seg["gain_db"] = -2
        prev["pan"] = -0.35
        n_overlap += 1

out = {"language": language,
       "speakers": sorted({s["speaker"] for s in segments}),
       "segments": [{"id": i, **s} for i, s in enumerate(segments)]}
(work / out_name).write_text(json.dumps(out, ensure_ascii=False, indent=1))

counts = {spk: sum(1 for s in out["segments"] if s["speaker"] == spk)
          for spk in out["speakers"]}
print(f"{len(segments)} segments ({n_overlap} concurrent), "
      f"speakers={counts} -> {out_name}")

if out_name == "segments.json":
    from export_transcripts import export_work  # noqa: E402
    export_work(work)
