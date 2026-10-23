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
from pathlib import Path

from cli_args import opt

import numpy as np

from audio_utils import decode
from segmenting import (GAP_BREAK, MAX_PAUSE, MAX_SENT_SECS, SENT_END,
                        TARGET_SECS, build_sentences, merge_short,
                        runon_cut_ids, split_long_flagged, words_text)

# cross-track housekeeping (none of this exists in the mix flow).
# Deleted 10-23 (CLAUDE.md §3, tracker d7): the mic-bleed echo dedupe and
# the quiet-hallucination drop caught 0 segments on all three regression
# sets (traps / baseline20 / e1v7: 0/0/0 and 0/0/0) — the word-level bleed
# filter and cross-track echo upstream already remove what they targeted.
# non-verbal vocalizations are language-independent: pasted from the solo
# stem as type:"orig" instead of being translated and synthesized
NONVERBAL = re.compile(r"^[-\s.,!?…]*(?:(?:m+[-\s]?h?m+|h+m+|uh[-\s]?huh|mhm+|ha(?:[-\s]?ha)+)[-.!?,…\s]*)+$", re.I)
ECHO_FRAG_WORDS = 0.7   # share of words found in neighbours' speech
ECHO_FRAG_RMS = 0.5     # of the speaker's median segment RMS

work = Path(sys.argv[1])
min_overlap = opt("--min-overlap", 0.3, float)
out_name = opt("--out", "segments.json")

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
    audio = decode(path, 16000)
    ws = [w for w in words if w["speaker"] == spk]
    rms = []
    for w in ws:
        span = audio[int(w["start"] * 16000): int(w["end"] * 16000)]
        rms.append(float(np.sqrt((span ** 2).mean())) if len(span) else 0.0)
    thr = max(BLEED_FLOOR, BLEED_FRACTION * float(np.percentile(rms, 95)))
    for w, r in zip(ws, rms):
        w["_rms"] = r  # own-stem loudness, for the cross-track echo pass
    good = [w for w, r in zip(ws, rms) if r >= thr]
    print(f"{spk}: bleed filter thr={thr:.4f} dropped "
          f"{len(ws) - len(good)}/{len(ws)} words")
    kept.extend(good)
words = sorted(kept, key=lambda w: (w["start"], w["end"]))
del kept

# word-level cross-track echo: a mic close to another speaker picks up
# whole phrases loud enough to pass the bleed filter (seen: the host's
# questions transcribed on the guest's stem at confidence ~1.0, which
# would then be dubbed in the GUEST's voice). The same word on two stems
# within ECHO_WORD_DT belongs to the stem where it is louder; the quieter
# copy is dropped when it is part of a run of ≥ ECHO_WORD_RUN such words —
# echoes come as phrases, while two people saying "yeah" at once don't.
ECHO_WORD_DT, ECHO_WORD_RATIO, ECHO_WORD_RUN = 0.6, 0.9, 3
import bisect  # noqa: E402

def _key(w):
    return "".join(c for c in w["word"].lower() if c.isalnum())

starts = [w["start"] for w in words]
echoed = set()
for i, w in enumerate(words):
    k = _key(w)
    if not k:
        continue
    j = bisect.bisect_left(starts, w["start"] - ECHO_WORD_DT)
    while j < len(words) and words[j]["start"] <= w["start"] + ECHO_WORD_DT:
        o = words[j]
        if (o["speaker"] != w["speaker"] and _key(o) == k
                and w["_rms"] < ECHO_WORD_RATIO * o["_rms"]):
            echoed.add(id(w))
            break
        j += 1
drop = set()
for spk in {w["speaker"] for w in words}:
    own = [w for w in words if w["speaker"] == spk]
    run = []
    for w in own + [None]:
        if w is not None and id(w) in echoed:
            run.append(w)
            continue
        if len(run) >= ECHO_WORD_RUN:
            drop.update(id(x) for x in run)
        run = []
n_echo_words = len(drop)
words = [w for w in words if id(w) not in drop]
print(f"cross-track echo: dropped {n_echo_words} words heard louder on another stem")

# per-speaker sentence building (segmenting.py: the same rules as
# segment_sentences.py, plus the run-on guards GAP_BREAK / split_long;
# MAX_SENT_SECS lives there too). A last-resort cut with no clause mark
# carries cut:"mid-sentence" into the segment (TODO #12).
segments = []
dropped_orphan = 0
for spk in sorted({w["speaker"] for w in words}):
    stream = [w for w in words if w["speaker"] == spk]
    sentences = build_sentences(stream, gap_break=GAP_BREAK)
    pieces = [pf for ws in sentences
              for pf in split_long_flagged(ws, MAX_SENT_SECS)]
    # stray bleed words split off by the gap rule come out as tiny
    # fragments with no sentence-final punctuation — not translatable
    kept = []
    for ws, mid in pieces:
        text = words_text(ws)
        if (len(ws) <= 3 and not text.rstrip().endswith(SENT_END)
                and not NONVERBAL.match(text)):
            dropped_orphan += 1
            continue
        kept.append((ws, mid))
    items = []
    for ws, mid in kept:
        text = words_text(ws)
        if not text:
            continue
        items.append({"start": round(ws[0]["start"], 3),
                      "end": round(ws[-1]["end"], 3),
                      "speaker": spk, "text": text,
                      **({"cut": "mid-sentence"} if mid else {})})
    segments.extend(merge_short(items, TARGET_SECS, MAX_PAUSE))

# ── cross-track housekeeping ────────────────────────────────────────
def track_rms(spk: str, segs: list) -> None:
    """Attach each segment's RMS measured on its OWN speaker's stem."""
    path = work / "tracks" / tracks_meta["speakers"][spk]
    audio = decode(path, 16000)
    for s in segs:
        span = audio[int(s["start"] * 16000): int(s["end"] * 16000)]
        s["_rms"] = float(np.sqrt((span ** 2).mean())) if len(span) else 0.0

for spk in {s["speaker"] for s in segments}:
    track_rms(spk, [s for s in segments if s["speaker"] == spk])

print(f"{dropped_orphan} orphan fragments (gap-split bleed strays) dropped")
segments.sort(key=lambda s: (s["start"], s["end"]))

# 1. echo FRAGMENTS: bleed loud enough to pass the word filter comes
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

segments = [s for s in segments if not s.get("_drop")]

# 2. non-verbal vocalizations (mm-hmm, hmm, laughter-as-text): paste the
#    original sound from the solo stem instead of synthesizing it
# whisper confidence (words carry "prob" since transcribe_tracks.py kept
# it): short low-confidence lines are crosstalk/quiet backchannels that
# whisper garbles ("Prostar my name is Kloé" for "second one") — pasting
# the speaker's own stem audio keeps the real voice and never translates
# garbage. Mid-confidence lines are translated but flagged low_conf.
PASTE_CONF, FLAG_CONF, PASTE_MAX_SECS = 0.5, 0.65, 4.0
words_by_spk = {}
for w in words:
    words_by_spk.setdefault(w["speaker"], []).append(w)
for s in segments:
    ws = [w for w in words_by_spk.get(s["speaker"], ())
          if "prob" in w and w["start"] >= s["start"] - 0.05 and w["end"] <= s["end"] + 0.05]
    if ws:
        s["_conf"] = sum(w["prob"] for w in ws) / len(ws)

from audio_utils import raw_stem_source  # noqa: E402
from bleed import OWN_OVER_BLEED, bleed_expected, calibrate  # noqa: E402

paste_speakers, paste_dir = raw_stem_source(work)
# Which copy to paste from is PER KIND (review 10-10 #3):
#   nonverbal (ha-ha, mm-hmm) — always RAW: cleaning mutes exactly these;
#   low_conf — often garbled BECAUSE of crosstalk, and the raw cut would
#     bring the other speaker's English back where the cleaned copy had
#     it muted. Decide per segment with the calibrated bleed model: RAW
#     when the owner's own level beats expected bleed, else the CLEANED
#     copy.
_speech_spans = {}
for s in segments:
    _speech_spans.setdefault(s["speaker"], []).append((s["start"], s["end"]))
_raw_rms = {}
for spk, fname in paste_speakers.items():
    _audio = decode(paste_dir / fname, 16000)
    _raw_rms[spk] = np.sqrt(
        (_audio[: len(_audio) // 4000 * 4000].reshape(-1, 4000) ** 2).mean(1))
_coeffs = calibrate(_raw_rms, _speech_spans, 0.25)


def _span_rms(spk, a, b):
    r = _raw_rms[spk][int(a / 0.25): max(int(b / 0.25), int(a / 0.25) + 1)]
    return float(r.mean()) if len(r) else 0.0


def low_conf_source(spk, a, b):
    """(dir, speakers_map, tag) for a low_conf paste: raw when the
    owner's own raw level beats what calibrated bleed would explain."""
    own = _span_rms(spk, a, b)
    others = {o: _span_rms(o, a, b) for o in _raw_rms if o != spk}
    expected = bleed_expected(_coeffs, spk, others)
    if expected is not None:
        raw_ok = own > OWN_OVER_BLEED * expected
    else:  # uncalibrated pair: fixed-ratio fallback, same as harvest
        raw_ok = max(others.values(), default=0.0) < 6.0 * own
    if raw_ok:
        return paste_dir, paste_speakers, "raw"
    return work / "tracks", tracks_meta["speakers"], "cleaned"


nonverbal_dir = work / "nonverbal"
n_nonverbal = n_lowconf = n_flagged = 0
for s in segments:
    conf = s.pop("_conf", None)
    low = (conf is not None and conf < PASTE_CONF
           and s["end"] - s["start"] <= PASTE_MAX_SECS)
    if (not low and conf is not None and conf < FLAG_CONF
            and not NONVERBAL.match(s["text"].strip())):
        s["low_conf"] = round(conf, 2)
        n_flagged += 1
    if not low and not NONVERBAL.match(s["text"].strip()):
        continue
    if low:
        src_dir, src_map, src_tag = low_conf_source(
            s["speaker"], s["start"], s["end"])
        if src_tag == "cleaned":
            print(f"  low_conf paste {s['speaker'][-2:]} {s['start']:.1f}s: "
                  "CLEANED copy (raw would bring crosstalk back)")
    else:
        src_dir, src_map, src_tag = paste_dir, paste_speakers, "raw"
    nonverbal_dir.mkdir(exist_ok=True)
    wav = nonverbal_dir / f"{s['speaker']}_{int(s['start'] * 10):06d}.wav"
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{s['start']:.3f}",
        "-t", f"{s['end'] - s['start']:.3f}",
        "-i", str(src_dir / src_map[s["speaker"]]),
        "-ar", "44100", "-ac", "1", str(wav)])
    s["type"] = "orig"
    s["file"] = f"nonverbal/{wav.name}"
    if low:
        s["label"] = "low_conf"
        s["conf"] = round(conf, 2)
        n_lowconf += 1
    else:
        s["label"] = "backchannel"
        n_nonverbal += 1

for s in segments:
    s.pop("_rms", None)

print(f"housekeeping: {len(dropped_frag)} echo fragments dropped, "
      f"{n_nonverbal} non-verbal segments pasted from stems, "
      f"{n_lowconf} low-confidence lines pasted from stems, "
      f"{n_flagged} flagged low_conf for the translator")

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

# mid-sentence cut report (segmenting.runon_cut_ids, shared with the
# doctor's gate): a cut inside a run-on is a defect.
runon_cuts, interrupted = runon_cut_ids(out["segments"])
print(f"mid-sentence ends: {interrupted} at interruptions (ok), "
      f"{len(runon_cuts)} inside run-ons"
      + (f" — ids {runon_cuts[:20]}: whisper lost punctuation there, "
         "fix with repunctuate_words.py and re-segment" if runon_cuts else ""))

counts = {spk: sum(1 for s in out["segments"] if s["speaker"] == spk)
          for spk in out["speakers"]}
print(f"{len(segments)} segments ({n_overlap} concurrent), "
      f"speakers={counts} -> {out_name}")

if out_name == "segments.json":
    from export_transcripts import export_work  # noqa: E402
    export_work(work)
