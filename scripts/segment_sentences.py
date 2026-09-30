#!/usr/bin/env python
"""Sentence-boundary segmentation with word-level speaker attribution.

Model: mlx-community/whisper-large-v3-turbo (word timestamps) +
pre-computed pyannote turns from diarization.json (run diarize.py first,
or copy its diarization.json into the work dir).

Usage: venv/bin/python scripts/segment_sentences.py work/<id> [--limit-secs N]

Unlike diarize.py (which keeps whisper's raw segment cuts and can break
mid-sentence), this script:
  1. re-transcribes with word timestamps,
  2. assigns EVERY WORD a speaker by overlap with diarization turns,
  3. builds sentences (cut at sentence-final punctuation or speaker change),
  4. merges short same-speaker sentences up to TARGET_SECS.
Cuts therefore never land inside a sentence. Writes segments.json
(+ words.json for debugging) and exports the EN transcript.
"""
import json
import sys
from pathlib import Path

import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"
TARGET_SECS = 12.0   # merge short same-speaker sentences up to this
MAX_PAUSE = 1.0      # never merge across a pause longer than this
SENT_END = (".", "!", "?", "…")

work = Path(sys.argv[1])
limit_secs = float(sys.argv[sys.argv.index("--limit-secs") + 1]) if "--limit-secs" in sys.argv else None

turns = json.loads((work / "diarization.json").read_text())
if limit_secs:
    turns = [t for t in turns if t["start"] < limit_secs]

# prefer words.json from transcribe.py (transcription is its own step);
# fall back to transcribing here only if it is missing
words_path = work / "words.json"
if words_path.exists():
    data = json.loads(words_path.read_text())
    words = data["words"] if isinstance(data, dict) else data
    language = data.get("language") if isinstance(data, dict) else None
    if limit_secs:
        words = [w for w in words if w["start"] < limit_secs]
    print(f"loaded {len(words)} words from words.json")
else:
    audio = work / "audio.wav"
    if limit_secs:
        import subprocess
        clip = work / f"audio_first{int(limit_secs)}s.wav"
        if not clip.exists():
            subprocess.check_call([
                "ffmpeg", "-y", "-v", "quiet", "-i", str(audio),
                "-t", str(limit_secs), "-c", "copy", str(clip),
            ])
        audio = clip
    print("no words.json — transcribing with word timestamps...")
    result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=MODEL, word_timestamps=True)
    language = result.get("language")
    words = [
        {"start": w["start"], "end": w["end"], "word": w["word"]}
        for seg in result["segments"]
        for w in seg.get("words", [])
    ]
    words_path.write_text(
        json.dumps({"language": language, "words": words}, ensure_ascii=False)
    )


def speaker_at(start: float, end: float) -> str | None:
    best, best_ov = None, 0.0
    for t in turns:
        ov = min(end, t["end"]) - max(start, t["start"])
        if ov > best_ov:
            best, best_ov = t["speaker"], ov
    return best


# tag words with speakers; words in diarization gaps inherit the nearest
# neighbour's speaker (decided later via sentence majority)
for w in words:
    w["speaker"] = speaker_at(w["start"], w["end"])

# build sentences: a sentence ends at sentence-final punctuation
sentences = []
cur = []
for w in words:
    cur.append(w)
    if w["word"].strip().endswith(SENT_END):
        sentences.append(cur)
        cur = []
if cur:
    sentences.append(cur)


def majority_speaker(ws) -> str:
    weights = {}
    for w in ws:
        if w["speaker"]:
            weights[w["speaker"]] = weights.get(w["speaker"], 0.0) + (w["end"] - w["start"])
    return max(weights, key=weights.get) if weights else "SPEAKER_00"


sent_items = [
    {
        "start": round(ws[0]["start"], 3),
        "end": round(ws[-1]["end"], 3),
        "speaker": majority_speaker(ws),
        "text": "".join(w["word"] for w in ws).strip(),
    }
    for ws in sentences
    if "".join(w["word"] for w in ws).strip()
]

# merge short same-speaker sentences (never across long pauses)
merged = []
for s in sent_items:
    if (
        merged
        and s["speaker"] == merged[-1]["speaker"]
        and s["start"] - merged[-1]["end"] <= MAX_PAUSE
        and s["end"] - merged[-1]["start"] <= TARGET_SECS
    ):
        merged[-1]["end"] = s["end"]
        merged[-1]["text"] += " " + s["text"]
    else:
        merged.append(dict(s))

segments = [{"id": i, **s} for i, s in enumerate(merged)]
speakers = sorted({s["speaker"] for s in segments})
out = {"language": language, "speakers": speakers, "segments": segments}
(work / "segments.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

counts = {spk: sum(1 for s in segments if s["speaker"] == spk) for spk in speakers}
mid_cut = sum(1 for s in segments if not s["text"].rstrip().endswith(SENT_END))
print(f"{len(sent_items)} sentences → {len(segments)} segments, "
      f"speakers={counts}, segments not ending on sentence punctuation: {mid_cut}")

from export_transcripts import export_work  # noqa: E402

export_work(work)
