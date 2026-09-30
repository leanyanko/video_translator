#!/usr/bin/env python
"""Export human-readable transcripts + segment data in both languages.

Model: none (pure JSON/text reshaping of existing segment files).

Usage: venv/bin/python scripts/export_transcripts.py work/<id> [work/<id2> ...]
Writes into work/<id>/transcripts/:
  transcript_en.txt  - full English transcript, speaker-labelled turns
  transcript_ru.txt  - full Russian transcript, same structure
  segments_en.json   - per-segment English data (id/start/end/speaker/text)
  segments_ru.json   - per-segment Russian data (same, with text_ru)
Skips whichever language isn't available yet and says so.
"""
import json
import shutil
import sys
from pathlib import Path


def fmt_ts(secs: float) -> str:
    s = int(secs)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def write_transcript(segments, text_key: str, path: Path) -> bool:
    if not segments or not all(text_key in s for s in segments):
        return False
    lines = []
    turn = None  # [start, speaker, [texts]]
    for s in segments:
        if turn and s["speaker"] == turn[1]:
            turn[2].append(s[text_key])
        else:
            if turn:
                lines.append(f"[{fmt_ts(turn[0])}] {turn[1]}:\n{' '.join(turn[2])}\n")
            turn = [s["start"], s["speaker"], [s[text_key]]]
    if turn:
        lines.append(f"[{fmt_ts(turn[0])}] {turn[1]}:\n{' '.join(turn[2])}\n")
    path.write_text("\n".join(lines))
    return True


def export_work(work: Path) -> None:
    """Export transcripts for every language whose segment file exists.

    Each run writes into transcripts/<timestamp>/ so earlier exports are
    never overwritten and runs can be compared.
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%m%d-%H%M%S")
    out = work / "transcripts" / stamp
    print(f"== {work.name} → transcripts/{stamp}/")

    for src, text_key, txt_name, json_name in [
        ("segments.json", "text", "transcript_en.txt", "segments_en.json"),
        ("segments_ru.json", "text_ru", "transcript_ru.txt", "segments_ru.json"),
    ]:
        src_path = work / src
        if not src_path.exists():
            print(f"  {src} missing — skipped ({txt_name})")
            continue
        out.mkdir(parents=True, exist_ok=True)
        segments = json.loads(src_path.read_text())["segments"]
        if write_transcript(segments, text_key, out / txt_name):
            shutil.copyfile(src_path, out / json_name)
            print(f"  {txt_name} + {json_name} ({len(segments)} segments)")
        else:
            print(f"  {src} lacks '{text_key}' texts — skipped")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        export_work(Path(arg))
