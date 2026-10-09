#!/usr/bin/env python
"""Back-transcribe every synthesized clip and flag the bad ones.

Model: mlx-community/whisper-large-v3-turbo (Russian, auto-downloaded).

Usage: venv/bin/python scripts/qc_tts.py work/<id> [--tts-dir tts_f5] [--only 1,2]

RU_INSTRUCTIONS §5 quality control, automated: F5 sometimes produces
mumbling, breaths, garbled or extra words — mostly at the END of a clip,
and more often on very short lines or when the pace was forced. Each clip
is transcribed back and compared with the script (text_ru, stress marks
removed). Flags:
  MISMATCH  word-level similarity to the script < SIM_MIN
  EXTRA     whisper heard noticeably more words than the script has
  TAIL      > TAIL_SECS of sound after whisper's last word (mumble/noise)
  LEAD      > TAIL_SECS of sound before whisper's first word
  SILENT    nothing recognized at all
Writes work/<id>/qc_tts.json ({id: {flags, sim, heard, script}}) and
prints the flagged clips worst first. Re-synthesize them with
  venv-f5/bin/python scripts/synthesize_f5.py work/<id> --only <ids> --seed 7
(after moving the bad wavs aside) and re-run this check with --only.
"""
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import mlx_whisper
import numpy as np
import soundfile as sf

MODEL = "mlx-community/whisper-large-v3-turbo"
SIM_MIN = 0.75
TAIL_SECS = 0.8
LOUD = 0.02  # amplitude treated as sound for lead/tail measurement

work = Path(sys.argv[1])
tts = work / (sys.argv[sys.argv.index("--tts-dir") + 1] if "--tts-dir" in sys.argv else "tts_f5")
only = ({int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}
        if "--only" in sys.argv else None)
segs = [s for s in json.loads((work / "segments_ru.json").read_text())["segments"]
        if s.get("type") != "orig" and (only is None or s["id"] in only)]


def words(t: str) -> list[str]:
    t = t.replace("+", "").lower().replace("ё", "е")
    return re.findall(r"[а-яa-z0-9]+", t)


out_path = work / "qc_tts.json"
results = json.loads(out_path.read_text()) if out_path.exists() and only else {}
for n, s in enumerate(segs, 1):
    wav = tts / f"seg_{s['id']:04d}.wav"
    if not wav.exists():
        continue
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    r = mlx_whisper.transcribe(audio if sr == 16000 else wav.as_posix(),
                               path_or_hf_repo=MODEL, language="ru", word_timestamps=True,
                               condition_on_previous_text=False)
    heard = " ".join(seg["text"].strip() for seg in r["segments"]).strip()
    ws = [w for seg in r["segments"] for w in seg.get("words", [])]
    script_w, heard_w = words(s["text_ru"]), words(heard)
    sim = SequenceMatcher(None, script_w, heard_w).ratio() if script_w else 0.0
    flags = []
    if not heard_w:
        flags.append("SILENT")
    else:
        if sim < SIM_MIN:
            flags.append("MISMATCH")
        if len(heard_w) > len(script_w) * 1.25 + 2:
            flags.append("EXTRA")
        loud = np.where(np.abs(audio) > LOUD)[0]
        if len(loud) and ws:
            lead = ws[0]["start"] - loud[0] / sr
            tail = loud[-1] / sr - ws[-1]["end"]
            if tail > TAIL_SECS:
                flags.append("TAIL")
            if lead > TAIL_SECS:
                flags.append("LEAD")
    results[str(s["id"])] = {"flags": flags, "sim": round(sim, 2), "dur": round(len(audio) / sr, 2),
                             "speaker": s["speaker"], "start": s["start"],
                             "heard": heard, "script": s["text_ru"].replace("+", "")}
    if n % 100 == 0:
        print(f"  {n}/{len(segs)}", flush=True)
out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))

bad = sorted(((k, v) for k, v in results.items() if v["flags"]), key=lambda kv: kv[1]["sim"])
print(f"{len(bad)}/{len(results)} clips flagged "
      + str({f: sum(f in v["flags"] for _, v in bad) for f in ("MISMATCH", "EXTRA", "TAIL", "LEAD", "SILENT")}))
for k, v in bad[:60]:
    print(f"  #{k:>5} {v['speaker'][-2:]} {v['start']:7.1f}s {','.join(v['flags']):14} sim {v['sim']:.2f}\n"
          f"         script: {v['script'][:90]}\n         heard : {v['heard'][:90]}")
print("bad ids:", ",".join(k for k, _ in bad))
