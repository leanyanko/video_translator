#!/usr/bin/env python
"""Back-transcribe every synthesized clip and flag the bad ones.

Model: mlx-community/whisper-large-v3-turbo (Russian, auto-downloaded).

Usage: venv/bin/python scripts/qc_tts.py work/<id> [--tts-dir tts_f5] [--only 1,2]

RU_INSTRUCTIONS §5 quality control, automated: F5 sometimes produces
mumbling, breaths, garbled or extra words — mostly at the END of a clip,
and more often on very short lines or when the pace was forced. Each
clip is transcribed back and compared with what was actually SUNG — the
accents.json text, where the dictionary deliberately respells names
(«Гельберт»); text_ru is a fallback variant, best of the two counts
(qc_text.py, TODO #6). Flags:
  MISMATCH  word-level similarity to the script < SIM_MIN
  EXTRA     whisper heard noticeably more words than the script has
  TAIL      > TAIL_SECS of sound after whisper's last word (mumble/noise)
  LEAD      > TAIL_SECS of sound before whisper's first word
  SILENT    nothing recognized at all
Writes work/<id>/qc_<tts-dir>.json, e.g. qc_tts_f5.json — one report
per engine, so QCing tts_qwen3 no longer overwrites the F5 report
(TODO #8). Entries: {id: {flags, sim, heard, script[, sung]}}. Prints
the flagged clips worst first. Re-synthesize them with
  venv-f5/bin/python scripts/synthesize_f5.py work/<id> --only <ids> --seed 7
(after moving the bad wavs aside) and re-run this check with --only.
"""
import json
import sys
from pathlib import Path

import mlx_whisper
import numpy as np
import soundfile as sf

from cli_args import opt
from manifest_bridge import voiced
from qc_text import script_sim

MODEL = "mlx-community/whisper-large-v3-turbo"
SIM_MIN = 0.75
TAIL_SECS = 0.8
LOUD = 0.02  # amplitude treated as sound for lead/tail measurement

work = Path(sys.argv[1])
tts = work / opt("--tts-dir", "tts_f5")
only = opt("--only", None, lambda v: {int(x) for x in v.split(",")})
segs = [s for s in json.loads((work / "segments_ru.json").read_text())["segments"]
        if voiced(s) and (only is None or s["id"] in only)]
accents_path = work / "accents.json"
accents = json.loads(accents_path.read_text()) if accents_path.exists() else {}

out_path = work / f"qc_{tts.name}.json"
legacy = work / "qc_tts.json"  # pre-TODO#8 name, read-only fallback
merge_src = out_path if out_path.exists() else \
    (legacy if tts.name == "tts_f5" and legacy.exists() else None)
results = json.loads(merge_src.read_text()) if merge_src and only else {}
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
    sung = accents.get(str(s["id"]), s["text_ru"])
    sim, heard_w, sung_w = script_sim(heard, sung, s["text_ru"])
    flags = []
    if not heard_w:
        flags.append("SILENT")
    else:
        if sim < SIM_MIN:
            flags.append("MISMATCH")
        if len(heard_w) > len(sung_w) * 1.25 + 2:
            flags.append("EXTRA")
        loud = np.where(np.abs(audio) > LOUD)[0]
        if len(loud) and ws:
            lead = ws[0]["start"] - loud[0] / sr
            tail = loud[-1] / sr - ws[-1]["end"]
            if tail > TAIL_SECS:
                flags.append("TAIL")
            if lead > TAIL_SECS:
                flags.append("LEAD")
    entry = {"flags": flags, "sim": round(sim, 2), "dur": round(len(audio) / sr, 2),
             "speaker": s["speaker"], "start": s["start"],
             "heard": heard, "script": s["text_ru"].replace("+", "")}
    if sung.replace("+", "") != s["text_ru"].replace("+", ""):
        entry["sung"] = sung.replace("+", "")  # dictionary respellings etc.
    results[str(s["id"])] = entry
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
