#!/usr/bin/env python
"""Synthesize Russian audio for every segment via the local Fish Speech
API server, using each segment's own speaker reference (multi-voice).

Usage: venv/bin/python scripts/synthesize.py work/<id> [--limit-secs N]
Reads segments_ru.json ("text_ru" + "speaker" per segment) and refs/.
Segments are processed grouped by speaker so the server's reference
cache stays warm. Writes tts/seg_XXXX.wav; resumable.
--limit-secs N only synthesizes segments starting before N seconds
(for preview samples).
"""
import json
import sys
import time
from pathlib import Path

import ormsgpack
import requests

URL = "http://127.0.0.1:8080/v1/tts"

work = Path(sys.argv[1])
limit_secs = None
if "--limit-secs" in sys.argv:
    limit_secs = float(sys.argv[sys.argv.index("--limit-secs") + 1])

segments = json.loads((work / "segments_ru.json").read_text())["segments"]
if limit_secs is not None:
    segments = [s for s in segments if s["start"] < limit_secs]

refs = {}
for f in (work / "refs").glob("*.wav"):
    refs[f.stem] = {
        "audio": f.read_bytes(),
        "text": (work / "refs" / f"{f.stem}.txt").read_text().strip(),
    }

outdir = work / "tts"
outdir.mkdir(exist_ok=True)

todo = [s for s in segments if not (outdir / f"seg_{s['id']:04d}.wav").exists()]
todo.sort(key=lambda s: (s["speaker"], s["start"]))  # group by speaker
print(f"{len(todo)} of {len(segments)} segments to synthesize")

t_start = time.time()
for n, seg in enumerate(todo, 1):
    out = outdir / f"seg_{seg['id']:04d}.wav"
    ref = refs[seg["speaker"]]
    req = {
        "text": seg["text_ru"],
        "references": [{"audio": ref["audio"], "text": ref["text"]}],
        "reference_id": None,
        "format": "wav",
        "chunk_length": 300,
        "max_new_tokens": 1024,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": 0.8,
        "streaming": False,
        "use_memory_cache": "on",
        "seed": None,
    }
    t0 = time.time()
    r = requests.post(
        URL,
        params={"format": "msgpack"},
        data=ormsgpack.packb(req),
        headers={"content-type": "application/msgpack"},
        timeout=600,
    )
    if r.status_code != 200:
        print(f"FAILED seg {seg['id']}: HTTP {r.status_code} {r.text[:200]}")
        sys.exit(1)
    out.write_bytes(r.content)
    rate = (time.time() - t_start) / n
    print(f"[{n}/{len(todo)}] seg {seg['id']:04d} {seg['speaker']} "
          f"({time.time()-t0:.1f}s, avg {rate:.1f}s/seg)", flush=True)

print("done")
