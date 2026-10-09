#!/usr/bin/env python
"""Synthesize Russian segments with Chatterbox Multilingual (Resemble AI).

Model: ResembleAI/chatterbox (multilingual, 23 languages incl. Russian;
MIT weights, output carries Resemble's inaudible Perth watermark).
Experimental alternative to synthesize_f5.py, same contract: reads
segments_ru.json + refs_f5/, writes <--outdir>/seg_XXXX.wav +
manifest.json. Stress marks are STRIPPED (testing the model's own
lexical stress). Our reference clips are ENGLISH speech, so cfg_weight
defaults to 0 per the vendor's guidance to avoid English-accent
transfer into Russian; override with --cfg.

Usage: venv-cbx/bin/python scripts/synthesize_chatterbox.py work/<id>
           [--outdir tts_cbx] [--only 1,2] [--device mps|cpu] [--cfg 0]
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch
import torchaudio

work = Path(sys.argv[1])
outname = sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else "tts_cbx"
device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else (
    "mps" if torch.backends.mps.is_available() else "cpu")
cfg = float(sys.argv[sys.argv.index("--cfg") + 1]) if "--cfg" in sys.argv else 0.0
only_ids = ({int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}
            if "--only" in sys.argv else None)

REFS = work / ("refs_f5" if (work / "refs_f5").exists() else "refs")
segments = [s for s in json.loads((work / "segments_ru.json").read_text())["segments"]
            if s.get("type") != "orig" and s.get("text_ru")]
outdir = work / outname
outdir.mkdir(exist_ok=True)
manifest_path = outdir / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}


def plain(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("+", "")).strip()


_ref_sha = {}


def ref_sha(spk):
    if spk not in _ref_sha:
        p = REFS / f"{spk}.wav"
        _ref_sha[spk] = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
    return _ref_sha[spk]


def seg_hash(s):
    return hashlib.sha1(
        f"cbx|{cfg}|{s['speaker']}|{plain(s['text_ru'])}|{ref_sha(s['speaker'])}".encode()
    ).hexdigest()


if only_ids is not None:
    todo = [s for s in segments if s["id"] in only_ids]
else:
    todo = [s for s in segments
            if not (outdir / f"seg_{s['id']:04d}.wav").exists()
            or manifest.get(str(s["id"])) != seg_hash(s)]
print(f"{len(todo)} of {len(segments)} segments to synthesize "
      f"(device={device}, cfg_weight={cfg})", flush=True)
if not todo:
    sys.exit(0)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

model = ChatterboxMultilingualTTS.from_pretrained(device=device)

t0 = time.time()
for n, s in enumerate(sorted(todo, key=lambda x: (x["speaker"], x["start"])), 1):
    wav = model.generate(
        plain(s["text_ru"]),
        language_id="ru",
        audio_prompt_path=str(REFS / f"{s['speaker']}.wav"),
        cfg_weight=cfg,
    )
    torchaudio.save(str(outdir / f"seg_{s['id']:04d}.wav"), wav, model.sr)
    manifest[str(s["id"])] = seg_hash(s)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"[{n}/{len(todo)}] seg {s['id']:04d} {s['speaker']} "
          f"(avg {(time.time()-t0)/n:.1f}s/seg)", flush=True)
print("done")
