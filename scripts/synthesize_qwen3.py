#!/usr/bin/env python
"""Synthesize Russian segments with Qwen3-TTS (voice cloning per speaker).

Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base (Apache 2.0 — commercial use OK,
unlike F5-RU's CC-BY-NC). Experimental alternative to synthesize_f5.py,
same contract: reads segments_ru.json + refs_f5/, writes
<--outdir>/seg_XXXX.wav + manifest.json. Stress marks are STRIPPED —
the experiment is exactly whether Qwen3's own lexical stress is good
enough to not need them (ElevenLabs showed it is possible).

Usage: venv-qwen/bin/python scripts/synthesize_qwen3.py work/<id>
           [--outdir tts_qwen3] [--only 1,2] [--device mps|cpu]
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

work = Path(sys.argv[1])
outname = sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else "tts_qwen3"
device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else (
    "mps" if torch.backends.mps.is_available() else "cpu")
only_ids = ({int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}
            if "--only" in sys.argv else None)

REFS = work / ("refs_f5" if (work / "refs_f5").exists() else "refs")
from manifest_bridge import voiced  # orig pastes and echo lines: not voiced
segments = [s for s in json.loads((work / "segments_ru.json").read_text())["segments"]
            if voiced(s) and s.get("text_ru")]
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
        f"qwen3|{s['speaker']}|{plain(s['text_ru'])}|{ref_sha(s['speaker'])}".encode()
    ).hexdigest()


if only_ids is not None:
    todo = [s for s in segments if s["id"] in only_ids]
else:
    todo = [s for s in segments
            if not (outdir / f"seg_{s['id']:04d}.wav").exists()
            or manifest.get(str(s["id"])) != seg_hash(s)]
print(f"{len(todo)} of {len(segments)} segments to synthesize "
      f"(device={device})", flush=True)
if not todo:
    sys.exit(0)

from qwen_tts import Qwen3TTSModel  # noqa: E402

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    dtype=torch.float32 if device == "cpu" else torch.float16,
)

ref_text = {}
t0 = time.time()
for n, s in enumerate(sorted(todo, key=lambda x: (x["speaker"], x["start"])), 1):
    spk = s["speaker"]
    if spk not in ref_text:
        ref_text[spk] = (REFS / f"{spk}.txt").read_text().strip()
    wavs, sr = model.generate_voice_clone(
        text=plain(s["text_ru"]),
        language="Russian",
        ref_audio=str(REFS / f"{spk}.wav"),
        ref_text=ref_text[spk],
    )
    sf.write(outdir / f"seg_{s['id']:04d}.wav", wavs[0], sr)
    manifest[str(s["id"])] = seg_hash(s)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"[{n}/{len(todo)}] seg {s['id']:04d} {spk} "
          f"(avg {(time.time()-t0)/n:.1f}s/seg)", flush=True)
print("done")
