#!/usr/bin/env python
"""Re-color F5 output with OpenVoice v2's tone-color converter.

Model: myshell-ai/OpenVoiceV2 "converter" checkpoint (tone-color
conversion only — its base TTS is not used; runs on CPU).

Usage: /opt/miniconda3/envs/openvoice/bin/python scripts/openvoice_convert.py work/<id> [--min-secs A] [--limit-secs B]
Source: tts_f5/seg_*.wav (already voice-cloned Russian).
Target timbre: refs/<SPEAKER>.wav (original speakers).
Output: tts_ov/seg_*.wav — for A/B against plain F5.
"""
import json
import sys
from pathlib import Path

import torch
from openvoice.api import ToneColorConverter

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints/openvoice-v2/converter"

work = Path(sys.argv[1])
min_secs = float(sys.argv[sys.argv.index("--min-secs") + 1]) if "--min-secs" in sys.argv else 0.0
limit_secs = float(sys.argv[sys.argv.index("--limit-secs") + 1]) if "--limit-secs" in sys.argv else None

segments = json.loads((work / "segments_ru.json").read_text())["segments"]
segments = [s for s in segments if s["start"] >= min_secs]
if limit_secs is not None:
    segments = [s for s in segments if s["start"] < limit_secs]

converter = ToneColorConverter(str(CKPT / "config.json"), device="cpu")
converter.load_ckpt(str(CKPT / "checkpoint.pth"))

outdir = work / "tts_ov"
outdir.mkdir(exist_ok=True)

speakers = sorted({s["speaker"] for s in segments})
src_se, tgt_se = {}, {}
for spk in speakers:
    # source timbre: what F5 actually produced for this speaker
    f5_samples = [
        str(work / "tts_f5" / f"seg_{s['id']:04d}.wav")
        for s in segments
        if s["speaker"] == spk and (work / "tts_f5" / f"seg_{s['id']:04d}.wav").exists()
    ][:3]
    src_se[spk] = converter.extract_se(f5_samples)
    # target timbre: the original speaker
    tgt_se[spk] = converter.extract_se([str(work / "refs" / f"{spk}.wav")])
    print(f"{spk}: SEs extracted from {len(f5_samples)} F5 samples")

done = 0
for seg in segments:
    src = work / "tts_f5" / f"seg_{seg['id']:04d}.wav"
    out = outdir / f"seg_{seg['id']:04d}.wav"
    if not src.exists() or out.exists():
        continue
    spk = seg["speaker"]
    converter.convert(
        audio_src_path=str(src),
        src_se=src_se[spk],
        tgt_se=tgt_se[spk],
        output_path=str(out),
    )
    done += 1
    print(f"seg {seg['id']:04d} {spk} converted", flush=True)

print(f"done: {done} segments")
