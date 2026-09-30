#!/usr/bin/env python
"""Synthesize Russian segments with the F5-TTS Russian finetune
(Misha24-10/F5-TTS_RUSSIAN, v2 checkpoint) — voice cloning per speaker.

Model: Misha24-10/F5-TTS_RUSSIAN "F5TTS_v1_Base_v2" checkpoint (CC-BY-NC)
on the F5TTS_v1_Base architecture, vocab from F5TTS_v1_Base; vocoder
charactr/vocos-mel-24khz (auto-downloaded). This is the PRODUCTION TTS.

Usage: venv-f5/bin/python scripts/synthesize_f5.py work/<id> [--min-secs A] [--limit-secs B]
Reads segments_ru.json + refs/, applies RUAccent stress marks, writes
tts_f5/seg_XXXX.wav. Resumable.
"""
import json
import sys
import time
from pathlib import Path

# torchaudio 2.11+ delegates load/save to torchcodec, which fails to link
# against Homebrew ffmpeg 8 — route both through soundfile instead.
import soundfile as _sf
import torch as _torch
import torchaudio as _ta


def _sf_load(path, *args, **kwargs):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return _torch.from_numpy(data.T), sr


def _sf_save(path, tensor, sample_rate, *args, **kwargs):
    _sf.write(str(path), tensor.detach().cpu().numpy().T, sample_rate)


_ta.load = _sf_load
_ta.save = _sf_save

from f5_tts.api import F5TTS

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints/f5-ru/F5TTS_v1_Base_v2/model_last_inference.safetensors"
VOCAB = ROOT / "checkpoints/f5-ru/F5TTS_v1_Base/vocab.txt"

work = Path(sys.argv[1])
min_secs = float(sys.argv[sys.argv.index("--min-secs") + 1]) if "--min-secs" in sys.argv else 0.0
limit_secs = float(sys.argv[sys.argv.index("--limit-secs") + 1]) if "--limit-secs" in sys.argv else None
only_ids = None
if "--only" in sys.argv:  # comma-separated ids; existing files are re-rolled
    only_ids = {int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 42

# F5 refs must be <=~11.5s with exactly matching text (F5 clips ref audio
# at 12s internally; longer refs desync text/audio and garble output)
REFS = work / ("refs_f5" if (work / "refs_f5").exists() else "refs")

segments = json.loads((work / "segments_ru.json").read_text())["segments"]
# slot = time until the next segment starts; used to pace generation
slots = {}
for i, s in enumerate(segments):
    nxt = segments[i + 1]["start"] if i + 1 < len(segments) else s["end"] + 2.0
    slots[s["id"]] = max(nxt - s["start"], 1.0)
segments = [s for s in segments if s["start"] >= min_secs]
if limit_secs is not None:
    segments = [s for s in segments if s["start"] < limit_secs]

outdir = work / "tts_f5"
outdir.mkdir(exist_ok=True)
if only_ids is not None:
    todo = [s for s in segments if s["id"] in only_ids]
else:
    todo = [s for s in segments if not (outdir / f"seg_{s['id']:04d}.wav").exists()]
todo.sort(key=lambda s: (s["speaker"], s["start"]))
print(f"{len(todo)} of {len(segments)} segments to synthesize")
if not todo:
    sys.exit(0)

# stress marks are pre-computed by scripts/accent_texts.py in a separate
# process — RUAccent and F5 deadlock when loaded together
accents_path = work / "accents.json"
accents = json.loads(accents_path.read_text()) if accents_path.exists() else {}

tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=str(CKPT), vocab_file=str(VOCAB))

# F5's multi-batch generation hangs/segfaults on MPS (torch 2.14), so
# split long texts at punctuation into pieces short enough to stay
# single-batch, synthesize each, and concatenate.
# NB: F5 budgets text in UTF-8 BYTES (Cyrillic = 2 bytes/char), so the
# cap is in bytes, well under its ~190-byte batch threshold for 11s refs
MAX_PIECE_BYTES = 150


def _blen(s: str) -> int:
    return len(s.encode("utf-8"))


def split_single_batch(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?…;]) +", text)
    pieces, cur = [], ""
    for p in parts:
        if cur and _blen(cur) + 1 + _blen(p) > MAX_PIECE_BYTES:
            pieces.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
        # a single sentence longer than the cap gets split at commas/dashes
        while _blen(cur) > MAX_PIECE_BYTES:
            limit = len(cur.encode("utf-8")[:MAX_PIECE_BYTES].decode("utf-8", "ignore"))
            cut = max(cur.rfind(c, 0, limit) for c in ",—:")
            if cut <= 0:
                cut = limit
            pieces.append(cur[: cut + 1].strip())
            cur = cur[cut + 1 :].strip()
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p]


import numpy as np
import soundfile as sf

ref_text_cache = {}
t_start = time.time()
for n, seg in enumerate(todo, 1):
    spk = seg["speaker"]
    ref_wav = REFS / f"{spk}.wav"
    if spk not in ref_text_cache:
        ref_text_cache[spk] = (REFS / f"{spk}.txt").read_text().strip()
    text = accents.get(str(seg["id"]), seg["text_ru"])
    t0 = time.time()
    # pace so estimated duration (~12.5 plain chars/sec) fits the slot
    plain_chars = len(seg["text_ru"])
    est_dur = plain_chars / 12.5
    speed = min(max(est_dur / (slots[seg["id"]] * 0.95), 1.0), 1.3)
    waves = []
    sr_out = 24000
    for piece in split_single_batch(text):
        wav, sr_out, _ = tts.infer(
            ref_file=str(ref_wav),
            ref_text=ref_text_cache[spk],
            gen_text=piece,
            speed=speed,
            show_info=lambda *a, **k: None,
            seed=SEED,
        )
        waves.append(np.asarray(wav))
    sf.write(
        outdir / f"seg_{seg['id']:04d}.wav", np.concatenate(waves), sr_out
    )
    rate = (time.time() - t_start) / n
    print(f"[{n}/{len(todo)}] seg {seg['id']:04d} {spk} "
          f"({time.time()-t0:.1f}s, avg {rate:.1f}s/seg)", flush=True)

print("done")
