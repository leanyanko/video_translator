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
segments = [s for s in segments if s.get("type") != "orig"]  # orig = pasted, not voiced
# slot = time until the next segment starts; used to pace generation.
# Overlap segments live outside the sequential timeline: they get their
# own span as slot and must not shrink their neighbours' slots.
timeline = [s for s in segments if not s.get("overlap")]
slots = {}
for i, s in enumerate(timeline):
    nxt = timeline[i + 1]["start"] if i + 1 < len(timeline) else s["end"] + 2.0
    slots[s["id"]] = max(nxt - s["start"], 1.0)
for s in segments:
    if s.get("overlap"):
        slots[s["id"]] = max(s["end"] - s["start"], 1.0)
segments = [s for s in segments if s["start"] >= min_secs]
if limit_secs is not None:
    segments = [s for s in segments if s["start"] < limit_secs]

outdir = work / "tts_f5"
outdir.mkdir(exist_ok=True)

# stress marks are pre-computed by scripts/accent_texts.py in a separate
# process — RUAccent and F5 deadlock when loaded together
accents_path = work / "accents.json"
accents = json.loads(accents_path.read_text()) if accents_path.exists() else {}

# ── consistency guards ──────────────────────────────────────────────
# Both exist because their absence shipped broken audio (e1v5): a run
# whose translation was edited after accent_texts.py sang the OLD text
# with the OLD stresses, and clips carried over from earlier runs were
# never invalidated.
import hashlib
import re as _re


from stress_rules import force_name_stress as _fns  # noqa: E402


def _norm(t: str) -> str:
    # same tolerance as check_pipeline.py: marks, ёfication, respellings
    t = _fns(t or "")
    t = t.replace("+", "").replace("ё", "е").replace("Ё", "Е").lower()
    return _re.sub(r"\s+", " ", _re.sub(r"[^\w ]+", " ", t)).strip()


# 1. accents.json must exist and match the CURRENT translation; a stale
#    or missing entry means accent_texts.py was not re-run after editing
bad = [s["id"] for s in segments
       if str(s["id"]) not in accents
       or _norm(accents[str(s["id"])]) != _norm(s.get("text_ru", ""))]
if bad and "--allow-plain" not in sys.argv:
    print(f"ERROR: {len(bad)} segment(s) have missing or STALE stress marks "
          f"(accents.json does not match segments_ru.json): {bad[:20]}"
          f"{'...' if len(bad) > 20 else ''}\n"
          "Run  venv-f5/bin/python scripts/accent_texts.py "
          f"{work}  first (or pass --allow-plain to sing without marks).")
    sys.exit(2)

# 2. content manifest: seg id -> sha1(speaker | accented text | ref hash).
#    A clip whose hash no longer matches is re-synthesized, so edited
#    translations, new accents or new refs can never ship stale audio.
#    No manifest yet (pre-manifest run) = existing clips are adopted as
#    the baseline, same semantics as redub.py.
manifest_path = outdir / "manifest.json"
_ref_sha = {}


def ref_sha(spk: str) -> str:
    if spk not in _ref_sha:
        p = REFS / f"{spk}.wav"
        _ref_sha[spk] = hashlib.sha1(p.read_bytes()).hexdigest()[:12] if p.exists() else "noref"
    return _ref_sha[spk]


def seg_hash(s) -> str:
    text = accents.get(str(s["id"]), s.get("text_ru", ""))
    return hashlib.sha1(
        f"{s['speaker']}|{text}|{ref_sha(s['speaker'])}".encode()).hexdigest()


if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
else:
    manifest = {}
    existing = [s for s in segments
                if (outdir / f"seg_{s['id']:04d}.wav").exists()]
    if existing and "--adopt-existing" in sys.argv:
        # explicit opt-in for pre-manifest dirs KNOWN to match the
        # current text/accents/refs (e.g. a run finished minutes ago)
        manifest = {str(s["id"]): seg_hash(s) for s in existing}
        print(f"manifest bootstrap: adopting {len(manifest)} existing clips "
              "as matching the current script (--adopt-existing)")
    elif existing:
        print(f"NOTE: {len(existing)} clips exist but there is no manifest — "
              "they cannot be verified against the current script and WILL "
              "be re-synthesized. If you are sure they match, rerun with "
              "--adopt-existing.")

if only_ids is not None:
    todo = [s for s in segments if s["id"] in only_ids]
else:
    todo = [s for s in segments
            if not (outdir / f"seg_{s['id']:04d}.wav").exists()
            or manifest.get(str(s["id"])) != seg_hash(s)]
    n_stale = sum(1 for s in todo if (outdir / f"seg_{s['id']:04d}.wav").exists())
    if n_stale:
        print(f"{n_stale} existing clip(s) are stale (text/accents/ref "
              "changed) and will be re-synthesized")
todo.sort(key=lambda s: (s["speaker"], s["start"]))
print(f"{len(todo)} of {len(segments)} segments to synthesize")
manifest_path.write_text(json.dumps(manifest, indent=1))
if not todo:
    sys.exit(0)

tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=str(CKPT), vocab_file=str(VOCAB))

# split long texts into single-batch pieces (see text_split.py for why)
from text_split import split_single_batch


import numpy as np
import soundfile as sf

ref_text_cache = {}
piece_cap = {}  # per speaker: our split cap, set BELOW F5's own
# F5 internally re-chunks gen_text at max_chars = ref_bytes/ref_sec *
# (22 - ref_sec) * speed and BATCHES the chunks — multi-batch segfaults
# on MPS. Keep every piece under that limit (speed>=1 only raises it)
# so each infer call stays a single batch.
t_start = time.time()
for n, seg in enumerate(todo, 1):
    spk = seg["speaker"]
    ref_wav = REFS / f"{spk}.wav"
    if spk not in ref_text_cache:
        ref_text_cache[spk] = (REFS / f"{spk}.txt").read_text().strip()
        info = sf.info(str(ref_wav))
        ref_sec = info.frames / info.samplerate
        f5_max = (len(ref_text_cache[spk].encode("utf-8"))
                  / ref_sec * max(22 - ref_sec, 1.0))
        piece_cap[spk] = f5_max
        print(f"{spk}: ref {ref_sec:.1f}s, F5 max_chars {f5_max:.0f} bytes",
              flush=True)
    text = accents.get(str(seg["id"]), seg["text_ru"])
    t0 = time.time()
    # pace so estimated duration (~12.5 plain chars/sec) fits the slot
    plain_chars = len(seg["text_ru"])
    est_dur = plain_chars / 12.5
    if seg.get("fill_span"):
        # an overlapped (interrupted) line must occupy its FULL real
        # span so the interrupter audibly talks over it — slow down if
        # the translation is shorter than the original speech
        span = max(seg["end"] - seg["start"], 1.0)
        speed = min(max(est_dur / span, 0.8), 1.3)
    else:
        speed = min(max(est_dur / (slots[seg["id"]] * 0.95), 1.0), 1.3)
    waves = []
    sr_out = 24000
    # F5's limit scales with speed; speed<1 (slowed fill_span) shrinks it
    cap = min(150, int(piece_cap[spk] * min(speed, 1.0) * 0.9))
    for piece in split_single_batch(text, cap):
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
    manifest[str(seg["id"])] = seg_hash(seg)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    rate = (time.time() - t_start) / n
    print(f"[{n}/{len(todo)}] seg {seg['id']:04d} {spk} "
          f"({time.time()-t0:.1f}s, avg {rate:.1f}s/seg)", flush=True)

print("done")
