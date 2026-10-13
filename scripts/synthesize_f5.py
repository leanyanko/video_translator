#!/usr/bin/env python
"""Synthesize Russian segments with the F5-TTS Russian finetune
(Misha24-10/F5-TTS_RUSSIAN, v2 checkpoint) — voice cloning per speaker.

Model: Misha24-10/F5-TTS_RUSSIAN "F5TTS_v1_Base_v2" checkpoint (CC-BY-NC)
on the F5TTS_v1_Base architecture, vocab from F5TTS_v1_Base; vocoder
charactr/vocos-mel-24khz (auto-downloaded). This is the PRODUCTION TTS.

Usage: venv-f5/bin/python scripts/synthesize_f5.py work/<id> [--min-secs A] [--limit-secs B]
Reads segments_ru.json + refs/, applies RUAccent stress marks, writes
tts_f5/seg_XXXX.wav (+ manifest.json, pace.json). Resumable.

Clip length is planned, not estimated (pacing.py): the speaker's own
tempo from his reference, squeezed only to fit the slot, handed to F5
as fix_duration. Overlap lines get their own span as slot.
"""
import json
import sys
import time
from pathlib import Path

from cli_args import flag, opt

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
min_secs = opt("--min-secs", 0.0, float)
limit_secs = opt("--limit-secs", None, float)
# --only: comma-separated ids; existing files are re-rolled
only_ids = opt("--only", None, lambda v: {int(x) for x in v.split(",")})
SEED = opt("--seed", 42, int)
# --room 1.1: ask F5 for 10 % more time per piece than the plan — one rung of
# the QC re-roll ladder (reroll_qc.py): a crammed rendering is what a too-tight
# fix_duration produces, and the same text at another duration renders anew
ROOM = opt("--room", 1.0, float)

# F5 refs must be <=~11.5s with exactly matching text (F5 clips ref audio
# at 12s internally; longer refs desync text/audio and garble output)
REFS = work / ("refs_f5" if (work / "refs_f5").exists() else "refs")

segments = json.loads((work / "segments_ru.json").read_text())["segments"]
# orig = pasted, not voiced; echo = dropped by assemble (synthesizing
# them burned ~5 min/run on clips that went straight to the bin)
from manifest_bridge import slot_ends, voiced as _voiced  # noqa: E402
# slot = time until the next voiced line or own-speaker paste starts
# (manifest_bridge.slot_ends, the same rule assemble fits atempo
# against — computed on the FULL list, pastes included); used to pace
# generation. Overlap segments live outside the sequential timeline:
# they get their own span as slot and must not shrink their neighbours'.
_ends = slot_ends(segments)
segments = [s for s in segments if _voiced(s)]
slots = {}
for s in segments:
    if s.get("overlap"):
        slots[s["id"]] = max(s["end"] - s["start"], 1.0)
    else:
        nxt = _ends[s["id"]]
        nxt = s["end"] + 2.0 if nxt is None else nxt
        slots[s["id"]] = max(nxt - s["start"], 1.0)
all_speech = list(segments)  # unfiltered: the manifest covers the whole dir
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
from stress_rules import force_name_stress as _fns  # noqa: E402
from manifest_bridge import (  # noqa: E402
    load_or_bridge, ref_hasher, seg_hash as _seg_hash, stale_ids)

# 1. accents.json must exist and match the CURRENT translation; a stale
#    or missing entry means accent_texts.py was not re-run after editing.
#    Scope: with --only, check only the requested ids — an unrelated
#    stale line must not block a surgical re-roll; a full run still
#    checks every segment (that IS the protection).
scope = (segments if only_ids is None
         else [s for s in segments if s["id"] in only_ids])
bad = stale_ids(scope, accents, respell=_fns)
if bad and not flag("--allow-plain"):
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
ref_sha = ref_hasher(REFS)


def seg_hash(s) -> str:
    return _seg_hash(s, accents, ref_sha)


# bridge over the UNFILTERED speech list: with --limit-secs/--min-secs
# the filtered subset would produce a manifest that leaves every clip
# outside the window unverified for the next full run
manifest, _bridged = load_or_bridge(work, outdir, all_speech, accents,
                                    ref_sha, refs_dir=REFS)
if manifest is None:
    # no manifest, no accents.prev.json record: pre-manifest clips of
    # unknown provenance. Cautious default, unless explicitly adopted.
    manifest = {}
    existing = [s for s in segments
                if (outdir / f"seg_{s['id']:04d}.wav").exists()]
    if existing and flag("--adopt-existing"):
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
if flag("--dry-run"):
    # plan only: report what WOULD synthesize (bridge/manifest check),
    # touch nothing, load no model
    print("dry run: ids", sorted(s["id"] for s in todo)[:25],
          "..." if len(todo) > 25 else "")
    sys.exit(0)
manifest_path.write_text(json.dumps(manifest, indent=1))
if not todo:
    sys.exit(0)

tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=str(CKPT), vocab_file=str(VOCAB))

# split long texts into single-batch pieces (see text_split.py for why)
from text_split import split_single_batch


import numpy as np
import soundfile as sf

# ── pacing (tracker d5; the arithmetic and its rationale live in pacing.py)
# Each clip is given an EXACT length via F5's fix_duration: the original
# line's span (lip sync), bounded to a plausible tempo, squeezed only to
# fit the slot. fix_duration counts from the reference as F5
# PREPROCESSES it (silence split, edge trim, +50 ms) — so the reference
# is run through the same preprocess once per speaker and that length
# is used, not the file's. tts_f5/pace.json records the plan per clip
# for the doctor and for re-measuring what atempo is left downstream.
from f5_tts.infer.utils_infer import preprocess_ref_audio_text  # noqa: E402
from audio_utils import VOICED_REL, join_trimmed  # noqa: E402
from pacing import (PIECE_GAP, allocate_pieces, pace_chars,  # noqa: E402
                    plan_clip)

_quiet = lambda *a, **k: None  # noqa: E731
ref_text_cache = {}
ref_sec_f5 = {}   # per speaker: length of the PREPROCESSED reference
piece_cap = {}    # per speaker: our split cap, set BELOW F5's own
pace_path = outdir / "pace.json"
pace_log = json.loads(pace_path.read_text()) if pace_path.exists() else {}
# F5 internally re-chunks gen_text at max_chars = ref_bytes/ref_sec *
# (22 - ref_sec) and BATCHES the chunks — multi-batch segfaults on MPS.
# Keep every piece under that limit so each infer call stays a single
# batch.
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
        processed, _ = preprocess_ref_audio_text(
            str(ref_wav), ref_text_cache[spk], show_info=_quiet)
        pinfo = sf.info(processed)
        ref_sec_f5[spk] = pinfo.frames / pinfo.samplerate
        print(f"{spk}: ref {ref_sec:.1f}s (F5 sees {ref_sec_f5[spk]:.1f}s), "
              f"F5 max_chars {f5_max:.0f} bytes", flush=True)
    text = accents.get(str(seg["id"]), seg["text_ru"])
    t0 = time.time()
    chars = pace_chars(text)
    span = seg["end"] - seg["start"]
    wanted, rate = plan_clip(chars, span, slots[seg["id"]])
    wanted *= ROOM
    cap = min(150, int(piece_cap[spk] * 0.9))
    pieces = split_single_batch(text, cap)
    waves = []
    sr_out = 24000
    for piece, secs in zip(pieces, allocate_pieces(pieces, wanted)):
        wav, sr_out, _ = tts.infer(
            ref_file=str(ref_wav),
            ref_text=ref_text_cache[spk],
            gen_text=piece,
            speed=1.0,
            fix_duration=ref_sec_f5[spk] + secs,
            show_info=_quiet,
            seed=SEED,
        )
        waves.append(np.asarray(wav))
    # F5 paints lead/tail silence around every call; each piece is
    # trimmed (the rule assemble trims by) and rejoined with a short
    # gap so pieces do not stack pauses and the speech keeps its plan
    out_wav = join_trimmed(waves, sr_out, PIECE_GAP)
    sf.write(outdir / f"seg_{seg['id']:04d}.wav", out_wav, sr_out)
    manifest[str(seg["id"])] = seg_hash(seg)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    # voiced = seconds of frames above -30 dB of the clip's own p99: speech
    # proper, as opposed to the trimmed total that still holds the pauses
    # (the pre-flight reports both — review 10-13)
    _fr = sr_out // 50
    _n = len(out_wav) // _fr
    _rms = np.sqrt((out_wav[: _n * _fr].reshape(_n, _fr) ** 2).mean(axis=1)) if _n else np.zeros(0)
    _voiced = float((_rms > VOICED_REL * np.percentile(_rms, 99)).sum() * _fr / sr_out) if _n else 0.0
    pace_log[str(seg["id"])] = {
        "chars": chars, "span": round(span, 2), "wanted": round(wanted, 2),
        "slot": round(slots[seg["id"]], 2), "rate": round(rate, 1),
        "pieces": len(pieces), "speech": round(len(out_wav) / sr_out, 2),
        "voiced": round(_voiced, 2), "room": ROOM}
    pace_path.write_text(json.dumps(pace_log, indent=1))
    per_seg = (time.time() - t_start) / n
    print(f"[{n}/{len(todo)}] seg {seg['id']:04d} {spk} {chars} chars, "
          f"original {span:.1f}s -> {wanted:.1f}s ({rate:.1f} ch/s) in a "
          f"{slots[seg['id']]:.1f}s slot ({time.time()-t0:.1f}s, "
          f"avg {per_seg:.1f}s/seg)", flush=True)

# pre-flight number in the log (CLAUDE.md §2): speech / original span,
# before assemble's atempo — target 0.95-1.10
_ratios = sorted(pace_log[str(s["id"])]["speech"] / pace_log[str(s["id"])]["span"]
                 for s in todo if pace_log.get(str(s["id"]), {}).get("span", 0) > 0.3)
_voiced_r = sorted(pace_log[str(s["id"])]["voiced"] / pace_log[str(s["id"])]["span"]
                   for s in todo if pace_log.get(str(s["id"]), {}).get("span", 0) > 0.3
                   and "voiced" in pace_log[str(s["id"])])
if _ratios:
    print(f"speech/span over {len(_ratios)} clips: median "
          f"{_ratios[len(_ratios) // 2]:.2f}, min {_ratios[0]:.2f}, "
          f"max {_ratios[-1]:.2f} (target 0.95-1.10)"
          + (f"; voiced/span median {_voiced_r[len(_voiced_r) // 2]:.2f}" if _voiced_r else ""))
print("done")
