#!/usr/bin/env python
"""Recover speech that one speaker says OVER another (overlapped speech).

Models: pyannote/speech-separation-ami-1.0 (gated; separation),
speechbrain ECAPA (stem→speaker matching, via refs), and
mlx-community/whisper-large-v3-turbo (per-stem transcription).

Usage:
  venv/bin/python scripts/overlap_speech.py work/<id> [--min-overlap 0.8]
      [--regions 7233-7242,3830-3840]   # explicit windows instead of scan

For every cross-speaker overlap region in diarization.json:
  1. separate the window into per-speaker stems,
  2. match stems to global speakers by voice embedding against refs_f5/,
  3. find which speaker's words are MISSING from segments.json there
     (whisper's mixed-stream transcript keeps only the dominant voice),
  4. transcribe THAT speaker's stem → the recovered hidden words,
  5. append `"overlap": true` segments (ids 8000+) to segments.json and
     stem wavs to work/<id>/overlap_stems/.

After this: translate the new segments (apply_translation), synthesize,
assemble — the assembler overlays overlap segments concurrently.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# pyannote 4.x passes token= into speechbrain loaders that reject it
import speechbrain.inference.interfaces as sbi
_orig_init = sbi.Pretrained.__init__
def _patched(self, modules=None, hparams=None, run_opts=None, freeze_params=True, **kw):
    kw.pop("token", None); kw.pop("use_auth_token", None)
    _orig_init(self, modules=modules, hparams=hparams, run_opts=run_opts,
               freeze_params=freeze_params)
sbi.Pretrained.__init__ = _patched

import librosa
import torch
from pyannote.audio import Pipeline
from speechbrain.inference.speaker import EncoderClassifier

CONTEXT = 1.5   # sec of context around the overlap for the separator
MIN_STEM_RMS = 0.01

work = Path(sys.argv[1])
min_ovl = float(sys.argv[sys.argv.index("--min-overlap") + 1]) if "--min-overlap" in sys.argv else 0.8
regions_arg = sys.argv[sys.argv.index("--regions") + 1] if "--regions" in sys.argv else None

turns = sorted(json.loads((work / "diarization.json").read_text()), key=lambda t: t["start"])
seg_data = json.loads((work / "segments.json").read_text())
segments = seg_data["segments"]
audio, sr = sf.read(work / "audio.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

if regions_arg:
    regions = []
    for r in regions_arg.split(","):
        a, b = map(float, r.split("-"))
        regions.append((a, b, None, None))
else:
    regions = []
    for i, a in enumerate(turns):
        for b in turns[i + 1:]:
            if b["start"] >= a["end"]:
                break
            if b["speaker"] != a["speaker"]:
                s, e = max(a["start"], b["start"]), min(a["end"], b["end"])
                if e - s >= min_ovl:
                    regions.append((s, e, a["speaker"], b["speaker"]))
print(f"{len(regions)} overlap regions to process")

sep = Pipeline.from_pretrained("pyannote/speech-separation-ami-1.0")
emb = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                     run_opts={"device": "cpu"})

def embed(wav16: np.ndarray) -> np.ndarray:
    v = emb.encode_batch(torch.from_numpy(wav16[None, :]))
    return v.squeeze().numpy()

ref_embs = {}
for f in (work / "refs_f5").glob("*.wav"):
    w, rsr = sf.read(f, dtype="float32")
    if w.ndim > 1: w = w.mean(axis=1)
    ref_embs[f.stem] = embed(librosa.resample(w, orig_sr=rsr, target_sr=16000))

def dominant_speaker(s: float, e: float) -> str | None:
    """Who owns the transcript here per segments.json (the mixed-stream winner)."""
    best, best_ov = None, 0.0
    for g in segments:
        ov = min(e, g["end"]) - max(s, g["start"])
        if ov > best_ov:
            best, best_ov = g["speaker"], ov
    return best

stem_dir = work / "overlap_stems"
stem_dir.mkdir(exist_ok=True)
import mlx_whisper

new_segs = []
next_id = max([s["id"] for s in segments] + [7999]) + 1
for (s, e, spk_a, spk_b) in regions:
    A, B = max(0.0, s - CONTEXT), e + CONTEXT
    clip = audio[int(A * sr): int(B * sr)]
    # forcing the true local speaker count is what makes separation work
    n_local = len({t["speaker"] for t in turns if t["start"] < B and t["end"] > A}) or 2
    out = sep({"waveform": torch.from_numpy(clip).unsqueeze(0), "sample_rate": sr},
              num_speakers=max(2, n_local))
    sources = getattr(out, "sources", None) or out[1]
    sdata = np.asarray(sources.data, dtype="float32")
    out_sr = int(round(sdata.shape[0] / (B - A)))

    # text already present near this region (for de-duplication)
    import re
    def norm_words(t):
        return set(re.sub(r"[^\w ]", "", t.lower()).split())
    nearby_words = set()
    for g in segments:
        if g["end"] > s - 4 and g["start"] < e + 4:
            nearby_words |= norm_words(g["text"])

    for k in range(sdata.shape[1]):
        stem = sdata[:, k]
        # transcribe ONLY the overlap span (±0.2s), not the context window
        lo = int(max(0.0, (s - A - 0.2)) * out_sr)
        hi = int(min(B - A, (e - A + 0.2)) * out_sr)
        piece = stem[lo:hi]
        if float(np.sqrt((piece ** 2).mean())) < MIN_STEM_RMS:
            continue
        sp16 = librosa.resample(piece, orig_sr=out_sr, target_sr=16000) if out_sr != 16000 else piece
        v = embed(sp16)
        sims = {name: float(np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r) + 1e-9))
                for name, r in ref_embs.items()}
        who = max(sims, key=sims.get)
        stem_path = stem_dir / f"ovl_{int(s)}_{who}_{k}.wav"
        sf.write(stem_path, piece / (np.abs(piece).max() or 1) * 0.85, out_sr)
        r = mlx_whisper.transcribe(str(stem_path),
                                   path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
        text = r["text"].strip()
        words = norm_words(text)
        if len(words) < 2:
            print(f"  {s:.1f}-{e:.1f} {who}: stem empty/1-word — skipped ({text!r})")
            continue
        # if most of these words already exist in the transcript nearby,
        # this stem is the DOMINANT voice, not hidden speech
        if len(words & nearby_words) / len(words) > 0.6:
            print(f"  {s:.1f}-{e:.1f} {who}: duplicates transcript — skipped ({text[:50]!r})")
            continue
        new_segs.append({
            "id": next_id, "start": round(s, 3), "end": round(e, 3),
            "speaker": who, "text": text, "overlap": True,
            "stem": stem_path.name, "sim": round(sims[who], 3),
        })
        print(f"  RECOVERED {s:.1f}-{e:.1f} {who} (sim {sims[who]:.2f}): {text[:80]}")
        next_id += 1

if new_segs:
    seg_data["segments"] = sorted(segments + new_segs, key=lambda x: x["start"])
    (work / "segments.json").write_text(json.dumps(seg_data, ensure_ascii=False, indent=1))
print(f"{len(new_segs)} overlap segments appended to segments.json — translate them next")
