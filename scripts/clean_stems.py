#!/usr/bin/env python
"""Keep only each stem's OWN speaker: mute other voices picked up by its mic.

Model: pyannote/wespeaker-voxceleb-resnet34-LM (speaker embeddings, CPU;
cached with the pyannote diarization pipeline).

Usage: venv/bin/python scripts/clean_stems.py work/<id> [--margin 0.05]
           [--report segments_ru.json]

Multitrack mode (RU_INSTRUCTIONS §3б): one mic per speaker, but every mic
also hears the others. Loud bleed passes the loudness filters and gets
transcribed — then dubbed in the WRONG voice (the host's question spoken
by the guest's clone). Words only match across stems when whisper heard
them identically, so a text-level echo filter can't catch it all. This
step decides by VOICE, before transcription:

1. voiced regions of every stem (RMS scan), cut into WIN-s windows;
2. one WeSpeaker embedding per window;
3. a voice profile per speaker = centroid of the loudest windows of their
   own stem (on one's own mic, the loudest speech is oneself), refined
   once on the windows that match it;
4. per 0.25 s frame, score = mean (own − best other similarity) over the
   windows covering it; below −--margin = another voice. Muted stretches
   then grow up to GROW_FRAMES into adjacent voiced frames that are not
   clearly the owner (the first/last word of a bleed phrase is ambiguous
   because its windows straddle silence);
5. non-owner frames are muted (20 ms fades) at the stem's native rate →
   work/<id>/tracks_clean/<stem>.flac + tracks_clean.json (same layout as
   tracks.json; the music stem is passed through). Point the pipeline at
   it by copying tracks_clean.json over tracks.json and the files into
   tracks/ — or run with work/<id>/tracks → tracks_clean symlinks.

--report <segments file>: only audit, per dubbed segment, the share of its
voiced frames that are NOT its speaker's voice — printed, and saved as
work/<id>/bleed_report.json ({segment id: share}); no stems written.
"""
import json
import subprocess
import sys
from pathlib import Path

from cli_args import opt

import numpy as np
import torch

from audio_utils import RMS_SILENCE, decode, frame_rms  # noqa: E402

SR = 16000
FRAME = 0.25          # sec, labelling resolution
WIN, HOP = 1.5, 0.5   # sec, embedding window / hop
PROFILE_TOP = 0.3     # loudest share of own windows used for the first profile
FADE = 0.02           # sec
GROW_FRAMES = 3       # how far (×FRAME) a muted stretch may grow into its edges
OWNER_SURE = 0.15     # frames scoring above this are clearly the owner: never grown into

work = Path(sys.argv[1])
margin = opt("--margin", 0.05, float)
report = opt("--report", None)
meta = json.loads((work / "tracks.json").read_text())
speakers = meta["speakers"]


from pyannote.audio import Model  # noqa: E402  (slow import)

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM").eval()


@torch.no_grad()
def embed(chunks: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(chunks), 64):
        x = torch.from_numpy(chunks[i: i + 64]).unsqueeze(1)
        e = model(x).numpy()
        out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    return np.concatenate(out) if out else np.zeros((0, 256), np.float32)


stems = {}
for spk, fname in speakers.items():
    audio = decode(work / "tracks" / fname, SR)
    fn = int(FRAME * SR)
    rms = frame_rms(audio, fn)
    voiced = rms > RMS_SILENCE
    # windows centred on voiced frames, every HOP
    step = int(HOP / FRAME)
    centers = [i for i in range(0, len(voiced), step)
               if voiced[max(0, i - step): i + step + 1].any()]
    wn = int(WIN * SR)
    chunks = np.zeros((len(centers), wn), np.float32)
    win_rms = np.zeros(len(centers))
    for k, c in enumerate(centers):
        a = max(0, int((c + 0.5) * fn) - wn // 2)
        seg = audio[a: a + wn]
        chunks[k, : len(seg)] = seg
        win_rms[k] = np.sqrt((seg ** 2).mean()) if len(seg) else 0
    print(f"{spk}: {voiced.sum() * FRAME / 60:.1f} min voiced, {len(centers)} windows",
          flush=True)
    stems[spk] = dict(audio=audio, voiced=voiced, centers=np.array(centers),
                      emb=embed(chunks), win_rms=win_rms, fname=fname)

# voice profiles: loudest own windows, refined once on matching windows
profiles = {}
for spk, st in stems.items():
    if len(st["emb"]) == 0:
        continue
    top = st["win_rms"] >= np.quantile(st["win_rms"], 1 - PROFILE_TOP)
    c = st["emb"][top].mean(0)
    c /= np.linalg.norm(c)
    close = st["emb"] @ c > 0.5
    if close.sum() >= 10:
        c = st["emb"][close & top].mean(0) if (close & top).sum() >= 10 else st["emb"][close].mean(0)
        c /= np.linalg.norm(c)
    profiles[spk] = c
names = list(profiles)
P = np.stack([profiles[s] for s in names])
print("profile similarity:", {f"{a[-2:]}~{b[-2:]}": round(float(profiles[a] @ profiles[b]), 2)
                               for i, a in enumerate(names) for b in names[i + 1:]})

masks = {}
for spk, st in stems.items():
    if spk not in profiles:
        continue
    sims = st["emb"] @ P.T                    # windows × speakers
    own = sims[:, names.index(spk)]
    others = np.delete(sims, names.index(spk), axis=1).max(1) if len(names) > 1 else own - 1
    foreign_win = others > own + margin
    # per-frame owner score = mean (own − best other) over the windows
    # covering the frame; negative beyond the margin = another voice
    n = len(st["voiced"])
    score_sum = np.zeros(n)
    score_n = np.zeros(n)
    half = int(WIN / FRAME / 2)
    for c, d in zip(st["centers"], own - others):
        score_sum[max(0, c - half): c + half + 1] += d
        score_n[max(0, c - half): c + half + 1] += 1
    score = np.where(score_n > 0, score_sum / np.maximum(score_n, 1), 1.0)
    foreign = st["voiced"] & (score < -margin)
    # a bleed phrase's first/last word sits in windows straddling silence,
    # so its score is ambiguous and it would survive as a crumb ("what",
    # "And that's"). Grow muted stretches into adjacent voiced frames
    # unless those clearly sound like the owner.
    for _ in range(GROW_FRAMES):
        grow = np.zeros(n, bool)
        grow[1:] |= foreign[:-1]
        grow[:-1] |= foreign[1:]
        foreign |= grow & st["voiced"] & (score < OWNER_SURE)
    masks[spk] = foreign
    print(f"{spk}: {foreign.sum() * FRAME / 60:.1f} of {st['voiced'].sum() * FRAME / 60:.1f} "
          "voiced min are other voices", flush=True)

if report:
    segs = json.loads((work / report).read_text())["segments"]
    rows = []
    for s in segs:
        if s.get("type") == "orig" or s["speaker"] not in masks:
            continue
        a, b = int(s["start"] / FRAME), int(s["end"] / FRAME) + 1
        v = stems[s["speaker"]]["voiced"][a:b]
        f = masks[s["speaker"]][a:b]
        if v.sum():
            rows.append((f.sum() / v.sum(), s))
    rows.sort(key=lambda r: -r[0])
    (work / "bleed_report.json").write_text(json.dumps(
        {str(s["id"]): round(float(f), 3) for f, s in rows}, indent=0))
    bad = [r for r in rows if r[0] > 0.5]
    part = [r for r in rows if 0.2 < r[0] <= 0.5]
    print(f"\n{len(bad)} segments mostly another voice (>50%), {len(part)} partly (20–50%)")
    for frac, s in bad[:40] + part[:15]:
        print(f"  {frac:4.0%} {s['id']:5d} {s['speaker'][-2:]} {s['start']:8.1f}  {s['text'][:90]}")
    sys.exit(0)

# write cleaned stems at native rate
out_dir = work / "tracks_clean"
out_dir.mkdir(exist_ok=True)
clean_meta = {"speakers": {}, "music": meta.get("music")}
for spk, st in stems.items():
    src = work / "tracks" / st["fname"]
    sr = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0",
                                      "-show_entries", "stream=sample_rate", "-of", "csv=p=0",
                                      str(src)], text=True).strip())
    full = decode(src, sr).copy()
    gain = np.ones(len(full), np.float32)
    fn = int(FRAME * sr)
    for i in np.where(masks.get(spk, np.zeros(0, bool)))[0]:
        gain[i * fn: (i + 1) * fn] = 0.0
    k = int(FADE * sr)
    gain = np.convolve(gain, np.ones(k) / k, mode="same").astype(np.float32)
    name = Path(st["fname"]).with_suffix(".flac").name
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1",
                    "-i", "-", str(out_dir / name)], input=(full * gain).tobytes(), check=True)
    clean_meta["speakers"][spk] = name
    print(f"{spk}: -> tracks_clean/{name}", flush=True)
if meta.get("music"):
    m = out_dir / meta["music"]
    if not m.exists():
        m.symlink_to((work / "tracks" / meta["music"]).resolve())
(work / "tracks_clean.json").write_text(json.dumps(clean_meta, indent=1))
print("wrote tracks_clean/ + tracks_clean.json")
