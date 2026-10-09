#!/usr/bin/env python
"""Pipeline doctor: verify that every stage of a work dir is consistent.

Model: none (file and timing arithmetic only).

Usage: venv/bin/python scripts/check_pipeline.py work/<id> [--strict]

Checks, in pipeline order (multitrack checks are skipped in mix mode):
  tracks     tracks.json speakers/music exist as files
  words      words_tracks.json (or words.json) present, non-empty
  segments   segments.json present; monotonic ids; spans sane (>0, <30 s);
             overlap segments actually start inside another speaker's line
  translate  segments_ru.json covers every speech segment; budget outliers
  accents    accents.json covers every speech segment AND matches the
             CURRENT text_ru (stale accents sing the old text — e1v5 bug)
  refs       one wav+txt per speaker; 4-11.6 s; head silence <=0.3 s
  tts        every speech segment has a clip; manifest.json hash matches
             speaker|accents|ref (stale clip = old text with old voice)
  orig       every type:"orig" file= exists
Exit code 0 = all pass; 1 = failures (listed). --strict also fails on
warnings. Run it before synthesis and before assembly — run_pipeline.py
does this automatically.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

work = Path(sys.argv[1])
strict = "--strict" in sys.argv
failures, warnings = [], []


def fail(stage, msg):
    failures.append(f"[{stage}] {msg}")


def warn(stage, msg):
    warnings.append(f"[{stage}] {msg}")


sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_rules import force_name_stress  # noqa: E402
from manifest_bridge import load_or_bridge, norm_text  # noqa: E402


def norm(t):
    return norm_text(t, respell=force_name_stress)


def jload(p):
    return json.loads(p.read_text())


# ── mode & tracks ───────────────────────────────────────────────────
tracks_meta = None
if (work / "tracks.json").exists():
    tracks_meta = jload(work / "tracks.json")
    for spk, f in tracks_meta.get("speakers", {}).items():
        if not (work / "tracks" / f).exists():
            fail("tracks", f"{spk}: tracks/{f} missing")
    m = tracks_meta.get("music")
    if m and not (work / "tracks" / m).exists():
        fail("tracks", f"music stem tracks/{m} missing")
mode = "multitrack" if tracks_meta else "mix"

# ── words ───────────────────────────────────────────────────────────
words_path = work / ("words_tracks.json" if mode == "multitrack" else "words.json")
if words_path.exists():
    _w = jload(words_path)
    words = _w["words"] if isinstance(_w, dict) else _w  # old runs: bare list
    if not words:
        fail("words", f"{words_path.name} is empty")
else:
    fail("words", f"{words_path.name} missing")

# ── segments ────────────────────────────────────────────────────────
segs = None
if (work / "segments.json").exists():
    segs = jload(work / "segments.json")["segments"]
    by_time = sorted(segs, key=lambda s: s["start"])
    for s in segs:
        span = s["end"] - s["start"]
        if span <= 0:
            fail("segments", f"seg {s['id']}: non-positive span {span:.2f}s")
        elif span > 30:
            warn("segments", f"seg {s['id']}: span {span:.1f}s (>30 s is hard to dub)")
    for s in segs:
        if not s.get("overlap"):
            continue
        covered = any(p["speaker"] != s["speaker"]
                      and p["start"] < s["start"] < p["end"]
                      for p in by_time if p is not s and not p.get("overlap"))
        if not covered:
            warn("segments", f"seg {s['id']} marked overlap but no other "
                             f"speaker is talking at {s['start']:.1f}s")
else:
    fail("segments", "segments.json missing")

# ── translation ─────────────────────────────────────────────────────
ru = None
if (work / "segments_ru.json").exists():
    ru = jload(work / "segments_ru.json")["segments"]
    speech = [s for s in ru if s.get("type") != "orig"]
    untranslated = [s["id"] for s in speech if not s.get("text_ru")]
    if untranslated:
        fail("translate", f"{len(untranslated)} segments untranslated: "
                          f"{untranslated[:15]}")
    if segs is not None:
        base_speech = {s["id"] for s in segs if s.get("type") != "orig"}
        missing = sorted(base_speech - {s["id"] for s in ru})
        if missing:
            fail("translate", f"{len(missing)} segments of segments.json "
                              f"absent from segments_ru.json: {missing[:15]}")
    for s in speech:
        if not s.get("text_ru"):
            continue
        dur = s["end"] - s["start"]
        n = len(s["text_ru"].replace("+", ""))
        if dur >= 1.5 and n > 1.3 * 13 * dur + 10:
            warn("translate", f"seg {s['id']}: {n} chars vs budget "
                              f"{13 * dur:.0f} ({n / (13 * dur):.0%})")
else:
    fail("translate", "segments_ru.json missing")

# ── accents ─────────────────────────────────────────────────────────
if ru is not None:
    if (work / "accents.json").exists():
        acc = jload(work / "accents.json")
        speech = [s for s in ru if s.get("type") != "orig" and s.get("text_ru")]
        missing = [s["id"] for s in speech if str(s["id"]) not in acc]
        stale = [s["id"] for s in speech if str(s["id"]) in acc
                 and norm(acc[str(s["id"])]) != norm(s["text_ru"])]
        unmarked = [s["id"] for s in speech if str(s["id"]) in acc
                    and "+" not in acc[str(s["id"])]
                    and len(acc[str(s["id"])]) > 12]
        if missing:
            fail("accents", f"{len(missing)} segments missing: {missing[:15]}")
        if stale:
            fail("accents", f"{len(stale)} segments STALE (translation edited "
                            f"after accent_texts.py): {stale[:15]}")
        if unmarked:
            warn("accents", f"{len(unmarked)} entries carry no '+' at all: "
                            f"{unmarked[:15]}")
    else:
        fail("accents", "accents.json missing — run accent_texts.py")

# ── refs ────────────────────────────────────────────────────────────
refs = work / ("refs_f5" if (work / "refs_f5").exists() else "refs")
if ru is not None:
    import numpy as np
    for spk in sorted({s["speaker"] for s in ru if s.get("type") != "orig"}):
        wav, txt = refs / f"{spk}.wav", refs / f"{spk}.txt"
        if not wav.exists() or not txt.exists():
            fail("refs", f"{spk}: wav or txt missing in {refs.name}/")
            continue
        if not txt.read_text().strip():
            fail("refs", f"{spk}.txt is empty")
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(wav), "-f", "f32le",
             "-ac", "1", "-ar", "16000", "-"], capture_output=True)
        a = np.frombuffer(out.stdout, dtype=np.float32)
        dur = len(a) / 16000
        if not 4.0 <= dur <= 11.6:
            (fail if dur > 11.6 else warn)(
                "refs", f"{spk}.wav is {dur:.1f}s (need 4-11.6 s)")
        n = len(a) // 800
        if n:
            rms = np.sqrt((a[: n * 800].reshape(n, 800) ** 2).mean(axis=1))
            voiced = rms > max(0.01, 0.1 * float(np.percentile(rms, 95)))
            head = float(np.argmax(voiced)) * 0.05 if voiced.any() else dur
            if head > 0.3:
                fail("refs", f"{spk}.wav: {head:.2f}s of silence before speech "
                             "— F5 will paint the same junk into every clip")

# ── tts + manifest ──────────────────────────────────────────────────
tts = work / "tts_f5"
if ru is not None and tts.exists():
    speech = [s for s in ru if s.get("type") != "orig" and s.get("text_ru")]
    missing = [s["id"] for s in speech
               if not (tts / f"seg_{s['id']:04d}.wav").exists()]
    if missing:
        warn("tts", f"{len(missing)} clips not yet synthesized: {missing[:15]}")
    mpath = tts / "manifest.json"
    if (work / "accents.json").exists() and (mpath.exists() or speech):
        acc = jload(work / "accents.json")
        # the EFFECTIVE manifest: the same bridge synthesize_f5.py will
        # use (read-only, silent) — otherwise an old-scheme or
        # pre-manifest dir fails the doctor before synthesize ever gets
        # the chance to migrate it
        ref_sha = {}

        def rsha(spk):
            if spk not in ref_sha:
                p = refs / f"{spk}.wav"
                ref_sha[spk] = (hashlib.sha1(p.read_bytes()).hexdigest()[:12]
                                if p.exists() else "noref")
            return ref_sha[spk]

        manifest, _ = load_or_bridge(work, tts, speech, acc, rsha,
                                     log=lambda *a: None, refs_dir=refs)
        if manifest is None:
            # no manifest and no trustworthy record: synthesize_f5.py's
            # cautious default will re-do these clips — that is the
            # remedy, so the doctor must not block the pipeline here
            warn("tts", "clips have no manifest and no accents record — "
                        "they will be re-synthesized on the next "
                        "synthesize_f5.py run")
            manifest = None
        stale = []
        for s in (speech if manifest is not None else ()):
            if not (tts / f"seg_{s['id']:04d}.wav").exists():
                continue
            text = acc.get(str(s["id"]), s.get("text_ru", ""))
            h = hashlib.sha1(
                f"{s['speaker']}|{text}|{rsha(s['speaker'])}".encode()).hexdigest()
            if manifest.get(str(s["id"])) != h:
                stale.append(s["id"])
        if stale:
            fail("tts", f"{len(stale)} clips STALE vs current text/accents/"
                        f"refs: {stale[:15]} — re-run synthesize_f5.py")
    elif not mpath.exists():
        warn("tts", "no manifest.json — clips cannot be verified against the "
                    "script (synthesize_f5.py writes one on its next run)")

# ── orig files ──────────────────────────────────────────────────────
if ru is not None:
    for s in ru:
        if s.get("type") == "orig" and "file" in s \
                and not (work / s["file"]).exists():
            fail("orig", f"seg {s['id']}: {s['file']} missing")

# ── verdict ─────────────────────────────────────────────────────────
print(f"mode: {mode}")
for w in warnings:
    print("WARN", w)
for f in failures:
    print("FAIL", f)
print(f"{len(failures)} failure(s), {len(warnings)} warning(s)")
sys.exit(1 if failures or (strict and warnings) else 0)
