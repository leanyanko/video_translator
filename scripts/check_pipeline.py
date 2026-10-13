#!/usr/bin/env python
"""Pipeline doctor: verify that every stage of a work dir is consistent.

Model: none (file and timing arithmetic only).

Usage: venv/bin/python scripts/check_pipeline.py work/<id> [--strict]

Checks, in pipeline order (multitrack checks are skipped in mix mode):
  tracks     tracks.json speakers/music exist as files
  words      words_tracks.json (or words.json) present, non-empty; no
             5-min window with collapsed punctuation / hallucination
             tempo / stretched words (whisper_guard.degraded_windows)
  segments   segments.json present; monotonic ids; spans sane (>0, <30 s);
             overlap segments actually start inside another speaker's line;
             mid-sentence cuts inside run-ons <= 5% (whisper lost
             punctuation there — repunctuate_words.py)
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

A failure one has DECIDED to ship (a known degradation) is accepted on
record via work/<id>/doctor_accept.json: a list of {"stage",
"contains", "reason"}; a matching failure becomes a warning tagged
[ACCEPTED: reason], and an entry that matches nothing is flagged as
stale. The gate stays strict by default.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

work = Path(sys.argv[1])
strict = "--strict" in sys.argv
# Before synthesis, STALE clips are the synthesizer's to-do list, not a
# failure — aborting there would block the very stage that fixes them.
# Before assembly they are a hard failure (stale audio must not ship).
pre_synth = "--pre-synthesis" in sys.argv
failures, warnings = [], []

# Accepted, on-record degradations (review 10-12 #3): a run one DECIDES
# to ship despite a failure (e.g. a known unpunctuated stretch in an
# old transcript) had no override short of editing code. work/<id>/
# doctor_accept.json = [{"stage": ..., "contains": ..., "reason": ...}]
# downgrades a MATCHING failure to a warning, carrying the reason. The
# gate stays strict by default; an entry that matched nothing is itself
# a warning, so stale acceptances cannot linger silently.
_accept_path = work / "doctor_accept.json"
accepts = []
if _accept_path.exists():
    accepts = [dict(a, used=False) for a in json.loads(_accept_path.read_text())]


def fail(stage, msg):
    for a in accepts:
        if a.get("stage") == stage and a.get("contains", "") in msg:
            a["used"] = True
            warnings.append(f"[{stage}] {msg} [ACCEPTED: "
                            f"{a.get('reason', 'no reason given')}]")
            return
    failures.append(f"[{stage}] {msg}")


def warn(stage, msg):
    warnings.append(f"[{stage}] {msg}")


sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_rules import force_name_stress  # noqa: E402
from manifest_bridge import (  # noqa: E402
    load_or_bridge, missing_clips, ref_hasher, seg_hash, stale_ids, voiced)


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
        # whisper degradation gate (TODO #12): until now the doctor only
        # checked that words exist — a 15-minute punctuation collapse
        # sailed through and surfaced as bad segmentation later. Windows
        # are measured per speaker in multitrack (islands), globally in mix.
        from whisper_guard import degraded_windows, hallucination_loops
        groups = {}
        for w in words:
            groups.setdefault(w.get("speaker"), []).append(w)
        for spk, ws in sorted(groups.items(), key=lambda kv: str(kv[0])):
            tag = f" {spk}" if spk else ""
            for d in degraded_windows(ws):
                (fail if d["severity"] == "fail" else warn)(
                    "words", f"whisper degraded{tag} "
                             f"{d['t0']/60:.0f}-{d['t1']/60:.0f} min: "
                             f"{d['reason']} — re-transcribe that stretch "
                             "or fix punctuation with repunctuate_words.py")
            for t, phrase in hallucination_loops(ws):
                warn("words", f"possible hallucination loop{tag} at "
                              f"{t/60:.0f} min: «{phrase}» repeats")
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
    # run-on cut gate (TODO #12): a segment cut mid-sentence inside a
    # run-on means whisper lost punctuation there — the halves are
    # untranslatable on their own. A few are survivable; a large share
    # means a degraded transcript shipped into segmentation.
    from segmenting import runon_cut_ids
    speech_n = sum(1 for s in segs if s.get("type") != "orig")
    runons, _ = runon_cut_ids(segs)
    if runons and speech_n:
        share = len(runons) / speech_n
        (fail if share > 0.05 else warn)(
            "segments", f"{len(runons)} mid-sentence cuts inside run-ons "
                        f"({share:.0%}): ids {runons[:15]} — restore "
                        "punctuation with repunctuate_words.py, re-segment")
    marked = [s["id"] for s in segs if s.get("cut") == "mid-sentence"]
    if marked:
        warn("segments", f"{len(marked)} segments carry cut:\"mid-sentence\" "
                         f"(last-resort split of a run-on): {marked[:15]} — "
                         "the translator must treat each half as a fragment")
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
        # echo lines are never voiced (TODO #9) — their accents don't matter
        speech = [s for s in ru if voiced(s) and s.get("text_ru")]
        missing = [s["id"] for s in speech if str(s["id"]) not in acc]
        bad = stale_ids(speech, acc, respell=force_name_stress)
        missing_set = set(missing)
        stale = [i for i in bad if i not in missing_set]
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
    from audio_utils import audition, decode
    for spk in sorted({s["speaker"] for s in ru if s.get("type") != "orig"}):
        wav, txt = refs / f"{spk}.wav", refs / f"{spk}.txt"
        if not wav.exists() or not txt.exists():
            fail("refs", f"{spk}: wav or txt missing in {refs.name}/")
            continue
        if not txt.read_text().strip():
            fail("refs", f"{spk}.txt is empty")
        try:
            a = decode(wav, 16000)
        except subprocess.CalledProcessError:
            fail("refs", f"{spk}.wav cannot be decoded")
            continue
        dur = len(a) / 16000
        if not 4.0 <= dur <= 11.6:
            (fail if dur > 11.6 else warn)(
                "refs", f"{spk}.wav is {dur:.1f}s (need 4-11.6 s)")
        if len(a) >= 800:
            # same rule make_refs.py auditioned the candidates with
            head, _, _ = audition(a, 16000)
            if head > 0.3:
                fail("refs", f"{spk}.wav: {head:.2f}s of silence before speech "
                             "— F5 will paint the same junk into every clip")

# ── tts + manifest ──────────────────────────────────────────────────
tts = work / "tts_f5"
if ru is not None and tts.exists():
    speech = [s for s in ru if voiced(s) and s.get("text_ru")]
    missing = missing_clips(speech, tts)
    if missing:
        # before synthesis this IS the synthesizer's to-do list; before
        # assembly a large share means synthesis failed wholesale and
        # the "finished" video would be music and pastes (TODO #4)
        share = len(missing) / len(speech) if speech else 0
        (warn if pre_synth or share <= 0.10 else fail)(
            "tts", f"{len(missing)}/{len(speech)} clips missing or empty "
                   f"({share:.0%}): {missing[:15]}"
                   + ("" if pre_synth else
                      " — run synthesize_f5.py (assemble --force-partial "
                      "to override)"))
    mpath = tts / "manifest.json"
    if (work / "accents.json").exists() and (mpath.exists() or speech):
        acc = jload(work / "accents.json")
        # the EFFECTIVE manifest: the same bridge synthesize_f5.py will
        # use (read-only, silent) — otherwise an old-scheme or
        # pre-manifest dir fails the doctor before synthesize ever gets
        # the chance to migrate it
        rsha = ref_hasher(refs)
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
            if manifest.get(str(s["id"])) != seg_hash(s, acc, rsha):
                stale.append(s["id"])
        if stale:
            (warn if pre_synth else fail)(
                "tts", f"{len(stale)} clips STALE vs current text/accents/"
                       f"refs: {stale[:15]} — "
                       + ("synthesize_f5.py will re-sing them"
                          if pre_synth else "re-run synthesize_f5.py"))
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
for a in accepts:
    if not a["used"]:
        warn("doctor", f"doctor_accept.json entry matched NOTHING "
                       f"(stale acceptance, remove it): {a.get('stage')} / "
                       f"«{a.get('contains')}»")
print(f"mode: {mode}")
for w in warnings:
    print("WARN", w)
for f in failures:
    print("FAIL", f)
print(f"{len(failures)} failure(s), {len(warnings)} warning(s)")
sys.exit(1 if failures or (strict and warnings) else 0)
