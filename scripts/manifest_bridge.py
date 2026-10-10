#!/usr/bin/env python
"""Bridge pre-manifest and old-scheme tts dirs into the single manifest.

Model: none (hashes and file mtimes). Used by synthesize_f5.py.

The problem: dirs synthesized before tts_f5/manifest.json existed (or
with the old redub scheme sha1(speaker|text)) cannot be verified, and
the cautious default re-synthesizes everything — hours. Stamping the
CURRENT text onto old clips (--adopt-existing) is worse when lines were
edited first: the edit becomes a silent no-op (the e1v5 stale-clip
class).

The way out: accents.prev.json. The synthesis gate guarantees that at
synthesis time accents.json matched segments_ru.json, and
accent_texts.py renames the old file to accents.prev.json before
overwriting — so prev is a faithful record of what existing clips were
SUNG from. Hashing speaker|prev_text|ref per clip lets the normal
manifest comparison re-synthesize exactly the edited lines.

Guard: prev must not be newer than the newest clip (otherwise accents
were re-run after synthesis without re-synthesizing, and prev may
already contain an un-sung edit) — then no bridge, cautious default.

Limitation, stated: a bridged entry uses the CURRENT reference hash, so
a ref change made BEFORE bridging is invisible (a change after bridging
invalidates normally). All pre-manifest dirs in this repo kept their
refs untouched since synthesis.
"""
import hashlib
import json
import os
import re
from pathlib import Path


def norm_text(t, respell=None):
    """Words-only comparison form shared by the freshness checks
    (synthesize gate, doctor, run_pipeline). Tolerates everything the
    accent stage legitimately changes: stress marks, ёfication,
    punctuation-vs-space jitter (R&D→RD, слово,слово→слово, слово) —
    stripping ALL non-alphanumerics including spaces still catches any
    real edit, because an edit changes letters. `respell` is
    stress_rules.force_name_stress where available (venv-f5 and venv
    both have it on sys.path next to this module).
    """
    if respell is not None:
        t = respell(t or "")
    t = (t or "").replace("+", "").replace("ё", "е").replace("Ё", "Е").lower()
    return re.sub(r"[^\w]+", "", t)


def voiced(seg: dict) -> bool:
    """Is this segment one we synthesize? Pasted originals (type:"orig")
    and echo lines (repeat_audit --mark-echo; assemble drops them) are
    not voiced — synthesizing them is pure waste (TODO #9)."""
    return seg.get("type") != "orig" and not seg.get("echo")


def stale_ids(segments: list, accents: dict, respell=None) -> list:
    """ids whose accents.json entry is missing or no longer matches the
    segment's CURRENT text_ru — i.e. accent_texts.py was not re-run
    after a translation edit (the e1v5 stale-audio class). The single
    home of this check: synthesize_f5's gate, the doctor and
    run_pipeline's need-accents test all call it. Scope it to the set
    actually being synthesized (--only) so an unrelated stale line
    cannot hold a surgical re-roll hostage (TODO #3)."""
    bad = []
    for s in segments:
        a = accents.get(str(s["id"]))
        if a is None or norm_text(a, respell) != norm_text(
                s.get("text_ru", ""), respell):
            bad.append(s["id"])
    return bad


def rotate_accents(work: Path, new_out: dict) -> bool:
    """Save accents.json as accents.prev.json ONLY when the new output
    differs — a no-op accent run must not destroy the record of what
    existing clips were sung from (prev would get a post-clip mtime and
    the bridge's trust guard would rightly refuse it). Returns whether
    a rotation happened."""
    path = work / "accents.json"
    if not path.exists():
        return False
    try:
        if json.loads(path.read_text()) == new_out:
            return False
    except Exception:
        pass
    os.replace(path, work / "accents.prev.json")
    return True


def new_hash(speaker: str, text: str, ref_sha: str) -> str:
    return hashlib.sha1(f"{speaker}|{text}|{ref_sha}".encode()).hexdigest()


def old_hash(speaker: str, text: str) -> str:
    return hashlib.sha1(f"{speaker}|{text}".encode()).hexdigest()


def ref_hasher(refs_dir: Path):
    """spk -> 12-hex sha1 of refs_dir/<spk>.wav ("noref" if absent), cached.
    The same function object must be used for one manifest comparison
    (synthesize_f5.py and check_pipeline.py each build one)."""
    cache = {}

    def ref_sha(spk: str) -> str:
        if spk not in cache:
            p = Path(refs_dir) / f"{spk}.wav"
            cache[spk] = (hashlib.sha1(p.read_bytes()).hexdigest()[:12]
                          if p.exists() else "noref")
        return cache[spk]

    return ref_sha


def seg_hash(seg: dict, accents: dict, ref_sha) -> str:
    """Manifest entry for a segment: speaker | accented text (falls back
    to text_ru when the accent stage skipped it) | ref hash."""
    text = accents.get(str(seg["id"]), seg.get("text_ru", ""))
    return new_hash(seg["speaker"], text, ref_sha(seg["speaker"]))


def load_or_bridge(work: Path, outdir: Path, segments: list, accents: dict,
                   ref_sha, log=print, refs_dir: Path | None = None):
    """Return (manifest dict, bridged_count) for synthesize_f5.py.

    manifest is None when there is no trustworthy basis at all — the
    caller then falls back to its cautious default (or an explicit
    --adopt-existing).
    """
    mpath = outdir / "manifest.json"
    prev_path = work / "accents.prev.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    by_id = {s["id"]: s for s in segments}

    clips = {s["id"]: outdir / f"seg_{s['id']:04d}.wav" for s in segments}
    clips = {i: p for i, p in clips.items() if p.exists()}

    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        if not _is_old_scheme(manifest, segments, accents, prev, clips):
            return manifest, 0
        # old redub scheme: each entry says which text the clip was sung
        # from — upgrade per id to the new scheme with the current ref
        upgraded, n = {}, 0
        for i in clips:
            s, entry = by_id[i], manifest.get(str(i))
            for text in (accents.get(str(i)), prev.get(str(i))):
                if text is not None and entry == old_hash(s["speaker"], text):
                    upgraded[str(i)] = new_hash(s["speaker"], text,
                                                ref_sha(s["speaker"]))
                    n += 1
                    break
        log(f"manifest bridge: upgraded {n} old-scheme entries "
            f"({len(clips) - n} clips left unverified)")
        return upgraded, n

    if not clips:
        return {}, 0
    newest_clip = max(p.stat().st_mtime for p in clips.values())
    # the record of what the clips were sung from: accents.prev.json if
    # the accent stage ran since, else accents.json itself — on a dir
    # where nothing was edited, accent_texts is skipped, and the current
    # accents (older than every clip) ARE what the clips were sung from
    # (the synthesis gate guaranteed they matched text_ru at the time)
    record, record_path = None, None
    for cand, cpath in ((prev, prev_path), (None, work / "accents.json")):
        if cand is None:
            cand = json.loads(cpath.read_text()) if cpath.exists() else {}
        if not cand:
            continue
        if cpath.stat().st_mtime > newest_clip + 1.0:
            log(f"manifest bridge: {cpath.name} is NEWER than the newest "
                "clip — it may contain an edit that was never synthesized; "
                "not trusting it as the clips' source text")
            continue
        record, record_path = cand, cpath
        break
    if record is None:
        return None, 0
    # a reference re-cut after synthesis means that speaker's clips were
    # sung with a DIFFERENT voice reference — leave them unbridged
    stale_ref = set()
    if refs_dir is not None:
        for spk in {s["speaker"] for s in segments}:
            rp = refs_dir / f"{spk}.wav"
            if rp.exists() and rp.stat().st_mtime > newest_clip + 1.0:
                stale_ref.add(spk)
        if stale_ref:
            log(f"manifest bridge: refs re-cut after synthesis for "
                f"{sorted(stale_ref)} — their clips stay unverified")
    bridged = {}
    for i in clips:
        s = by_id[i]
        text = record.get(str(i))
        if text is not None and s["speaker"] not in stale_ref:
            bridged[str(i)] = new_hash(s["speaker"], text, ref_sha(s["speaker"]))
    log(f"manifest bridge: {len(bridged)} pre-manifest clips keyed to their "
        f"recorded source text ({record_path.name}); edited lines will "
        f"re-synthesize, {len(clips) - len(bridged)} unrecorded clips too")
    return bridged, len(bridged)


def _is_old_scheme(manifest, segments, accents, prev, clips) -> bool:
    """Old scheme = entries are sha1(speaker|text) of a known text."""
    by_id = {s["id"]: s for s in segments}
    checked = matched = 0
    for i in list(clips)[:50]:
        entry = manifest.get(str(i))
        if entry is None:
            continue
        checked += 1
        s = by_id[i]
        for text in (accents.get(str(i)), prev.get(str(i))):
            if text is not None and entry == old_hash(s["speaker"], text):
                matched += 1
                break
    return checked > 0 and matched >= max(3, checked // 2)
