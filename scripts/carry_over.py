#!/usr/bin/env python
"""Carry translations and synthesized clips over from an earlier run.

Model: none (matching over segments files + file copies).

Usage: venv/bin/python scripts/carry_over.py work/<new> work/<old>
           [--out carry.json]

After re-segmenting (new ids), most segments are still identical to the
earlier run's. A segment matches when speaker and the English words are
the same (ignoring punctuation and case) and start/end are within ±TOL;
then:
  - its old text_ru goes into <out> (default work/<new>/carry.json),
    a translation map for apply_translation.py;
  - the old run's tts_f5/seg_<old id>.wav is copied to the new run's
    tts_f5/seg_<new id>.wav, so synthesize_f5.py (resumable: it skips
    existing files) only synthesizes what is new or changed;
  - a manifest entry is written for each copied clip (see
    emit_manifest), so the manifest-era synthesizer keeps them.
If the new dir has no voice references yet (the normal case: run_pipeline
cuts refs_f5/ AFTER the translation gate, and carry_over is the tool you
use AT the gate), the old run's references are copied over first — same
episode, same speakers, and identical refs are the precondition for
keeping the clips at all.
Stress marks are not copied — rerun accent_texts.py on the new run.
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_bridge import new_hash  # noqa: E402

TOL = 0.3  # sec; re-transcription (e.g. from cleaned stems) shifts word edges a bit


def words(t: str) -> str:
    """Text without punctuation/case: whisper re-runs differ in those."""
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", t.lower()).split())


def refs_dir(base: Path):
    for name in ("refs_f5", "refs"):
        if (base / name).is_dir():
            return base / name
    return None


def refsha(base: Path, spk: str):
    d = refs_dir(base)
    p = d / f"{spk}.wav" if d else None
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12] \
        if p and p.exists() else None


def copy_refs_if_missing(new: Path, old: Path) -> bool:
    """The normal flow reaches carry_over before make_refs has run: the
    new dir has no references, so no manifest entry could ever be
    written (review 10-10 #1 — TODO #5 was a silent no-op without this).
    Copy the old run's references over (mtimes preserved: the bridge's
    ref-newer-than-clips guard must keep seeing pre-clip times);
    run_pipeline sees refs_f5/ and skips make_refs."""
    if refs_dir(new) is not None:
        return False
    src = refs_dir(old)
    if src is None:
        return False
    shutil.copytree(src, new / src.name, copy_function=shutil.copy2)
    print(f"voice references: copied {src} -> {new / src.name} "
          "(new run had none; identical refs keep the carried clips)")
    return True


def emit_manifest(new: Path, old: Path, mapping: list) -> int:
    """Manifest entries for carried clips (review 10-09 #5): without
    them the manifest-era synthesizer re-sings every copy for hours.
    Each entry is hash(speaker | OLD run's accented text | NEW ref) and
    is written ONLY when
      - the speaker's reference is byte-identical in both runs (a new
        voice reference means the clip SHOULD re-synthesize), and
      - the OLD run's manifest confirms the clip really was sung from
        that text with that ref (review 10-10 #2: the accents text is
        otherwise taken on trust) — no old manifest, no entry.
    Stale-safe either way: a missing/wrong entry only causes a re-synth,
    never stale audio.
    """
    old_acc_p = old / "accents.json"
    old_acc = json.loads(old_acc_p.read_text()) if old_acc_p.exists() else {}
    old_mp = old / "tts_f5" / "manifest.json"
    old_manifest = json.loads(old_mp.read_text()) if old_mp.exists() else None
    if old_manifest is None and mapping:
        print(f"NOTE: {old} has no tts_f5/manifest.json — carried clips "
              "cannot be verified against what they were sung from; no "
              "manifest entries written, they will re-synthesize.")
        return 0

    same_ref = {}
    mpath = new / "tts_f5" / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    n = 0
    for s, o in mapping:
        spk = s["speaker"]
        if spk not in same_ref:
            a, b = refsha(old, spk), refsha(new, spk)
            same_ref[spk] = a is not None and a == b
        if not same_ref[spk]:
            continue
        text = old_acc.get(str(o["id"]), o.get("text_ru", ""))
        entry = new_hash(spk, text, refsha(new, spk))
        # refs are identical, so the old run's entry for this clip must
        # equal the very hash we are about to write — else the clip was
        # NOT sung from this text (edited after synthesis, --only re-roll
        # of a different text, ...) and must honestly re-sing
        if old_manifest.get(str(o["id"])) != entry:
            continue
        manifest[str(s["id"])] = entry
        n += 1
    mpath.write_text(json.dumps(manifest, indent=1))
    return n


def main() -> None:
    new, old = Path(sys.argv[1]), Path(sys.argv[2])
    out = (Path(sys.argv[sys.argv.index("--out") + 1])
           if "--out" in sys.argv else new / "carry.json")

    old_segs = [s for s in json.loads((old / "segments_ru.json").read_text())["segments"]
                if s.get("type") != "orig" and s.get("text_ru")]
    index = {}
    for s in old_segs:
        index.setdefault((s["speaker"], words(s["text"])), []).append(s)

    new_segs = [s for s in json.loads((new / "segments.json").read_text())["segments"]
                if s.get("type") != "orig"]
    copy_refs_if_missing(new, old)
    carry, copied, mapping = {}, 0, []
    (new / "tts_f5").mkdir(exist_ok=True)
    for s in new_segs:
        for o in index.get((s["speaker"], words(s["text"])), ()):
            if abs(o["start"] - s["start"]) <= TOL and abs(o["end"] - s["end"]) <= TOL:
                carry[str(s["id"])] = o["text_ru"]
                src = old / "tts_f5" / f"seg_{o['id']:04d}.wav"
                dst = new / "tts_f5" / f"seg_{s['id']:04d}.wav"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                if dst.exists():
                    mapping.append((s, o))
                break

    n_manifest = emit_manifest(new, old, mapping)
    if copied and not n_manifest:
        print(f"WARNING: {copied} clips copied but 0 manifest entries written "
              "— every copy will RE-SYNTHESIZE (differing refs or no old "
              "manifest). The copies saved nothing.")
    out.write_text(json.dumps(carry, ensure_ascii=False, indent=0))
    print(f"{len(carry)}/{len(new_segs)} segments carried over ({copied} clips copied, "
          f"{n_manifest} manifest entries) -> {out}; "
          f"{len(new_segs) - len(carry)} need translation + synthesis")


if __name__ == "__main__":
    main()
