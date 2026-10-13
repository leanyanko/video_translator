#!/usr/bin/env python
"""Re-roll the clips QC flagged — F5's rendering lottery, closed by QC
for the failures QC can measure (tracker d6; TODO #17 keeps the rest).

Model: none here (calls synthesize_f5.py and qc_tts.py).

Usage: venv/bin/python scripts/reroll_qc.py work/<id> [--tts-dir tts_f5]
           [--sim-min 0.9] [--dry-run]

Why: F5 on MPS is not deterministic, and the same line at another
duration or seed renders anew — the same text, reference and seed gave
QC 0.59–0.80 («проб овали») in one run and 1.00 in the next (traps
line 9, 10-13). Whisper's back-transcription hears the gross failures:
every bad rendering scored under SIM_REROLL, every good one 1.00, while
SIM_MIN alone let «проб овали» (0.80) ship. So: clips with any QC flag
or sim < SIM_REROLL (qc_text, one home with SIM_MIN) are re-sung up the
LADDER — as is (the lottery alone), then with 10 % more time per piece
(a crammed rendering is what a too-tight duration produces), then with
another seed — one synthesize call per rung for all remaining clips
(the model loads once per call); after each rung a clip is kept only if
QC got better, else the previous wav and QC entry come back. Stops per
clip when it is clean. MPS kills synthesis at random (that is why
run_pipeline uses synth_supervisor.sh): a rung that dies is retried
once, then logged and treated as "no improvement" — the backups make
that safe. Output of every call goes to tmp/reroll/<rung>.log, the
verdicts to work/<id>/reroll_report.json (the doctor FAILs on clips
still flagged). What QC cannot hear (soft pronunciation faults, a
crumpled phrase with its words intact) stays with the ear and the
reference (s27, TODO #17).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cli_args import flag, opt
from qc_text import SIM_REROLL

LADDER = (("same", []), ("room", ["--room", "1.1"]), ("seed", ["--seed", "7"]))
ROOT = Path(__file__).resolve().parent.parent
PY_F5 = ROOT / "venv-f5/bin/python"
PY = ROOT / "venv/bin/python"
SCRIPTS = ROOT / "scripts"


def pick(qc: dict, sim_min: float = SIM_REROLL) -> list:
    """ids (str) whose QC entry carries a flag or sim < sim_min, worst first."""
    bad = [(k, v) for k, v in qc.items() if v.get("flags") or v.get("sim", 1.0) < sim_min]
    return [k for k, _ in sorted(bad, key=lambda kv: (len(kv[1].get("flags", [])) == 0,
                                                       kv[1].get("sim", 0.0)))]


def better(new: dict, old: dict) -> bool:
    """Fewer flags wins; same flags — higher sim wins."""
    nf, of = len(new.get("flags", [])), len(old.get("flags", []))
    return nf < of or (nf == of and new.get("sim", 0.0) > old.get("sim", 0.0))


def clean(entry: dict, sim_min: float) -> bool:
    return not entry.get("flags") and entry.get("sim", 0.0) >= sim_min


def _run(cmd: list, log: Path, tries: int = 2) -> bool:
    """Run `cmd`, appending output to `log`; retry once (MPS deaths are
    random). False when it still fails — the caller logs and moves on."""
    for attempt in range(1, tries + 1):
        with log.open("a") as f:
            f.write(f"\n### attempt {attempt}: {' '.join(map(str, cmd))}\n")
            f.flush()
            rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            return True
        print(f"    call failed (rc={rc}, attempt {attempt}) — see {log}", flush=True)
    return False


def main() -> None:
    t_start = time.time()
    work = Path(sys.argv[1])
    tts_name = opt("--tts-dir", "tts_f5")
    sim_min = opt("--sim-min", SIM_REROLL, float)
    dry = flag("--dry-run")
    tts = work / tts_name
    qc_path = work / f"qc_{tts_name}.json"
    if not qc_path.exists():
        sys.exit(f"no {qc_path.name} — run qc_tts.py first")
    qc = json.loads(qc_path.read_text())
    todo = pick(qc, sim_min)
    print(f"re-roll: {len(todo)} of {len(qc)} clips flagged or under sim {sim_min}: "
          + ", ".join(todo[:20]) + ("..." if len(todo) > 20 else ""), flush=True)
    if dry or not todo:
        return
    keep = work / "tmp" / "reroll"
    keep.mkdir(parents=True, exist_ok=True)
    for old in keep.glob("*.log"):
        old.unlink()

    def wav(sid):
        return tts / f"seg_{int(sid):04d}.wav"

    def backup(sid):
        return keep / f"seg_{int(sid):04d}.wav"

    best = {sid: dict(qc[sid]) for sid in todo}
    report = {sid: [{"rung": "start", "sim": qc[sid].get("sim"), "flags": qc[sid].get("flags", [])}]
              for sid in todo}
    for sid in todo:
        shutil.copy(wav(sid), backup(sid))
    remaining, per_rung = list(todo), {}
    for rung, extra in LADDER:
        if not remaining:
            break
        per_rung[rung] = len(remaining)
        ids = ",".join(remaining)
        print(f"  rung «{rung}»: {len(remaining)} clip(s)", flush=True)
        log = keep / f"{rung}.log"
        ok = (_run([PY_F5, SCRIPTS / "synthesize_f5.py", work, "--only", ids] + extra, log)
              and _run([PY, SCRIPTS / "qc_tts.py", work, "--tts-dir", tts_name, "--only", ids], log))
        if not ok:
            # the rung did not finish: whatever it re-sang is unverified —
            # back to the previous wavs, QC entries untouched, next rung
            for sid in remaining:
                shutil.copy(backup(sid), wav(sid))
                report[sid].append({"rung": rung, "failed": True})
            print(f"  rung «{rung}» failed twice — previous clips restored", flush=True)
            continue
        qc = json.loads(qc_path.read_text())
        still = []
        for sid in remaining:
            entry = qc[sid]
            report[sid].append({"rung": rung, "sim": entry.get("sim"), "flags": entry.get("flags", [])})
            if better(entry, best[sid]):
                best[sid] = dict(entry)
                shutil.copy(wav(sid), backup(sid))
            else:
                shutil.copy(backup(sid), wav(sid))
                qc[sid] = best[sid]
            if not clean(best[sid], sim_min):
                still.append(sid)
        qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=1))
        remaining = still
    fixed = [s for s in todo if clean(best[s], sim_min)]
    improved = [s for s in todo if s not in fixed and better(best[s], report[s][0])]
    for sid in todo:
        report[sid].append({"rung": "kept", "sim": best[sid].get("sim"), "flags": best[sid].get("flags", [])})
        backup(sid).unlink(missing_ok=True)  # the kept wav is in tts/; logs stay
    secs = time.time() - t_start
    (work / "reroll_report.json").write_text(json.dumps(
        {"sim_min": sim_min, "seconds": round(secs), "per_rung": per_rung,
         "clean": fixed, "still_flagged": [s for s in todo if s not in fixed], "clips": report},
        ensure_ascii=False, indent=1))
    print(f"re-roll: {len(fixed)} clean, {len(improved)} improved but still flagged, "
          f"{len(todo) - len(fixed) - len(improved)} unchanged of {len(todo)}; "
          f"rungs {per_rung}; {secs / 60:.1f} min (reroll_report.json)", flush=True)


if __name__ == "__main__":
    main()
