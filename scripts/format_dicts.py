#!/usr/bin/env python
"""Canonical format for the data/ dictionaries (TODO #10).

Model: none.

Usage: venv/bin/python scripts/format_dicts.py [--check]

One ad-hoc json.dump with a different indent once rewrote every line of
homographs.json: an 842-line diff for two real edits, and the review
went blind. The canonical form is fixed here once:
    json.dumps(obj, ensure_ascii=False, indent=1) + "\n"
with keys in their SOURCE order (json preserves insertion order; the
formatter never sorts). Dictionaries are edited by hand or by code that
reproduces exactly this form — tests/test_dict_format.py fails on any
file that drifts, so the next dictionary edit diffs in N lines where N
is the number of entries actually changed.

This is also the one place that knows what an ENTRY should look like
(review 10-11): validate() checks content, not only bytes — a dead
homograph branch or a stress value without a mark should fail the same
way a bad byte does. Stems of <= 3 characters only WARN: they are the
prefix-match risk («тед» claims anything starting with тед), but the
three existing ones (дэн, тед, иаг) are deliberate.

--check: exit 1 listing non-canonical files and content errors, change
nothing (the mode the test uses). Default: rewrite drifted files in
place (content errors still only report — data edits are by hand).
"""
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"


def drifted(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    return raw != canonical(json.loads(raw))


def _one_mark_per_part(value: str) -> bool:
    """Every hyphen-separated part carries exactly one '+': «Р+ида-
    Солом+она» is one entry with two legitimate stresses — a flat
    one-'+'-per-value rule would reject it."""
    return all(part.count("+") == 1 for part in value.split("-"))


def validate() -> tuple[list, list]:
    """(errors, warnings) across the dictionaries' CONTENT."""
    errors, warnings = [], []

    def entries(name):
        d = json.loads((DATA / name).read_text(encoding="utf-8"))
        return [(k, v) for k, v in d.items() if k != "_comment"]

    for k, v in entries("stress_names.json"):
        if k != k.lower():
            errors.append(f"stress_names: key «{k}» is not lowercase")
        if not _one_mark_per_part(v):
            errors.append(f"stress_names: «{k}»: «{v}» must carry exactly "
                          "one '+' per hyphen-part")
        if len(k) <= 3:
            warnings.append(f"stress_names: stem «{k}» is <= 3 chars — "
                            "prefix match may claim unrelated words")

    for k, v in entries("homographs.json"):
        if k != k.lower():
            errors.append(f"homographs: key «{k}» is not lowercase")
        marked_seen = set()
        for var in v:
            m, en = var.get("marked"), var.get("en")
            if not m or "+" not in m:
                errors.append(f"homographs: «{k}»: variant without a "
                              f"stressed 'marked' form: {var}")
                continue
            if not en or not [c for c in en if c]:
                errors.append(f"homographs: «{k}»: «{m}» has no 'en' cues — "
                              "a dead branch that can never win")
            if m in marked_seen:
                errors.append(f"homographs: «{k}»: duplicate marked form "
                              f"«{m}» — the second variant is dead")
            marked_seen.add(m)

    for k, _ in entries("forbidden_words.json"):
        if k != k.lower():
            errors.append(f"forbidden_words: key «{k}» is not lowercase")
    return errors, warnings


def main() -> None:
    check = "--check" in sys.argv
    bad = []
    for path in sorted(DATA.glob("*.json")):
        if not drifted(path):
            continue
        bad.append(path)
        if not check:
            path.write_text(canonical(json.loads(path.read_text(
                encoding="utf-8"))), encoding="utf-8")
            print(f"reformatted {path}")
    errors, warnings = validate()
    for w in warnings:
        print("WARN", w)
    for e in errors:
        print("ERROR", e)
    if check and (bad or errors):
        sys.exit("NOT canonical: " + ", ".join(str(p) for p in bad)
                 + f"\nfix format with  venv/bin/python scripts/format_dicts.py"
                 + ("; content errors are fixed by hand" if errors else ""))
    if errors:
        sys.exit(1)
    if not bad:
        print(f"all {len(list(DATA.glob('*.json')))} dictionaries canonical")


if __name__ == "__main__":
    main()
