#!/usr/bin/env python
"""Find lines that are REPEATED by another speaker (mic-bleed symptom).

Model: none (text similarity over a segments file).

Usage: venv/bin/python scripts/repeat_audit.py work/<id> [segments_ru.json]
           [--window 3] [--min-sim 0.6] [--mark-echo]

Two segments of different speakers that start within --window seconds
of each other and whose English text overlaps strongly are the same
utterance heard on two mics — after dubbing, the viewer hears the line
twice in two voices. Pasted stem clips (type "orig") are included: their
transcribed text is checked the same way. Prints every pair and a count;
0 is the target for a clean run.

Most remaining pairs are real ECHOES: a listener repeating or completing
the speaker's words while he talks ("…to take off the knot" / "Take it
off"). In English they coincide; dubbed, the speaker's Russian puts that
word elsewhere, so the echo becomes an audible repeat by another voice.
--mark-echo flags the echo side with "echo": true in the segments file
(assemble.py skips it; the speaker carries the line alone) when it is the
shorter line, ≤ ECHO_MAX_SECS, starts inside the other line, and ≥
ECHO_SHARE of its content words occur in the other line. Replies that
merely share small words ("They are repairable.") are kept.
"""
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

work = Path(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "segments_ru.json"
window = float(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 3.0
min_sim = float(sys.argv[sys.argv.index("--min-sim") + 1]) if "--min-sim" in sys.argv else 0.6

segs = json.loads((work / name).read_text())["segments"]
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).split()
# filler-only lines ("yeah", "mm-hmm") are said by everyone — not repeats
FILLER = {"yeah", "yes", "okay", "ok", "mmhmm", "hmm", "mm", "mmm", "right", "uhhuh",
          "no", "so", "and", "oh", "wow", "sure", "yep", "exactly", "ha"}

ECHO_MAX_SECS, ECHO_SHARE = 3.5, 0.6
STOP = {"the", "a", "an", "it", "its", "is", "are", "was", "were", "be", "they", "theyre", "we",
        "you", "i", "he", "she", "that", "thats", "this", "to", "of", "in", "on", "at", "for",
        "with", "what", "whats", "about", "like", "just", "but", "do", "dont", "did", "its", "there",
        "theres", "have", "has", "can", "if", "or", "all", "one", "not"}


def echo_of(a, b):
    """The echo segment of pair (a, b), or None."""
    short, long_ = sorted((a, b), key=lambda s: s["end"] - s["start"])
    if short["end"] - short["start"] > ECHO_MAX_SECS:
        return None
    if not (long_["start"] < short["start"] < long_["end"]):
        return None
    content = set(norm(short.get("text", ""))) - FILLER - STOP
    if not content:
        return None
    other = set(norm(long_.get("text", "")))
    return short if len(content & other) / len(content) >= ECHO_SHARE else None


pairs = []
segs = sorted(segs, key=lambda s: s["start"])
for i, a in enumerate(segs):
    wa = norm(a.get("text", ""))
    if len(set(wa) - FILLER) < 2:
        continue
    for b in segs[i + 1:]:
        # compare lines that OVERLAP in time (or start within --window): an
        # echo can sit deep inside the other speaker's long line
        if b["start"] > max(a["end"], a["start"] + window):
            break
        if b["speaker"] == a["speaker"]:
            continue
        wb = norm(b.get("text", ""))
        if len(set(wb) - FILLER) < 2:
            continue
        m = SequenceMatcher(None, wa, wb)
        shared = sum(bl.size for bl in m.get_matching_blocks())
        content = {w for bl in m.get_matching_blocks()
                   for w in wa[bl.a: bl.a + bl.size]} - FILLER
        sim = shared / min(len(wa), len(wb))
        if sim >= min_sim and shared >= 2 and content:
            pairs.append((sim, a, b))

echoes = {}
for sim, a, b in pairs:
    e = echo_of(a, b)
    if e is not None:
        echoes[e["id"]] = e
for sim, a, b in pairs:
    tag = "  ECHO→drop #%d" % echo_of(a, b)["id"] if echo_of(a, b) is not None else ""
    print(f"{sim:4.0%}{tag}  {a['start']:7.1f} {a['speaker'][-2:]}#{a['id']} {a.get('type', '')}: {a.get("text", "")[:60]}")
    print(f"       {b['start']:7.1f} {b['speaker'][-2:]}#{b['id']} {b.get('type', '')}: {b.get("text", "")[:60]}")
if "--mark-echo" in sys.argv:
    data = json.loads((work / name).read_text())
    for s in data["segments"]:
        if s["id"] in echoes:
            s["echo"] = True
    (work / name).write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"marked {len(echoes)} echo segments in {name}: {sorted(echoes)}")
print(f"{len(pairs)} repeated lines across speakers, {len(echoes)} of them echoes "
      f"({sum(1 for _, a, b in pairs if 'orig' in (a.get('type'), b.get('type')))} involve pasted stem clips)")
