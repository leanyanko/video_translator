#!/usr/bin/env python
"""Extract a voice-clone reference clip for EVERY speaker.

Model: none (ffmpeg slicing over segments.json timings).

Usage: venv/bin/python scripts/make_refs.py work/<id> [--tracks] [--no-asr]
           [--speaker SPEAKER_XX]   (rebuild just one speaker's ref;
           the tts manifest then re-synthesizes only that speaker)
For each speaker, picks their longest run of consecutive segments with
tight gaps (continuous, uninterrupted speech) and extracts
refs/<SPEAKER>.wav + refs/<SPEAKER>.txt.

--tracks (multitrack mode): cut each clip from that speaker's SOLO stem
(work/<id>/tracks + tracks.json) instead of the mix — the reference is
then free of music and other voices, which clones noticeably cleaner.

Every candidate that passes the audio audition is BACK-TRANSCRIBED and
accepted only if its transcript text matches the audio (ref_check,
tracker s7); the .txt is written from that transcription, not from the
segment texts. Also reports chars/s — a slow, pause-heavy sample clones
a slow speaker (s27).
"""
import json
import subprocess
import sys
from pathlib import Path

from cli_args import flag, opt

MIN_SECS, MAX_SECS = 6.0, 20.0
MAX_GAP = 0.5

work = Path(sys.argv[1])
if flag("--max-secs"):
    MAX_SECS = opt("--max-secs", cast=float)
    MIN_SECS = min(MIN_SECS, MAX_SECS / 2)
refs_name = opt("--outdir", "refs")
data = json.loads((work / "segments.json").read_text())
refs_dir = work / refs_name
refs_dir.mkdir(exist_ok=True)

track_of = {}
if flag("--tracks"):
    tracks_meta = json.loads((work / "tracks.json").read_text())
    track_of = {spk: work / "tracks" / f
                for spk, f in tracks_meta["speakers"].items()}

# A ref whose AUDIO does not match its text head-on ruins every clip:
# F5 imitates the reference pattern, so a ref that opens with 2-3 s of
# silence or someone else's mumble makes the model paint the same junk
# before every line (seen in e1v5: Dan's ref had a 2.85 s silent head —
# every clip got a LEAD artifact and the dub felt delayed). Candidates
# are therefore AUDITIONED, longest first, and the first one that passes
# the audio checks wins.
from audio_utils import audition as _audition, decode  # noqa: E402
from pacing import PACE_MIN  # noqa: E402
from ref_check import (REF_TEXT_MATCH_MIN, check_reference,  # noqa: E402
                       slow_references, whisper_transcribe)

MAX_HEAD_SIL = 0.3   # sec of silence allowed before the first word
MAX_TAIL_SIL = 0.6   # sec allowed after the last word
MIN_VOICED = 0.55    # min share of voiced frames inside the clip


def audition(source, start, end):
    """(head_sil, tail_sil, voiced_share) of the span — the rule lives in
    audio_utils.audition; check_pipeline.py re-checks the chosen ref
    with the very same function."""
    a = decode(source, 16000, start=start, end=end)
    if len(a) < 800:  # nothing decodable: treat as all silence
        return end - start, end - start, 0.0
    return _audition(a, 16000)


def transcribe_cut(source, start, end):
    """(heard_text, last_word_end, audio_secs) of the candidate span —
    cut to a temp wav the same way the final ref is cut."""
    tmp = refs_dir / "_candidate.wav"
    asr_calls[0] += 1
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "quiet", "-i", str(source),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-ac", "1", "-ar", "16000", str(tmp)])
    heard, last = whisper_transcribe(tmp, data.get("language"))
    secs = len(decode(tmp, 16000)) / 16000
    tmp.unlink(missing_ok=True)
    return heard, last, secs


asr_calls = [0]
tempo = {}  # spk -> chars/s of the chosen reference (relative slow check below)
only_spk = opt("--speaker", None)
# --no-asr: skip the back-transcription (unit tests with synthetic audio,
# offline runs). The doctor still verifies text vs audio (s7), so this
# cannot ship an unchecked reference silently.
NO_ASR = flag("--no-asr")


def verdict_for(texts, source, start, end, dur):
    if NO_ASR:
        heard = " ".join(texts)
        return {"ok": True, "match": 1.0, "tail": 0.0, "heard": heard,
                "chars_per_sec": round(len(heard) / max(dur, 0.1), 1)}
    v = transcribe_cut(source, start, end + min(0.15, max(0.0, 11.6 - dur)))
    return check_reference(" ".join(texts), v[0], v[1], v[2])

for spk in data["speakers"]:
    if only_spk and spk != only_spk:
        continue
    segs = [s for s in data["segments"] if s["speaker"] == spk
            and s.get("type") != "orig"]
    runs = []  # (duration, start, end, texts)
    i = 0
    while i < len(segs):
        j = i
        while (
            j + 1 < len(segs)
            and segs[j + 1]["start"] - segs[j]["end"] <= MAX_GAP
            and segs[j + 1]["end"] - segs[i]["start"] <= MAX_SECS
        ):
            j += 1
        dur = segs[j]["end"] - segs[i]["start"]
        # audio and text must cover exactly the same span — never clip
        # audio mid-segment, skip runs that exceed the cap instead
        if dur <= MAX_SECS:
            runs.append((dur, segs[i]["start"], segs[j]["end"],
                         [s["text"] for s in segs[i: j + 1]]))
        i = j + 1

    source = track_of.get(spk, work / "audio.wav")
    runs.sort(key=lambda r: r[0], reverse=True)
    candidates = [r for r in runs if r[0] >= MIN_SECS] or runs
    chosen, verdict = None, None
    for dur, start, end, texts in candidates:
        head, tail, voiced = audition(source, start, end)
        if not (head <= MAX_HEAD_SIL and tail <= MAX_TAIL_SIL and voiced >= MIN_VOICED):
            print(f"  {spk}: rejected {start:.1f}-{end:.1f}s "
                  f"(head {head:.2f}s, tail {tail:.2f}s, voiced {voiced:.0%})")
            continue
        # text must say what the audio says (s7): cut, back-transcribe,
        # compare with the segment texts; the transcription becomes the .txt
        v_check = verdict_for(texts, source, start, end, dur)
        if not v_check["ok"]:
            print(f"  {spk}: rejected {start:.1f}-{end:.1f}s (text/audio match "
                  f"{v_check['match']:.0%} < {REF_TEXT_MATCH_MIN:.0%}: heard "
                  f"«{v_check['heard'][:70]}»)")
            continue
        chosen, verdict = (dur, start, end, texts, head, tail, voiced), v_check
        break
    if chosen is None:
        dur, start, end, texts = runs[0]
        head, tail, voiced = audition(source, start, end)
        verdict = verdict_for(texts, source, start, end, dur)
        chosen = (dur, start, end, texts, head, tail, voiced)
        print(f"WARNING: {spk}: no candidate passed the checks — using the "
              f"longest anyway (head {head:.2f}s, voiced {voiced:.0%}, "
              f"text match {verdict['match']:.0%}); the doctor will flag it")
    dur, start, end, texts, head, tail, voiced = chosen
    if dur < MIN_SECS:
        print(f"WARNING: {spk} best clip is only {dur:.1f}s — cloning may be weak")
    # pad the cut past the last word's decay: cutting exactly at the
    # word-end timestamp clips the final consonant, and F5 then imitates
    # the abrupt ending (seen: English ref tail leaking into Russian
    # clips). 0.15 s into the following pause is inaudible and safe —
    # the audition above already rejected runs with noisy tails.
    # ...but never past F5's ~11.6 s ref limit
    end_padded = end + max(0.0, min(0.15, 11.6 - dur))
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "quiet", "-i", str(source),
        "-ss", f"{start:.3f}", "-to", f"{end_padded:.3f}",
        "-ac", "1", "-ar", "44100", str(refs_dir / f"{spk}.wav"),
    ])
    # the .txt is what whisper HEARD in the cut — the only text F5 can
    # align to the audio without "finishing" it at the start of every clip
    (refs_dir / f"{spk}.txt").write_text(verdict["heard"] + "\n")
    tempo[spk] = verdict["chars_per_sec"]
    print(f"{spk}: {start:.1f}s-{end:.1f}s ({dur:.1f}s, head {head:.2f}s, "
          f"voiced {voiced:.0%}, text match {verdict['match']:.0%}, "
          f"{verdict['chars_per_sec']:.1f} chars/s, {asr_calls[0]} whisper call(s) so far)")

# a slow sample clones a slow speaker (s27): judged against the OTHER
# speakers of this run, the absolute floor only for a lone speaker
for spk in slow_references(tempo, PACE_MIN):
    print(f"WARNING: {spk}: reference runs {tempo[spk]:.1f} chars/s against "
          f"{ {k: v for k, v in tempo.items() if k != spk} } — F5 will clone the "
          "drawl; re-cut from a fluent moment (make_refs --speaker)")
