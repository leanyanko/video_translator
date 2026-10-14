# Clip timing: how a dubbed line gets its length (tracker d5, s16, d6)

Status: rule in `scripts/pacing.py` + `synthesize_f5.py` + `assemble.py`
(pace7, 2026-10-13). This file is the investigation record; the code
carries only the rule and the tracker ids. Companion: TODO.md #14–#17,
`docs/whisper-gap-skip-leak.md` (the pastes that this work exposed).

## Symptom

Every episode had lines sped up for no reason (atempo 1.35 on lines
with room), others drawling (one speaker most of all), and the Russian
line did not end where the speaker's lips stopped. Anna flagged it on
the 2:41 traps clip.

## Three stacked causes

**1. `assemble` counted pasted originals as slot ends (s16).** A line's
slot is the time until the next line. `assemble` took "next line" to mean
the next *segment*, including `type:"orig"` pastes (laughs, backchannel
harvested from raw stems). A 0.5 s "mm-hmm" pasted into an 11 s line cut
its slot to 0.5 s → 1.35× for nothing. `synthesize_f5` filtered pastes
before computing *its* slot, so the two stages disagreed. e1v6: 121/817
lines had an other-speaker paste as the next item; 99 lines sat at the
cap. Fix: `manifest_bridge.slot_ends()` — one rule both stages import; a
slot ends at the next voiced line or a paste by the **same** speaker
(his own laugh is sequential in his timeline). Measured on e1v6 with the
existing clips: lines at the 1.35 cap 99 → 32, overrun past slot 312 s →
49 s.

**2. F5 budgets clip length in UTF‑8 bytes.** F5 does not measure the
clip — it sets it:
`duration = ref_sec + ref_sec / len(ref_text.utf8) × len(gen_text.utf8) / speed`.
English references are 1 byte/char, Russian lines 1.92 bytes/char
(with stress marks). Every line was budgeted ~1.9× the time its speaker
needs; baseline20 clips ran at 0.53× the reference's own chars/sec for
all three speakers. The old pacing (`est = chars/12.5`, speed 1.0–1.3)
fought the phantom with speed 1.3, `assemble` fought the rest with
atempo 1.35, and lines still overran. Lip length matched only by
coincidence: 1.9 / 1.3 / 1.35 ≈ 1.08. Two more rules in the same F5
function: any piece under 10 bytes (five Cyrillic letters) is sung at
speed 0.3; and a slow reference multiplies on top.

**3. F5 paints ~0.5 s of silence around every call** and crams the words
into the rest. Harmless on a 10 s line, fatal on a 1.6 s one (an intro
line came out at 21 chars/s and was garbled).

## The rule (pace7)

A dubbed line lasts as long as the **original line** — that is lip sync
— bounded to a plausible tempo: never faster than `PACE_MAX` (13
chars/s, the same number as the translator's budget; a long translation
overruns into the slot rather than being garbled, and F5 is *never*
asked to articulate faster — any further squeeze is `assemble`'s
atempo, a time-stretch of a clean rendering) and never slower than
`PACE_MIN` (9; a hesitant original does not make the dub drawl). The
plan goes to F5 as an exact `fix_duration = preprocessed_ref_sec +
share`, per piece (pieces split at sentence ends), each piece over-asked
by `SILENCE_PAD` (0.5 s) and trimmed back with
`audio_utils.speech_bounds` (speech found at −20 dB of the clip's own
p99, bounds extended while the signal decays above −40 dB so unvoiced
word endings survive), rejoined with `PIECE_GAP` (0.35 s). `fix_duration`
also bypasses the 10‑byte rule. Clip length matches the plan to ±0.03 s.
A pacing version salt (`PACE_VERSION`) in the manifest hash makes every
clip sung under an older rule stale.

Pre-flight on traps (from-scratch run, pace7): speech/span median 1.08,
voiced/span 0.83, median tempo 13.0, QC 1/23 flagged (a 0.6 s one-word
overlap echo whisper cannot score), mean sim 0.947. Reference run v5:
4/22, 0.918.

## Tried and rejected — do not go back

- **pace2: the speaker's REFERENCE tempo as the target.** Lost lip sync
  (Dan 20–30 % shorter than his lips), kept the slow speaker slow (his
  reference was slow), swallowed words. Audibly worse.
- **PACE_MAX 15.** Endings swallowed by ear on three clips; 13 is also
  the translator's budget, so one constant.
- **Single-threshold per-piece trim (−20 dB + 50 ms).** Cut unvoiced
  endings mid-decay (−30…−15 dB before joins); replaced by the decay
  rule. Note: the ear said "swallowed endings" and the first three
  hypotheses (trim, punctuation, painted breath) were each disproved by
  measurement on six renderings of one piece — see below.
- **Giving F5 more room to slow it down.** +25 % duration did not slow
  the speech (1.54 vs 1.56 s of speech); F5 keeps its rate and pads
  silence. Tempo cannot be bought with duration.
- **Per-speaker pad calibrated from the first piece.** n = 1 with a
  0.3–1.8 s spread is noisier than the documented constant; the residual
  shortfall is edge silence beyond the pad (reviewer correction: the
  trimmed total still contains internal pauses, so a 0.93 ratio means
  the edges carried more than the pad). A running per-speaker estimate
  is the candidate if it is ever needed.

## The rendering lottery (d6, TODO #17)

The same text, reference, seed and `fix_duration` render differently
from run to run on MPS: line 9 scored QC 0.59–0.80 («проб овали») in
one run and 1.00 in the next; six re-renderings with identical settings
were all fine by ear after twelve model/reference variants had been
judged bad. One rendering of line 7 rushed its first phrase (15 chars/s)
and left a 0.3 s hole (−73 dB) before continuing; renderings at another
duration spoke the phrase in 2.5–2.9 s with a natural pause. Punctuation
(colon / comma / dash / period) and a painted breath were ruled out:
all four punctuation variants were acceptable, and a broadband-burst
detector flagged the fricatives of the good renderings, not the bad
ones.

What QC measures: whisper similarity caught every bad rendering of line
9 (< 0.9) and none of the good ones. `reroll_qc.py` re-sings flagged /
low-sim clips up a ladder (as is → +10 % room → another seed), keeps a
clip only when QC improves, and the doctor FAILs on clips still flagged.
What QC does not measure: a crumpled phrase with its words intact (line
7 had sim 1.00), soft pronunciation faults the ear hears. An acoustic
hole alone does not discriminate (acceptable renderings also carry
0.5–0.65 s pauses at −70…−78 dB); the rushed phrase before the hole
does — no reliable threshold yet. #17 stays open.

## Reference quality (s27)

One speaker's reference was a slow, pause-heavy 10 s sample (73 %
voiced, 10 chars/s vs 16–22 for the others). F5 clones the sample's
manner, not only the timbre. Re-cut from a fluent moment of the same
speaker (8.0 s, 78 % voiced, 18 chars/s, text verified by whisper); old
sample kept in `refs_f5_old/`. By ear the two were comparable on two
lines — the lottery dominated — but the dense sample is the better
input, and `make_refs` should report chars/s (open).

## Open

- Timings are not in the clip fingerprint (s28); the bridge stamps the
  current salt on legacy clips (s29); overlap slot floors differ (s24);
  harvest runs after synthesis though same-speaker pastes end slots
  (s26); timing-chain audit document and end-to-end test (d12).
- Constants 9/13/0.5/0.35/0.12 are hypotheses until they hold on
  baseline20 and one full episode (CLAUDE.md §1, s25).
