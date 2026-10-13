# Original-language leak at 22:32 (e1v6): Whisper gap-skip × non-speech restore

Status: **investigated, not yet fixed** (2026-10-12). Backlog: TODO #13.

At 22:32 of `work/e1v6/video_ru_h264.mp4` the dub plays ~28 seconds of
Dan's ORIGINAL English speech. This document reconstructs the full
causal chain, the evidence for each link, what was ruled out, the blast
radius, and the fix plan. Nothing in the pipeline has been changed yet.

## Symptom

Between 1345.9 s and 1373.6 s the final video contains four pasted
spans of the raw Dan stem — live English — instead of dubbed Russian.
The words Dan says there appear in NO artifact: not in
`words_tracks.json`, not in `segments.json`, not in the translation.

## The causal chain (two mechanisms, in series)

### Link 1 — Whisper silently skipped 29.7 s inside one pass

`segments_tracks_raw.json` (Whisper's raw output, before any of our
filtering) already has the hole:

```
1337.4-1345.2  "So, but if you don't understand the theory ... not just steel."
1374.9-1376.3  "Okay."            <- next Dan segment, 29.7 s later
```

The last word before the hole is ` steel.` (ends 1344.9), the next is
` Okay.` (starts 1374.9, prob 0.373). Nothing of our own dropped words
here — Whisper never emitted them.

What the skipped speech actually says (verified by transcribing the
window 1344–1376 in isolation — it transcribes PERFECTLY as a short
clip): *"So, again, if somebody asks you, could you make an
electromagnet which will attract copper or aluminum, you say, no, no,
no ... because of Lenz's law, it will actually repel aluminum and
copper, not attract ... it will repel it. See? Pushing it away."* —
an entire point about Lenz's law, lost.

#### Ruled out

* **clean_stems muting the owner** (the TODO #1 class): NO. The
  CLEANED stem (`tracks/`) is loud through the whole window — per-second
  RMS 0.03–0.26, 80 % of 0.1-s frames above `RMS_SILENCE` (0.0056).
  The audio reached Whisper audible.
* **Island boundaries**: NO. The window lies strictly inside one
  transcription island (1039.8–1558.8 s), so it was part of a single
  guarded pass; no island cut falls near it.
* **Our word filters** (bleed threshold, cross-track echo, orphan
  drop): NO. Those run after transcription and log what they drop; the
  raw Whisper output is already missing the words.

#### Why both existing guards were blind

`whisper_guard` has two signals, and this failure mode evades both:

1. **Punctuation-collapse detector** (`first_poor`): the words before
   and after the hole are normally punctuated — the 80-word window
   share never drops below 4 %. A *skip* is not a *collapse*.
2. **Stretched-word splice** (`fix_stretched`): fires on a "word"
   whose own span exceeds 2 s — the usual signature when Whisper
   bridges skipped speech with one long word. Here Whisper left a clean
   INTER-WORD GAP instead: ` steel.` ends normally, ` Okay.` starts
   normally, no word spans the hole. (The guard did fire correctly
   elsewhere in this same run: spliced regions at 2619 s, 5337 s,
   1132 s — this is a different shape of the same disease.)

The doctor's new degradation gate (5-minute windows, TODO #12) is also
blind by construction: the surrounding window still has plenty of
punctuated words, so punctuation share, tempo and stretched-word count
all look healthy.

### Link 2 — non-speech restore faithfully pasted the hole as "sounds"

`harvest_laughs.py` restores the owner's non-speech by the invariant
*"voiced on the RAW stem minus the owner's SPEECH SPANS = the owner's
non-verbal sounds"*. That invariant assumes the transcription is
complete. With 29.7 s of real speech absent from the spans, the whole
stretch became a legitimate candidate, and every gate passed honestly:

* **Anti-bleed**: own RMS 0.094–0.115 vs others 0.001 → KEEP (correct —
  it IS Dan's own voice on Dan's mic).
* **Laugh discriminator** (`--restore-sounds laughs`, the default):
  `is_laugh` = (dur ≥ 0.7 s AND ≥ 2 envelope bursts) OR PANNs ≥ 0.05.
  PANNs said 0.00 (not a laugh!), but the acoustic signature passed:
  speech pulses at syllable rate, so a 12.2-s span shows `bursts=20`.
  The signature was designed to separate laughter from single grunts
  ("mm", throat sounds) — it CANNOT separate laughter from speech,
  because under the invariant speech was never supposed to reach it.

Log evidence (`work/e1v6/pipeline.log`):

```
LAUGH 00  1345.9-1358.1 dur=12.2s bursts=20 panns=0.00 nonspeech
LAUGH 00  1358.4-1360.1 dur=1.7s  bursts=4  panns=0.00 nonspeech
LAUGH 00  1360.4-1364.8 dur=4.4s  bursts=12 panns=0.00 nonspeech
LAUGH 00  1365.4-1373.6 dur=8.2s  bursts=9  panns=0.00 nonspeech
```

Result: segments 9510–9513 (`type:"orig"`, `auto_track:true`,
files `laughs/SPEAKER_00_013459.wav` …) pasted raw English into the dub.

A red flag we currently do not check: a "laugh" lasting **12.2 seconds**.
Real laughs in this corpus are 0.7–3 s.

## Blast radius: this is the ONLY such spot in the 2-hour episode

Scanned every inter-word gap > 15 s per speaker and measured voiced
fraction of the CLEANED stem inside each:

* Mike/Misha (SPEAKER_02/03): dozens of long gaps — all genuine
  listening (their stems are silent there; islands handle it).
* Dan (SPEAKER_00): 9 gaps > 15 s. Eight are real pauses (0–2 % voiced
  frames). Exactly ONE is anomalous: **1345.2–1374.9, 80 % voiced**.
* All `auto_track` pastes longer than 4 s in e1v6: only these (12.2 s,
  8.2 s, 4.4 s — ids 9510/9513/9512).

## Fix plan (agreed direction, not yet implemented)

Two independent belts, attacking each link:

1. **Gap detector in `whisper_guard`** (root cause): after a pass,
   find inter-word gaps > N s (start with N≈10) where the source stem
   is voiced (cheap RMS check over the gap, same scan as islands);
   re-transcribe each such gap as a short isolated clip via the
   existing `run(a, b)` callback and splice the words in — symmetric to
   `fix_stretched`, which already proves the splice machinery. The
   isolated re-transcription demonstrably works (see above). Add the
   same gap measurement to the doctor (`degraded_windows` companion):
   voiced-but-wordless gap > N s → FAIL before anything downstream can
   inherit the hole.
2. **Sanity caps in `harvest_laughs`** (defense in depth): a laughs-mode
   candidate longer than ~4 s with PANNs = 0.00 is almost certainly not
   a laugh — drop it (or demote to `--restore-sounds all` only). This
   belt matters because ANY future upstream miss (new model, new
   failure shape) funnels into the same paste path.

Repair for e1v6 itself (surgical, ~15 min): re-transcribe the window,
re-segment locally, translate the ~3 new segments, synthesize them
(`--only`), re-assemble. The carried manifest keeps the other 1028
clips untouched.

## Reproduction notes

* Lost window audio: `ffmpeg -ss 1344 -t 32 -i "work/e1v6/tracks/Gelbart
  Episode Dan Audio Only.flac" -ac 1 -ar 16000 lost.wav` → transcribes
  cleanly with mlx-whisper large-v3-turbo in isolation.
* The hole in raw output: `segments_tracks_raw.json`, SPEAKER_00,
  between 1345.2 and 1374.9.
* Harvest verdicts: `pipeline.log` lines with `1345.9`/`013459`.
