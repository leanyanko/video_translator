# video_translator — YouTube → Russian dubbing pipeline

Dubs a spoken-word video (a YouTube link, or an editor's project with one
audio stem per speaker) into Russian, with every speaker voiced by a clone
of their own voice. Runs fully locally on Apple Silicon (MLX for Whisper,
MPS for TTS). Tested on an M4 Pro with 24 GB; a two-hour episode takes
roughly a working day of machine time, most of it synthesis.

This file is the source of truth for how the pipeline works and how to run
it. `RU_INSTRUCTIONS.md` is a Russian-language guide for translation
contractors, derived from it.

---

## 1. What happens to a video

```
download ─► transcribe ─► segment ─► TRANSLATE ─► stress marks ─► voice refs
                                        (gate)                        │
         video_ru.mp4 ◄─ assemble ◄─ QC ◄─ synthesize (F5-TTS) ◄───────┘
                                            ▲ doctor          ▲ doctor
```

1. **Transcribe** with Whisper (word timestamps). Long audio is cut into
   ~20-minute chunks at speech pauses; Whisper degrades on long single
   passes (punctuation disappears after ~90 min), so a punctuation-collapse
   detector restarts it from the last sentence end where it derails.
2. **Segment** into sentences: cuts only at sentence-final punctuation or
   long silences, short same-speaker sentences merged up to ~12 s.
3. **Translate** — the one stage the pipeline does not run itself. An LLM
   or a human produces `segment id → Russian text` chunks following the
   rules in §5. `run_pipeline` stops here and resumes when every segment
   has a translation.
4. **Stress marks** (RUAccent + the dictionaries in `data/`): Russian TTS
   mispronounces unmarked homographs and foreign names.
5. **Voice references**: one ≤ 11.5 s clip per speaker, auditioned for a
   clean start, with the exact transcript the clip contains.
6. **Synthesize** every segment with the Russian F5-TTS fine-tune, pacing
   each line to the time slot it has on the original timeline.
7. **QC**: every clip is transcribed back and compared with its script.
8. **Assemble**: clips are placed at their original start times, overruns
   are time-fitted, concurrent speech is overlaid, original laughs and the
   music bed are mixed back in, and the result is muxed with the untouched
   video.

### Two input modes

Chosen automatically by what is in `work/<id>/`: `tracks/` + `tracks.json`
present → multitrack, otherwise mix.

| | **Mix** (YouTube audio only) | **Multitrack** (editor's per-speaker stems) |
|---|---|---|
| Who is speaking | pyannote diarization, word-level majority vote | known from the stem; no diarization |
| Overlapping speech | recovered from diarization vs. Whisper geometry | read clean off each solo stem; overlaps from timeline geometry |
| Mic bleed | n/a | `clean_stems.py` mutes other voices by speaker embedding before transcription |
| Laughs / reactions | detected in diarization gaps only | harvested clean from each stem; can be overlaid on speech |
| Music | lost | the "Music and SFX" stem is mixed back in |

Ask the editor for stems whenever possible: one file per speaker plus
"Music and SFX", timeline-aligned and level-balanced. Mix mode exists for
videos where they are not available.

### Gates — what stops a bad dub from shipping

- **Doctor** (`scripts/check_pipeline.py work/<id>`): consistency of every
  stage — stress marks match the current translation, references are
  4–11.6 s with no silent head, every segment has a clip whose manifest
  hash matches, original-audio inserts exist. `run_pipeline` runs it before
  synthesis and before assembly and aborts on failures. Run it by hand
  whenever in doubt.
- **Manifest** (`tts_f5/manifest.json`): per clip, a hash of
  `speaker | accented text | reference audio`. Edit a line, re-run the
  accents or re-cut a reference and exactly the affected clips are
  re-synthesized; nothing stale can be assembled. Work dirs from before the
  manifest are migrated by `scripts/manifest_bridge.py` from the recorded
  accent files (`accents.prev.json`); `--adopt-existing` is only for
  hand-copied clips with no record.
- **Freshness**: synthesis refuses to run if `accents.json` disagrees with
  `segments_ru.json` (re-run `accent_texts.py`; `run_pipeline` does).
- **QC** (`scripts/qc_tts.py`): back-transcription flags MISMATCH / EXTRA /
  LEAD / TAIL / SILENT clips in `qc_tts_f5.json`; fix with
  `synthesize_f5.py --only <ids> --seed 7` and re-assemble.

---

## 2. One-time setup

```bash
brew install ffmpeg portaudio

# main environment: download, Whisper, pyannote, assembly
python3.12 -m venv venv
venv/bin/pip install -r requirements-venv.txt

# synthesis environment (F5-TTS + RUAccent — they deadlock in one process)
python3.12 -m venv venv-f5
venv-f5/bin/pip install -r requirements-f5.txt
```

`requirements-*.txt` pin the direct dependencies at the tested versions;
`requirements-*.lock.txt` are full freezes that recreate the environments
exactly. Known quirks already handled in the scripts: new `torchaudio`
does not link against Homebrew ffmpeg 8 (audio I/O goes through
`soundfile`); RUAccent's ONNX model needs a `token_type_ids` input the
library forgets to pass (patched in `accent_texts.py`).

### Hugging Face

Create a Read token at https://huggingface.co/settings/tokens, then
`venv/bin/hf auth login`. Accept the terms on all three gated pages
(each is a separate click; approval is instant) or diarization will not
start:

1. https://huggingface.co/pyannote/speaker-diarization-3.1
2. https://huggingface.co/pyannote/segmentation-3.0
3. https://huggingface.co/pyannote/speaker-diarization-community-1

Download the production TTS checkpoint once:

```bash
venv/bin/hf download Misha24-10/F5-TTS_RUSSIAN --local-dir checkpoints/f5-ru
```

Whisper (`mlx-community/whisper-large-v3-turbo`), pyannote, the Vocos
vocoder and RUAccent download themselves on first use.

### Licences

- `Misha24-10/F5-TTS_RUSSIAN` — **CC-BY-NC-4.0, non-commercial only.**
- pyannote — MIT, gated (terms above).
- Fish Speech / OpenAudio checkpoints (legacy) — CC-BY-NC-SA, not needed.

---

## 3. Running a video

```bash
venv/bin/python scripts/run_pipeline.py "<youtube-url>"   # from a URL
venv/bin/python scripts/run_pipeline.py --work work/<id>  # existing dir
#   --limit-secs N   preview: only the first N seconds (mix mode)
```

Resumable: every stage that already has its output is skipped. The run
stops at the translation gate and prints how many segments still need
text; apply the translation (§5) and run the same command again. Stage
timings are appended to `work/<id>/timings.json`. Synthesis runs under
`scripts/synth_supervisor.sh`, which restarts it after the occasional
MPS crash; for an overnight run start it with `nohup`.

Outputs in `work/<id>/`: `video_ru.mp4` (upload master), `video_ru_h264.mp4`
(QuickTime-friendly re-encode), `transcripts/<mmdd-hhmmss>/` (EN+RU text and
segment JSON for every translation application).

Re-voicing after editing `segments_ru.json` (only changed lines are
re-synthesized, voices reused from `refs_f5/`):

```bash
venv/bin/python scripts/redub.py work/<id>
# standalone: scripts/redub.py <video.mp4> <segments_ru.json> --refs work/<prev>/refs_f5
```

### The same steps individually

Every script's docstring is its usage and names the model it uses.

```bash
# mix mode
venv/bin/python scripts/download.py "<youtube-url>"        # → work/<id>/{video.mp4,audio.wav}
venv/bin/python scripts/diarize.py work/<id>               # → diarization.json (~15 min / 2 h)
venv/bin/python scripts/transcribe.py work/<id>            # → words.json (~15 min / 2 h; chunks cut at diarization gaps)
venv/bin/python scripts/segment_sentences.py work/<id>     # → segments.json
# multitrack mode (instead of the three lines above; see §4 for preparation)
venv/bin/python scripts/clean_stems.py work/<id>           # → tracks_clean/ (run_pipeline swaps it in as tracks/)
venv/bin/python scripts/transcribe_tracks.py work/<id> --language en   # → words_tracks.json
venv/bin/python scripts/segment_tracks.py work/<id>        # → segments.json (+ overlap flags, nonverbal inserts)

# shared tail
venv/bin/python scripts/apply_translation.py work/<id> <chunk.json>   # merges into segments_ru.json, exports transcripts
venv-f5/bin/python scripts/accent_texts.py work/<id>       # → accents.json (separate process from F5!)
venv/bin/python scripts/make_refs.py work/<id> --max-secs 11.5 --outdir refs_f5 [--tracks]
venv/bin/python scripts/check_pipeline.py work/<id>        # doctor
venv-f5/bin/python scripts/synthesize_f5.py work/<id>      # → tts_f5/ (~19 s/segment; --only ids --seed N; --dry-run)
venv/bin/python scripts/qc_tts.py work/<id>                # → qc_tts_f5.json
venv/bin/python scripts/harvest_laughs.py work/<id> --threshold 0.12   # multitrack: clean laughs → laughs/, segments_ru.json
venv/bin/python scripts/assemble.py work/<id> --tts-dir tts_f5 --out video_ru.mp4 [--music "work/<id>/tracks/<Music>.mp3"]
ffmpeg -i work/<id>/video_ru.mp4 -c:v h264_videotoolbox -b:v 6000k -c:a copy -movflags +faststart work/<id>/video_ru_h264.mp4
```

Other tools: `carry_over.py <new> <old>` pre-fills translations and
clips from an earlier run after re-segmentation; `repunctuate_words.py`
exports Whisper run-ons for punctuation repair and applies the fix without
touching word timings; `repeat_audit.py` finds lines heard on two mics and
marks echoes so they are dubbed once; `export_transcripts.py` re-exports
transcripts at any time.

---

## 4. Multitrack preparation

```
work/<id>/tracks/                 one mp3/wav per speaker + the music stem
work/<id>/tracks.json             {"speakers": {"SPEAKER_00": "<Dan>.mp3", ...},
                                   "music": "<Music and SFX>.mp3"}
```

Identify who is who by listening to two or three long segments per stem.
`run_pipeline` then cleans the stems (originals kept in `tracks_raw/`),
transcribes each solo stem (only its voiced islands), segments per
speaker, marks concurrent speech with pan/gain flags that `assemble.py`
honours, pastes non-verbal sounds ("mm-hmm", laughs) from the stem as
`type: "orig"` inserts instead of synthesizing them, and mixes the music
stem back in at the editor's levels (`--music-gain` to adjust).

If `segment_tracks.py` reports "mid-sentence ends … inside run-ons",
Whisper lost punctuation there: `repunctuate_words.py export`, restore
punctuation (LLM or human; only punctuation and case may change), `apply`,
re-segment.

---

## 5. Translation rules

Translation happens outside the scripts and is applied in chunks: a JSON
map `{"<segment id>": "<Russian text>", ...}` to `apply_translation.py`.

1. **Timing budget = the original line's duration.** At most ~13 Russian
   characters per second of the segment's own span (end − start); pauses
   between lines are not part of the budget. The budget is also a floor:
   the Russian must **fill** the original speech (≈ 90–100 %), keeping
   false starts, repetitions and self-corrections — a compressed line ends
   while the speaker is still visibly talking. `apply_translation.py`
   prints `SHORT` (< 80 %) and `LONG` (> 115 %) per segment.
2. **Stress marks are automatic**, three layers, higher wins: RUAccent;
   the dictionaries `data/stress_names.json` (foreign names by stem, so all
   inflections are covered) and `data/homographs.json` (same spelling,
   different stress — disambiguated by cue words in the English original:
   *shower* → `д+уше`, *soul* → `душ+е`); translator-placed `+` in the
   text itself, which overrides everything. Extend both files whenever an
   error is found; mark by hand only what the files do not know.
3. **Forbidden words** — `data/forbidden_words.json` lists words the TTS
   mispronounces regardless of marks; rephrase (`apply_translation.py`
   warns).
4. Names and terms follow established Russian usage; write numbers out in
   words — the TTS reads digits badly.

---

## 6. Quality control routine

1. Read `qc_tts_f5.json` after every synthesis: `MISMATCH` (heard text far
   from the script), `EXTRA` (extra words), `LEAD`/`TAIL` (noise before or
   after the line), `SILENT`. Re-roll the ids with
   `synthesize_f5.py --only <ids> --seed 7`, re-run `qc_tts.py --only <ids>`.
2. Listen through. Pronunciation errors are fixed by editing `accents.json`
   or the dictionaries, re-rolling the affected ids, and re-assembling.
3. Translation edits go into `segments_ru.json` (or a new chunk through
   `apply_translation.py`); `redub.py` or `run_pipeline.py` re-synthesizes
   exactly the changed lines.

---

## 7. Repository layout

```
scripts/      one script per stage (docstring = usage; "Model:" line = which model)
  run_pipeline.py, redub.py                   orchestrators
  audio_utils.py, segmenting.py, manifest_bridge.py,
  stress_rules.py, text_split.py, cli_args.py shared helpers
  synthesize_chatterbox.py, synthesize_qwen3.py experimental TTS engines (benchmarks)
  synthesize.py, openvoice_convert.py, detect_laughs.py, overlap_speech.py,
  mark_overlaps.py, stress_report.py          legacy / mix-mode-only tools
data/         stress_names.json, homographs.json, forbidden_words.json
tests/        python -m unittest discover -s tests
work/<id>/    everything per video (git-ignored): audio, words, segments,
              segments_ru, accents, refs_f5/, tts_f5/, video_ru*.mp4, transcripts/
checkpoints/  model weights (git-ignored; §2)
TODO.md       backlog with evidence and verification criteria
tracker/      local progress board for the review findings (git-ignored)
RU_INSTRUCTIONS.md   Russian guide for translation contractors (derived from this file)
```

## 8. Known limitations

- Translation quality is the ceiling, and it is produced outside the
  pipeline (§5).
- Segments are identified by position; re-segmenting after translation
  needs `carry_over.py` to bring translations and clips across.
- Mix mode cannot restore music or laughter that overlaps speech, and
  short interjections over another speaker are attributed to that speaker.
- macOS / Apple Silicon only (MLX, MPS, VideoToolbox).
