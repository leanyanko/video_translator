# YouTube → Russian dubbing pipeline (Fish Speech / OpenAudio S1-mini)

Dubs a YouTube video into Russian, cloning the original speaker's voice.
Runs fully locally on Apple Silicon (MPS).

## One-time setup (already done by Claude, except HF login)

1. `python3 -m venv venv && venv/bin/pip install yt-dlp mlx-whisper -e ./fish-speech`
2. `brew install ffmpeg portaudio`
3. Hugging Face (manual): accept license at
   https://huggingface.co/fishaudio/s1-mini then `venv/bin/hf auth login`
4. Download weights:
   `venv/bin/hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini`

## Per-video flow

```bash
# 1. download video + audio → prints WORKDIR=work/<id>
venv/bin/python scripts/download.py "<youtube-url>"

# 2. transcribe with WORD timestamps → words.json (+ segments_raw.json)
venv/bin/python scripts/transcribe.py work/<id>

# 3. diarize → work/<id>/diarization.json (speaker turns)
#    (needs accepted conditions on hf.co/pyannote/speaker-diarization-3.1,
#     hf.co/pyannote/segmentation-3.0 and speaker-diarization-community-1)
venv/bin/python scripts/diarize.py work/<id>

# 4. DEFAULT segmentation: sentence-boundary cuts + word-level speaker
#    attribution from words.json + diarization.json → segments.json
venv/bin/python scripts/segment_sentences.py work/<id>
#    (re-segmenting with other settings reuses words.json — no re-transcribe)

# 5. translate: Claude produces translation maps (segment id → Russian
#    text) in-session and applies each chunk with:
#      venv/bin/python scripts/apply_translation.py work/<id> <chunk.json>
#    This writes/merges segments_ru.json AND saves work/<id>/transcripts/
#    right away — transcripts are captured at translation time.

# 6. stress marks (separate process — RUAccent deadlocks next to F5)
venv-f5/bin/python scripts/accent_texts.py work/<id>

# 7. voice references: ≤11.5s per speaker, text exactly matching the clip
venv/bin/python scripts/make_refs.py work/<id> --max-secs 11.5 --outdir refs_f5

# 8. synthesize (production TTS: F5-TTS Russian; ~19s/segment, resumable;
#    --only <ids> --seed N for spot repairs) → work/<id>/tts_f5/seg_*.wav
venv-f5/bin/python scripts/synthesize_f5.py work/<id>

# 9. time-fit, place on timeline, mux → work/<id>/video_ru.mp4
venv/bin/python scripts/assemble.py work/<id> --tts-dir tts_f5 --out video_ru.mp4

# (legacy Fish Speech s1-mini flow: tools/api_server.py + synthesize.py —
#  superseded by F5; see scripts' docstrings)

# (transcripts are saved automatically: EN after step 3, EN+RU on every
#  apply_translation.py run. To re-export manually:)
venv/bin/python scripts/export_transcripts.py work/<id> [work/<id2> ...]
```

Notes
- fish-speech/ is pinned to commit 781bf1c ("Finetune support of
  OpenAudio-S1") — the last version compatible with the s1-mini
  checkpoint's tiktoken tokenizer. Do NOT git pull it to main.
- Multi-speaker: diarize.py tags every segment with a speaker;
  synthesize.py clones each speaker's own voice from refs/.
- Model weights are CC-BY-NC-SA (non-commercial).
- `synthesize.py` is resumable — re-run it and it skips finished segments.
- If MPS fails, restart the server with `--device cpu` (slower).
