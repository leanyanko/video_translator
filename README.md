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

# 2. transcribe with timestamps → work/<id>/segments_raw.json
venv/bin/python scripts/transcribe.py work/<id>

# 3. diarize + tag speakers + merge → work/<id>/segments.json
#    (needs accepted conditions on hf.co/pyannote/speaker-diarization-3.1
#     and hf.co/pyannote/segmentation-3.0)
venv/bin/python scripts/diarize.py work/<id> [max_speakers]

# 4. translate: Claude produces translation maps (segment id → Russian
#    text) in-session and applies each chunk with:
#      venv/bin/python scripts/apply_translation.py work/<id> <chunk.json>
#    This writes/merges segments_ru.json AND saves work/<id>/transcripts/
#    right away — transcripts are captured at translation time.

# 5. voice reference per speaker → refs/<SPEAKER>.wav + .txt
venv/bin/python scripts/make_refs.py work/<id>

# 6. start the TTS server (leave running; loads the model once)
cd fish-speech && ../venv/bin/python tools/api_server.py \
  --llama-checkpoint-path ../checkpoints/openaudio-s1-mini \
  --decoder-checkpoint-path ../checkpoints/openaudio-s1-mini/codec.pth \
  --decoder-config-name modded_dac_vq \
  --device mps

# 7. synthesize every segment, each with its speaker's cloned voice
#    (--limit-secs N for a preview sample) → work/<id>/tts/seg_*.wav
venv/bin/python scripts/synthesize.py work/<id>

# 8. time-fit, place on timeline, mux → work/<id>/video_ru.mp4
venv/bin/python scripts/assemble.py work/<id>

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
