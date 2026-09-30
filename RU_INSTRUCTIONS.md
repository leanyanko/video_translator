# Инструкция по пайплайну дубляжа (YouTube → русский)

Пайплайн полностью локальный (тестировался на Mac M4 Pro, 24 ГБ; нужен
Apple Silicon — синтез идёт через MPS/Metal). Он берёт видео с YouTube,
распознаёт речь, определяет спикеров, переводит на русский и озвучивает
каждого спикера клоном его собственного голоса.

Результаты по эпизоду лежат в `work/<video-id>/`:
`video_ru.mp4` (итоговый дубляж), `video_ru_h264.mp4` (то же, но H.264 —
для QuickTime), `transcripts/<ммдд-ччммсс>/` (стенограммы и сегментация
EN+RU на каждый запуск).

---

## 1. Что установить (один раз)

```bash
# системные пакеты
brew install ffmpeg portaudio

# основное окружение: yt-dlp, whisper, pyannote, сборка
python3 -m venv venv
./venv/bin/pip install yt-dlp mlx-whisper pyannote.audio pydub soundfile "huggingface_hub[cli]"

# окружение для синтеза (F5-TTS + расстановка ударений)
python3 -m venv venv-f5
./venv-f5/bin/pip install f5-tts ruaccent

# опционально: OpenVoice v2 (тембровая пост-обработка), нужен conda
# ($CONDA — путь к вашей установке conda; ниже в примерах /opt/miniconda3)
conda create -y -n openvoice python=3.10
/opt/miniconda3/envs/openvoice/bin/pip install torch torchaudio librosa soundfile "numpy<2" wavmark \
    inflect unidecode eng_to_ipa pypinyin cn2an jieba langid
/opt/miniconda3/envs/openvoice/bin/pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git
```

Известные особенности окружения:
- `torchaudio` новых версий не дружит с ffmpeg 8 из brew — скрипты уже
  обходят это через `soundfile`, ничего делать не надо.
- RUAccent и F5 нельзя загружать в один процесс (зависает) — поэтому
  ударения считаются отдельным скриптом заранее.

---

## 2. Hugging Face: аккаунт, лицензии, модели

Нужен аккаунт на huggingface.co и токен уровня Read
(https://huggingface.co/settings/tokens). Залогиниться:

```bash
./venv/bin/hf auth login   # вставить токен по запросу
```

### 2.1 Гейтед-модели — принять условия вручную (по клику на странице)

Диаризация не заработает, пока залогиненный аккаунт не примет условия
на ВСЕХ ТРЁХ страницах (это три отдельных формы):

1. https://huggingface.co/pyannote/speaker-diarization-3.1
2. https://huggingface.co/pyannote/segmentation-3.0
3. https://huggingface.co/pyannote/speaker-diarization-community-1
   (тянется библиотекой pyannote 4.x как зависимость)

Одобрение мгновенное. После этого модели скачаются сами при первом
запуске `diarize.py` (токен подхватывается из `hf auth login`).

### 2.2 Модели, которые надо скачать заранее (не гейтед)

```bash
# основная TTS: русский файнтюн F5-TTS (5000 ч русской речи)
./venv/bin/hf download Misha24-10/F5-TTS_RUSSIAN --local-dir checkpoints/f5-ru

# опционально: OpenVoice v2 (тембровый конвертер)
./venv/bin/hf download myshell-ai/OpenVoiceV2 --local-dir checkpoints/openvoice-v2
```

Whisper (mlx-community/whisper-large-v3-turbo) скачается сам при первом
запуске транскрипции.

### 2.3 Лицензии — важно

- F5-TTS_RUSSIAN: **CC-BY-NC-4.0 — только некоммерческое использование.**
- pyannote: MIT, но требует принятия условий (см. выше).
- Устаревшие модели из README (fishaudio/s1-mini, s2-pro) для продакшена
  НЕ нужны: качество s1-mini забраковано, s2-pro на Mac непрактичен
  (~48 мин на сегмент).

### 2.4 Какая модель используется в каком скрипте

| Скрипт | Модель | Примечание |
|---|---|---|
| `download.py` | — | только yt-dlp + ffmpeg |
| `transcribe.py` | `mlx-community/whisper-large-v3-turbo` | скачается сама при первом запуске |
| `diarize.py` | `pyannote/speaker-diarization-3.1` | **гейтед** — принять условия на всех трёх страницах из п. 2.1; скачается сама |
| `apply_translation.py` | — | перевод делает LLM или человек по правилам раздела 4 |
| `accent_texts.py` | RUAccent (omograph «turbo» + словарь) | скачается сама; на ~9% сегментов падает (баг ONNX) — они идут без ударений |
| `make_refs.py` | — | нарезка ffmpeg по таймингам |
| `synthesize_f5.py` | `Misha24-10/F5-TTS_RUSSIAN` (чекпоинт v2) + вокодер `charactr/vocos-mel-24khz` | **основная TTS**; чекпоинт скачать заранее (п. 2.2), CC-BY-NC; вокодер скачается сам |
| `synthesize.py` | Fish Speech `fishaudio/openaudio-s1-mini` | **устаревший** — качество забраковано, оставлен для справки |
| `openvoice_convert.py` | `myshell-ai/OpenVoiceV2` (конвертер) | опциональный тембровый проход, CPU; скачать заранее (п. 2.2) |
| `assemble.py` | — | ffmpeg/ffprobe + pydub |
| `export_transcripts.py` | — | только перекладка JSON/текста |

---

## 3. Запуск пайплайна по шагам

```bash
# 1. скачать видео + вытащить аудио → coхраняет WORKDIR=work/<id>
venv/bin/python scripts/download.py "<youtube-url>"

# 2. транскрипция с ПОСЛОВНЫМИ таймкодами → words.json
#    (Whisper, ~10 мин на 2-часовой эпизод)
venv/bin/python scripts/transcribe.py work/<id>

# 3. диаризация → diarization.json, тайм-линия спикеров
#    (~15 мин; см. п.2.1 про лицензии)
venv/bin/python scripts/diarize.py work/<id>

# 4. сегментация ПО ПРЕДЛОЖЕНИЯМ (по умолчанию): разрезы только на
#    границах предложений + пословная привязка спикеров, из words.json
#    и diarization.json → segments.json; английская стенограмма
#    сохраняется автоматически. Пересегментация с другими параметрами
#    переиспользует words.json — Whisper заново не запускается.
venv/bin/python scripts/segment_sentences.py work/<id>

# 5. ПЕРЕВОД — см. «Правила перевода» ниже. Каждый чанк применяется так:
venv/bin/python scripts/apply_translation.py work/<id> <chunk.json>

# 6. ударения (обязательно отдельным процессом — RUAccent и F5 в одном
#    процессе зависают, см. «Известные особенности окружения» в разделе 1)
venv-f5/bin/python scripts/accent_texts.py work/<id>

# 7. референсы голосов: ≤11.5 сек на спикера, текст точно совпадает с аудио
venv/bin/python scripts/make_refs.py work/<id> --max-secs 11.5 --outdir refs_f5

# 8. синтез (долго: ~19 сек/сегмент, 6–9 часов на эпизод; возобновляемый)
#    на ночь запускать через nohup, чтобы пережил закрытие терминала:
nohup venv-f5/bin/python scripts/synthesize_f5.py work/<id> > work/<id>/f5_run.log 2>&1 &

# 8б. точечный перегон сегментов после правок (id через запятую):
venv-f5/bin/python scripts/synthesize_f5.py work/<id> --only 59,341 --seed 7

# 9. (опционально) тембровый проход OpenVoice (~20 мин, CPU) → tts_ov/
/opt/miniconda3/envs/openvoice/bin/python scripts/openvoice_convert.py work/<id>

# 10. сборка дорожки и финальное видео
venv/bin/python scripts/assemble.py work/<id> --tts-dir tts_f5 --out video_ru.mp4
#    (для варианта с OpenVoice: --tts-dir tts_ov --out video_ru_ov.mp4)

# 11. перекодировка для QuickTime (исходники YouTube часто VP9)
ffmpeg -i work/<id>/video_ru.mp4 -c:v h264_videotoolbox -b:v 6000k \
  -c:a copy -movflags +faststart work/<id>/video_ru_h264.mp4
```

Стенограммы можно пересохранить вручную в любой момент:
`venv/bin/python scripts/export_transcripts.py work/<id>` — каждый запуск
пишет в новую папку `transcripts/<ммдд-ччммсс>/`, ничего не перетирая.

---

## 4. Правила перевода (критично для качества)

Перевод делается вне скриптов (LLM или человек) и подаётся чанками —
JSON-словарь «id сегмента → русский текст» в `apply_translation.py`.

1. **Тайминг-бюджет.** На сегмент — не больше ~13 русских символов на
   секунду слота (слот = от начала сегмента до начала следующего).
   Русский длиннее английского; перевод «слово в слово» гарантированно
   не влезет, и дубляж начнёт отставать от губ. Переводить надо сжато,
   как в профессиональном дубляже.
2. **Ударения в омографах ставить прямо в переводе**: `+` перед ударной
   гласной — `в д+уше` (не в душе́!), `ст+оит заниматься`, `Ванк+увер`,
   `з+амок/зам+ок` и т.п. Автоматика (RUAccent) такие случаи угадывает
   плохо, а на ~9% сегментов вообще падает (известный баг ONNX) — эти
   сегменты уходят в синтез без ударений.
3. Имена и термины — по устоявшейся русской традиции (Тед Мейман,
   Цейс, закон Ленца, быстрорежущая сталь…). Числа писать словами:
   «тысяча девятьсот шестидесятого года» — TTS читает цифры хуже.

## 5. Контроль качества

- После синтеза скрипт-проверка: прогнать несколько случайных сегментов
  обратно через Whisper и сверить текст (язык должен быть `ru`, текст —
  совпадать со сценарием).
- Слуховая вычитка: любые ошибки произношения чинятся правкой
  `accents.json` + перегоном конкретных id (`--only`, ~1 мин/сегмент),
  затем одна пересборка (шаги 9–10).

## Известные ограничения

- Короткие реплики-вставки («поддакивания») поверх чужой речи Whisper
  приклеивает к сегменту основного спикера — они звучат его голосом.
  Лечение (пословная привязка спикеров) запланировано, пока не внедрено.
- Музыка и смех, перекрытые речью, в дубляж не переносятся. Возврат
  оригинального звука в паузы выключен (`--fill-gaps` в assemble.py):
  туда протекают хвосты английских слов.
