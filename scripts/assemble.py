#!/usr/bin/env python
"""Lay the synthesized Russian segments onto the video's timeline and mux.

Model: none (ffprobe/ffmpeg + pydub audio arithmetic).

Usage: venv/bin/python scripts/assemble.py work/<id>
Each clip is placed at its segment's start time. If a clip is longer than
the slot before the next DUBBED line or the speaker's own type:"orig"
paste (other speakers' pastes and overlap lines are laid on top and do
not end a slot), it is sped up with ffmpeg atempo
(capped at MAX_TEMPO so it still sounds natural; beyond that it simply
overflows into the following pause). Produces dubbed.wav and video_ru.mp4.
"""
import json
import subprocess
import sys
from pathlib import Path

from cli_args import flag, opt

from pydub import AudioSegment

from audio_utils import probe_duration

MAX_TEMPO = 1.35  # max speed-up before we allow overflow instead
SR = 44100

work = Path(sys.argv[1])
tts_name = opt("--tts-dir", "tts")
out_name = opt("--out", "video_ru.mp4")
segments = json.loads((work / "segments_ru.json").read_text())["segments"]
# "echo": a listener repeating the speaker's words as he says them
# (scripts/repeat_audit.py --mark-echo) — dubbed, it would be heard twice
segments = [s for s in segments if not s.get("echo")]
# "overlap": true segments are deliberately CONCURRENT speech (speaker B
# talking over speaker A): they are excluded from the sequential timeline
# (no slot math, no cursor) and overlaid afterwards at their own start
overlap_segments = [s for s in segments if s.get("overlap")]
segments = [s for s in segments if not s.get("overlap")]
tts_dir = work / tts_name
tmp = work / "tmp"
tmp.mkdir(exist_ok=True)

# partial-build gate (TODO #4): when synthesis failed wholesale, the old
# behaviour printed a WARNING per clip and still shipped a "finished"
# video of music and pastes (10-09: Chatterbox died on a dependency,
# tts_cbx stayed empty, assemble reported DONE on 0/22 clips). A few
# missing clips are survivable noise; a large share is a broken run.
from manifest_bridge import missing_clips, slot_ends  # noqa: E402

voiced = [s for s in segments + overlap_segments if s.get("type") != "orig"]
# slot end per line = the next voiced line or own-speaker paste (same
# rule synthesize paces against); other speakers' pastes are laid on
# top, never slot-bearing
slot_end = slot_ends(segments)
_missing = missing_clips(voiced, tts_dir)
if _missing and voiced and not flag("--force-partial"):
    share = len(_missing) / len(voiced)
    if share > 0.10:
        sys.exit(f"ABORT: {len(_missing)}/{len(voiced)} voiced segments have "
                 f"no usable clip in {tts_dir.name}/ ({share:.0%} > 10%) — "
                 f"ids {sorted(_missing)[:20]}"
                 f"{'...' if len(_missing) > 20 else ''}. Run synthesize "
                 "first, or pass --force-partial to assemble anyway.")
if _missing:
    print(f"NOTE: {len(_missing)}/{len(voiced)} clips missing/empty — "
          "assembling without them"
          + (" (--force-partial)" if flag("--force-partial") else ""))


video_dur = probe_duration(work / "video.mp4")
canvas = AudioSegment.silent(duration=int(video_dur * 1000), frame_rate=SR).set_channels(2)

import numpy as np
import soundfile as sf_

TRIM_THRESHOLD = 0.01  # amplitude fraction (~-40 dBFS): below = silence
TRIM_MARGIN = 0.05  # keep 50 ms of natural air around the speech


def trim_silence(path: Path, out_path: Path) -> Path | None:
    """Cut leading/trailing silence so speech starts at sample ~0.

    The TTS model paints silence around the speech (imitating the
    reference's pauses); alignment must use where speech actually starts.
    Returns None for an all-silent clip.
    """
    data, sr = sf_.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    # threshold relative to the clip's own loudness: F5 also paints
    # quiet lead/tail mumble and breaths ABOVE the absolute floor, which
    # then delays the line by their length (qc_tts.py's LEAD flag). RMS
    # over 20 ms frames, not raw samples — a single click must not count
    # as speech onset.
    n = len(data) // (sr // 50)
    frame = sr // 50
    rms = np.sqrt((data[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    thr = max(TRIM_THRESHOLD, 0.1 * float(np.percentile(rms, 99)))
    voiced_f = np.where(rms > thr)[0]
    if len(voiced_f) == 0:
        return None
    lo = max(0, voiced_f[0] * frame - int(TRIM_MARGIN * sr))
    hi = min(len(data), (voiced_f[-1] + 1) * frame + int(TRIM_MARGIN * sr))
    sf_.write(out_path, data[lo:hi], sr)
    return out_path


original_audio = None  # loaded lazily for type:"orig" segments

placed = []  # (start_sec, end_sec) of every dubbed clip, for gap-filling
cursor = 0.0  # end time of the previously placed clip
for i, seg in enumerate(segments):
    if seg.get("type") == "orig":
        # restore the ORIGINAL audio span (laughter, applause, jingle);
        # never synthesized, never time-fitted, never moves the cursor
        if "file" in seg:
            # external source (e.g. a separated laugh stem); the whole
            # file is placed at seg["start"]
            piece = (AudioSegment.from_wav(work / seg["file"])
                     .set_frame_rate(SR).set_channels(1))
        else:
            if original_audio is None:
                original_audio = (AudioSegment.from_wav(work / "audio.wav")
                                  .set_frame_rate(SR).set_channels(1))
            piece = original_audio[int(seg["start"] * 1000): int(seg["end"] * 1000)]
        if "gain_db" in seg:
            piece = piece + seg["gain_db"]
        if seg.get("label") == "laugh-micro":
            piece = piece + 6  # dB; breaths are quiet, the dub would mask them
        fade = min(80, len(piece) // 2)
        canvas = canvas.overlay(piece.fade_in(fade).fade_out(fade),
                                position=int(seg["start"] * 1000))
        continue
    raw_path = tts_dir / f"seg_{seg['id']:04d}.wav"
    if not raw_path.exists():
        print(f"WARNING: missing {raw_path.name}, skipping")
        continue
    clip_path = trim_silence(raw_path, tmp / f"trim_{raw_path.name}")
    if clip_path is None:
        print(f"WARNING: {raw_path.name} is silent, skipping")
        continue

    # lip sync beats gapless audio: every line starts at its speaker's
    # own timestamp, tolerating at most 1s delay from an overrunning
    # predecessor (the overlap tail is quiet crosstalk, lag is worse)
    place_at = max(seg["start"], min(cursor, seg["start"] + 1.0))
    next_start = slot_end.get(seg["id"])
    if next_start is None:
        next_start = video_dur
    slot = max(next_start - place_at, 0.5)

    dur = probe_duration(clip_path)
    tempo = min(max(dur / slot, 1.0), MAX_TEMPO)
    if tempo > 1.02:
        # distinct name: clip_path may itself live in tmp/ (trimmed),
        # and ffmpeg must never write the file it is reading
        fitted = tmp / f"fit_{raw_path.name}"
        subprocess.check_call([
            "ffmpeg", "-y", "-v", "quiet", "-i", str(clip_path),
            "-filter:a", f"atempo={tempo:.4f}", "-ar", str(SR), str(fitted),
        ])
        clip_path, dur = fitted, dur / tempo

    clip = AudioSegment.from_wav(clip_path).set_frame_rate(SR).set_channels(1)
    if "pan" in seg:
        clip = clip.set_channels(2).pan(seg["pan"])
    canvas = canvas.overlay(clip, position=int(place_at * 1000))
    placed.append((place_at, place_at + dur))
    cursor = place_at + dur

# optionally fill gaps between dubbed lines with the ORIGINAL audio.
# Off by default: whisper's segment edges leak English word tails and
# breaths into the gaps, which sounds worse than silence. (A future
# laughter-only variant would need a laughter detector to pick regions.)
if flag("--fill-gaps"):
    MIN_GAP = 0.4  # sec; ignore hairline gaps
    FADE_MS = 120
    original = AudioSegment.from_wav(work / "audio.wav").set_frame_rate(SR).set_channels(1)
    placed.sort()
    gaps, prev_end = [], 0.0
    for s, e in placed:
        if s - prev_end >= MIN_GAP:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if video_dur - prev_end >= MIN_GAP:
        gaps.append((prev_end, video_dur))
    for gs, ge in gaps:
        piece = original[int(gs * 1000) : int(ge * 1000)]
        fade = min(FADE_MS, len(piece) // 2)
        canvas = canvas.overlay(piece.fade_in(fade).fade_out(fade), position=int(gs * 1000))
    print(f"gap-filled {len(gaps)} regions with original audio "
          f"({sum(ge-gs for gs, ge in gaps):.0f}s total)")

# overlay the concurrent-speech segments (speaker-over-speaker)
for seg in overlap_segments:
    raw_path = tts_dir / f"seg_{seg['id']:04d}.wav"
    if not raw_path.exists():
        print(f"WARNING: overlap seg {seg['id']} missing, skipping")
        continue
    clip_path = trim_silence(raw_path, tmp / f"trim_{raw_path.name}")
    if clip_path is None:
        continue
    span = max(seg["end"] - seg["start"], 0.5)
    dur = probe_duration(clip_path)
    tempo = min(max(dur / span, 1.0), MAX_TEMPO)
    if tempo > 1.02:
        fitted = tmp / f"fit_{raw_path.name}"
        subprocess.check_call([
            "ffmpeg", "-y", "-v", "quiet", "-i", str(clip_path),
            "-filter:a", f"atempo={tempo:.4f}", "-ar", str(SR), str(fitted)])
        clip_path = fitted
    clip = AudioSegment.from_wav(clip_path).set_frame_rate(SR).set_channels(1)
    if "gain_db" in seg:
        clip = clip + seg["gain_db"]
    if "pan" in seg:
        clip = clip.set_channels(2).pan(seg["pan"])
    canvas = canvas.overlay(clip, position=int(seg["start"] * 1000))
if overlap_segments:
    print(f"overlaid {len(overlap_segments)} concurrent-speech segments")

# optional full-length music/SFX bed (multitrack mode): the editor's
# "Music and SFX" stem is timeline-aligned and already mixed at
# speech-relative levels, so it is overlaid as-is; --music-gain adjusts by ear
if flag("--music"):
    music_path = Path(opt("--music"))
    music_gain = opt("--music-gain", 0.0, float)
    bed = AudioSegment.from_file(music_path).set_frame_rate(SR).set_channels(2)
    if music_gain:
        bed = bed + music_gain
    if flag("--mute-lone-sfx"):
        # Duck LONE FOLEY in the bed (page flips poking out in dub
        # pauses). The decision lives in audio_utils.lone_sfx_gain —
        # "pokes out above the bed's own level next to speech", NOT
        # "quiet": a level-only rule pumped underscore music by 40 dB
        # in every pause (review 10-10). Gains are ramped (20 ms) and
        # the bed/canvas are processed in CHUNKS so a 2-hour episode
        # never holds two float32 copies in memory.
        from audio_utils import lone_sfx_gain, smooth_gain
        FR_S = 0.1
        fr = int(SR * FR_S)
        full = 1 << (8 * bed.sample_width - 1)

        def frames_rms_chunked(seg, m):
            out = np.empty(m, np.float32)
            step = 600  # frames per chunk (~60 s)
            raw = seg.raw_data
            ch, wid = seg.channels, seg.sample_width
            dt = {1: np.int8, 2: np.int16, 4: np.int32}[wid]
            # normalization is PER SEGMENT: the bed may be 32-bit FLAC
            # while the canvas is 16-bit — one shared divisor crushed
            # the canvas to ~1e-5 and the gate saw eternal silence
            seg_full = 1 << (8 * wid - 1)
            for i in range(0, m, step):
                j = min(i + step, m)
                buf = np.frombuffer(
                    raw[i * fr * ch * wid: j * fr * ch * wid], dtype=dt)
                x = buf.astype(np.float32).reshape(-1, ch).mean(axis=1) / seg_full
                out[i:j] = np.sqrt((x.reshape(j - i, fr) ** 2).mean(axis=1))
            return out

        # pydub len() is MILLISECONDS — frame counts must come from the
        # raw byte buffers (this very bug made the gate process 3.6 s
        # of a 161 s clip and duck nothing)
        def n_frames(seg):
            return len(seg.raw_data) // (fr * seg.channels * seg.sample_width)

        n = n_frames(bed)
        m = min(n, n_frames(canvas))
        bed_rms = frames_rms_chunked(bed, n)
        dub_loud = np.zeros(n, bool)
        dub_loud[:m] = frames_rms_chunked(canvas, m) > 10 ** (-38 / 20)
        if flag("--sfx-debug"):
            print(f"  sfxdbg canvas: ch={canvas.channels} w={canvas.sample_width} "
                  f"fr={canvas.frame_rate} max={canvas.max} n={n} m={m}")
            print(f"  sfxdbg bed   : ch={bed.channels} w={bed.sample_width} "
                  f"fr={bed.frame_rate}")
            cvr = frames_rms_chunked(canvas, m)
            for i in range(40, min(70, m)):
                print(f"  sfxdbg t={i/10:.1f} canvas={cvr[i]:.5f} "
                      f"bed={bed_rms[i]:.5f} loud={int(dub_loud[i])}")
        gain_f = lone_sfx_gain(bed_rms, dub_loud)
        if (gain_f < 1.0).any():
            dt = {1: np.int8, 2: np.int16, 4: np.int32}[bed.sample_width]
            raw = bed.raw_data
            ch, wid = bed.channels, bed.sample_width
            out = bytearray(raw)
            ramp = int(0.02 * SR)
            gains = smooth_gain(gain_f, fr, ramp)
            step = 600
            for i in range(0, n, step):
                j = min(i + step, n)
                sl = slice(i * fr * ch * wid, j * fr * ch * wid)
                buf = np.frombuffer(raw[sl], dtype=dt).astype(np.float32)
                g = np.repeat(gains[i * fr: j * fr], ch)
                buf *= g[: len(buf)]
                out[sl] = np.clip(buf, -full, full - 1).astype(dt).tobytes()
            bed = bed._spawn(bytes(out))
        ducked = float((gain_f < 1.0).sum()) * FR_S
        print(f"lone-SFX gate: ducked {ducked:.1f}s of poking foley "
              "in dub pauses")
    canvas = canvas.overlay(bed, position=0)
    print(f"mixed music bed {music_path.name} ({music_gain:+.1f} dB)")

dubbed = work / "dubbed.wav"
canvas.export(dubbed, format="wav")

out_video = work / out_name
subprocess.check_call([
    "ffmpeg", "-y", "-i", str(work / "video.mp4"), "-i", str(dubbed),
    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    "-shortest", str(out_video),
])
print(f"DONE: {out_video}")
