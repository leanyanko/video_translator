"""Unit tests for scripts/audio_utils.py (ffmpeg-made fixtures)."""
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from audio_utils import (RMS_SILENCE, audition, decode, frame_rms,  # noqa: E402
                         join_trimmed, merge_regions, probe_duration,
                         scan_voiced_regions, speech_bounds)


class SpeechBoundsTest(unittest.TestCase):
    """The trim rule synthesize_f5 applies per piece (pacing): F5's
    painted lead/tail silence must go, 50 ms of air stays."""

    def test_lead_and_tail_silence_are_cut(self):
        sr = 24000
        tone = 0.3 * np.sin(np.linspace(0, 2 * np.pi * 220 * 1.0, sr))
        clip = np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr)]).astype(np.float32)
        lo, hi = speech_bounds(clip, sr)
        self.assertAlmostEqual(lo / sr, 0.5 - 0.05, delta=0.03)
        self.assertAlmostEqual(hi / sr, 1.5 + 0.05, delta=0.03)

    def test_quiet_breath_below_relative_threshold_is_not_onset(self):
        sr = 24000
        breath = 0.012 * np.sin(np.linspace(0, 2 * np.pi * 220 * 0.5, sr // 2))
        tone = 0.5 * np.sin(np.linspace(0, 2 * np.pi * 220 * 1.0, sr))
        clip = np.concatenate([breath, tone]).astype(np.float32)
        lo, _ = speech_bounds(clip, sr)
        self.assertGreater(lo / sr, 0.4)  # starts at the tone, not the breath

    def test_all_silent(self):
        self.assertIsNone(speech_bounds(np.zeros(24000, dtype=np.float32), 24000))

    def test_join_trimmed_keeps_speech_plus_gaps(self):
        # the invariant pacing rests on: pieces' painted silence goes,
        # the speech that remains = sum of pieces + the gaps between them
        sr = 24000
        def piece(speech_secs, lead=0.5, tail=0.6):
            tone = 0.3 * np.sin(np.linspace(0, 2 * np.pi * 220 * speech_secs,
                                            int(speech_secs * sr)))
            return np.concatenate([np.zeros(int(lead * sr)), tone,
                                   np.zeros(int(tail * sr))]).astype(np.float32)
        out = join_trimmed([piece(2.0), piece(1.0), piece(0.5)], sr, 0.15)
        expected = 2.0 + 1.0 + 0.5 + 2 * 0.15 + 3 * 2 * 0.05  # + 50 ms air per edge
        self.assertAlmostEqual(len(out) / sr, expected, delta=0.06)

    def test_join_trimmed_empty(self):
        self.assertEqual(len(join_trimmed([], 24000, 0.15)), 0)


class TestWithFixture(unittest.TestCase):
    """tone 1 s | silence 1 s | tone 1 s | silence 1 s  (16 kHz mono)."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.wav = Path(cls._td.name) / "fx.wav"
        sr = 16000
        t = np.arange(sr) / sr
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        sil = np.zeros(sr)
        pcm = (np.concatenate([tone, sil, tone, sil]) * 32767).astype("<i2")
        with wave.open(str(cls.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_probe_duration(self):
        self.assertAlmostEqual(probe_duration(self.wav), 4.0, delta=0.05)

    def test_decode_full_and_window(self):
        a = decode(self.wav, 16000)
        self.assertEqual(a.dtype, np.float32)
        self.assertAlmostEqual(len(a) / 16000, 4.0, delta=0.05)
        w = decode(self.wav, 16000, start=1.0, dur=1.0)
        self.assertAlmostEqual(len(w) / 16000, 1.0, delta=0.05)
        self.assertLess(np.abs(w).max(), 1e-3)  # the silent second
        w2 = decode(self.wav, 8000, start=2.0, end=3.0)
        self.assertAlmostEqual(len(w2) / 8000, 1.0, delta=0.05)
        self.assertGreater(np.abs(w2).max(), 0.3)  # the second tone

    def test_scan_voiced_regions_finds_the_two_tones(self):
        regions = scan_voiced_regions(self.wav, 0.25, RMS_SILENCE)
        self.assertEqual(len(regions), 2)
        (a0, b0), (a1, b1) = regions
        self.assertAlmostEqual(a0, 0.0, delta=0.3)
        self.assertAlmostEqual(b0, 1.0, delta=0.3)
        self.assertAlmostEqual(a1, 2.0, delta=0.3)
        self.assertAlmostEqual(b1, 3.0, delta=0.3)


class TestPureFunctions(unittest.TestCase):
    def test_frame_rms(self):
        a = np.concatenate([np.zeros(100), np.ones(100)]).astype(np.float32)
        self.assertTrue(np.allclose(frame_rms(a, 100), [0.0, 1.0]))
        self.assertEqual(len(frame_rms(a, 150)), 1)  # tail dropped
        self.assertEqual(len(frame_rms(np.zeros(10), 100)), 0)

    def test_merge_regions_gap_and_span(self):
        r = [(0, 1), (1.5, 2), (2.2, 9), (9.3, 12)]
        self.assertEqual(merge_regions(r, 0.6), [(0, 12)])                 # every gap <= 0.6
        self.assertEqual(merge_regions(r, 0.4), [(0, 1), (1.5, 12)])       # first gap is 0.5
        self.assertEqual(merge_regions(r, 0.6, max_span=5.0),
                         [(0, 2), (2.2, 9), (9.3, 12)])                   # span cap breaks the chain
        self.assertEqual(merge_regions([], 1.0), [])

    def test_audition_head_tail_voiced(self):
        sr = 16000
        rng = np.random.default_rng(0)
        voice = (rng.standard_normal(2 * sr) * 0.2).astype(np.float32)
        a = np.concatenate([np.zeros(int(0.5 * sr)), voice, np.zeros(sr)]).astype(np.float32)
        head, tail, voiced = audition(a, sr)
        self.assertAlmostEqual(head, 0.5, delta=0.06)
        self.assertAlmostEqual(tail, 1.0, delta=0.06)
        self.assertAlmostEqual(voiced, 2 / 3.5, delta=0.03)

    def test_audition_all_silent_and_too_short(self):
        self.assertEqual(audition(np.zeros(16000, np.float32), 16000), (1.0, 1.0, 0.0))
        self.assertEqual(audition(np.zeros(10, np.float32), 16000)[2], 0.0)


if __name__ == "__main__":
    unittest.main()
