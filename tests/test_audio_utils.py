"""Unit tests for scripts/audio_utils.py (ffmpeg-made fixtures)."""
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from audio_utils import (RMS_SILENCE, audition, decode, frame_rms,  # noqa: E402
                         merge_regions, probe_duration, scan_voiced_regions)


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
