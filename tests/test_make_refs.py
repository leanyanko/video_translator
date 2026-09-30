"""Tests for scripts/make_refs.py (subprocess; uses a real ffmpeg-made wav).

Contracts covered:
  - one reference per speaker, wav + matching txt
  - reference audio never exceeds --max-secs
  - reference text is exactly the segments covered by the clip (audio and
    text describe the same span — the F5 desync guard)
  - the longest continuous run wins
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_refs.py"

SEGMENTS = {
    "language": "en",
    "speakers": ["S0", "S1"],
    "segments": [
        # S0: two runs — a short one (2s) and a longer continuous one (8s)
        {"id": 0, "start": 0.0, "end": 2.0, "speaker": "S0", "text": "short run."},
        {"id": 1, "start": 5.0, "end": 9.0, "speaker": "S0", "text": "long part one."},
        {"id": 2, "start": 9.2, "end": 13.0, "speaker": "S0", "text": "long part two."},
        # S1: a single 3s segment
        {"id": 3, "start": 15.0, "end": 18.0, "speaker": "S1", "text": "other voice."},
    ],
}


def wav_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], text=True)
    return float(out.strip())


class TestMakeRefs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.work = Path(cls._td.name) / "vid"
        cls.work.mkdir()
        (cls.work / "segments.json").write_text(json.dumps(SEGMENTS))
        subprocess.check_call(
            ["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=20",
             "-ac", "1", "-ar", "44100", str(cls.work / "audio.wav")])
        r = subprocess.run(
            [sys.executable, str(SCRIPT), str(cls.work),
             "--max-secs", "11.5", "--outdir", "refs_f5"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        cls.refs = cls.work / "refs_f5"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_one_ref_pair_per_speaker(self):
        names = sorted(p.name for p in self.refs.iterdir())
        self.assertEqual(names, ["S0.txt", "S0.wav", "S1.txt", "S1.wav"])

    def test_duration_never_exceeds_cap(self):
        for spk in ("S0", "S1"):
            self.assertLessEqual(wav_duration(self.refs / f"{spk}.wav"), 11.5 + 0.1)

    def test_longest_continuous_run_is_chosen_and_text_matches_span(self):
        # S0's best run is segments 1+2 (5.0–13.0, 8s) — not the 2s opener
        self.assertAlmostEqual(wav_duration(self.refs / "S0.wav"), 8.0, delta=0.2)
        self.assertEqual(
            (self.refs / "S0.txt").read_text().strip(),
            "long part one. long part two.",
        )

    def test_single_segment_speaker_text_matches(self):
        self.assertEqual((self.refs / "S1.txt").read_text().strip(), "other voice.")
        self.assertAlmostEqual(wav_duration(self.refs / "S1.wav"), 3.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
