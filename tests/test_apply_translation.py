"""Tests for scripts/apply_translation.py (run as a subprocess, as in real use).

Contracts covered:
  - chunked application merges into segments_ru.json
  - unknown segment ids are rejected with a non-zero exit
  - hand edits to existing RU segments (times, texts) survive re-application
  - hand-added split segments (ids absent from segments.json) survive
  - transcripts are exported on every application
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "apply_translation.py"

BASE = {
    "language": "en",
    "speakers": ["S0", "S1"],
    "segments": [
        {"id": 0, "start": 0.0, "end": 4.0, "speaker": "S0", "text": "One."},
        {"id": 1, "start": 4.0, "end": 8.0, "speaker": "S1", "text": "Two."},
        {"id": 2, "start": 8.0, "end": 12.0, "speaker": "S0", "text": "Three."},
    ],
}


def run(work: Path, chunk: dict):
    chunk_file = work / "chunk.json"
    chunk_file.write_text(json.dumps(chunk, ensure_ascii=False))
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(work), str(chunk_file)],
        capture_output=True, text=True,
    )


def read_ru(work: Path):
    return {s["id"]: s for s in json.loads((work / "segments_ru.json").read_text())["segments"]}


class TestApplyTranslation(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.work = Path(self._td.name) / "vid"
        self.work.mkdir()
        (self.work / "segments.json").write_text(json.dumps(BASE, ensure_ascii=False))

    def tearDown(self):
        self._td.cleanup()

    def test_chunked_application_merges(self):
        r1 = run(self.work, {"0": "Раз."})
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertEqual(set(read_ru(self.work)), {0})

        r2 = run(self.work, {"1": "Два.", "2": "Три."})
        self.assertEqual(r2.returncode, 0, r2.stderr)
        ru = read_ru(self.work)
        self.assertEqual(
            {i: s["text_ru"] for i, s in ru.items()},
            {0: "Раз.", 1: "Два.", 2: "Три."},
        )
        self.assertIn("complete", r2.stdout)

    def test_retranslation_overwrites_text(self):
        run(self.work, {"0": "Черновик."})
        run(self.work, {"0": "Чистовик."})
        self.assertEqual(read_ru(self.work)[0]["text_ru"], "Чистовик.")

    def test_unknown_id_is_rejected(self):
        r = run(self.work, {"99": "Лишний."})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("99", r.stderr + r.stdout)
        self.assertFalse((self.work / "segments_ru.json").exists())

    def test_hand_edits_survive_reapplication(self):
        run(self.work, {"0": "Раз.", "1": "Два."})
        ru_path = self.work / "segments_ru.json"
        data = json.loads(ru_path.read_text())
        for s in data["segments"]:
            if s["id"] == 0:
                s["end"] = 2.5  # hand-trimmed boundary
                s["text"] = "One (trimmed)."
        ru_path.write_text(json.dumps(data, ensure_ascii=False))

        run(self.work, {"2": "Три."})
        ru = read_ru(self.work)
        self.assertEqual(ru[0]["end"], 2.5)
        self.assertEqual(ru[0]["text"], "One (trimmed).")

    def test_hand_added_split_segment_survives(self):
        run(self.work, {"0": "Раз.", "1": "Два.", "2": "Три."})
        ru_path = self.work / "segments_ru.json"
        data = json.loads(ru_path.read_text())
        data["segments"].append({
            "id": 500, "start": 2.5, "end": 4.0, "speaker": "S1",
            "text": "split tail", "text_ru": "хвост",
        })
        ru_path.write_text(json.dumps(data, ensure_ascii=False))

        run(self.work, {"1": "Два!"})
        ru = read_ru(self.work)
        self.assertIn(500, ru)
        self.assertEqual(ru[500]["text_ru"], "хвост")
        # and segments stay sorted by start time
        starts = [s["start"] for s in json.loads(ru_path.read_text())["segments"]]
        self.assertEqual(starts, sorted(starts))

    def test_transcripts_exported_on_apply(self):
        run(self.work, {"0": "Раз.", "1": "Два.", "2": "Три."})
        runs = list((self.work / "transcripts").iterdir())
        self.assertGreaterEqual(len(runs), 1)
        self.assertTrue(any((r / "transcript_ru.txt").exists() for r in runs))


if __name__ == "__main__":
    unittest.main()
