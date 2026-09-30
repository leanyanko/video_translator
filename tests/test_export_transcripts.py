"""Unit tests for scripts/export_transcripts.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from export_transcripts import export_work, fmt_ts, write_transcript

SEGS = [
    {"id": 0, "start": 0.0, "end": 5.0, "speaker": "S0", "text": "Hello.", "text_ru": "Привет."},
    {"id": 1, "start": 5.0, "end": 9.0, "speaker": "S0", "text": "More.", "text_ru": "Ещё."},
    {"id": 2, "start": 9.0, "end": 12.0, "speaker": "S1", "text": "Reply.", "text_ru": "Ответ."},
    {"id": 3, "start": 12.0, "end": 15.0, "speaker": "S0", "text": "Back.", "text_ru": "Снова."},
]


class TestFmtTs(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(fmt_ts(0), "00:00:00")

    def test_hours_minutes_seconds(self):
        self.assertEqual(fmt_ts(3723.9), "01:02:03")


class TestWriteTranscript(unittest.TestCase):
    def test_merges_consecutive_same_speaker_turns(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t.txt"
            self.assertTrue(write_transcript(SEGS, "text", out))
            content = out.read_text()
            # 3 turns: S0 (merged 0+1), S1, S0
            self.assertEqual(content.count("S0:"), 2)
            self.assertEqual(content.count("S1:"), 1)
            self.assertIn("Hello. More.", content)
            self.assertIn("[00:00:00] S0:", content)
            self.assertIn("[00:00:09] S1:", content)

    def test_missing_text_key_returns_false(self):
        broken = [dict(s) for s in SEGS]
        del broken[1]["text_ru"]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t.txt"
            self.assertFalse(write_transcript(broken, "text_ru", out))
            self.assertFalse(out.exists())

    def test_empty_returns_false(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(write_transcript([], "text", Path(td) / "t.txt"))


class TestExportWork(unittest.TestCase):
    def _work(self, td, with_ru=True):
        work = Path(td) / "vid123"
        work.mkdir()
        (work / "segments.json").write_text(
            json.dumps({"language": "en", "segments": SEGS})
        )
        if with_ru:
            (work / "segments_ru.json").write_text(
                json.dumps({"language": "ru", "segments": SEGS})
            )
        return work

    def test_exports_both_languages_into_timestamped_dir(self):
        with tempfile.TemporaryDirectory() as td:
            work = self._work(td)
            export_work(work)
            runs = list((work / "transcripts").iterdir())
            self.assertEqual(len(runs), 1)
            names = {p.name for p in runs[0].iterdir()}
            self.assertEqual(names, {
                "transcript_en.txt", "transcript_ru.txt",
                "segments_en.json", "segments_ru.json",
            })

    def test_timestamp_format_mmdd_hhmmss(self):
        with tempfile.TemporaryDirectory() as td:
            work = self._work(td)
            export_work(work)
            name = next((work / "transcripts").iterdir()).name
            self.assertRegex(name, r"^\d{4}-\d{6}$")

    def test_ru_missing_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            work = self._work(td, with_ru=False)
            export_work(work)
            names = {p.name for p in next((work / "transcripts").iterdir()).iterdir()}
            self.assertEqual(names, {"transcript_en.txt", "segments_en.json"})

    def test_each_run_gets_its_own_directory(self):
        # contract: a new export never touches an earlier run's folder
        with tempfile.TemporaryDirectory() as td:
            work = self._work(td)
            export_work(work)
            old_dir = next((work / "transcripts").iterdir())
            renamed = old_dir.rename(old_dir.parent / "0101-000000")
            export_work(work)
            runs = {p.name for p in (work / "transcripts").iterdir()}
            self.assertIn("0101-000000", runs)
            self.assertEqual(len(runs), 2)
            self.assertEqual(
                {p.name for p in renamed.iterdir()},
                {"transcript_en.txt", "transcript_ru.txt",
                 "segments_en.json", "segments_ru.json"},
            )


if __name__ == "__main__":
    unittest.main()
