"""The doctor (scripts/check_pipeline.py) decides whether a dub ships; until
now it was tested only by running it. Each test builds a minimal MIX-mode
work dir that passes clean, breaks exactly one invariant, and asserts the
doctor's verdict line. Whisper is never called: the reference check reads
its <spk>.check.json cache, written here with matching mtimes."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_pipeline.py"
sys.path.insert(0, str(ROOT / "scripts"))
from manifest_bridge import ref_hasher, seg_hash  # noqa: E402

SR = 16000


def tone_wav(path: Path, secs: float, freq: float = 220.0, amp: float = 0.3) -> None:
    t = np.arange(int(secs * SR)) / SR
    pcm = (amp * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def build_work(root: Path) -> Path:
    """A clean 2-speaker mix-mode dir: 3 lines, consistent manifest."""
    work = root / "vid"
    (work / "tts_f5").mkdir(parents=True)
    (work / "refs_f5").mkdir()
    segs = [
        {"id": 0, "start": 0.0, "end": 4.0, "speaker": "S0", "text": "One two three four five."},
        {"id": 1, "start": 4.5, "end": 8.0, "speaker": "S1", "text": "Six seven eight nine ten."},
        {"id": 2, "start": 8.5, "end": 12.0, "speaker": "S0", "text": "Eleven twelve thirteen."},
    ]
    ru_text = {0: "Раз, два, три, четыре, пять, шесть, семь, восемь.",
               1: "Шесть, семь, восемь, девять, десять, одиннадцать.",
               2: "Одиннадцать, двенадцать, тринадцать, четырнадцать."}
    acc = {str(i): t.replace("Раз", "Р+аз").replace("Шесть", "Ш+есть").replace("Одиннадцать", "Од+иннадцать")
           for i, t in ru_text.items()}
    words = [{"start": s["start"] + 0.2 * k, "end": s["start"] + 0.2 * k + 0.15, "word": w, "prob": 0.9,
              "speaker": s["speaker"]} for s in segs for k, w in enumerate(s["text"].split())]
    (work / "words.json").write_text(json.dumps({"language": "en", "words": words}))
    (work / "segments.json").write_text(json.dumps({"language": "en", "speakers": ["S0", "S1"], "segments": segs}))
    ru = [dict(s, text_ru=ru_text[s["id"]]) for s in segs]
    (work / "segments_ru.json").write_text(json.dumps({"segments": ru}, ensure_ascii=False))
    (work / "accents.json").write_text(json.dumps(acc, ensure_ascii=False))
    for spk, txt in (("S0", "a reference line for the first voice"), ("S1", "a reference line for the second voice")):
        wav = work / "refs_f5" / f"{spk}.wav"
        tone_wav(wav, 6.0)
        tp = work / "refs_f5" / f"{spk}.txt"
        tp.write_text(txt + "\n")
        # the doctor's reference check reads this cache when the mtimes match
        (work / "refs_f5" / f"{spk}.check.json").write_text(json.dumps(
            {"match": 1.0, "tail": 0.1, "chars_per_sec": 14.0, "ok": True, "heard": txt,
             "stamp": [wav.stat().st_mtime, tp.stat().st_mtime]}))
    rs = ref_hasher(work / "refs_f5")
    manifest = {}
    for s in ru:
        tone_wav(work / "tts_f5" / f"seg_{s['id']:04d}.wav", 3.0)
        manifest[str(s["id"])] = seg_hash(s, acc, rs)
    (work / "tts_f5" / "manifest.json").write_text(json.dumps(manifest))
    return work


def doctor(work: Path, *flags) -> tuple:
    r = subprocess.run([sys.executable, str(SCRIPT), str(work), *flags],
                       capture_output=True, text=True, cwd=ROOT)
    lines = (r.stdout + r.stderr).splitlines()
    return r.returncode, [l for l in lines if l.startswith("FAIL")], [l for l in lines if l.startswith("WARN")]


class DoctorTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.work = build_work(Path(self._td.name))

    def tearDown(self):
        self._td.cleanup()

    def test_clean_dir_passes(self):
        rc, fails, _ = doctor(self.work)
        self.assertEqual((rc, fails), (0, []))

    def test_stale_clip_fails_before_assembly_warns_before_synthesis(self):
        acc = json.loads((self.work / "accents.json").read_text())
        ru = json.loads((self.work / "segments_ru.json").read_text())
        ru["segments"][1]["text_ru"] = "Совсем другой текст этой строки, да."
        acc["1"] = "Совс+ем друг+ой т+екст +этой строк+и, д+а."
        (self.work / "segments_ru.json").write_text(json.dumps(ru, ensure_ascii=False))
        (self.work / "accents.json").write_text(json.dumps(acc, ensure_ascii=False))
        rc, fails, _ = doctor(self.work)
        self.assertEqual(rc, 1)
        self.assertTrue(any("[tts]" in f and "STALE" in f for f in fails), fails)
        rc, fails, warns = doctor(self.work, "--pre-synthesis")
        self.assertEqual((rc, fails), (0, []))
        self.assertTrue(any("STALE" in w for w in warns))

    def test_missing_clips_beyond_ten_percent_fail(self):
        (self.work / "tts_f5" / "seg_0001.wav").unlink()
        rc, fails, _ = doctor(self.work)
        self.assertEqual(rc, 1)
        self.assertTrue(any("[tts]" in f and "missing" in f for f in fails), fails)

    def test_qc_flag_after_reroll_fails_per_clip_and_accept_downgrades(self):
        qc = {"0": {"flags": [], "sim": 1.0, "speaker": "S0", "heard": "x"},
              "1": {"flags": ["MISMATCH"], "sim": 0.4, "speaker": "S1", "heard": "garbled"},
              "2": {"flags": ["TAIL"], "sim": 1.0, "speaker": "S0", "heard": "y"}}
        (self.work / "qc_tts_f5.json").write_text(json.dumps(qc))
        # before the re-roll ran: a warning only
        rc, fails, warns = doctor(self.work)
        self.assertEqual((rc, fails), (0, []))
        self.assertTrue(any("flagged by QC" in w for w in warns))
        (self.work / "reroll_report.json").write_text(json.dumps({"clips": {}}))
        rc, fails, _ = doctor(self.work)
        self.assertEqual(rc, 1)
        self.assertEqual(len([f for f in fails if "flagged by QC" in f]), 2)  # one line per clip
        (self.work / "doctor_accept.json").write_text(json.dumps(
            [{"stage": "tts", "contains": "seg 1 MISMATCH", "reason": "test"}]))
        rc, fails, warns = doctor(self.work)
        self.assertEqual(len([f for f in fails if "flagged by QC" in f]), 1)
        self.assertTrue(any("ACCEPTED: test" in w for w in warns))

    def test_untranscribed_speech_fails_unless_covered_or_bleed(self):
        report = {"speech_checked": True, "speakers_checked": ["S0", "S1"], "pasted_with_words": [],
                  "untranscribed_speech": [
                      {"speaker": "S1", "start": 13.0, "end": 14.5, "heard": "and that is true", "words": 4, "logprob": -0.5, "bleed": False},
                      {"speaker": "S1", "start": 15.0, "end": 15.6, "heard": "yeah", "words": 1, "logprob": -0.5, "bleed": False},
                      {"speaker": "S0", "start": 4.6, "end": 7.9, "heard": "six seven eight nine", "words": 4, "logprob": -0.5, "bleed": True},
                      {"speaker": "S1", "start": 16.0, "end": 17.0, "heard": "noise words here", "words": 3, "logprob": -4.0, "bleed": False},
                  ]}
        (self.work / "harvest_report.json").write_text(json.dumps(report))
        rc, fails, warns = doctor(self.work)
        self.assertEqual(rc, 1)
        self.assertEqual(len([f for f in fails if "untranscribed speech" in f]), 1)  # only the clear phrase
        self.assertTrue(any("bleed" in w for w in warns))
        self.assertTrue(any("unclear" in w for w in warns))
        # covered by a (hand-added) speech segment of the same speaker -> resolved
        ru = json.loads((self.work / "segments_ru.json").read_text())
        ru["segments"].append({"id": 3, "start": 13.0, "end": 14.5, "speaker": "S1",
                               "text": "and that is true", "text_ru": "И это правда."})
        (self.work / "segments_ru.json").write_text(json.dumps(ru, ensure_ascii=False))
        acc = json.loads((self.work / "accents.json").read_text())
        acc["3"] = "И +это пр+авда."
        (self.work / "accents.json").write_text(json.dumps(acc, ensure_ascii=False))
        rc, fails, _ = doctor(self.work, "--pre-synthesis")
        self.assertEqual([f for f in fails if "untranscribed speech" in f], [])

    def test_long_auto_paste_fails(self):
        ru = json.loads((self.work / "segments_ru.json").read_text())
        (self.work / "laughs").mkdir()
        tone_wav(self.work / "laughs" / "S1_000130.wav", 0.5)
        ru["segments"].append({"id": 9500, "start": 13.0, "end": 19.0, "speaker": "S1", "type": "orig",
                               "auto_track": True, "file": "laughs/S1_000130.wav", "label": "nonspeech"})
        (self.work / "segments_ru.json").write_text(json.dumps(ru, ensure_ascii=False))
        rc, fails, _ = doctor(self.work)
        self.assertTrue(any("[orig]" in f and "auto paste" in f for f in fails), fails)

    def test_reference_text_mismatch_fails(self):
        chk = self.work / "refs_f5" / "S1.check.json"
        v = json.loads(chk.read_text())
        v.update(match=0.9, ok=False, heard="a reference line for the second voice yeah")
        chk.write_text(json.dumps(v))
        rc, fails, _ = doctor(self.work)
        self.assertTrue(any("[refs]" in f and "does not match the audio" in f for f in fails), fails)

    def test_stale_acceptance_is_reported(self):
        (self.work / "doctor_accept.json").write_text(json.dumps(
            [{"stage": "tts", "contains": "seg 7 MISMATCH", "reason": "old"}]))
        rc, fails, warns = doctor(self.work)
        self.assertEqual((rc, fails), (0, []))
        self.assertTrue(any("matched NOTHING" in w for w in warns))


if __name__ == "__main__":
    unittest.main()
