"""manifest_bridge: pre-manifest/old-scheme dirs migrate without the two
failure modes — full re-synthesis (hours) and silently blessing edited
lines as fresh (the e1v5 stale-clip class).
"""
import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from manifest_bridge import (load_or_bridge, missing_clips,  # noqa: E402
                             new_hash, old_hash, norm_text, rotate_accents,
                             slot_ends, stale_ids, voiced)

SEGS = [{"id": 1, "speaker": "SPEAKER_00"}, {"id": 2, "speaker": "SPEAKER_00"},
        {"id": 3, "speaker": "SPEAKER_03"}]
REF = lambda spk: "refsha-" + spk[-2:]  # noqa: E731


def make_dir(tmp, prev=None, manifest=None, clip_ids=(1, 2, 3),
             prev_age=100, accents=None, accents_age=100):
    work = Path(tmp)
    out = work / "tts_f5"
    out.mkdir()
    now = time.time()
    for i in clip_ids:
        p = out / f"seg_{i:04d}.wav"
        p.write_bytes(b"RIFF")
        os.utime(p, (now, now))
    if prev is not None:
        pp = work / "accents.prev.json"
        pp.write_text(json.dumps(prev, ensure_ascii=False))
        os.utime(pp, (now - prev_age, now - prev_age))
    if accents is not None:
        ap = work / "accents.json"
        ap.write_text(json.dumps(accents, ensure_ascii=False))
        os.utime(ap, (now - accents_age, now - accents_age))
    if manifest is not None:
        (out / "manifest.json").write_text(json.dumps(manifest))
    return work, out


class BridgeTest(unittest.TestCase):
    def test_pre_manifest_bridges_from_prev_and_catches_edits(self):
        prev = {"1": "ст+арый текст один", "2": "ст+арый текст два",
                "3": "ст+арый текст три"}
        cur = dict(prev, **{"2": "н+овый текст два"})  # line 2 edited
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp, prev=prev)
            m, n = load_or_bridge(work, out, SEGS, cur, REF, log=lambda *a: None)
            self.assertEqual(n, 3)
            # unedited lines match the current hash -> clip kept
            self.assertEqual(m["1"], new_hash("SPEAKER_00", cur["1"], REF("SPEAKER_00")))
            # the edited line is keyed to its OLD text -> clip re-synthesized
            self.assertNotEqual(m["2"], new_hash("SPEAKER_00", cur["2"], REF("SPEAKER_00")))
            self.assertEqual(m["2"], new_hash("SPEAKER_00", prev["2"], REF("SPEAKER_00")))

    def test_prev_newer_than_clips_is_not_trusted(self):
        prev = {"1": "а", "2": "б", "3": "в"}
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp, prev=prev, prev_age=-100)  # prev NEWER
            m, n = load_or_bridge(work, out, SEGS, prev, REF, log=lambda *a: None)
            self.assertIsNone(m)

    def test_no_prev_no_manifest_gives_none(self):
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp)
            m, n = load_or_bridge(work, out, SEGS, {}, REF, log=lambda *a: None)
            self.assertIsNone(m)

    def test_new_scheme_manifest_passes_through_untouched(self):
        cur = {"1": "т1", "2": "т2", "3": "т3"}
        good = {str(s["id"]): new_hash(s["speaker"], cur[str(s["id"])],
                                       REF(s["speaker"])) for s in SEGS}
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp, manifest=good)
            m, n = load_or_bridge(work, out, SEGS, cur, REF, log=lambda *a: None)
            self.assertEqual(m, good)
            self.assertEqual(n, 0)

    def test_old_scheme_manifest_upgrades_per_id(self):
        prev = {"1": "ст+арый один", "2": "ст+арый два", "3": "ст+арый три"}
        cur = dict(prev, **{"2": "н+овый два"})
        # old redub hashed speaker|text of what it sang: ids 1,3 = current
        # text, id 2 = the old text (sung before the edit)
        old_m = {"1": old_hash("SPEAKER_00", cur["1"]),
                 "2": old_hash("SPEAKER_00", prev["2"]),
                 "3": old_hash("SPEAKER_03", cur["3"])}
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp, prev=prev, manifest=old_m)
            m, n = load_or_bridge(work, out, SEGS, cur, REF, log=lambda *a: None)
            self.assertEqual(n, 3)
            self.assertEqual(m["1"], new_hash("SPEAKER_00", cur["1"], REF("SPEAKER_00")))
            self.assertEqual(m["2"], new_hash("SPEAKER_00", prev["2"], REF("SPEAKER_00")))
            self.assertNotEqual(m["2"], new_hash("SPEAKER_00", cur["2"], REF("SPEAKER_00")))

    def test_empty_dir_returns_empty_manifest(self):
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp, clip_ids=())
            m, n = load_or_bridge(work, out, SEGS, {}, REF, log=lambda *a: None)
            self.assertEqual(m, {})


class ReviewBatchTest(unittest.TestCase):
    def test_no_prev_old_accents_are_the_record(self):
        # reviewer gap 2: unedited pre-manifest dir, accent stage skipped
        acc = {"1": "т+екст один", "2": "т+екст два", "3": "т+екст три"}
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp, accents=acc)
            m, n = load_or_bridge(work, out, SEGS, acc, REF, log=lambda *a: None)
            self.assertEqual(n, 3)
            for s in SEGS:
                self.assertEqual(m[str(s["id"])],
                                 new_hash(s["speaker"], acc[str(s["id"])],
                                          REF(s["speaker"])))

    def test_accents_newer_than_clips_not_trusted(self):
        acc = {"1": "а", "2": "б", "3": "в"}
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp, accents=acc, accents_age=-100)
            m, n = load_or_bridge(work, out, SEGS, acc, REF, log=lambda *a: None)
            self.assertIsNone(m)

    def test_ref_recut_after_clips_leaves_speaker_unbridged(self):
        acc = {"1": "а один", "2": "б два", "3": "в три"}
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp, accents=acc)
            refs = Path(tmp) / "refs"
            refs.mkdir()
            now = time.time()
            for spk, age in (("SPEAKER_00", 500), ("SPEAKER_03", -500)):
                p = refs / f"{spk}.wav"
                p.write_bytes(b"RIFF")
                os.utime(p, (now - age, now - age))
            m, n = load_or_bridge(work, out, SEGS, acc, REF,
                                  log=lambda *a: None, refs_dir=refs)
            self.assertIn("1", m)       # old ref: bridged
            self.assertNotIn("3", m)    # re-cut ref: unverified

    def test_bridge_covers_clips_outside_the_requested_subset(self):
        # reviewer: --limit-secs must not shrink the written manifest;
        # the caller passes the UNFILTERED list — entries cover all clips
        acc = {"1": "а один", "2": "б два", "3": "в три"}
        with TemporaryDirectory() as tmp:
            work, out = make_dir(tmp, accents=acc)
            m, n = load_or_bridge(work, out, SEGS, acc, REF, log=lambda *a: None)
            self.assertEqual(set(m), {"1", "2", "3"})

    def test_rotate_accents_noop_keeps_record(self):
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "accents.json").write_text(json.dumps({"1": "т+екст"},
                                                          ensure_ascii=False))
            self.assertFalse(rotate_accents(work, {"1": "т+екст"}))
            self.assertFalse((work / "accents.prev.json").exists())
            self.assertTrue(rotate_accents(work, {"1": "друг+ой"}))
            self.assertTrue((work / "accents.prev.json").exists())

    def test_norm_tolerates_symbol_and_space_jitter(self):
        self.assertEqual(norm_text("типаж, R&D."), norm_text("тип+аж RD"))
        self.assertEqual(norm_text("слово,слово"), norm_text("слово, слово"))
        self.assertNotEqual(norm_text("старый текст"), norm_text("новый текст"))


class StaleIdsTest(unittest.TestCase):
    """TODO #3: the freshness gate must not hold --only hostage — scope
    it to the set being synthesized; the full run still checks all."""

    SEGS = [{"id": 5, "speaker": "S", "text_ru": "новый текст пять"},
            {"id": 42, "speaker": "S", "text_ru": "текст сорок два"}]
    ACC = {"5": "ст+арый т+екст п+ять",  # 5 edited, accents not re-run
           "42": "т+екст с+орок дв+а"}

    def test_full_run_catches_the_stale_line(self):
        self.assertEqual(stale_ids(self.SEGS, self.ACC), [5])

    def test_only_scope_is_not_blocked_by_an_unrelated_edit(self):
        scope = [s for s in self.SEGS if s["id"] == 42]
        self.assertEqual(stale_ids(scope, self.ACC), [])

    def test_missing_entry_is_stale(self):
        self.assertEqual(stale_ids([{"id": 7, "text_ru": "х"}], {}), [7])

    def test_respelling_is_not_an_edit(self):
        from stress_rules import force_name_stress
        segs = [{"id": 1, "text_ru": "Дэна Гельбарта"}]
        acc = {"1": "Д+эна Г+ельберта"}  # dictionary respelling
        self.assertEqual(stale_ids(segs, acc, respell=force_name_stress), [])


class MissingClipsTest(unittest.TestCase):
    """TODO #4: absent and trivially-empty clips both count as missing —
    a crashed synth leaves 0-byte wavs that exists() alone blesses."""

    def test_absent_and_empty_are_missing(self):
        with TemporaryDirectory() as tmp:
            tts = Path(tmp)
            (tts / "seg_0001.wav").write_bytes(b"\0" * 4096)  # fine
            (tts / "seg_0002.wav").write_bytes(b"RIFF")       # crashed
            segs = [{"id": 1}, {"id": 2}, {"id": 3}]          # 3 absent
            self.assertEqual(missing_clips(segs, tts), [2, 3])


class VoicedTest(unittest.TestCase):
    """TODO #9: orig pastes and echo lines are never synthesized."""

    def test_voiced(self):
        self.assertTrue(voiced({"id": 1, "text_ru": "х"}))
        self.assertFalse(voiced({"id": 2, "type": "orig"}))
        self.assertFalse(voiced({"id": 3, "echo": True, "text_ru": "х"}))


class SlotEndsTest(unittest.TestCase):
    """s16: an 11 s line with a 0.5 s "mm-hmm" paste dropped into it
    must keep its slot up to the next DUBBED line, not the paste."""

    SEGS = [
        {"id": 1, "start": 10.0, "end": 21.0, "text_ru": "длинная"},
        {"id": 9500, "start": 10.5, "end": 11.0, "type": "orig",
         "auto_track": True, "file": "laughs/x.wav"},
        {"id": 2, "start": 15.0, "end": 16.0, "text_ru": "поверх",
         "overlap": True},
        {"id": 3, "start": 17.0, "end": 18.0, "text_ru": "эхо", "echo": True},
        {"id": 4, "start": 22.0, "end": 25.0, "text_ru": "следующая"},
        {"id": 5, "start": 26.0, "end": 27.0, "type": "orig"},
    ]

    def test_other_speaker_pastes_overlap_and_echo_do_not_end_a_slot(self):
        segs = [{**s, "speaker": "B" if s["id"] in (9500, 5) else "A"}
                for s in self.SEGS]
        ends = slot_ends(segs)
        self.assertEqual(ends[1], 22.0)
        self.assertIsNone(ends[4])  # last dubbed line: caller's tail
        self.assertEqual(set(ends), {1, 4})  # only sequential voiced lines

    def test_same_speaker_paste_ends_the_slot(self):
        # his own laugh 0.5 s after his line: the dub must not run over it
        segs = [{**s, "speaker": "A"} for s in self.SEGS]
        segs[1] = {**segs[1], "start": 21.5, "end": 22.0}
        ends = slot_ends(segs)
        self.assertEqual(ends[1], 21.5)
        self.assertEqual(ends[4], 26.0)  # trailing own paste ends it too

    def test_unsorted_input(self):
        segs = [{**s, "speaker": "A"} for s in reversed(self.SEGS)]
        self.assertEqual(slot_ends(segs)[1], 10.5)  # own paste inside the line

    def test_empty(self):
        self.assertEqual(slot_ends([]), {})


if __name__ == "__main__":
    unittest.main()
