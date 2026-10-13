"""Unit tests for scripts/segmenting.py (shared sentence build/merge)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from segmenting import (GAP_BREAK, MAX_PAUSE, SENT_END, TARGET_SECS,  # noqa: E402
                        build_sentences, merge_short, runon_cut_ids,
                        split_long, split_long_flagged, words_text)


def W(start, end, word):
    return {"start": start, "end": end, "word": word}


class TestBuildSentences(unittest.TestCase):
    def test_cuts_at_sentence_end_only(self):
        ws = [W(0, 1, " One"), W(1, 2, " two."), W(2, 3, " Three"), W(3, 4, " four?"), W(4, 5, " tail")]
        out = build_sentences(ws)
        self.assertEqual([words_text(s) for s in out], ["One two.", "Three four?", "tail"])

    def test_gap_break_also_cuts(self):
        ws = [W(0, 1, " a"), W(1, 2, " b"), W(2 + GAP_BREAK + 0.1, 6, " c."), W(6, 7, " d")]
        self.assertEqual(len(build_sentences(ws)), 2)            # no gap rule: "a b c." | "d"
        self.assertEqual(len(build_sentences(ws, GAP_BREAK)), 3)  # gap rule: "a b" | "c." | "d"

    def test_every_sentence_end_mark(self):
        for mark in SENT_END:
            out = build_sentences([W(0, 1, f" x{mark}"), W(1, 2, " y")])
            self.assertEqual(len(out), 2, mark)

    def test_empty(self):
        self.assertEqual(build_sentences([]), [])


class TestSplitLong(unittest.TestCase):
    def test_short_untouched(self):
        ws = [W(0, 1, " a"), W(1, 2, " b")]
        self.assertEqual(split_long(ws, 25.0), [ws])

    def test_prefers_largest_pause_after_clause_mark(self):
        # biggest pause (3 s) is after "b", but "a," is a clause boundary
        ws = [W(0, 1, " a,"), W(2, 3, " b"), W(6, 7, " c"), W(7, 30, " d")]
        parts = split_long(ws, 10.0)
        self.assertEqual(words_text(parts[0]), "a,")

    def test_falls_back_to_largest_pause_without_clause_marks(self):
        ws = [W(0, 1, " a"), W(2, 3, " b"), W(6, 7, " c"), W(7, 30, " d")]
        parts = split_long(ws, 10.0)
        self.assertEqual(words_text(parts[0]), "a b")

    def test_recurses_until_under_cap(self):
        ws = [W(i * 5, i * 5 + 4, f" w{i}") for i in range(12)]  # 59 s, no marks
        parts = split_long(ws, 20.0)
        for p in parts:
            self.assertLessEqual(p[-1]["end"] - p[0]["start"], 20.0)
        self.assertEqual(sum(len(p) for p in parts), len(ws))


class TestMergeShort(unittest.TestCase):
    def item(self, start, end, spk="S0", text="t."):
        return {"start": start, "end": end, "speaker": spk, "text": text}

    def test_merges_same_speaker_within_limits(self):
        out = merge_short([self.item(0, 4, text="One."), self.item(4.5, 8, text="Two.")])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "One. Two.")
        self.assertEqual(out[0]["end"], 8)

    def test_speaker_change_blocks_merge(self):
        out = merge_short([self.item(0, 4), self.item(4.5, 8, spk="S1")])
        self.assertEqual(len(out), 2)

    def test_pause_blocks_merge(self):
        out = merge_short([self.item(0, 4), self.item(4 + MAX_PAUSE + 0.1, 8)])
        self.assertEqual(len(out), 2)

    def test_target_span_blocks_merge(self):
        out = merge_short([self.item(0, 8), self.item(8.5, 8.5 + TARGET_SECS)])
        self.assertEqual(len(out), 2)

    def test_input_not_mutated(self):
        items = [self.item(0, 4, text="One."), self.item(4.5, 8, text="Two.")]
        merge_short(items)
        self.assertEqual(items[0]["text"], "One.")
        self.assertEqual(items[0]["end"], 4)

    def test_cut_flag_follows_the_tail(self):
        # TODO #12: the mid-sentence flag describes a piece's END — after
        # a merge the tail's end is the one that counts
        head = dict(self.item(0, 4), cut="mid-sentence")
        tail = self.item(4.5, 8)  # ends cleanly
        self.assertNotIn("cut", merge_short([head, tail])[0])
        head2, tail2 = self.item(0, 4), dict(self.item(4.5, 8), cut="mid-sentence")
        self.assertEqual(merge_short([head2, tail2])[0]["cut"], "mid-sentence")


class TestSplitLongFlagged(unittest.TestCase):
    def test_clause_cut_is_not_flagged(self):
        ws = [W(0, 1, " a,"), W(2, 3, " b"), W(6, 7, " c"), W(7, 30, " d")]
        parts = split_long_flagged(ws, 10.0)
        self.assertFalse(parts[0][1])  # cut after the clause mark: fine

    def test_markless_cut_is_flagged(self):
        ws = [W(0, 1, " a"), W(2, 3, " b"), W(6, 7, " c"), W(7, 30, " d")]
        parts = split_long_flagged(ws, 10.0)
        self.assertTrue(parts[0][1])   # last-resort mid-sentence cut
        self.assertFalse(parts[-1][1])  # the final piece ends naturally


class TestRunonCutIds(unittest.TestCase):
    SEG = staticmethod(lambda i, spk, a, b, text: {
        "id": i, "speaker": spk, "start": a, "end": b, "text": text})

    def test_cut_inside_runon_is_flagged(self):
        segs = [self.SEG(0, "S0", 0, 10, "без знака конца"),
                self.SEG(1, "S0", 10.5, 20, "продолжение.")]
        runons, interrupted = runon_cut_ids(segs)
        self.assertEqual((runons, interrupted), ([0], 0))

    def test_interruption_is_legitimate(self):
        segs = [self.SEG(0, "S0", 0, 10, "без знака конца"),
                self.SEG(2, "S1", 9.0, 12.0, "перебил."),
                self.SEG(1, "S0", 10.5, 20, "продолжение.")]
        runons, interrupted = runon_cut_ids(segs)
        self.assertEqual((runons, interrupted), ([], 1))

    def test_sentence_end_is_clean(self):
        segs = [self.SEG(0, "S0", 0, 10, "фраза целиком."),
                self.SEG(1, "S0", 10.5, 20, "вторая.")]
        self.assertEqual(runon_cut_ids(segs), ([], 0))


if __name__ == "__main__":
    unittest.main()
