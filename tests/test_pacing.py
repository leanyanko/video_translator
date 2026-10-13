"""pacing: a dubbed line lasts as long as the original (lip sync),
bounded to a plausible tempo; squeezed only to fit the slot (tracker d5)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from pacing import (MAX_SQUEEZE, PACE_MAX, PACE_MIN, PIECE_MIN_SECS,  # noqa: E402
                    SILENCE_PAD, allocate_pieces, pace_chars, plan_clip)


class PaceCharsTest(unittest.TestCase):
    def test_stress_marks_and_whitespace_do_not_count(self):
        self.assertEqual(pace_chars("Д+а,  д+а."), pace_chars("Да, да."))
        self.assertEqual(pace_chars("  a  b "), 3)


class PlanClipTest(unittest.TestCase):
    def test_original_span_when_tempo_is_plausible(self):
        # 120 chars over the original's 10 s = 12 ch/s: keep the lips
        self.assertEqual(plan_clip(120, 10.0, 20.0), (10.0, 12.0))

    def test_long_translation_overruns_at_pace_max(self):
        # 150 chars cannot fit 5 s at PACE_MAX: take chars/PACE_MAX
        wanted, rate = plan_clip(150, 5.0, 20.0)
        self.assertAlmostEqual(wanted, 150 / PACE_MAX)
        self.assertAlmostEqual(rate, PACE_MAX)

    def test_hesitant_original_does_not_make_the_dub_drawl(self):
        # 45 chars over 10 s would be 4.5 ch/s: shortened to PACE_MIN
        wanted, rate = plan_clip(45, 10.0, 20.0)
        self.assertAlmostEqual(wanted, 45 / PACE_MIN)
        self.assertAlmostEqual(rate, PACE_MIN)

    def test_squeezed_to_the_slot_but_capped(self):
        # 150 chars, original 5 s, slot 9 s: 10 s at PACE_MAX does not
        # fit 8.55 s -> 8.55 s (rate 17.5, within PACE_MAX * MAX_SQUEEZE)
        wanted, rate = plan_clip(150, 5.0, 9.0)
        self.assertAlmostEqual(wanted, 8.55)
        self.assertAlmostEqual(rate, 150 / 8.55)
        # slot 3 s: the cap binds, the line overruns
        wanted, rate = plan_clip(150, 5.0, 3.0)
        self.assertAlmostEqual(wanted, 150 / PACE_MAX / MAX_SQUEEZE)
        self.assertAlmostEqual(rate, PACE_MAX * MAX_SQUEEZE)


class AllocatePiecesTest(unittest.TestCase):
    def test_proportional_with_pad_and_floor(self):
        secs = allocate_pieces(["а" * 30, "б" * 10, "Да."], 4.0)
        budget = 4.0 + SILENCE_PAD
        self.assertAlmostEqual(secs[0], budget * 30 / 43)
        self.assertAlmostEqual(secs[1], budget * 10 / 43)
        self.assertEqual(secs[2], PIECE_MIN_SECS)  # 0.3 s natural -> floor

    def test_single_piece_gets_everything(self):
        self.assertEqual(allocate_pieces(["текст"], 2.5), [2.5 + SILENCE_PAD])


if __name__ == "__main__":
    unittest.main()
