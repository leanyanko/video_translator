"""pacing: a dubbed line lasts as long as the original (lip sync),
bounded to a plausible tempo; squeezed only to fit the slot (tracker d5)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from pacing import (PACE_MAX, PACE_MIN, PIECE_MIN_SECS, SILENCE_PAD,  # noqa: E402
                    allocate_pieces, pace_chars, plan_clip)


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

    def test_squeezed_to_the_slot_but_never_past_pace_max(self):
        # 150 chars, original 5 s: at PACE_MAX the line needs 150/PACE_MAX;
        # a slot whose 95 % is a bit shorter squeezes it only down to that
        # — the rest overruns for assemble's atempo (no native squeeze)
        fastest = 150 / PACE_MAX
        wanted, rate = plan_clip(150, 5.0, fastest * 0.9 / 0.95)
        self.assertAlmostEqual(wanted, fastest)
        self.assertAlmostEqual(rate, PACE_MAX)
        wanted, rate = plan_clip(150, 5.0, 3.0)
        self.assertAlmostEqual(wanted, fastest)  # tight slot: still PACE_MAX


class AllocatePiecesTest(unittest.TestCase):
    def test_share_plus_pad_per_piece_with_floor(self):
        # the pad is PER CALL: F5 paints silence around each piece
        secs = allocate_pieces(["а" * 30, "б" * 10, "Да."], 4.0)
        self.assertAlmostEqual(secs[0], 4.0 * 30 / 43 + SILENCE_PAD)
        self.assertAlmostEqual(secs[1], 4.0 * 10 / 43 + SILENCE_PAD)
        self.assertAlmostEqual(secs[2], PIECE_MIN_SECS + SILENCE_PAD)  # 0.28 s -> floor

    def test_single_piece_gets_everything(self):
        self.assertEqual(allocate_pieces(["текст"], 2.5), [2.5 + SILENCE_PAD])


if __name__ == "__main__":
    unittest.main()
