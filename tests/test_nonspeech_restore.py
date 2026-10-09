"""harvest_laughs (non-speech restore): the keep/drop core is pure.

The invariant replaces detection (TODO #1 resolution): voiced islands
minus speech spans, anti-bleed by cross-stem loudness ratio. Anchor
numbers come from measured cases in work/exp_traps_v3.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from harvest_laughs import bleed_dropped, subtract_speech  # noqa: E402


class SubtractSpeechTest(unittest.TestCase):
    def test_laugh_between_speech_survives(self):
        # island 10-16, speech 10-11.5 and 14.5-16 -> middle 1.75..14.25
        out = subtract_speech([(10.0, 16.0)], [(10.0, 11.5), (14.5, 16.0)])
        self.assertEqual(len(out), 1)
        a, b = out[0]
        self.assertAlmostEqual(a, 11.75)
        self.assertAlmostEqual(b, 14.25)

    def test_fully_covered_island_vanishes(self):
        self.assertEqual(subtract_speech([(5.0, 8.0)], [(4.0, 9.0)]), [])

    def test_island_with_no_speech_is_kept_whole(self):
        self.assertEqual(subtract_speech([(11.3, 14.4)], []), [(11.3, 14.4)])

    def test_short_leftovers_are_dropped(self):
        out = subtract_speech([(10.0, 12.0)], [(10.0, 11.4)])  # leftover 0.35-ε
        self.assertEqual(out, [])


class BleedRatioTest(unittest.TestCase):
    def test_misha_laugh_over_dan_speech_is_kept(self):
        # measured: own 0.054, Dan's stem 0.2 -> ratio ~3.7 < 6
        self.assertFalse(bleed_dropped(0.054, 0.20))

    def test_dan_bleed_on_misha_stem_is_dropped(self):
        # measured: own 0.017, Dan's stem 0.30 -> ratio ~18
        self.assertTrue(bleed_dropped(0.017, 0.30))

    def test_solo_sound_with_silent_others_is_kept(self):
        self.assertFalse(bleed_dropped(0.05, 0.0))


if __name__ == "__main__":
    unittest.main()


class LaughSignatureTest(unittest.TestCase):
    def test_pulsed_envelope_is_a_laugh(self):
        from harvest_laughs import envelope_bursts, is_laugh
        env = [0.01, 0.06, 0.01, 0.055, 0.012, 0.05, 0.01]  # ha-ha-ha
        self.assertGreaterEqual(envelope_bursts(env), 2)
        self.assertTrue(is_laugh(2.3, envelope_bursts(env), 0.0))

    def test_flat_hum_is_not_a_laugh(self):
        from harvest_laughs import envelope_bursts, is_laugh
        env = [0.02, 0.05, 0.055, 0.05, 0.02]  # one flat "mm"
        self.assertEqual(envelope_bursts(env), 1)
        self.assertFalse(is_laugh(0.5, envelope_bursts(env), 0.0))

    def test_breathy_laugh_saved_by_low_panns_vote(self):
        from harvest_laughs import is_laugh
        self.assertTrue(is_laugh(0.6, 1, 0.06))  # acoustic no, panns yes

    def test_short_grunt_fails_both_votes(self):
        from harvest_laughs import is_laugh
        self.assertFalse(is_laugh(0.5, 1, 0.01))
