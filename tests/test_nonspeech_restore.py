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



def pulse_train(n_pulses, period_frames=8, width=3, hi=0.06, lo=0.008):
    """Realistic 30 ms-frame laugh envelope: 4-6 Hz pulses = period of
    ~6-8 frames (reviewer: the old 0.25 s frames sat at Nyquist)."""
    env = []
    for _ in range(n_pulses):
        env += [lo] * (period_frames - width) + [hi] * width
    return env + [lo] * 3


class LaughSignatureTest(unittest.TestCase):
    def test_4hz_pulse_train_is_a_laugh(self):
        from harvest_laughs import envelope_bursts, is_laugh
        env = pulse_train(4)  # "ha-ha-ha-ha" at ~4 Hz on 30 ms frames
        self.assertGreaterEqual(envelope_bursts(env), 3)
        self.assertTrue(is_laugh(1.0, envelope_bursts(env), 0.0))

    def test_flat_hum_is_not_a_laugh(self):
        from harvest_laughs import envelope_bursts, is_laugh
        # one sustained "mm": rises, holds ~0.6 s, falls — 20 frames
        env = [0.01] + [0.05] * 20 + [0.01]
        self.assertEqual(envelope_bursts(env), 1)
        self.assertFalse(is_laugh(0.66, envelope_bursts(env), 0.0))

    def test_breathy_laugh_saved_by_low_panns_vote(self):
        from harvest_laughs import is_laugh
        self.assertTrue(is_laugh(0.6, 1, 0.06))  # acoustic no, panns yes

    def test_short_grunt_fails_both_votes(self):
        from harvest_laughs import is_laugh
        self.assertFalse(is_laugh(0.5, 1, 0.01))


class BleedCalibrationTest(unittest.TestCase):
    def test_solo_spans_subtract_other_speakers(self):
        from harvest_laughs import solo_spans
        speech = {"A": [(0.0, 10.0)], "B": [(4.0, 6.0)]}
        self.assertEqual(solo_spans(speech, "A"), [(0.0, 4.0), (6.0, 10.0)])

    def test_calibrated_quiet_laugh_over_loud_speech_kept(self):
        from harvest_laughs import bleed_expected, OWN_OVER_BLEED
        # mic separation -26 dB: coeff 0.05; Dan talks at 0.3
        coeffs = {("M", "D"): 0.05}
        exp = bleed_expected(coeffs, "M", {"D": 0.3})
        self.assertAlmostEqual(exp, 0.015)
        own = 0.05  # quiet laugh, old fixed 6x rule would have dropped it
        self.assertGreater(own, OWN_OVER_BLEED * exp)

    def test_calibrated_pure_bleed_dropped(self):
        from harvest_laughs import bleed_expected, OWN_OVER_BLEED
        coeffs = {("M", "D"): 0.05}
        exp = bleed_expected(coeffs, "M", {"D": 0.3})
        own = 0.017  # measured pure-bleed level
        self.assertLessEqual(own, OWN_OVER_BLEED * exp)  # the actual DROP decision

    def test_uncalibrated_audible_pair_falls_back(self):
        from harvest_laughs import bleed_expected
        self.assertIsNone(bleed_expected({}, "M", {"D": 0.3}))

    def test_silent_others_need_no_calibration(self):
        from harvest_laughs import bleed_expected
        self.assertEqual(bleed_expected({}, "M", {"D": 0.001}), 0.0)


class RawStemSourceTest(unittest.TestCase):
    def test_prefers_raw_when_present(self):
        import json, tempfile
        from audio_utils import raw_stem_source
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp)
            (w / "tracks.json").write_text(json.dumps(
                {"speakers": {"S": "clean.flac"}}))
            (w / "tracks_raw.json").write_text(json.dumps(
                {"speakers": {"S": "raw.flac"}}))
            spk, d = raw_stem_source(w)
            self.assertEqual(spk["S"], "raw.flac")
            self.assertEqual(d.name, "tracks_raw")

    def test_falls_back_to_tracks(self):
        import json, tempfile
        from audio_utils import raw_stem_source
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp)
            (w / "tracks.json").write_text(json.dumps(
                {"speakers": {"S": "only.flac"}}))
            spk, d = raw_stem_source(w)
            self.assertEqual(d.name, "tracks")

class LoneSfxGateTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np

    def test_underscore_music_is_not_pumped(self):
        # continuous bed at -30 dBFS under speech AND through the pause
        from audio_utils import lone_sfx_gain
        bed = [0.03] * 40
        dub = [True] * 10 + [False] * 20 + [True] * 10
        g = lone_sfx_gain(bed, dub)
        self.assertTrue((g == 1.0).all(), "quiet music must pass untouched")

    def test_poking_foley_in_pause_is_ducked(self):
        from audio_utils import lone_sfx_gain
        bed = [0.004] * 15 + [0.04] * 4 + [0.004] * 21  # burst in silence
        dub = [True] * 10 + [False] * 20 + [True] * 10
        g = lone_sfx_gain(bed, dub)
        self.assertTrue((g[15:19] < 1.0).all(), "foley burst must duck")
        self.assertTrue((g[:10] == 1.0).all())

    def test_long_transient_is_music_not_foley(self):
        from audio_utils import lone_sfx_gain
        bed = [0.004] * 12 + [0.04] * 20 + [0.004] * 8  # 2 s "transient"
        dub = [True] * 8 + [False] * 26 + [True] * 6
        g = lone_sfx_gain(bed, dub, max_run=15)
        self.assertTrue((g == 1.0).all(), ">1.5s of sound is music")

    def test_smooth_gain_ramps_steps(self):
        from audio_utils import smooth_gain
        g = smooth_gain([1.0, 0.01, 1.0], frame_len=100, ramp_len=40)
        self.assertEqual(len(g), 300)
        mid = g[80:120]
        self.assertTrue((self.np.diff(mid) <= 1e-6).all())  # monotone down
        self.assertTrue(0.01 < g[99] < 1.0, "step must be ramped, not hard")

    def test_foley_over_underscore_music_is_ducked(self):
        # review 10-10: run-cap must walk POKE frames, not the whole
        # audible pause — with music playing the pause is always long
        from audio_utils import lone_sfx_gain
        bed = [0.03] * 12 + [0.12] * 3 + [0.03] * 25  # burst over bed
        dub = [True] * 6 + [False] * 28 + [True] * 6
        g = lone_sfx_gain(bed, dub)
        self.assertTrue((g[12:15] < 1.0).all(), "burst over music must duck")
        self.assertTrue((g[:12] == 1.0).all() and (g[15:] == 1.0).all(),
                        "the underscore itself must stay")

    def test_edge_sound_before_first_speech_is_not_ducked(self):
        # review 10-10: missing anchor must mean "no judgement", not
        # "reference zero" — an intro stinger is not foley
        from audio_utils import lone_sfx_gain
        bed = [0.0] * 5 + [0.2] * 10 + [0.0] * 10 + [0.0] * 15
        dub = [False] * 25 + [True] * 15
        g = lone_sfx_gain(bed, dub)
        self.assertTrue((g == 1.0).all(), "edge stinger must pass untouched")

class LowConfSourceRuleTest(unittest.TestCase):
    """segment_tracks pastes low_conf from RAW only when the owner's
    level beats calibrated bleed — otherwise raw would re-import the
    crosstalk that made the line low-confidence in the first place."""

    def test_own_voice_beats_bleed_raw(self):
        from bleed import bleed_expected, OWN_OVER_BLEED
        exp = bleed_expected({("M", "D"): 0.05}, "M", {"D": 0.2})
        self.assertGreater(0.06, OWN_OVER_BLEED * exp)   # raw

    def test_crosstalk_garble_goes_cleaned(self):
        from bleed import bleed_expected, OWN_OVER_BLEED
        exp = bleed_expected({("M", "D"): 0.05}, "M", {"D": 0.3})
        self.assertLessEqual(0.02, OWN_OVER_BLEED * exp)  # cleaned

if __name__ == "__main__":
    unittest.main()
