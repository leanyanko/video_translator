"""ref_check: a reference is accepted only when its text says what its
audio says (s7)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ref_check import (REF_TEXT_MATCH_MIN, check_reference,  # noqa: E402
                       word_match)


class WordMatchTest(unittest.TestCase):
    def test_punctuation_and_case_do_not_count(self):
        self.assertEqual(word_match("Why do you, what do you do now? You have",
                                    "why do you what do you do? Now you have"), 1.0)

    def test_trailing_word_the_audio_lacks_fails(self):
        # baseline20 SPEAKER_02: «... a recruiting machine. Yeah.» vs audio without «Yeah»
        text = "it seems to be recruiting actually. He's like, he's a recruiting machine. Yeah."
        heard = "it seems to be recruiting, actually. He's like, he's a recruiting machine."
        self.assertLess(word_match(text, heard), REF_TEXT_MATCH_MIN)

    def test_missing_opening_words_fail(self):
        # traps SPEAKER_00: audio opens with «Yeah, yeah.» the text omitted
        text = "The advantage was archival. That's a good point."
        heard = "Yeah, yeah. The advantage was archival. That's a good point."
        self.assertLess(word_match(text, heard), REF_TEXT_MATCH_MIN)


class SlowReferenceTest(unittest.TestCase):
    def test_relative_to_the_panel(self):
        from ref_check import slow_references
        # the s27 case: 9.9 ch/s against 16-22 fires; the absolute floor (9) did not
        self.assertEqual(slow_references({"A": 16.1, "B": 19.4, "C": 22.5, "D": 9.9}, 9.0), ["D"])

    def test_uniformly_slow_panel_does_not_fire(self):
        from ref_check import slow_references
        self.assertEqual(slow_references({"A": 10.0, "B": 9.5, "C": 10.5}, 9.0), [])

    def test_single_speaker_uses_the_floor(self):
        from ref_check import slow_references
        self.assertEqual(slow_references({"A": 8.0}, 9.0), ["A"])
        self.assertEqual(slow_references({"A": 12.0}, 9.0), [])


class CheckReferenceTest(unittest.TestCase):
    def test_verdict(self):
        v = check_reference("a b c d e f g h i j", "a b c d e f g h i j",
                            last_word_end=9.6, audio_secs=10.0)
        self.assertTrue(v["ok"])
        self.assertAlmostEqual(v["tail"], 0.4)
        self.assertEqual(v["chars_per_sec"], 1.9)

    def test_mismatch_is_not_ok(self):
        v = check_reference("a b c d e f g h i j yeah", "a b c d e f g h i j", 9.6, 10.0)
        self.assertFalse(v["ok"])


if __name__ == "__main__":
    unittest.main()
