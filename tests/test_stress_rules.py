"""Unit tests for scripts/stress_rules.py (automatic stress from lists)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from stress_rules import apply_homographs, apply_name_stress, resolve


class TestNameStress(unittest.TestCase):
    def test_inflected_forms_are_covered(self):
        self.assertEqual(apply_name_stress("он жил в Ванкувере"), "он жил в Ванк+увере")
        self.assertEqual(apply_name_stress("над Ванкувером"), "над Ванк+увером")

    def test_case_is_preserved(self):
        self.assertIn("Ванк+увер", apply_name_stress("Ванкувер"))

    def test_already_marked_token_is_untouched(self):
        self.assertEqual(apply_name_stress("в Ванкув+ере"), "в Ванкув+ере")

    def test_unknown_words_pass_through(self):
        text = "обычное предложение без имён"
        self.assertEqual(apply_name_stress(text), text)


class TestHomographs(unittest.TestCase):
    def test_english_cue_selects_variant(self):
        self.assertEqual(
            apply_homographs("думаешь в душе", "you think in the shower"),
            "думаешь в д+уше",
        )
        self.assertEqual(
            apply_homographs("в душе я знал", "deep down in my soul I knew"),
            "в душ+е я знал",
        )

    def test_no_cue_leaves_word_alone(self):
        self.assertEqual(
            apply_homographs("этим стоит заниматься", "completely unrelated"),
            "этим стоит заниматься",
        )

    def test_capitalized_wordform(self):
        out = apply_homographs("Стоит попробовать", "it is worth trying")
        self.assertEqual(out, "Ст+оит попробовать")

    def test_translator_mark_wins_over_list(self):
        self.assertEqual(
            apply_homographs("сто+ит на столе", "worth a lot"),
            "сто+ит на столе",
        )


class TestResolve(unittest.TestCase):
    def test_combined(self):
        out = resolve(
            "она живёт в Ванкувере, и об этом стоит подумать в душе",
            "she lives in Vancouver, worth thinking about in the shower",
        )
        self.assertIn("Ванк+увере", out)
        self.assertIn("ст+оит", out)
        self.assertIn("д+уше", out)

    def test_missing_english_text(self):
        out = resolve("замок на двери в Ванкувере", None)
        self.assertIn("Ванк+увере", out)   # names never need English
        self.assertIn("замок", out)        # homograph left for RUAccent
        self.assertNotIn("з+амок", out)


if __name__ == "__main__":
    unittest.main()
