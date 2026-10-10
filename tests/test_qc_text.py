"""qc_text: the QC compares against what was SUNG (accents text with
dictionary respellings), so a correct «Гельберт» clip is never flagged
MISMATCH (TODO #6); digit tokens expand to Russian words so «20» ↔
«двадцать» matches.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from qc_text import ru_num_words, script_sim, words  # noqa: E402


class NumWordsTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(ru_num_words(0), ["ноль"])
        self.assertEqual(ru_num_words(5), ["пять"])
        self.assertEqual(ru_num_words(11), ["одиннадцать"])
        self.assertEqual(ru_num_words(20), ["двадцать"])
        self.assertEqual(ru_num_words(42), ["сорок", "два"])
        self.assertEqual(ru_num_words(400), ["четыреста"])

    def test_thousands(self):
        self.assertEqual(ru_num_words(1960),
                         ["тысяча", "девятьсот", "шестьдесят"])
        self.assertEqual(ru_num_words(2000), ["две", "тысячи"])
        self.assertEqual(ru_num_words(5000), ["пять", "тысяч"])
        self.assertEqual(ru_num_words(21000),
                         ["двадцать", "одна", "тысяча"])
        self.assertEqual(ru_num_words(400_000_000),
                         ["четыреста", "миллионов"])

    def test_words_expand_digits(self):
        self.assertEqual(words("в 1960 году"),
                         ["в", "тысяча", "девятьсот", "шестьдесят", "году"])
        self.assertEqual(words("двадцать лет"), words("20 лет"))


class ScriptSimTest(unittest.TestCase):
    def test_respelled_name_is_not_a_mismatch(self):
        # the dictionary deliberately sings «Гельберта»; whisper hears it
        sim, _, _ = script_sim("встречайте дэн гельберта",
                               sung="Встреч+айте: Д+эн Г+ельберта.",
                               plain="Встречайте: Дэн Гельбарта.")
        self.assertEqual(sim, 1.0)

    def test_real_garbage_is_still_flagged(self):
        sim, _, _ = script_sim("совсем другие слова здесь",
                               sung="Встреч+айте: Д+эн Г+ельберта.",
                               plain="Встречайте: Дэн Гельбарта.")
        self.assertLess(sim, 0.75)

    def test_plain_variant_counts_when_better(self):
        # no accents entry case: sung falls back to plain — still works
        sim, _, _ = script_sim("двадцать лет спустя",
                               sung="20 лет спустя", plain="20 лет спустя")
        self.assertEqual(sim, 1.0)

    def test_extra_sizing_uses_sung_words(self):
        _, heard_w, sung_w = script_sim("раз два три",
                                        sung="сл+ово", plain="слово")
        self.assertEqual(len(sung_w), 1)
        self.assertEqual(len(heard_w), 3)


if __name__ == "__main__":
    unittest.main()
