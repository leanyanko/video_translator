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


class QcNoiseTest(unittest.TestCase):
    """The QC-noise classes measured on baseline20 (10-13): fine clips that
    scored 0.5-0.89 and were re-rolled for nothing. Each must now pass
    the re-roll threshold (0.9) while real garbage stays below."""

    def sim(self, heard, script):
        return script_sim(heard, script, script)[0]

    def test_inflection_only_differences(self):
        self.assertGreaterEqual(self.sim("Но это крайний случай.", "Но это крайние случаи."), 0.9)
        self.assertGreaterEqual(self.sim("Да, но гравитацию компенсирует.",
                                         "Да, но гравитацию компенсируют."), 0.9)
        self.assertGreaterEqual(self.sim("Потребительская, да. Да.", "Потребительское, да. Да."), 0.9)

    def test_latin_acronym_vs_spelled_out(self):
        self.assertGreaterEqual(self.sim("Это ровно CD или DVD на ленте.",
                                         "Это ровно си-ди или ди-ви-ди на ленте."), 0.9)

    def test_one_word_clip_without_its_particle(self):
        self.assertGreaterEqual(self.sim("дальше", "А дальше?"), 0.9)
        self.assertGreaterEqual(self.sim("сердце", "В сердце."), 0.9)

    def test_real_faults_still_flagged(self):
        self.assertLess(self.sim("обратить прибора здесь", "Да, а прототип прибора здесь, да?"), 0.9)
        self.assertLess(self.sim("вы тоже проб овали", "вы тоже пробовали"), 0.9)
        self.assertLess(self.sim("Потому что я неамбициозный человек.",
                                 "Потому что я, ну, не амбициозный человек."), 0.9)


class QcPairsFixtureTest(unittest.TestCase):
    """tests/fixtures/qc_pairs.json: ~220 (heard, sung, script) pairs from
    baseline20 + traps with known verdicts. Every leniency change to
    script_sim is measured against the same set (the d7 idea, applied
    where the data exists)."""

    @classmethod
    def setUpClass(cls):
        import json
        from qc_text import SIM_MIN, SIM_REROLL
        cls.SIM_MIN, cls.SIM_REROLL = SIM_MIN, SIM_REROLL
        path = Path(__file__).resolve().parent / "fixtures" / "qc_pairs.json"
        cls.pairs = json.loads(path.read_text())["pairs"]

    def test_bad_pairs_stay_under_the_reroll_threshold(self):
        for p in self.pairs:
            if p["expect"] == "bad":
                sim = script_sim(p["heard"], p["sung"], p["script"])[0]
                self.assertLess(sim, self.SIM_REROLL, p["src"])

    def test_ok_pairs_are_never_flagged(self):
        for p in self.pairs:
            if p["expect"] == "ok":
                sim = script_sim(p["heard"], p["sung"], p["script"])[0]
                self.assertGreaterEqual(sim, self.SIM_MIN, f"{p['src']}: {p['heard'][:40]}")

    def test_ok_pairs_under_the_reroll_threshold_do_not_grow(self):
        # the documented residual: long lines with several small whisper
        # slips (baseline20 ids 3, 5, 44, 106, 127 at 0.83-0.90)
        low = [p["src"] for p in self.pairs if p["expect"] == "ok"
               and script_sim(p["heard"], p["sung"], p["script"])[0] < self.SIM_REROLL]
        self.assertLessEqual(len(low), 5, low)

    def test_latin_w_matches_the_translators_spelling(self):
        self.assertGreaterEqual(script_sim("это WD-сорок", "это дабл-ю ди сорок", "это дабл-ю ди сорок")[0], 0.9)


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
