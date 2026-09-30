"""Unit tests for scripts/text_split.py (F5 single-batch splitting)."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from text_split import MAX_PIECE_BYTES, _blen, split_single_batch


def normalize(text: str) -> str:
    """Words only — splitting may drop/normalize inter-piece whitespace."""
    return " ".join(re.split(r"[ ]+", text.strip()))


class TestSplitSingleBatch(unittest.TestCase):
    def test_short_text_is_one_piece(self):
        text = "Видимо, Ванкувер притягивает людей."
        self.assertEqual(split_single_batch(text), [text])

    def test_every_piece_fits_byte_budget(self):
        text = (
            "И Мейман был уверен: рубин не просто генерирует — это один из "
            "лучших материалов. — Вы его знали? — Он был моим другом до "
            "конца. Так лазер и попал ко мне. И это очень длинная история "
            "о том, как всё начиналось в тысяча девятьсот шестидесятом году."
        )
        pieces = split_single_batch(text)
        self.assertGreater(len(pieces), 1)
        for p in pieces:
            self.assertLessEqual(_blen(p), MAX_PIECE_BYTES, repr(p))

    def test_no_words_lost(self):
        text = (
            "Первое предложение о лазерах. Второе предложение про рубин и "
            "усиление! Третье предложение — про инверсную населённость? "
            "Четвёртое, совсем короткое. Пятое завершает мысль о физике."
        )
        pieces = split_single_batch(text)
        self.assertEqual(
            normalize(" ".join(pieces)).replace(" ", ""),
            normalize(text).replace(" ", ""),
        )

    def test_cyrillic_counts_two_bytes(self):
        # enough Cyrillic chars to exceed the BYTE budget while a naive
        # char-based counter would still think the text fits
        n = MAX_PIECE_BYTES // 2  # chars whose UTF-8 weight ≈ full budget
        text = "а" * (n - 10) + ", " + "б" * (n - 10)
        self.assertLess(len(text), MAX_PIECE_BYTES)  # fits by chars...
        self.assertGreater(_blen(text), MAX_PIECE_BYTES)  # ...not by bytes
        pieces = split_single_batch(text)
        self.assertGreater(len(pieces), 1)
        for p in pieces:
            self.assertLessEqual(_blen(p), MAX_PIECE_BYTES)

    def test_long_sentence_without_periods_splits_at_comma(self):
        text = ("очень длинное предложение без точек, " * 4).strip().rstrip(",")
        pieces = split_single_batch(text)
        self.assertGreater(len(pieces), 1)
        for p in pieces:
            self.assertLessEqual(_blen(p), MAX_PIECE_BYTES)

    def test_unbreakable_run_is_hard_split_on_byte_boundary(self):
        text = "х" * 200  # no punctuation at all
        pieces = split_single_batch(text)
        self.assertEqual("".join(pieces), text)
        for p in pieces:
            self.assertLessEqual(_blen(p), MAX_PIECE_BYTES)

    def test_empty_and_whitespace(self):
        self.assertEqual(split_single_batch(""), [])
        self.assertEqual(split_single_batch("   "), [])

    def test_stress_marks_survive(self):
        text = "В+идимо, есть в Ванк+увере чт+о-то волш+ебное. " * 3
        pieces = split_single_batch(text.strip())
        self.assertIn("Ванк+увере", " ".join(pieces))


if __name__ == "__main__":
    unittest.main()
