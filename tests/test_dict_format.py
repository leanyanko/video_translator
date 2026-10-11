"""Every data/*.json stays in the canonical format (TODO #10): a
dictionary edit must diff in as many lines as entries changed, never in
a total reformat that blinds the review. Fix drift with
    venv/bin/python scripts/format_dicts.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from format_dicts import DATA, canonical, drifted, validate  # noqa: E402


class DictFormatTest(unittest.TestCase):
    def test_all_dictionaries_canonical(self):
        files = sorted(DATA.glob("*.json"))
        self.assertTrue(files, "data/ has no dictionaries?")
        bad = [p.name for p in files if drifted(p)]
        self.assertEqual(bad, [],
                         "run scripts/format_dicts.py to fix: " + ", ".join(bad))

    def test_content_valid(self):
        # a dead homograph branch or a markless stress value fails CI
        # the same way a bad byte does (review 10-11); short stems only
        # warn — дэн/тед/иаг are deliberate
        errors, _ = validate()
        self.assertEqual(errors, [])

    def test_canonical_preserves_key_order(self):
        # the formatter must never sort: source order is meaningful
        self.assertEqual(canonical({"б": 1, "а": 2}),
                         '{\n "б": 1,\n "а": 2\n}\n')


if __name__ == "__main__":
    unittest.main()
