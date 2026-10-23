"""run_pipeline: the shared tail's harvest decision (d10)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_pipeline import harvest_due, tail  # noqa: E402


class HarvestDueTest(unittest.TestCase):
    def test_runs_when_stamp_changed(self):
        self.assertTrue(harvest_due("laughs:aaa", "laughs:bbb", True, "laughs"))

    def test_skips_when_stamp_same_and_pastes_present(self):
        self.assertFalse(harvest_due("laughs:aaa", "laughs:aaa", True, "laughs"))

    def test_runs_when_no_auto_pastes_even_with_same_stamp(self):
        # the e1v7 case: a prior --restore-sounds none left a stamp; the
        # laughs run must not be skipped while the dir has no pastes
        self.assertTrue(harvest_due("laughs:aaa", "laughs:aaa", False, "laughs"))

    def test_none_mode_does_not_rerun_for_missing_pastes(self):
        self.assertFalse(harvest_due("none:aaa", "none:aaa", False, "none"))

    def test_tail_is_importable_without_side_effects(self):
        self.assertTrue(callable(tail))


if __name__ == "__main__":
    unittest.main()
