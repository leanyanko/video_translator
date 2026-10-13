"""reroll_qc: which clips to re-sing and when a new rendering wins (d6)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from reroll_qc import SIM_REROLL, better, clean, pick  # noqa: E402


class PickTest(unittest.TestCase):
    QC = {"1": {"sim": 1.0, "flags": []},
          "9": {"sim": 0.80, "flags": []},            # «проб овали» — under 0.9, no flag
          "10": {"sim": 0.0, "flags": ["MISMATCH"]},
          "17": {"sim": 0.93, "flags": []}}

    def test_flagged_and_low_sim_are_picked_worst_first(self):
        self.assertEqual(pick(self.QC), ["10", "9"])

    def test_threshold_is_configurable(self):
        self.assertEqual(pick(self.QC, 0.95), ["10", "9", "17"])


class BetterTest(unittest.TestCase):
    def test_fewer_flags_beats_higher_sim(self):
        self.assertTrue(better({"sim": 0.7, "flags": []}, {"sim": 0.9, "flags": ["TAIL"]}))

    def test_same_flags_higher_sim_wins(self):
        self.assertTrue(better({"sim": 1.0, "flags": []}, {"sim": 0.8, "flags": []}))
        self.assertFalse(better({"sim": 0.8, "flags": []}, {"sim": 1.0, "flags": []}))

    def test_clean(self):
        self.assertTrue(clean({"sim": SIM_REROLL, "flags": []}, SIM_REROLL))
        self.assertFalse(clean({"sim": 1.0, "flags": ["LEAD"]}, SIM_REROLL))


if __name__ == "__main__":
    unittest.main()
