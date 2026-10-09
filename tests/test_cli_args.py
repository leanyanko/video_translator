"""Unit tests for scripts/cli_args.py."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from cli_args import flag, opt  # noqa: E402


class TestOpt(unittest.TestCase):
    argv = ["x.py", "work/id", "--seed", "7", "--only", "1,2", "--gain", "-3", "--dry-run"]

    def test_value_cast_and_default(self):
        self.assertEqual(opt("--seed", 42, int, self.argv), 7)
        self.assertEqual(opt("--seedless", 42, int, self.argv), 42)
        self.assertEqual(opt("--only", None, lambda v: {int(x) for x in v.split(",")}, self.argv), {1, 2})
        self.assertEqual(opt("--gain", 0.0, float, self.argv), -3.0)  # negative values are values

    def test_flag(self):
        self.assertTrue(flag("--dry-run", self.argv))
        self.assertFalse(flag("--wet-run", self.argv))

    def test_missing_value_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            opt("--dry-run", None, None, self.argv)      # last arg, nothing follows
        with self.assertRaises(SystemExit):
            opt("--a", None, None, ["x.py", "--a", "--b"])  # next token is a flag
        with self.assertRaises(SystemExit):
            opt("--seed", 1, int, ["x.py", "--seed", "seven"])


if __name__ == "__main__":
    unittest.main()
