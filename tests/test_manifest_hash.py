"""ref_hasher / seg_hash in scripts/manifest_bridge.py — the one hash scheme
synthesize_f5.py, check_pipeline.py and the bridge must all agree on."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from manifest_bridge import new_hash, ref_hasher, seg_hash  # noqa: E402


class TestHashes(unittest.TestCase):
    def test_ref_hasher_matches_file_and_caches(self):
        with tempfile.TemporaryDirectory() as td:
            refs = Path(td)
            (refs / "S0.wav").write_bytes(b"abc")
            rs = ref_hasher(refs)
            self.assertEqual(rs("S0"), hashlib.sha1(b"abc").hexdigest()[:12])
            self.assertEqual(rs("S9"), "noref")
            (refs / "S0.wav").write_bytes(b"changed")
            self.assertEqual(rs("S0"), hashlib.sha1(b"abc").hexdigest()[:12])  # cached per comparison
            self.assertNotEqual(ref_hasher(refs)("S0"), rs("S0"))            # a new hasher sees the change

    def test_seg_hash_uses_accented_text_then_text_ru(self):
        rs = lambda spk: "r" + spk  # noqa: E731
        seg = {"id": 5, "speaker": "S0", "text_ru": "plain"}
        self.assertEqual(seg_hash(seg, {"5": "acc+ented"}, rs), new_hash("S0", "acc+ented", "rS0"))
        self.assertEqual(seg_hash(seg, {}, rs), new_hash("S0", "plain", "rS0"))
        self.assertEqual(new_hash("S0", "t", "r"), hashlib.sha1(b"S0|t|r").hexdigest())


if __name__ == "__main__":
    unittest.main()
