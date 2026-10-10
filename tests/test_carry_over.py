"""carry_over manifest entries (TODO #5 + review 10-10): refs are
copied over when the new dir has none (otherwise the standard flow —
carry at the translation gate, refs cut later — wrote ZERO entries and
every copied clip re-sang), and an entry is written only when the OLD
run's manifest confirms the clip was sung from that text with that ref.
"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from carry_over import copy_refs_if_missing, emit_manifest, refsha  # noqa: E402
from manifest_bridge import new_hash  # noqa: E402


def make_old(tmp, acc, manifest, ref=b"REFBYTES"):
    old = Path(tmp) / "old"
    (old / "refs_f5").mkdir(parents=True)
    (old / "refs_f5" / "SPEAKER_00.wav").write_bytes(ref)
    old.joinpath("accents.json").write_text(json.dumps(acc, ensure_ascii=False))
    (old / "tts_f5").mkdir()
    if manifest is not None:
        (old / "tts_f5" / "manifest.json").write_text(json.dumps(manifest))
    return old


def make_new(tmp, ref=None):
    new = Path(tmp) / "new"
    (new / "tts_f5").mkdir(parents=True)
    if ref is not None:
        (new / "refs_f5").mkdir()
        (new / "refs_f5" / "SPEAKER_00.wav").write_bytes(ref)
    return new


MAPPING = [({"id": 7, "speaker": "SPEAKER_00"},
            {"id": 3, "speaker": "SPEAKER_00", "text_ru": "текст"})]


class CarryManifestTest(unittest.TestCase):
    def entry(self, old):
        return new_hash("SPEAKER_00", "т+екст", refsha(old, "SPEAKER_00"))

    def test_normal_flow_copies_refs_and_emits(self):
        # the standard order: carry runs BEFORE make_refs, new has no refs
        with TemporaryDirectory() as tmp:
            old = make_old(tmp, {"3": "т+екст"}, None)
            (old / "tts_f5" / "manifest.json").write_text(
                json.dumps({"3": self.entry(old)}))
            new = make_new(tmp, ref=None)
            self.assertTrue(copy_refs_if_missing(new, old))
            self.assertTrue((new / "refs_f5" / "SPEAKER_00.wav").exists())
            n = emit_manifest(new, old, MAPPING)
            self.assertEqual(n, 1)
            m = json.loads((new / "tts_f5" / "manifest.json").read_text())
            self.assertEqual(m["7"], self.entry(old))

    def test_existing_refs_are_not_overwritten(self):
        with TemporaryDirectory() as tmp:
            old = make_old(tmp, {}, {})
            new = make_new(tmp, ref=b"OTHERVOICE")
            self.assertFalse(copy_refs_if_missing(new, old))
            self.assertEqual((new / "refs_f5" / "SPEAKER_00.wav").read_bytes(),
                             b"OTHERVOICE")

    def test_old_manifest_disagreement_skips_entry(self):
        # the old clip was NOT sung from the accents text -> re-sing
        with TemporaryDirectory() as tmp:
            old = make_old(tmp, {"3": "т+екст"},
                           {"3": new_hash("SPEAKER_00", "другой текст",
                                          "somerefsha123")})
            new = make_new(tmp, ref=b"REFBYTES")
            self.assertEqual(emit_manifest(new, old, MAPPING), 0)

    def test_no_old_manifest_means_no_entries(self):
        with TemporaryDirectory() as tmp:
            old = make_old(tmp, {"3": "т+екст"}, None)
            new = make_new(tmp, ref=b"REFBYTES")
            self.assertEqual(emit_manifest(new, old, MAPPING), 0)

    def test_differing_ref_skips_speaker(self):
        with TemporaryDirectory() as tmp:
            old = make_old(tmp, {"3": "т+екст"}, {})
            (old / "tts_f5" / "manifest.json").write_text(
                json.dumps({"3": self.entry(old)}))
            new = make_new(tmp, ref=b"NEWVOICE")  # re-cut reference
            self.assertEqual(emit_manifest(new, old, MAPPING), 0)


if __name__ == "__main__":
    unittest.main()
