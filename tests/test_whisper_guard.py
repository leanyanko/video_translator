"""whisper_guard: the collapse detector/restart shared by both
transcribers, and the doctor's degradation measurement (TODO #12).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from whisper_guard import (PUNCT_WIN, STRETCH_SECS,  # noqa: E402
                           degraded_windows, first_poor, guarded_pass,
                           hallucination_loops)


def mk_words(t0, t1, punct_until=None, rate=2.0):
    """Synthetic word stream, `rate` words/sec; words after
    `punct_until` (absolute sec) carry no punctuation at all."""
    ws, t, i = [], t0, 0
    step = 1.0 / rate
    while t < t1:
        word = f"слово{i}"
        if (punct_until is None or t < punct_until) and i % 8 == 7:
            word += "."
        ws.append({"start": round(t, 3), "end": round(t + step * 0.8, 3),
                   "word": " " + word})
        t += step
        i += 1
    return ws


class FirstPoorTest(unittest.TestCase):
    def test_healthy_stream_passes(self):
        self.assertIsNone(first_poor(mk_words(0, 300)))

    def test_collapse_is_located(self):
        ws = mk_words(0, 400, punct_until=200)
        bad = first_poor(ws)
        self.assertIsNotNone(bad)
        # the flagged window starts near the collapse, not at the stream head
        self.assertGreater(ws[bad]["start"], 150)

    def test_short_stream_skipped(self):
        self.assertIsNone(first_poor(mk_words(0, 10)))


class GuardedPassTest(unittest.TestCase):
    def test_restart_recovers_punctuation(self):
        # model whisper's conditioning: each PASS derails 120 s after ITS
        # OWN start — a restart therefore recovers
        calls = []

        def run(a, b):
            calls.append(a)
            return [], mk_words(a, b, punct_until=a + 120)

        _, ws = guarded_pass(run, 0.0, 400.0, log=lambda *a, **k: None)
        self.assertGreater(len(calls), 1, "no restart happened")
        # the final stream has no 80-word window below 4% punctuation
        self.assertIsNone(first_poor(ws))

    def test_hopeless_pass_does_not_loop_forever(self):
        def run(a, b):  # collapsed from the very first word of every pass
            return [], mk_words(a, b, punct_until=a)

        _, ws = guarded_pass(run, 0.0, 300.0, log=lambda *a, **k: None)
        self.assertTrue(ws)  # kept the words it had, exited

    def test_stretched_word_is_spliced(self):
        base = mk_words(0, 60)
        stretched = dict(base[50], end=base[50]["start"] + 8.0,
                         word=" растянутое.")
        broken = base[:50] + [stretched]

        def run(a, b):
            if a == 0.0:  # the full pass returns the broken stream
                return [], broken
            return [], [w for w in base if a - 0.05 <= w["start"] < b]

        _, ws = guarded_pass(run, 0.0, 60.0, log=lambda *a, **k: None)
        self.assertFalse([w for w in ws
                          if w["end"] - w["start"] > STRETCH_SECS])
        self.assertGreater(len(ws), len(broken))


class DegradedWindowsTest(unittest.TestCase):
    def test_collapsed_window_flagged(self):
        ws = mk_words(0, 300) + mk_words(300, 600, punct_until=300)
        bad = degraded_windows(ws, win_secs=300)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["t0"], 300)
        self.assertIn("punctuation", bad[0]["reason"])
        self.assertEqual(bad[0]["severity"], "fail")

    def test_sparse_window_is_silence_not_degradation(self):
        # 20 words in 5 minutes = island gap, not a collapse
        ws = mk_words(0, 10, rate=2.0) + mk_words(590, 600, rate=2.0)
        self.assertEqual(degraded_windows(ws, win_secs=300), [])

    def test_stretched_words_flagged(self):
        ws = mk_words(0, 300)
        for i in (10, 20, 30):
            ws[i] = dict(ws[i], end=ws[i]["start"] + 5.0)
        bad = degraded_windows(ws, win_secs=300)
        self.assertEqual(len(bad), 1)
        self.assertIn("stretched", bad[0]["reason"])
        self.assertEqual(bad[0]["severity"], "warn")  # calibrated on e1v6

    def test_hallucination_tempo_flagged(self):
        ws = mk_words(0, 300, rate=5.0)  # 300 words/min, punctuated
        bad = degraded_windows(ws, win_secs=300)
        self.assertEqual(len(bad), 1)
        self.assertIn("words/min", bad[0]["reason"])


class LoopTest(unittest.TestCase):
    def test_loop_detected(self):
        ws = mk_words(0, 30)
        phrase = [" так,", " и", " вот."]
        for i, w in enumerate(ws[:18]):
            w["word"] = phrase[i % 3]
        hits = hallucination_loops(ws)
        self.assertTrue(hits)
        self.assertEqual(hits[0][1], "так и вот")

    def test_normal_speech_clean(self):
        self.assertEqual(hallucination_loops(mk_words(0, 300)), [])

    def test_filler_triples_are_real_speech(self):
        # people DO say «yeah yeah yeah» and «thanks thanks thanks»
        for word in (" yeah", " thanks.", " ha,"):
            ws = mk_words(0, 10)
            for w in ws:
                w["word"] = word
            self.assertEqual(hallucination_loops(ws), [], word)

    def test_mixed_filler_loop_still_detected(self):
        # a filler inside a non-filler 3-gram does not grant immunity
        ws = mk_words(0, 10)
        phrase = [" easy", " yellaris", " wah"]
        for i, w in enumerate(ws):
            w["word"] = phrase[i % 3]
        self.assertTrue(hallucination_loops(ws))


if __name__ == "__main__":
    unittest.main()
