"""Smoke tests for the pipeline runner."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline  # noqa: E402
from m4_features import N_FEATURES  # noqa: E402
from m5_scorer import DATA_PATH  # noqa: E402

SAMPLE = (
    "The cat sat on the mat. It was a warm afternoon, and the dog slept. "
    "However, the kitchen door stayed open all day."
)
# Over pipeline.MIN_WORDS, so the scorers give a verdict.
LONG_SAMPLE = (
    "The proposed method improves segmentation accuracy on noisy images. "
    "Although the results are promising, we note that the training cost "
    "remains high. Using a small graph, we show that the approach scales "
    "to larger inputs without retuning. We also compare against three "
    "baselines on two public benchmarks and report where each one fails, "
    "with a short discussion of the remaining errors."
)


class TestRunPipeline(unittest.TestCase):
    def test_core_stage_order(self):
        names = [name.split()[0] for name, _, _ in pipeline.CORE_STAGES]
        self.assertEqual(names, ["M1", "M2", "M3", "M4", "M5"])

    def test_ctx_holds_every_stage_result(self):
        ctx = pipeline.run_pipeline(SAMPLE, show=False)
        for key in ("text", "tokens", "sym_tab", "meta", "features", "rules"):
            self.assertIn(key, ctx)
        self.assertEqual(len(ctx["features"]), N_FEATURES)

    def test_rule_score_is_a_probability(self):
        ctx = pipeline.run_pipeline(SAMPLE, show=False)
        self.assertGreaterEqual(ctx["rules"]["score"], 0.0)
        self.assertLessEqual(ctx["rules"]["score"], 1.0)

    def test_empty_text_does_not_crash(self):
        ctx = pipeline.run_pipeline("", show=False)
        self.assertEqual(ctx["tokens"], [])
        self.assertEqual(len(ctx["features"]), N_FEATURES)

    def test_show_false_prints_nothing(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            pipeline.run_pipeline(SAMPLE, show=False)
        self.assertEqual(buffer.getvalue(), "")

    def test_main_prints_every_stage(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            pipeline.main(["--text", SAMPLE])
        out = buffer.getvalue()
        for name, _, _ in pipeline.CORE_STAGES:
            self.assertIn(f"== {name} ==", out)


class TestMinWordsAbstain(unittest.TestCase):
    def test_short_text_abstains_in_both_scorers(self):
        ctx = pipeline.run_pipeline(SAMPLE, show=False)
        self.assertLess(ctx["n_words"], pipeline.MIN_WORDS)
        self.assertTrue(ctx["abstain"])
        self.assertEqual(ctx["verdicts"]["M5a"], "abstain")
        self.assertIn(ctx["verdicts"]["M5b"], ("abstain", None))

    def test_long_text_gets_a_verdict(self):
        ctx = pipeline.run_pipeline(LONG_SAMPLE, show=False)
        self.assertGreaterEqual(ctx["n_words"], pipeline.MIN_WORDS)
        self.assertFalse(ctx["abstain"])
        self.assertIn(ctx["verdicts"]["M5a"], ("ai", "human"))
        self.assertIn(ctx["verdicts"]["M5b"], ("ai", "human", None))

    def test_boundary_is_min_words_inclusive(self):
        at = " ".join(["word"] * pipeline.MIN_WORDS) + "."
        below = " ".join(["word"] * (pipeline.MIN_WORDS - 1)) + "."
        self.assertFalse(pipeline.run_pipeline(at, show=False)["abstain"])
        self.assertTrue(pipeline.run_pipeline(below, show=False)["abstain"])

    def test_hashes_and_urls_count_as_one_word_each(self):
        text = ("Fixed the flaky build, see commit "
                "9f2c4e1a7b3d5f6e8a0c2b4d6f8e1a3c5b7d9f0e and "
                "https://ci.example.com/job/42/run/7/log-2a3b4c5d.")
        ctx = pipeline.run_pipeline(text, show=False)
        self.assertEqual(ctx["n_words"], 9)
        self.assertTrue(ctx["abstain"])

    def test_pieces_without_letters_do_not_count(self):
        ctx = pipeline.run_pipeline("> > 42 1,000 -- word", show=False)
        self.assertEqual(ctx["n_words"], 1)

    def test_built_in_sample_gets_a_verdict(self):
        ctx = pipeline.run_pipeline(pipeline.SAMPLE, show=False)
        self.assertFalse(ctx["abstain"])

    def test_empty_text_abstains(self):
        ctx = pipeline.run_pipeline("", show=False)
        self.assertEqual(ctx["n_words"], 0)
        self.assertEqual(ctx["verdicts"]["M5a"], "abstain")

    def test_short_text_prints_no_verdict(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            pipeline.main(["--text", "I went to the store yesterday."])
        out = buffer.getvalue()
        self.assertIn("abstain: only 6 words", out)
        self.assertNotIn("M5a rule score", out)
        self.assertNotIn("M5b model score", out)

    def test_long_text_prints_a_verdict(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            pipeline.main(["--text", LONG_SAMPLE])
        out = buffer.getvalue()
        self.assertIn("M5a rule score", out)
        self.assertNotIn("abstain:", out)


class TestExtensionPoint(unittest.TestCase):
    def test_an_added_stage_runs_last_and_sees_earlier_results(self):
        def run_extra(ctx):
            ctx["extra"] = ctx["rules"]["score"]

        stages = pipeline.CORE_STAGES + (("M6 extra", run_extra, lambda c: None),)
        ctx = pipeline.run_pipeline(SAMPLE, stages=stages, show=False)
        self.assertEqual(ctx["extra"], ctx["rules"]["score"])

    def test_default_stages_start_as_core(self):
        self.assertEqual(tuple(pipeline.STAGES), pipeline.CORE_STAGES)


@unittest.skipUnless(Path(DATA_PATH).exists(), "dataset not generated")
class TestTestSplitRow(unittest.TestCase):
    def test_returns_text_and_a_known_label(self):
        text, label = pipeline.test_split_row(0)
        self.assertTrue(text.strip())
        self.assertIn(label, ("ai", "human"))

    def test_out_of_range_exits(self):
        with self.assertRaises(SystemExit):
            pipeline.test_split_row(10_000)

    def test_eval_reports_both_scorers(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            pipeline.main(["--eval"])
        out = buffer.getvalue()
        self.assertIn("held-out test split: 190 rows", out)
        self.assertIn(f"abstained (under {pipeline.MIN_WORDS} words)", out)
        self.assertIn("M5a rules", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
