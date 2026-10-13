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
        self.assertIn("M5a rules", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
