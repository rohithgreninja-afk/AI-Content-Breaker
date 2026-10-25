"""Smoke tests for the pipeline runner."""

from __future__ import annotations

import io
import csv
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
        for name, _, _ in pipeline.STAGES:
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

    def test_m6_is_the_last_stage_and_core_stays_the_scorer(self):
        self.assertEqual(tuple(pipeline.STAGES[:5]), pipeline.CORE_STAGES)
        self.assertEqual(pipeline.STAGES[5][0], "M6 robustness")
        self.assertEqual(
            [name.split()[0] for name, _, _ in pipeline.CORE_STAGES],
            ["M1", "M2", "M3", "M4", "M5"],
        )


class TestSavedHoldout(unittest.TestCase):
    """Evaluation must never invent a split for a stale or unrelated model."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "train_clean.csv"
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "label", "text"])
            writer.writeheader()
            writer.writerows([
                {"id": "a", "label": "human", "text": "training text"},
                {"id": "b", "label": "ai", "text": "test text"},
            ])
        self.metadata = {"dataset_sha256": "current", "train_ids": ["a"],
                         "test_ids": ["b"]}
        self.provenance = {"dataset_sha256": "current", "row_ids": ["a", "b"],
                           "groups": ["group-a", "group-b"]}
        for mock in (
            patch.object(pipeline, "DATA_PATH", str(self.path)),
            patch.object(pipeline, "MODEL_PATH", str(self.path)),
            patch.object(pipeline.ModelScorer, "load",
                         return_value=SimpleNamespace(metadata=self.metadata)),
            patch.object(pipeline, "dataset_metadata", return_value=self.provenance),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def test_uses_saved_test_ids(self):
        self.assertEqual([r["id"] for r in pipeline._test_rows()], ["b"])

    def test_rejects_changed_dataset(self):
        self.metadata["dataset_sha256"] = "old"
        with self.assertRaisesRegex(ValueError, "does not match"):
            pipeline._test_rows()

    def test_rejects_legacy_artifact_without_provenance(self):
        self.metadata.clear()
        with self.assertRaisesRegex(ValueError, "does not match"):
            pipeline._test_rows()

    def test_rejects_train_test_overlap(self):
        self.metadata["test_ids"] = ["a"]
        with self.assertRaisesRegex(ValueError, "invalid split"):
            pipeline._test_rows()

    def test_rejects_repeated_saved_test_ids(self):
        self.metadata["test_ids"] = ["b", "b"]
        with self.assertRaisesRegex(ValueError, "invalid split"):
            pipeline._test_rows()

    def test_rejects_changed_group_manifest(self):
        self.metadata["manifest_sha256"] = "original"
        self.provenance["manifest_sha256"] = "edited"
        with self.assertRaisesRegex(ValueError, "dataset manifest"):
            pipeline._test_rows()

    def test_scores_with_the_validated_instance(self):
        self.addCleanup(pipeline._MODEL.clear)
        pipeline._MODEL["model"] = ("stale", "")
        pipeline._test_rows()
        self.assertEqual(pipeline._model()[0].metadata, self.metadata)

    def test_rejects_related_texts_across_split(self):
        self.provenance["groups"] = ["same-question", "same-question"]
        with self.assertRaisesRegex(ValueError, "related texts"):
            pipeline._test_rows()


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
        n_test = len(pipeline._test_rows())
        self.assertIn(f"held-out test split: {n_test} rows", out)
        self.assertIn(f"abstained (under {pipeline.MIN_WORDS} words)", out)
        self.assertIn("M5a rules", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
