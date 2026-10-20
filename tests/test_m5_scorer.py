"""Smoke tests for M5: rule scorer and logistic-regression classifier.

The model tests train on a small synthetic set so they stay fast and do not
depend on data/train_clean.csv (which is gitignored and regenerable).
"""

from __future__ import annotations

import math
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m4_features import FEATURE_NAMES, extract_features, to_vector  # noqa: E402
from m5_scorer import (  # noqa: E402
    AI_LABEL,
    HUMAN_LABEL,
    RULES,
    ModelScorer,
    RuleScorer,
    _mean_squared,
    _cv_indices,
    _print_model_report,
    _rule_fires,
    audit_rules,
    calibrate_rules,
    dataset_metadata,
    dataset_sha256,
    evaluate_rules,
    split_indices,
)

# Two deliberately different registers. Not a benchmark, just enough signal
# for the plumbing tests to have something to separate.
HUMANISH = (
    "So I went down to the shop - the one by the bridge - and it was shut. "
    "Again! Third time this week, honestly. The sign said \"back in 10\", "
    "which, sure, fine, but I'd already waited twenty. I gave up and walked."
)
AIISH = (
    "The implementation of comprehensive data management strategies provides "
    "significant advantages for modern organizations. These methodologies "
    "enable improved operational efficiency across multiple departments. "
    "Furthermore, the integration of automated systems facilitates enhanced "
    "decision making processes throughout the organizational structure."
)


def _synthetic_dataset(n: int = 24):
    """Alternating labelled feature dicts.

    Each row gets a different tail so the feature vectors are not duplicates;
    identical rows make every scorer look perfect and hide real behaviour.
    """
    tails_ai = (
        " Result {i} confirms the stated improvement across the dataset.",
        " Additionally, metric {i} demonstrates substantial optimization.",
        " The framework consequently delivers measurable value at scale {i}.",
    )
    tails_human = (
        " Anyway, {i} tries later, still nothing. Typical.",
        " I waited {i} minutes, gave up, got chips instead.",
        " Told my mate about it; he laughed, said \"{i} quid says it's shut\".",
    )
    features, labels = [], []
    for i in range(n):
        if i % 2:
            text = AIISH + tails_ai[i % len(tails_ai)].format(i=i)
            labels.append(AI_LABEL)
        else:
            text = HUMANISH + tails_human[i % len(tails_human)].format(i=i)
            labels.append(HUMAN_LABEL)
        features.append(extract_features(text))
    return features, labels


class TestRuleDefinitions(unittest.TestCase):
    """The RULES table itself must stay well formed."""

    def test_every_rule_names_a_real_feature(self) -> None:
        for name, _, _, _ in RULES:
            self.assertIn(name, FEATURE_NAMES, name)

    def test_directions_and_weights_are_valid(self) -> None:
        for name, direction, threshold, weight in RULES:
            self.assertIn(direction, ("high", "low"), name)
            self.assertIsInstance(threshold, float, name)
            self.assertGreater(weight, 0.0, name)

    def test_no_duplicate_rules(self) -> None:
        keys = [(name, direction) for name, direction, _, _ in RULES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_unknown_feature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RuleScorer(rules=(("not_a_feature", "high", 1.0, 1.0),))

    def test_rule_fires_boundaries(self) -> None:
        self.assertTrue(_rule_fires(0.5, "high", 0.5))   # inclusive
        self.assertFalse(_rule_fires(0.49, "high", 0.5))
        self.assertTrue(_rule_fires(0.5, "low", 0.5))    # inclusive
        self.assertFalse(_rule_fires(0.51, "low", 0.5))
        with self.assertRaises(ValueError):
            _rule_fires(0.5, "sideways", 0.5)


class TestRuleScorer(unittest.TestCase):
    """M5a behaviour."""

    def setUp(self) -> None:
        self.scorer = RuleScorer()

    def test_score_is_a_probability(self) -> None:
        for text in (HUMANISH, AIISH, "", "   ", "Short."):
            score = self.scorer.score(text)
            self.assertGreaterEqual(score, 0.0, repr(text))
            self.assertLessEqual(score, 1.0, repr(text))
            self.assertFalse(math.isnan(score), repr(text))

    def test_explain_accounts_for_every_rule(self) -> None:
        explanation = self.scorer.explain(extract_features(AIISH))
        self.assertEqual(
            len(explanation["fired"]) + len(explanation["missed"]),
            len(self.scorer.rules))
        # The signed total is the sum of the contributions.
        contributions = sum(
            row[3] for row in explanation["fired"] + explanation["missed"])
        self.assertAlmostEqual(contributions, explanation["total"], places=9)
        self.assertAlmostEqual(explanation["max_total"],
                               self.scorer.max_total, places=9)

    def test_score_matches_explain(self) -> None:
        features = extract_features(AIISH)
        self.assertEqual(self.scorer.score_features(features),
                         self.scorer.explain(features)["score"])

    def test_all_rules_firing_scores_above_half(self) -> None:
        """A text that fires everything must land clearly on the AI side."""
        total = self.scorer.max_total
        score = 1.0 / (1.0 + math.exp(
            -self.scorer.scale * (total + self.scorer.bias)))
        self.assertGreater(score, 0.5)
        # And nothing firing must land clearly on the human side.
        score_none = 1.0 / (1.0 + math.exp(
            -self.scorer.scale * (-total + self.scorer.bias)))
        self.assertLess(score_none, 0.5)

    def test_bias_moves_the_decision_point(self) -> None:
        features = extract_features(HUMANISH)
        low = RuleScorer(bias=-50.0).score_features(features)
        high = RuleScorer(bias=50.0).score_features(features)
        self.assertLess(low, 0.01)
        self.assertGreater(high, 0.99)

    def test_scale_cannot_flip_a_prediction(self) -> None:
        """Scale is calibration only: every scale > 0 keeps the same side."""
        features = extract_features(AIISH)
        sides = {RuleScorer(scale=s).score_features(features) >= 0.5
                 for s in (0.05, 0.2, 0.5, 2.0)}
        self.assertEqual(len(sides), 1)

    def test_predict_returns_a_label(self) -> None:
        self.assertIn(self.scorer.predict(AIISH), (AI_LABEL, HUMAN_LABEL))
        self.assertEqual(self.scorer.predict(AIISH, cutoff=0.0), AI_LABEL)
        self.assertEqual(self.scorer.predict(AIISH, cutoff=1.01), HUMAN_LABEL)

    def test_is_deterministic(self) -> None:
        self.assertEqual(self.scorer.score(AIISH), self.scorer.score(AIISH))


class TestRuleEvaluation(unittest.TestCase):
    """evaluate_rules / audit_rules / calibrate_rules."""

    def setUp(self) -> None:
        self.features, self.labels = _synthetic_dataset()

    def test_evaluate_reports_bounded_metrics(self) -> None:
        result = evaluate_rules(self.features, self.labels)
        self.assertEqual(result["n"], len(self.labels))
        self.assertEqual(len(result["scores"]), len(self.labels))
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertGreaterEqual(result[key], 0.0, key)
            self.assertLessEqual(result[key], 1.0, key)

    def test_evaluate_handles_an_empty_set(self) -> None:
        result = evaluate_rules([], [])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["precision"], 0.0)

    def test_audit_covers_every_rule(self) -> None:
        rows = audit_rules(self.features, self.labels)
        self.assertEqual(len(rows), len(RULES))
        self.assertEqual({row["feature"] for row in rows},
                         {name for name, _, _, _ in RULES})
        for row in rows:
            self.assertAlmostEqual(row["lift"],
                                   row["ai_rate"] - row["human_rate"],
                                   places=9)
        # Sorted by lift, descending.
        lifts = [row["lift"] for row in rows]
        self.assertEqual(lifts, sorted(lifts, reverse=True))

    def test_calibrate_returns_a_best_from_its_grid(self) -> None:
        calibration = calibrate_rules(self.features, self.labels)
        self.assertIn(calibration["best"], calibration["grid"])
        self.assertIn(calibration["best_bias"], calibration["bias_grid"])
        # Brier is what picks the scale.
        self.assertEqual(calibration["best"]["brier"],
                         min(row["brier"] for row in calibration["grid"]))

    def test_accuracy_is_invariant_to_scale(self) -> None:
        """Scale cannot move a label, because it is monotonic through 0.5.

        Uses an off-grid bias on purpose: at a bias that exactly cancels a
        rule total, the signed total lands ~1e-16 from the boundary and float
        rounding inside exp() decides the side. That tie is inherent to any
        squashing function, so the invariance claim is about non-tied scores.
        """
        base = RuleScorer(bias=0.37)
        accuracies = set()
        for scale in (0.05, 0.1, 0.2, 0.42, 0.8, 2.0):
            scorer = RuleScorer(base.rules, base.bias, scale)
            accuracies.add(
                evaluate_rules(self.features, self.labels, scorer)["accuracy"])
        self.assertEqual(len(accuracies), 1)

    def test_exact_boundary_scores_one_half(self) -> None:
        """A total that exactly cancels the bias sits on the cutoff."""
        scorer = RuleScorer(rules=(("sent_cv_len", "low", 0.5, 1.0),), bias=1.0)
        # One rule of weight 1.0: missing it gives total -1.0, +bias 1.0 -> 0.
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features["sent_cv_len"] = 0.9  # above the threshold, so it misses
        explanation = scorer.explain(features)
        self.assertEqual(explanation["total"], -1.0)
        self.assertEqual(explanation["score"], 0.5)

    def test_brier_score(self) -> None:
        self.assertEqual(_mean_squared([1.0, 0.0], [1, 0]), 0.0)
        self.assertEqual(_mean_squared([0.0, 1.0], [1, 0]), 1.0)
        self.assertAlmostEqual(_mean_squared([0.5, 0.5], [1, 0]), 0.25)
        self.assertEqual(_mean_squared([], []), 0.0)

    def test_split_is_stratified_and_disjoint(self) -> None:
        train, test = split_indices(self.labels)
        self.assertEqual(set(train) & set(test), set())
        self.assertEqual(len(train) + len(test), len(self.labels))
        ai_share = sum(1 for i in test if self.labels[i] == AI_LABEL) / len(test)
        self.assertAlmostEqual(ai_share, 0.5, delta=0.2)


class TestModelScorer(unittest.TestCase):
    """M5b behaviour, on the synthetic set."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.features, cls.labels = _synthetic_dataset(40)
        cls.vectors = [to_vector(f) for f in cls.features]
        cls.model = ModelScorer()
        cls.report = cls.model.fit(cls.vectors, cls.labels)

    def test_report_shape(self) -> None:
        self.assertEqual(self.report["n_features"], len(FEATURE_NAMES))
        self.assertEqual(self.report["n_train"] + self.report["n_test"],
                         len(self.labels))
        self.assertEqual(self.report["C"], ModelScorer.DEFAULT_C)
        for key in ("train_accuracy", "test_accuracy", "test_roc_auc",
                    "cv_mean"):
            self.assertGreaterEqual(self.report[key], 0.0, key)
            self.assertLessEqual(self.report[key], 1.0, key)
        self.assertEqual(len(self.report["confusion_matrix"]), 2)
        self.assertIn("precision", self.report["report"])

    def test_scores_are_probabilities(self) -> None:
        for text in (HUMANISH, AIISH, "", "Short."):
            score = self.model.score(text)
            self.assertGreaterEqual(score, 0.0, repr(text))
            self.assertLessEqual(score, 1.0, repr(text))

    def test_separates_the_two_registers(self) -> None:
        """Sanity check that fitting did something, not a benchmark."""
        self.assertGreater(self.model.score(AIISH), self.model.score(HUMANISH))

    def test_predict_returns_a_label(self) -> None:
        self.assertIn(self.model.predict(AIISH), (AI_LABEL, HUMAN_LABEL))

    def test_top_coefficients(self) -> None:
        top = self.model.top_coefficients(7)
        self.assertEqual(len(top), 7)
        for name, weight in top:
            self.assertIn(name, FEATURE_NAMES)
            self.assertIsInstance(weight, float)
        # Sorted by descending magnitude.
        magnitudes = [abs(w) for _, w in top]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_unfitted_model_refuses_to_score(self) -> None:
        blank = ModelScorer()
        with self.assertRaises(RuntimeError):
            blank.score(AIISH)
        with self.assertRaises(RuntimeError):
            blank.top_coefficients()
        with self.assertRaises(RuntimeError):
            blank.save(os.devnull)

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m5.joblib")
            self.model.save(path)
            self.assertTrue(os.path.exists(path))
            reloaded = ModelScorer.load(path)
            self.assertAlmostEqual(reloaded.score(AIISH),
                                   self.model.score(AIISH), places=9)

    def test_load_rejects_a_mismatched_feature_order(self) -> None:
        """A model saved against a different M4 must not be scored with."""
        import joblib

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stale.joblib")
            joblib.dump({"pipeline": self.model.pipeline,
                         "feature_names": FEATURE_NAMES[:-1]}, path)
            with self.assertRaises(ValueError):
                ModelScorer.load(path)

    def test_select_C_searches_its_grid(self) -> None:
        result = ModelScorer.select_C(self.vectors, self.labels,
                                      grid=(0.1, 1.0, 10.0))
        self.assertIn(result["best_C"], (0.1, 1.0, 10.0))
        self.assertEqual(len(result["grid"]), 3)
        self.assertEqual(result["best_cv"],
                         max(row[1] for row in result["grid"]))

    def test_grouped_fit_keeps_pairs_out_of_train_and_cv(self) -> None:
        import numpy as np

        groups = [f"question-{i // 2}" for i in range(len(self.labels))]
        model = ModelScorer()
        report = model.fit(self.vectors, self.labels, groups=groups)
        train, test = report["train_indices"], report["test_indices"]
        self.assertFalse({groups[i] for i in train} & {groups[i] for i in test})
        self.assertEqual((train, test), split_indices(self.labels, groups=groups))
        # Scaling sees the training rows only, even when a held-out value is extreme.
        X = np.asarray(self.vectors)
        np.testing.assert_allclose(model.pipeline.named_steps["scale"].mean_,
                                   X[train].mean(axis=0))
        y = np.asarray([label == AI_LABEL for label in self.labels])[train]
        train_groups = np.asarray(groups)[train]
        for fit, validation in _cv_indices(X[train], y, train_groups):
            self.assertFalse(set(train_groups[fit]) & set(train_groups[validation]))
        self.assertEqual(len(report["cv_scores"]), 5)

    def test_load_rejects_incompatible_tagger_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m5.joblib")
            model = ModelScorer(self.model.pipeline, metadata={"tagger": "nltk"})
            model.save(path)
            with patch("m5_scorer.tagger_mode", return_value="fallback"):
                with self.assertRaisesRegex(ValueError, "requires nltk"):
                    ModelScorer.load(path)
            model.metadata = {"schema_version": 999}
            model.save(path)
            with self.assertRaisesRegex(ValueError, "schema"):
                ModelScorer.load(path)

    def test_metadata_survives_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m5.joblib")
            metadata = {"dataset_sha256": "fingerprint", "test_ids": ["a", "b"]}
            ModelScorer(self.model.pipeline, metadata=metadata).save(path)
            self.assertEqual(ModelScorer.load(path).metadata, metadata)

    def test_load_rejects_changed_feature_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m5.joblib")
            model = ModelScorer(self.model.pipeline,
                                metadata={"feature_extractor_sha256": "stale"})
            model.save(path)
            with self.assertRaisesRegex(ValueError, "extractor changed"):
                ModelScorer.load(path)

    def test_cli_report_binds_saved_model_to_exact_training_rows(self) -> None:
        import io
        from contextlib import redirect_stdout

        groups = [f"pair-{i // 2}" for i in range(len(self.labels))]
        rows = [{"id": f"row-{i}", "domain": "test:domain", "label": label,
                 "text": "word " * (20 if i % 4 < 2 else 60)}
                for i, label in enumerate(self.labels)]
        provenance = {"dataset_sha256": "csv-hash", "manifest_sha256": "manifest-hash",
                      "groups": groups, "dataset_sources": {"test": {"revision": "pinned"}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m5.joblib")
            with patch("m5_scorer.MODEL_PATH", path), redirect_stdout(io.StringIO()):
                # save()'s default path is bound at definition; supply the test path explicitly.
                original_save = ModelScorer.save
                with patch.object(ModelScorer, "save", lambda model: original_save(model, path)):
                    report = _print_model_report(self.features, self.labels, save=True,
                                                 provenance=provenance, rows=rows)
            saved = ModelScorer.load(path)
            self.assertEqual(saved.metadata["dataset_sha256"], "csv-hash")
            self.assertEqual(saved.metadata["test_ids"],
                             [rows[i]["id"] for i in report["test_indices"]])
            self.assertFalse(set(saved.metadata["train_groups"]) &
                             set(saved.metadata["test_groups"]))
            with open(os.path.join(tmp, "training_report.json"), encoding="utf-8") as handle:
                record = json.load(handle)
            evaluation = record["evaluation"]["overall"]
            self.assertEqual(evaluation["n_scored"] + evaluation["n_abstained"],
                             report["n_test"])
            self.assertEqual(record["test_source_counts"], {"test": report["n_test"]})
            self.assertIn("scikit-learn", record["versions"])


class TestDatasetProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "train_clean.csv")
        with open(self.path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "model", "label", "domain", "text"])
            writer.writerow(["a", "human", "human", "hc3:test", HUMANISH])
            writer.writerow(["b", "chatgpt", "ai", "hc3:test", AIISH])
        self.manifest = {"csv_sha256": dataset_sha256(self.path),
                         "row_groups": {"a": "same-question", "b": "same-question"},
                         "sources": {"hc3": {"revision": "pinned"}}}

    def write_manifest(self):
        with open(os.path.join(self.tmp.name, "train_manifest.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle)

    def test_manifest_preserves_csv_group_order(self):
        self.write_manifest()
        metadata = dataset_metadata(self.path)
        self.assertEqual(metadata["row_ids"], ["a", "b"])
        self.assertEqual(metadata["groups"], ["same-question"] * 2)
        self.assertEqual(metadata["dataset_sources"], self.manifest["sources"])

    def test_changed_dataset_rejects_stale_manifest(self):
        self.write_manifest()
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "differs from its manifest"):
            dataset_metadata(self.path)

    def test_manifest_digest_ignores_platform_newlines_and_json_layout(self):
        self.write_manifest()
        expected = dataset_metadata(self.path)["manifest_sha256"]
        path = os.path.join(self.tmp.name, "train_manifest.json")
        with open(path, "w", encoding="utf-8", newline="\r\n") as handle:
            json.dump(self.manifest, handle, indent=2, sort_keys=True)
        self.assertEqual(dataset_metadata(self.path)["manifest_sha256"], expected)
        self.manifest["row_groups"]["a"] = "changed"
        self.write_manifest()
        self.assertNotEqual(dataset_metadata(self.path)["manifest_sha256"], expected)

    def test_missing_or_incomplete_manifest_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "manifest missing"):
            dataset_metadata(self.path)
        self.manifest["row_groups"].pop("b")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "group IDs"):
            dataset_metadata(self.path)

    def test_unknown_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            split_indices(["ai", "unknown", "human", "human"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
