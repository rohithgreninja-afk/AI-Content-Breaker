"""Leakage regression tests for training-only model selection.

Synthetic rows and estimator spies make excluded-row access observable without
reading the real holdout, extracting language features, or fitting candidates.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import m5_training as training  # noqa: E402
from m4_features import FEATURE_NAMES  # noqa: E402


def protocol() -> dict:
    """Small valid protocol using the same fixed gate, cutoff, and folds."""
    return {
        "schema_version": 1,
        "cross_validation": {"method": "StratifiedGroupKFold", "n_splits": 5,
                             "shuffle": True, "random_state": 42},
        "min_words": 50,
        "probability_cutoff": 0.5,
        "baseline": "baseline",
        "minimum_primary_gain_over_baseline": 0.01,
        "maximum_all_row_regression": 0.01,
        "candidates": [
            {"name": "baseline", "config": {"family": "logreg", "C": 10.0}},
            {"name": "candidate", "config": {"family": "logreg", "C": 1.0}},
        ],
    }


def result(name: str, gated: float, all_rows: float, **extra) -> dict:
    """Create a candidate record with explicit selection metrics."""
    return {"name": name, "metrics": {
        "all_rows": {"accuracy": all_rows},
        "at_least_50_words": {"accuracy": gated},
    }, **extra}


class TestFixedPartition(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": name} for name in ("a", "b", "c", "d")]
        self.provenance = {"dataset_sha256": "csv-hash", "manifest_sha256": "manifest-hash",
                           "groups": ["group-a", "group-b", "group-c", "group-d"]}
        self.metadata = {"dataset_sha256": "csv-hash", "manifest_sha256": "manifest-hash",
                         "train_ids": ["c", "a"], "test_ids": ["d", "b"]}

    def test_preserves_exact_saved_ids_and_order(self):
        train, test = training.fixed_partition(self.rows, self.provenance, self.metadata)
        self.assertEqual(train, [2, 0])
        self.assertEqual(test, [3, 1])

    def test_rejects_stale_data_or_group_provenance(self):
        for key in ("dataset_sha256", "manifest_sha256"):
            with self.subTest(key=key):
                metadata = {**self.metadata, key: "stale"}
                with self.assertRaisesRegex(ValueError, "stale " + key):
                    training.fixed_partition(self.rows, self.provenance, metadata)

    def test_rejects_related_documents_across_boundary(self):
        self.provenance["groups"][3] = "group-a"
        with self.assertRaisesRegex(ValueError, "related groups overlap"):
            training.fixed_partition(self.rows, self.provenance, self.metadata)

    def test_rejects_repeated_or_overlapping_split_ids(self):
        invalid = (
            {"train_ids": ["a", "a", "c"]},
            {"test_ids": ["d", "b", "b"]},
            {"test_ids": ["a", "b", "d"]},
            {"test_ids": ["d"]},
            {"train_ids": []},
        )
        for change in invalid:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "invalid saved train/test IDs"):
                    training.fixed_partition(self.rows, self.provenance,
                                             {**self.metadata, **change})


class TestTrainingFeatureBoundary(unittest.TestCase):
    def test_heldout_text_never_reaches_extractor(self):
        rows = [
            {"id": "heldout-a", "text": "DO NOT EXTRACT HELDOUT", "label": "human",
             "domain": "test:excluded"},
            {"id": "train-a", "text": "alpha " * 50, "label": "human", "domain": "mage:story"},
            {"id": "heldout-b", "text": "DO NOT EXTRACT HELDOUT", "label": "ai",
             "domain": "test:excluded"},
            {"id": "train-b", "text": "beta " * 20, "label": "ai", "domain": "hc3:answer"},
        ]
        extracted = []

        def extractor(text):
            if "HELDOUT" in text:
                raise AssertionError("selection extracted a heldout text")
            extracted.append(text)
            value = float(len(extracted))
            return {name: value for name in FEATURE_NAMES}

        data = training.training_arrays(rows, [3, 1], ["test-a", "train-a", "test-b", "train-b"],
                                        extractor=extractor)
        self.assertEqual(extracted, [rows[3]["text"], rows[1]["text"]])
        self.assertEqual(data["ids"].tolist(), ["train-b", "train-a"])
        self.assertEqual(data["groups"].tolist(), ["train-b", "train-a"])
        self.assertEqual(data["sources"].tolist(), ["hc3", "mage"])
        self.assertEqual(data["words"].tolist(), [20, 50])
        self.assertEqual(data["y"].tolist(), [1, 0])
        self.assertEqual(data["X"].shape, (2, len(FEATURE_NAMES)))


class TestTrainingOnlyCrossValidation(unittest.TestCase):
    def test_fits_and_predictions_use_separate_groups_and_only_training_rows(self):
        # Ten related human/AI pairs provide both classes in every grouped fold.
        indices = np.arange(20)
        groups = np.asarray([f"document-{index // 2}" for index in indices])
        data = {
            "X": np.column_stack((indices, indices % 2)).astype(float),
            "y": indices % 2,
            "groups": groups,
            "ids": np.asarray([f"training-{index}" for index in indices]),
            "words": np.where(indices % 2 == 0, 30, 60),
            "sources": np.where(indices % 4 < 2, "mage", "hc3"),
        }
        fit_events, validation_events = [], []
        training_rows = set(indices.tolist())
        heldout_rows = {100, 101}

        class SpyEstimator:
            def fit(self, vectors, labels):
                self.fit_rows = set(vectors[:, 0].astype(int).tolist())
                self.fit_groups = {groups[index] for index in self.fit_rows}
                if not self.fit_rows <= training_rows or self.fit_rows & heldout_rows:
                    raise AssertionError("fitting reached the external holdout")
                fit_events.append(self.fit_rows)
                return self

            def predict_proba(self, vectors):
                validation = set(vectors[:, 0].astype(int).tolist())
                validation_groups = {groups[index] for index in validation}
                if (not validation <= training_rows or validation & heldout_rows
                        or validation & self.fit_rows or validation_groups & self.fit_groups):
                    raise AssertionError("validation rows or groups leaked into fitting")
                validation_events.append(validation)
                positive = np.where(vectors[:, 1] == 1, 0.9, 0.1)
                return np.column_stack((1 - positive, positive))

        with patch.object(training, "build_estimator", side_effect=lambda config: SpyEstimator()), \
                redirect_stdout(io.StringIO()):
            record = training.select_candidates(data, protocol())

        self.assertEqual(len(fit_events), 10)  # Five fresh fits for each candidate.
        self.assertEqual(len(validation_events), 10)
        self.assertEqual(validation_events[:5], validation_events[5:])
        for fit, validation in zip(fit_events, validation_events):
            self.assertEqual(fit | validation, training_rows)
        for start in (0, 5):
            visits = [index for fold in validation_events[start:start + 5] for index in fold]
            self.assertCountEqual(visits, indices.tolist())
        self.assertEqual(set(record["fold_assignment"]), set(data["ids"]))
        self.assertEqual(set(record["fold_assignment"].values()), set(range(5)))
        self.assertFalse(record["test_evaluated_during_selection"])
        self.assertEqual(record["n_training_rows"], 20)
        self.assertEqual(record["n_training_groups"], 10)
        self.assertEqual(record["n_scored_training_rows"], 10)
        self.assertEqual(record["protocol_sha256"], training.json_digest(protocol()))


class TestPredeclaredSelectionPolicy(unittest.TestCase):
    def test_below_minimum_primary_gain_retains_baseline(self):
        rows = [result("baseline", 0.70, 0.65), result("candidate", 0.705, 0.69)]
        self.assertEqual(training.choose_candidate(rows, protocol())["name"], "baseline")

    def test_all_row_regression_limit_excludes_candidate(self):
        rows = [result("baseline", 0.70, 0.65), result("candidate", 0.90, 0.62)]
        self.assertEqual(training.choose_candidate(rows, protocol())["name"], "baseline")

    def test_selection_ignores_any_test_result_and_never_switches_on_it(self):
        rows = [result("baseline", 0.70, 0.65, test_accuracy=1.0),
                result("candidate", 0.74, 0.67, test_accuracy=0.0)]
        self.assertEqual(training.choose_candidate(rows, protocol())["name"], "candidate")

    def test_all_row_accuracy_breaks_primary_ties(self):
        policy = protocol()
        policy["candidates"].append({"name": "other", "config": {"family": "logreg", "C": 0.1}})
        rows = [result("baseline", 0.70, 0.65), result("candidate", 0.74, 0.67),
                result("other", 0.74, 0.68)]
        self.assertEqual(training.choose_candidate(rows, policy)["name"], "other")

    def test_exact_ties_preserve_declared_candidate_order(self):
        policy = protocol()
        policy["candidates"].append({"name": "other", "config": {"family": "logreg", "C": 0.1}})
        rows = [result("baseline", 0.70, 0.65), result("candidate", 0.74, 0.67),
                result("other", 0.74, 0.67)]
        self.assertEqual(training.choose_candidate(rows, policy)["name"], "candidate")

    def test_gate_cutoff_and_cv_policy_cannot_drift(self):
        training.validate_protocol(protocol())
        variants = (("min_words", 100), ("probability_cutoff", 0.6), ("schema_version", 2),
                    ("cross_validation", {**protocol()["cross_validation"], "random_state": 7}))
        for key, value in variants:
            with self.subTest(key=key):
                changed = deepcopy(protocol())
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "unsupported selection protocol"):
                    training.validate_protocol(changed)


if __name__ == "__main__":
    unittest.main()
