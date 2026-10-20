"""Leakage regression tests for training-only model selection.

Synthetic rows and estimator spies make excluded-row access observable without
reading the real holdout, extracting language features, or fitting candidates.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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


class TestSelectionWordGateEdges(unittest.TestCase):
    @staticmethod
    def data():
        indices = np.arange(10)
        return {"X": np.column_stack((indices, indices % 2)).astype(float),
                "y": indices % 2,
                "groups": np.asarray([f"pair-{index // 2}" for index in indices]),
                "ids": np.asarray([f"train-{index}" for index in indices]),
                "words": np.full(10, 60), "sources": np.asarray(["mage"] * 10)}

    def test_no_gated_training_rows_fails_before_folds_or_fitting(self):
        data = self.data()
        data["words"][:] = 49
        with patch.object(training, "build_estimator") as factory, \
                patch.object(training, "_cv_indices") as folds:
            with self.assertRaisesRegex(ValueError, "no training texts pass the fixed word gate"):
                training.select_candidates(data, protocol())
            factory.return_value.fit.assert_not_called()
            folds.assert_not_called()

    def test_empty_gated_validation_fold_is_reported_without_changing_gate(self):
        data = self.data()
        data["words"][:2] = 49
        folds = [(np.asarray([index for index in range(10) if index // 2 != fold]),
                  np.asarray([2 * fold, 2 * fold + 1])) for fold in range(5)]

        class ToyEstimator:
            def fit(self, vectors, labels):
                return self

            def predict_proba(self, vectors):
                positive = np.where(vectors[:, 1] == 1, 0.9, 0.1)
                return np.column_stack((1 - positive, positive))

        policy = protocol()
        policy["candidates"] = policy["candidates"][:1]
        output = io.StringIO()
        with patch.object(training, "_cv_indices", return_value=folds), \
                patch.object(training, "build_estimator", side_effect=lambda config: ToyEstimator()), \
                redirect_stdout(output):
            selection = training.select_candidates(data, policy)
        self.assertIn("n/a (no gated rows)", output.getvalue())
        first_fold = selection["candidates"][0]["fold_metrics"][0]
        self.assertIsNone(first_fold["at_least_50_words"]["accuracy"])
        self.assertEqual(selection["n_scored_training_rows"], 8)
        self.assertEqual(selection["candidates"][0]["metrics"]["at_least_50_words"]["n"], 8)


class TestFinalizeSelectionBoundary(unittest.TestCase):
    """Reject changed experiment records before training or touching test text."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.protocol_path = Path(temporary.name) / "protocol.json"
        self.selection_path = Path(temporary.name) / "selection.json"
        self.policy = protocol()
        ids = [f"train-{index}" for index in range(10)]
        self.identity = {"training_ids": ids, "dataset_sha256": "fixed-corpus"}
        self.data = {"ids": np.asarray(ids),
                     "groups": np.asarray([f"group-{index // 2}" for index in range(10)])}
        candidates = [
            {**self.policy["candidates"][0], **result("baseline", 0.70, 0.65)},
            {**self.policy["candidates"][1], **result("candidate", 0.74, 0.67)},
        ]
        self.selection = {
            "protocol_sha256": training.json_digest(self.policy),
            "candidates": candidates, "winner": "candidate",
            "estimator_config": self.policy["candidates"][1]["config"],
            "test_evaluated_during_selection": False,
            "training_identity": deepcopy(self.identity),
            "training_feature_cache_sha256": "fixed-features",
            "fold_assignment": {row_id: index // 2 for index, row_id in enumerate(ids)},
        }
        self.protocol_path.write_text(json.dumps(self.policy), encoding="utf-8")
        self.write_selection()
        mocks = {
            "PROTOCOL_PATH": patch.object(training, "PROTOCOL_PATH", self.protocol_path),
            "SELECTION_PATH": patch.object(training, "SELECTION_PATH", self.selection_path),
            "committed": patch.object(training, "committed_json", return_value="recorded-commit"),
            "prepare": patch.object(training, "prepare_training", return_value=(
                self.data, {"identity": self.identity, "cache_sha256": "fixed-features"})),
            "extract": patch.object(training, "extract_features",
                                    side_effect=AssertionError("heldout text was transformed")),
            "read_rows": patch.object(training, "load_dataset_rows",
                                      side_effect=AssertionError("heldout data stage was reached")),
        }
        for name, mock in mocks.items():
            setattr(self, name, mock.start())
            self.addCleanup(mock.stop)

    def write_selection(self):
        self.selection_path.write_text(json.dumps(self.selection), encoding="utf-8")

    def assert_no_test_stage(self):
        self.extract.assert_not_called()
        self.read_rows.assert_not_called()

    def test_uncommitted_protocol_or_selection_fails_before_preparing_features(self):
        for committed_replies in ([ValueError("commit protocol before final test evaluation")],
                                  ["protocol-commit", ValueError("commit selection before final test evaluation")]):
            with self.subTest(replies=committed_replies):
                self.committed.side_effect = committed_replies
                with self.assertRaisesRegex(ValueError, "commit .* before final test evaluation"):
                    training.finalize()
                self.prepare.assert_not_called()
                self.assert_no_test_stage()

    def test_changed_protocol_hash_fails_before_preparing_features(self):
        self.selection["protocol_sha256"] = "changed-protocol"
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "does not match the committed protocol"):
            training.finalize()
        self.prepare.assert_not_called()
        self.assert_no_test_stage()

    def test_changed_training_identity_or_feature_cache_fails_before_test_stage(self):
        for change in ({"training_identity": {"dataset_sha256": "other-corpus"}},
                       {"training_feature_cache_sha256": "other-features"}):
            with self.subTest(change=change):
                original = deepcopy(self.selection)
                self.selection.update(change)
                self.write_selection()
                with self.assertRaisesRegex(ValueError, "changed after selection"):
                    training.finalize()
                self.assert_no_test_stage()
                self.selection = original

    def test_invalid_fold_numbers_fail_before_test_stage(self):
        for fold in (-1, 5, "0", True):
            with self.subTest(fold=fold):
                self.selection["fold_assignment"]["train-0"] = fold
                self.write_selection()
                with self.assertRaisesRegex(ValueError, "invalid saved validation fold number"):
                    training.finalize()
                self.assert_no_test_stage()

    def test_related_group_in_multiple_folds_fails_before_test_stage(self):
        self.selection["fold_assignment"]["train-1"] = 1
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "split a related group"):
            training.finalize()
        self.assert_no_test_stage()

    def test_missing_fold_or_training_id_fails_before_test_stage(self):
        original = deepcopy(self.selection["fold_assignment"])
        self.selection["fold_assignment"] = dict.fromkeys(original, 0)
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "do not cover all five folds"):
            training.finalize()
        self.assert_no_test_stage()
        self.selection["fold_assignment"] = original
        self.selection["fold_assignment"].pop("train-9")
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "cover exactly the training IDs"):
            training.finalize()
        self.assert_no_test_stage()


class TestCommittedJsonGuard(unittest.TestCase):
    """Exercise the actual Git-content check using synthetic command responses."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "record.json"
        self.path.write_text('{"name": "frozen", "revision": 1}', encoding="utf-8")
        mock = patch.object(training, "ROOT", self.root)
        mock.start()
        self.addCleanup(mock.stop)

    def test_untracked_record_is_rejected(self):
        with patch.object(subprocess, "check_output", side_effect=subprocess.CalledProcessError(128, "git")):
            with self.assertRaisesRegex(ValueError, "commit record.json before final test evaluation"):
                training.committed_json(self.path)

    def test_edited_record_is_rejected(self):
        committed = b'{"name": "different", "revision": 1}'
        with patch.object(subprocess, "check_output", side_effect=[committed, "commit-sha\n"]):
            with self.assertRaisesRegex(ValueError, "commit changes to record.json"):
                training.committed_json(self.path)

    def test_formatting_only_changes_preserve_committed_identity(self):
        committed = b'{\n  "revision": 1,\n  "name": "frozen"\n}'
        with patch.object(subprocess, "check_output", side_effect=[committed, "commit-sha\n"]):
            self.assertEqual(training.committed_json(self.path), "commit-sha")


if __name__ == "__main__":
    unittest.main()
