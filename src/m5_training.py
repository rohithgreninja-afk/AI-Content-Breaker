"""Select a classifier using only the saved training partition.

--select writes a training-only experiment report. Commit that report before
--finalize fits the frozen winner and evaluates the fixed internal holdout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from m4_features import FEATURE_NAMES, extract_features, to_vector
from m3_parser import tagger_mode
from m5_estimators import build_estimator
from m5_scorer import (DATA_PATH, MODEL_PATH, REPO_ROOT, ModelScorer, _classification_metrics,
                       _cv_indices, _heldout_metrics, dataset_metadata, dataset_sha256, feature_extractor_sha256,
                       load_dataset_rows, runtime_versions)

ROOT = Path(REPO_ROOT)
PROTOCOL_PATH = ROOT / "reports" / "model_selection_protocol.json"
SELECTION_PATH = ROOT / "reports" / "model_selection.json"
CACHE_PATH = ROOT / "data" / "training_features.npz"


def json_digest(value: dict) -> str:
    """Fingerprint JSON content independently of formatting or platform."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(value: dict, path: Path) -> None:
    """Atomically save a JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def word_count(text: str) -> int:
    """Use the inference pipeline's alphabetic whitespace-piece word unit."""
    return sum(any(character.isalpha() for character in word) for word in text.split())


def fixed_partition(rows: list[dict], provenance: dict, metadata: dict) -> tuple[list[int], list[int]]:
    """Validate and reuse saved IDs without constructing a new test split."""
    for key in ("dataset_sha256", "manifest_sha256"):
        if metadata.get(key) != provenance.get(key):
            raise ValueError(f"saved partition has stale {key}")
    train_ids, test_ids = metadata.get("train_ids", []), metadata.get("test_ids", [])
    ids = [row["id"] for row in rows]
    if (not train_ids or not test_ids or len(set(ids)) != len(ids)
            or len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids)
            or set(train_ids) & set(test_ids) or set(train_ids + test_ids) != set(ids)):
        raise ValueError("invalid saved train/test IDs")
    by_id = {row_id: index for index, row_id in enumerate(ids)}
    train = [by_id[row_id] for row_id in train_ids]
    test = [by_id[row_id] for row_id in test_ids]
    groups = provenance["groups"]
    if {groups[i] for i in train} & {groups[i] for i in test}:
        raise ValueError("related groups overlap training and test")
    return train, test


def training_arrays(rows: list[dict], train: list[int], groups: list[str],
                    extractor=extract_features) -> dict:
    """Extract only explicitly supplied training rows, never held-out text."""
    vectors = []
    for position, index in enumerate(train, 1):
        vectors.append(to_vector(extractor(rows[index]["text"])))
        if position % 500 == 0:
            print(f"training features: {position}/{len(train)}", flush=True)
    data = {"X": np.asarray(vectors, dtype=float),
            "y": np.asarray([int(rows[i]["label"] == "ai") for i in train]),
            "ids": np.asarray([rows[i]["id"] for i in train]),
            "groups": np.asarray([groups[i] for i in train]),
            "words": np.asarray([word_count(rows[i]["text"]) for i in train]),
            "sources": np.asarray([rows[i]["domain"].split(":", 1)[0] for i in train])}
    if not np.isfinite(data["X"]).all():
        raise ValueError("non-finite training features")
    return data


def prepare_training() -> tuple[dict, dict]:
    """Build or validate a cache containing saved training rows only."""
    rows, provenance = load_dataset_rows(), dataset_metadata()
    anchor = ModelScorer.load()
    train, _ = fixed_partition(rows, provenance, anchor.metadata)
    identity = {"dataset_sha256": provenance["dataset_sha256"],
                "manifest_sha256": provenance["manifest_sha256"],
                "feature_extractor_sha256": feature_extractor_sha256(),
                "tagger": tagger_mode(), "versions": runtime_versions(),
                "training_ids": [rows[i]["id"] for i in train],
                "training_groups": [provenance["groups"][i] for i in train]}
    metadata_path = CACHE_PATH.with_suffix(".json")
    if CACHE_PATH.exists() and metadata_path.exists():
        cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (cache_metadata.get("identity") == identity
                and cache_metadata.get("cache_sha256") == dataset_sha256(str(CACHE_PATH))):
            with np.load(CACHE_PATH, allow_pickle=False) as cache:
                data = {key: cache[key] for key in cache.files}
            if data["ids"].tolist() != identity["training_ids"]:
                raise ValueError("training feature cache has incorrect IDs")
            return data, cache_metadata
    data = training_arrays(rows, train, provenance["groups"])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **data)
    temporary.replace(CACHE_PATH)
    cache_metadata = {"identity": identity, "cache_sha256": dataset_sha256(str(CACHE_PATH))}
    write_json(cache_metadata, metadata_path)
    return data, cache_metadata


def validate_protocol(protocol: dict) -> None:
    """Fail closed if a run changes the committed split/scoring safeguards."""
    expected_cv = {"method": "StratifiedGroupKFold", "n_splits": 5,
                   "shuffle": True, "random_state": 42}
    if (protocol.get("schema_version") != 1 or protocol.get("cross_validation") != expected_cv
            or protocol.get("min_words") != 50 or protocol.get("probability_cutoff") != 0.5):
        raise ValueError("unsupported selection protocol")
    names = [candidate["name"] for candidate in protocol["candidates"]]
    if len(set(names)) != len(names) or protocol["baseline"] not in names:
        raise ValueError("invalid candidate names or baseline")
    for candidate in protocol["candidates"]:
        build_estimator(candidate["config"])


def _scores(y: np.ndarray, probabilities: np.ndarray, words: np.ndarray) -> dict:
    labels = ["ai" if value else "human" for value in y]
    gated = words >= 50
    return {"all_rows": _classification_metrics(labels, probabilities),
            "at_least_50_words": _classification_metrics(
                [label for label, keep in zip(labels, gated) if keep], probabilities[gated])}


def choose_candidate(results: list[dict], protocol: dict) -> dict:
    """Apply the predeclared rule using training out-of-fold metrics only."""
    baseline = next(row for row in results if row["name"] == protocol["baseline"])
    eligible = [row for row in results
                if row["metrics"]["all_rows"]["accuracy"] >=
                baseline["metrics"]["all_rows"]["accuracy"] - protocol["maximum_all_row_regression"]]
    winner = max(eligible, key=lambda row: (
        row["metrics"]["at_least_50_words"]["accuracy"],
        row["metrics"]["all_rows"]["accuracy"]))
    gain = (winner["metrics"]["at_least_50_words"]["accuracy"]
            - baseline["metrics"]["at_least_50_words"]["accuracy"])
    if gain < protocol["minimum_primary_gain_over_baseline"]:
        winner = baseline
    return winner


def select_candidates(data: dict, protocol: dict) -> dict:
    """Evaluate the frozen candidates using only training-group folds."""
    validate_protocol(protocol)
    X, y, groups = data["X"], data["y"], data["groups"]
    if not np.any(data["words"] >= protocol["min_words"]):
        raise ValueError("no training texts pass the fixed word gate")
    folds = _cv_indices(X, y, groups, random_state=42)
    assignment = {}
    for number, (fit, validation) in enumerate(folds):
        if set(groups[fit]) & set(groups[validation]):
            raise ValueError("related groups cross a validation fold")
        for index in validation:
            assignment[str(data["ids"][index])] = number
    results = []
    for candidate in protocol["candidates"]:
        probabilities = np.full(len(y), np.nan)
        fold_scores = []
        for number, (fit, validation) in enumerate(folds, 1):
            estimator = build_estimator(candidate["config"])
            with threadpool_limits(limits=2):
                estimator.fit(X[fit], y[fit])
                probabilities[validation] = estimator.predict_proba(X[validation])[:, 1]
            score = _scores(y[validation], probabilities[validation], data["words"][validation])
            fold_scores.append(score)
            accuracy = score["at_least_50_words"]["accuracy"]
            shown = f"{accuracy:.4f}" if accuracy is not None else "n/a (no gated rows)"
            print(f"{candidate['name']} fold {number}/5 "
                  f"gated={shown}", flush=True)
        if not np.isfinite(probabilities).all():
            raise ValueError("CV did not predict every training row exactly once")
        source_scores = {source: _scores(y[data["sources"] == source],
                         probabilities[data["sources"] == source],
                         data["words"][data["sources"] == source])
                         for source in sorted(set(data["sources"]))}
        results.append({**candidate, "metrics": _scores(y, probabilities, data["words"]),
                        "fold_metrics": fold_scores, "per_source": source_scores})
    winner = choose_candidate(results, protocol)
    return {"schema_version": 1, "protocol_sha256": json_digest(protocol),
            "n_training_rows": len(y), "n_training_groups": len(set(groups)),
            "n_scored_training_rows": int((data["words"] >= 50).sum()),
            "fold_assignment": assignment, "candidates": results,
            "winner": winner["name"], "estimator_config": winner["config"],
            "test_evaluated_during_selection": False,
            "cv_note": "Selection CV is optimistic for the selected winner; it is not a nested-CV or independent-test estimate."}


def committed_json(path: Path) -> str:
    """Require the experiment record on disk to equal its committed version."""
    import subprocess

    relative = path.resolve().relative_to(ROOT).as_posix()
    try:
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        revision = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"commit {relative} before final test evaluation") from error
    if json_digest(json.loads(committed)) != json_digest(json.loads(path.read_text(encoding="utf-8"))):
        raise ValueError(f"commit changes to {relative} before final test evaluation")
    return revision


def finalize() -> None:
    """Fit the committed winner, then evaluate the fixed holdout without tuning."""
    from datetime import datetime, timezone
    from importlib.metadata import version

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    protocol_commit = committed_json(PROTOCOL_PATH)
    selection_commit = committed_json(SELECTION_PATH)
    if selection.get("protocol_sha256") != json_digest(protocol):
        raise ValueError("selection does not match the committed protocol")
    recipes = [{"name": result["name"], "config": result["config"]}
               for result in selection["candidates"]]
    if recipes != protocol["candidates"]:
        raise ValueError("selection candidate recipes differ from the protocol")
    winner = choose_candidate(selection["candidates"], protocol)
    if (winner["name"] != selection["winner"]
            or winner["config"] != selection["estimator_config"]
            or selection.get("test_evaluated_during_selection") is not False):
        raise ValueError("invalid training-only winner record")
    data, cache_metadata = prepare_training()
    if (selection["training_identity"] != cache_metadata["identity"]
            or selection["training_feature_cache_sha256"] != cache_metadata["cache_sha256"]):
        raise ValueError("training data or feature cache changed after selection")
    if set(selection["fold_assignment"]) != set(data["ids"]):
        raise ValueError("selection folds do not cover exactly the training IDs")
    folds_by_group = {}
    for row_id, group in zip(data["ids"], data["groups"]):
        fold = selection["fold_assignment"][row_id]
        if type(fold) is not int or not 0 <= fold < 5:
            raise ValueError("invalid saved validation fold number")
        if group in folds_by_group and folds_by_group[group] != fold:
            raise ValueError("saved validation folds split a related group")
        folds_by_group[group] = fold
    if set(folds_by_group.values()) != set(range(5)):
        raise ValueError("saved validation folds do not cover all five folds")

    rows, provenance = load_dataset_rows(), dataset_metadata()
    anchor = ModelScorer.load()
    train, test = fixed_partition(rows, provenance, anchor.metadata)
    previous_model_sha256 = dataset_sha256(MODEL_PATH)
    previous_report_path = ROOT / "models" / "training_report.json"
    previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
    if any(previous_report.get(key) != value for key, value in anchor.metadata.items()):
        raise ValueError("previous model/report provenance differs")
    model = ModelScorer(pipeline=build_estimator(winner["config"]))
    reference = ModelScorer(pipeline=build_estimator({"family": "logreg", "C": 10.0}))
    with threadpool_limits(limits=2):
        model.pipeline.fit(data["X"], data["y"])
        reference.pipeline.fit(data["X"], data["y"])
        train_probabilities = model.pipeline.predict_proba(data["X"])[:, 1]
    # Selection and full-training fits are finished before the first test text
    # is transformed. No subsequent decision reads these test results.
    features = [None] * len(rows)
    for index, vector in zip(train, data["X"]):
        features[index] = dict(zip(FEATURE_NAMES, vector))
    for position, index in enumerate(test, 1):
        features[index] = extract_features(rows[index]["text"])
        if position % 500 == 0:
            print(f"final holdout features: {position}/{len(test)}", flush=True)
    with threadpool_limits(limits=2):
        evaluation = _heldout_metrics(model, features, rows, test)
        baseline = _heldout_metrics(reference, features, rows, test)
    model.metadata = {
        "schema_version": 1, "dataset_sha256": provenance["dataset_sha256"],
        "manifest_sha256": provenance["manifest_sha256"],
        "dataset_sources": provenance["dataset_sources"],
        "train_ids": [rows[i]["id"] for i in train], "test_ids": [rows[i]["id"] for i in test],
        "train_groups": sorted({provenance["groups"][i] for i in train}),
        "test_groups": sorted({provenance["groups"][i] for i in test}),
        "tagger": tagger_mode(), "feature_names": list(FEATURE_NAMES),
        "feature_extractor_sha256": feature_extractor_sha256(),
        "split_method": "stratified_group", "random_state": 42, "test_size": 0.25,
        "versions": runtime_versions(), "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_thread_limit": 2, "threadpoolctl_version": version("threadpoolctl"),
        "estimator_config": winner["config"], "selection_sha256": json_digest(selection),
        "training_protocol_sha256": json_digest(protocol), "selection_commit": selection_commit,
        "protocol_commit": protocol_commit,
        "training_feature_cache_sha256": cache_metadata["cache_sha256"],
    }
    cv_scores = [fold["all_rows"]["accuracy"] for fold in winner["fold_metrics"]]
    overall = evaluation["overall"]["all_rows"]["model"]
    metrics = {"C": winner["config"].get("C"), "n_train": len(train), "n_test": len(test),
               "n_features": len(FEATURE_NAMES), "split_method": "stratified_group",
               "train_accuracy": float(np.mean((train_probabilities >= 0.5) == data["y"])),
               "test_accuracy": overall["accuracy"], "test_roc_auc": overall["roc_auc"],
               "cv_mean": float(np.mean(cv_scores)), "cv_std": float(np.std(cv_scores)),
               "cv_scores": cv_scores, "train_indices": train, "test_indices": test,
               "confusion_matrix": overall["confusion_matrix"]}
    report = {**model.metadata, "metrics": metrics, "c_selection": None,
              "test_source_counts": dict(Counter(rows[i]["domain"].split(":", 1)[0] for i in test)),
              "evaluation": evaluation, "baseline_evaluation": baseline,
              "baseline_estimator_config": {"family": "logreg", "C": 10.0},
              "baseline_note": "The C=10 reference is freshly refitted using the two-thread training control; its numerical solver path can differ slightly from the previously published model.",
              "previous_published_evaluation": previous_report["evaluation"],
              "previous_model_sha256": previous_model_sha256,
              "model_selection": {"winner": winner["name"], "metrics": winner["metrics"],
                                  "test_used_for_selection": False, "cv_note": selection["cv_note"]},
              "holdout_note": "The same previously published internal holdout is evaluated only after the training-only winner is committed. It is not a new unseen external benchmark."}
    model.save()
    write_json(report, ROOT / "models" / "training_report.json")
    print(f"Saved frozen winner {winner['name']}: all-row accuracy={overall['accuracy']:.4f}, "
          f"gated={evaluation['overall']['at_least_50_words']['model']['accuracy']:.4f}")


def main(argv=None) -> None:
    """Run explicit selection or final fitting, keeping the test stage separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--select", action="store_true")
    action.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    if args.select:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        validate_protocol(protocol)
        data, cache_metadata = prepare_training()
        selection = select_candidates(data, protocol)
        selection.update(training_identity=cache_metadata["identity"],
                         training_feature_cache_sha256=cache_metadata["cache_sha256"])
        write_json(selection, SELECTION_PATH)
        print(f"Selected {selection['winner']} using training CV only -> {SELECTION_PATH}")
    else:
        finalize()


if __name__ == "__main__":
    main()
