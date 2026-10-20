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
                       _cv_indices, dataset_metadata, dataset_sha256, feature_extractor_sha256,
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
            print(f"{candidate['name']} fold {number}/5 "
                  f"gated={score['at_least_50_words']['accuracy']:.4f}", flush=True)
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
