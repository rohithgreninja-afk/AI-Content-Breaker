"""Refit the committed classifier when the local model artifact is unavailable.

Unlike model selection, this recovery command never evaluates candidate recipes
or reads holdout text. The committed report supplies the original partition when
there is no loadable model to use as the anchor required by --finalize.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from m3_parser import tagger_mode
from m4_features import FEATURE_NAMES
from m5_estimators import build_estimator
from m5_scorer import (MODEL_PATH, REPO_ROOT, ModelScorer,
                       dataset_metadata, dataset_sha256, feature_extractor_sha256,
                       load_dataset_rows, runtime_versions)
from m5_training import (CACHE_PATH, PROTOCOL_PATH, SELECTION_PATH,
                         choose_candidate, committed_json, fixed_partition,
                         json_digest, training_arrays, validate_protocol,
                         write_json)

REPORT_PATH = Path(REPO_ROOT) / "models" / "training_report.json"


def refit_committed_model(output_path: str = MODEL_PATH) -> None:
    """Validate the original training identity and refit only its saved rows."""
    for path in (PROTOCOL_PATH, SELECTION_PATH, REPORT_PATH):
        committed_json(path)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    if (selection.get("protocol_sha256") != json_digest(protocol)
            or report.get("selection_sha256") != json_digest(selection)
            or report.get("training_protocol_sha256") != json_digest(protocol)
            or selection.get("test_evaluated_during_selection") is not False):
        raise ValueError("Committed training records disagree; restore the original records before training.")
    recipes = [{"name": item["name"], "config": item["config"]}
               for item in selection["candidates"]]
    winner = choose_candidate(selection["candidates"], protocol)
    if (recipes != protocol["candidates"] or winner["name"] != selection["winner"]
            or winner["config"] != selection["estimator_config"]
            or report.get("estimator_config") != winner["config"]
            or report.get("feature_names") != list(FEATURE_NAMES)):
        raise ValueError("Committed winner or feature order changed; restore the original training records.")

    print("Validating dataset and the saved training partition…", flush=True)
    rows, provenance = load_dataset_rows(), dataset_metadata()
    train, _ = fixed_partition(rows, provenance, report)
    identity = {
        "dataset_sha256": provenance["dataset_sha256"],
        "manifest_sha256": provenance["manifest_sha256"],
        "feature_extractor_sha256": feature_extractor_sha256(),
        "tagger": tagger_mode(), "versions": runtime_versions(),
        "training_ids": [rows[index]["id"] for index in train],
        "training_groups": [provenance["groups"][index] for index in train],
    }
    if identity != selection["training_identity"]:
        raise ValueError("Training identity changed. Restore the pinned dataset, Python dependencies, "
                         "NLTK tagger and feature code documented in README.md; "
                         "new model selection must use the explicit CLI workflow.")
    folds = selection["fold_assignment"]
    if set(folds) != set(identity["training_ids"]):
        raise ValueError("Saved validation folds do not cover the training partition.")
    groups = {}
    for row_id, group in zip(identity["training_ids"], identity["training_groups"]):
        fold = folds[row_id]
        if type(fold) is not int or not 0 <= fold < 5 or (group in groups and groups[group] != fold):
            raise ValueError("Saved validation folds split a group or have an invalid fold.")
        groups[group] = fold
    if set(groups.values()) != set(range(5)):
        raise ValueError("Saved validation folds do not cover all five folds.")

    cache_metadata_path = CACHE_PATH.with_suffix(".json")
    cache_valid = False
    if CACHE_PATH.exists() and cache_metadata_path.exists():
        cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        cache_valid = (cache_metadata.get("identity") == identity
                       and cache_metadata.get("cache_sha256") == dataset_sha256(str(CACHE_PATH)))
    if cache_valid:
        print("Loading verified training features…", flush=True)
        with np.load(CACHE_PATH, allow_pickle=False) as cache:
            data = {key: cache[key] for key in cache.files}
    else:
        print("Extracting features from training rows only…", flush=True)
        data = training_arrays(rows, train, provenance["groups"])
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE_PATH.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **data)
        temporary.replace(CACHE_PATH)
        cache_metadata = {"identity": identity, "cache_sha256": dataset_sha256(str(CACHE_PATH))}
        write_json(cache_metadata, cache_metadata_path)
    if (cache_metadata["cache_sha256"] != selection["training_feature_cache_sha256"]
            or report.get("training_feature_cache_sha256") != cache_metadata["cache_sha256"]
            or data["ids"].tolist() != identity["training_ids"]
            or data["X"].shape != (len(train), len(FEATURE_NAMES))
            or not np.isfinite(data["X"]).all()):
        raise ValueError("Training feature cache differs from the committed selection; training stopped.")

    print(f"Fitting {winner['name']} on {len(train):,} training texts…", flush=True)
    model = ModelScorer(pipeline=build_estimator(winner["config"]))
    with threadpool_limits(limits=2):
        model.pipeline.fit(data["X"], data["y"])
    # Preserve the original provenance and evaluation report for this exact
    # deterministic recipe. No new test evaluation or selection is performed.
    metadata_keys = set(identity) - {"training_ids", "training_groups"}
    metadata_keys.update({"schema_version", "dataset_sources", "train_ids", "test_ids",
                          "train_groups", "test_groups", "feature_names", "split_method",
                          "random_state", "test_size", "trained_at_utc", "training_thread_limit",
                          "threadpoolctl_version", "estimator_config", "selection_sha256",
                          "training_protocol_sha256", "selection_commit", "protocol_commit",
                          "training_feature_cache_sha256"})
    model.metadata = {key: report[key] for key in metadata_keys if key in report}
    model.save(output_path)
    ModelScorer.load(output_path)
    print("Model trained and saved. The committed evaluation report is retained.", flush=True)


if __name__ == "__main__":
    refit_committed_model()
