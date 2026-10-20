"""Verify the shipped classifier against its exact training rows and holdout.

Run after M0 and M5: python scripts/verify_training.py
Refits the recorded estimator recipe, checks fitted state and predictions,
and compares combined training with a MAGE-only fit on the same heldout rows.
Neither this comparison nor the holdout chooses any training hyperparameter.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m4_features import to_vector  # noqa: E402
from m5_scorer import (  # noqa: E402
    DATA_PATH, MODEL_PATH, ModelScorer, _classification_metrics,
    dataset_metadata, dataset_sha256, load_dataset, load_dataset_rows,
)
import pipeline  # noqa: E402


def _iteration_details(estimator, prefix: str = "model") -> list[dict]:
    """Check completion using each classifier's actual training controls."""
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if isinstance(estimator, Pipeline):
        return [record for name, step in estimator.steps
                for record in _iteration_details(step, f"{prefix}.{name}")]
    if isinstance(estimator, VotingClassifier):
        return [record for name, step in estimator.named_estimators_.items()
                for record in _iteration_details(step, f"{prefix}.{name}")]
    if isinstance(estimator, LogisticRegression):
        iterations = int(estimator.n_iter_.max())
        if iterations >= estimator.max_iter:
            raise ValueError("saved classifier reached its iteration limit; retrain to convergence")
        return [{"estimator": prefix, "kind": "logistic_regression",
                 "iterations": iterations, "max_iterations": estimator.max_iter,
                 "status": "converged"}]
    if isinstance(estimator, HistGradientBoostingClassifier):
        if estimator.early_stopping is not False:
            raise ValueError("verified boosting must disable internal early-stopping validation")
        if estimator.n_iter_ != estimator.max_iter:
            raise ValueError("boosting did not complete its recorded fixed iteration budget")
        return [{"estimator": prefix, "kind": "hist_gradient_boosting",
                 "iterations": int(estimator.n_iter_), "max_iterations": estimator.max_iter,
                 "status": "fixed_budget_completed", "early_stopping": False}]
    if isinstance(estimator, ExtraTreesClassifier):
        if len(estimator.estimators_) != estimator.n_estimators:
            raise ValueError("tree ensemble did not complete its recorded tree budget")
        return [{"estimator": prefix, "kind": "extra_trees",
                 "trees": len(estimator.estimators_), "status": "fixed_budget_completed"}]
    return []


def _fitted_differences(saved, fresh, prefix: str = "") -> dict:
    """Compare linear parameters or the complete fitted trees of supported recipes."""
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    differences = {}
    def compare(name, left, right):
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape:
            raise ValueError(f"saved fitted state changed shape: {prefix}{name}")
        if left.dtype.names or left.dtype.kind not in "fciub":
            if not np.array_equal(left, right):
                raise ValueError(f"saved fitted state does not reproduce: {prefix}{name}")
            return 0.0
        difference = float(np.max(np.abs(left.astype(float) - right.astype(float)))) if left.size else 0.0
        if not np.allclose(left, right, rtol=1e-9, atol=1e-10):
            raise ValueError(f"saved fitted state does not reproduce: {prefix}{name}")
        return difference

    if isinstance(saved, Pipeline):
        for name, step in saved.steps:
            differences.update(_fitted_differences(step, fresh.named_steps[name], f"{prefix}{name}."))
    elif isinstance(saved, VotingClassifier):
        for name, step in saved.named_estimators_.items():
            differences.update(_fitted_differences(step, fresh.named_estimators_[name], f"{prefix}{name}."))
    elif isinstance(saved, (StandardScaler, LogisticRegression)):
        attributes = ("mean_", "scale_") if isinstance(saved, StandardScaler) else ("coef_", "intercept_")
        for attribute in attributes:
            differences[f"{prefix}{attribute}"] = compare(attribute, getattr(saved, attribute), getattr(fresh, attribute))
    elif isinstance(saved, HistGradientBoostingClassifier):
        differences[f"{prefix}baseline_prediction"] = compare(
            "baseline_prediction", saved._baseline_prediction, fresh._baseline_prediction)
        if len(saved._predictors) != len(fresh._predictors):
            raise ValueError("boosting tree count differs after refitting")
        for attribute in ("nodes", "binned_left_cat_bitsets", "raw_left_cat_bitsets"):
            values = []
            for left_stage, right_stage in zip(saved._predictors, fresh._predictors):
                if len(left_stage) != len(right_stage):
                    raise ValueError("boosting class tree count differs after refitting")
                values.extend(compare(attribute, getattr(left, attribute), getattr(right, attribute))
                              for left, right in zip(left_stage, right_stage))
            differences[f"{prefix}{attribute}"] = max(values, default=0.0)
        differences[f"{prefix}bin_thresholds"] = max(
            (compare("bin_thresholds", left, right) for left, right in
             zip(saved._bin_mapper.bin_thresholds_, fresh._bin_mapper.bin_thresholds_)), default=0.0)
    elif isinstance(saved, ExtraTreesClassifier):
        if len(saved.estimators_) != len(fresh.estimators_):
            raise ValueError("ensemble tree count differs after refitting")
        for attribute in ("nodes", "values"):
            differences[f"{prefix}tree_{attribute}"] = max(
                (compare(attribute, left.tree_.__getstate__()[attribute], right.tree_.__getstate__()[attribute])
                 for left, right in zip(saved.estimators_, fresh.estimators_)), default=0.0)
    else:
        raise ValueError(f"no fitted-state verification for {type(saved).__name__}")
    return differences


def _refit_and_compare(model, vectors, labels, train_indices, report: dict) -> dict:
    """Refit on saved training indices and compare state plus all supplied predictions."""
    from threadpoolctl import threadpool_limits

    config = model.metadata.get("estimator_config")
    if config is None:
        refitted = ModelScorer._build_pipeline(C=report["metrics"]["C"])
    else:
        from m5_estimators import build_estimator
        refitted = build_estimator(config)
    details = _iteration_details(model.pipeline)
    with threadpool_limits(limits=2):
        refitted.fit(vectors[train_indices], labels[train_indices])
        differences = _fitted_differences(model.pipeline, refitted)
        saved_probabilities = model.pipeline.predict_proba(vectors)
        fresh_probabilities = refitted.predict_proba(vectors)
    difference = float(np.max(np.abs(saved_probabilities - fresh_probabilities)))
    if not np.allclose(saved_probabilities, fresh_probabilities, rtol=1e-12, atol=1e-12):
        raise ValueError("saved model predictions do not reproduce after refitting")
    return {"parameter_differences": differences, "training_iterations": details,
            "prediction_max_abs_difference": difference}


def verify() -> dict:
    model = ModelScorer.load()
    provenance = dataset_metadata()
    rows = load_dataset_rows()
    # Includes exact dataset/manifest, ID coverage, and related-group checks.
    heldout = pipeline._test_rows()
    with (ROOT / "models" / "training_report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    for key, value in model.metadata.items():
        if report.get(key) != value:
            raise ValueError(f"model/report provenance mismatch: {key}")

    seen, lengths = set(), defaultdict(lambda: defaultdict(Counter))
    prefixes = defaultdict(set)
    for row, group in zip(rows, provenance["groups"]):
        key = " ".join(row["text"].casefold().split())
        if key in seen:
            raise ValueError("duplicate normalized training text")
        seen.add(key)
        n_words = len(row["text"].split())
        if not 20 <= n_words <= 300:
            raise ValueError("training text outside documented word limits")
        lengths[row["domain"]][row["label"]][n_words] += 1
        prefixes[" ".join(key.split()[:20])].add(group)
    for domain, classes in lengths.items():
        if classes["human"] != classes["ai"]:
            raise ValueError(f"class counts or lengths differ in {domain}")
    if any(len(groups) > 1 for groups in prefixes.values()):
        raise ValueError("shared opening assigned to multiple split groups")

    print(f"Extracting features for {len(rows)} rows to verify saved training...", flush=True)
    features, labels = load_dataset()
    X = np.asarray([to_vector(feature) for feature in features], dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("non-finite feature values")
    y = np.asarray([int(label == "ai") for label in labels])
    by_id = {row["id"]: i for i, row in enumerate(rows)}
    train = [by_id[row_id] for row_id in model.metadata["train_ids"]]
    test = [by_id[row["id"]] for row in heldout]
    refit = _refit_and_compare(model, X, y, train, report)

    combined_probs = model.pipeline.predict_proba(X[test])[:, 1]
    combined = _classification_metrics([labels[i] for i in test], combined_probs)
    recorded = report["evaluation"]["overall"]["all_rows"]["model"]
    if combined != recorded:
        raise ValueError("saved model predictions do not match recorded holdout metrics")

    mage_train = [i for i in train if rows[i]["domain"].startswith("mage:")]
    if not mage_train:
        raise ValueError("combined training has no MAGE training rows")
    # Fixed historical C: comparison cannot select hyperparameters on test data.
    baseline = ModelScorer._build_pipeline(C=30.0)
    baseline.fit(X[mage_train], y[mage_train])
    baseline_probs = baseline.predict_proba(X[test])[:, 1]
    comparison = {}
    for source in ["overall"] + sorted(provenance["dataset_sources"]):
        positions = [j for j, i in enumerate(test) if source == "overall"
                     or rows[i]["domain"].startswith(source + ":")]
        scored = [j for j in positions if sum(any(c.isalpha() for c in word)
                  for word in rows[test[j]]["text"].split()) >= pipeline.MIN_WORDS]

        def metrics(indices):
            truth = [labels[test[j]] for j in indices]
            return {"mage_only": _classification_metrics(truth, baseline_probs[indices]),
                    "combined": _classification_metrics(truth, combined_probs[indices])}

        comparison[source] = {"all_rows": metrics(positions),
                              "at_least_50_words": metrics(scored)}

    return {
        "dataset_sha256": dataset_sha256(DATA_PATH),
        "model_sha256": dataset_sha256(MODEL_PATH),
        "n_rows": len(rows), "n_train": len(train), "n_test": len(test),
        "n_domains": len(lengths), "n_groups": len(set(provenance["groups"])),
        "labels": dict(Counter(labels)),
        "sources": {name: metadata["rows"] for name, metadata
                    in provenance["dataset_sources"].items()},
        "checks": {"finite_features": True, "exact_duplicates": 0,
                   "shared_prefixes_across_groups": 0, "train_test_group_overlap": 0,
                   "per_domain_class_and_length_balance": True,
                   "model_report_metrics_match": True, "training_controls_completed": True},
        "training_iteration_details": refit["training_iterations"],
        "refit_parameter_max_abs_differences": refit["parameter_differences"],
        "refit_prediction_max_abs_difference": refit["prediction_max_abs_difference"],
        "same_holdout_comparison": comparison,
        "comparison_note": "MAGE-only comparison refits C=30 on MAGE rows in the final "
                           "training partition. The combined model's recipe was selected "
                           "using training-only grouped CV. Both score identical final "
                           "heldout rows; this diagnostic never selects a model. This is "
                           "an internal corpus comparison, not an external benchmark.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "training_validation.json")
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified training and saved comparison -> {args.output}")


if __name__ == "__main__":
    main()
