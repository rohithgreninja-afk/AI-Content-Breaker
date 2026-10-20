"""Verify the shipped classifier against its exact training rows and holdout.

Run after M0 and M5: python scripts/verify_training.py
Refits once at the already selected C, checks the saved coefficients/scaler,
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
    C = report["metrics"]["C"]
    refitted = ModelScorer._build_pipeline(C=C)
    refitted.fit(X[train], y[train])
    differences = {}
    for step, attribute in (("scale", "mean_"), ("scale", "scale_"),
                            ("logreg", "coef_"), ("logreg", "intercept_")):
        saved = getattr(model.pipeline.named_steps[step], attribute)
        fresh = getattr(refitted.named_steps[step], attribute)
        differences[f"{step}.{attribute}"] = float(np.max(np.abs(saved - fresh)))
        if not np.allclose(saved, fresh, rtol=1e-9, atol=1e-10):
            raise ValueError(f"saved parameters do not reproduce: {step}.{attribute}")

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
                   "model_report_metrics_match": True},
        "refit_parameter_max_abs_differences": differences,
        "same_holdout_comparison": comparison,
        "comparison_note": "MAGE-only comparison refits C=30 on MAGE rows in the final "
                           "training partition; combined model uses training-only CV to "
                           "choose C. Both score identical final heldout rows. This is "
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
