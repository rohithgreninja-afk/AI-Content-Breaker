"""M5: scoring and classification over M4 feature vectors.

Two scorers that share one feature extractor and one interface
(text -> probability in 0.0-1.0 that the text is machine-generated):

M5a  RuleScorer    hand-tuned thresholds over named M4 features. No training,
                   fully inspectable: it reports which rules fired and what
                   each contributed.
M5b  ModelScorer   logistic regression on M4 vectors, with a train/test split
                   and an accuracy report. Persists to models/ as a .joblib.

Tuning is deliberately data-driven but hand-held: RULES below is a plain list
of (feature, direction, threshold, weight) rows, each one a claim about the
corpus that `python src/m5_scorer.py --audit` re-checks.

CLI:
    python src/m5_scorer.py                 rule + model report on the sample
    python src/m5_scorer.py --audit         per-rule hit rates, for tuning
    python src/m5_scorer.py --train         fit, report, and save the model
    python src/m5_scorer.py --train --select-c  choose C using training groups
    python src/m5_scorer.py --score FILE    score one text file with both
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m1_lexer import tokenize  # noqa: E402
from m2_symtab import build_symtab  # noqa: E402
from m3_parser import parse_tokens, tagger_mode  # noqa: E402
from m4_features import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
    extract_features_from,
    to_vector,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "train_clean.csv")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "m5_logreg.joblib")

# Label convention, fixed for both scorers: 1 = machine-generated.
AI_LABEL = "ai"
HUMAN_LABEL = "human"


# --------------------------------------------------------------------------
# M5a: rule-based scorer
# --------------------------------------------------------------------------

# (feature, direction, threshold, weight)
#   direction "high": fires when value >= threshold (feature is higher in AI)
#   direction "low":  fires when value <= threshold (feature is lower in AI)
#
# Each threshold is the cut point that maximised lift (AI hit rate minus human
# hit rate) on the TRAINING split of the MAGE sample (3550 rows), subject to
# firing on 15-95% of AI rows so no rule is a near-constant. Weights follow
# the measured lift in tiers: >=0.30 -> 1.6, >=0.25 -> 1.4, >=0.20 -> 1.2,
# >=0.15 -> 0.9, >=0.10 -> 0.7, below that -> 0.4. Rules under 0.05 lift are
# dropped. Fitting on train only keeps the held-out figure honest.
#
# On MAGE every surviving rule lands in the lowest tier (best lift 0.09): no
# single stylometric feature separates human from machine text across 10
# domains and 27 generators. The earlier RAID-tuned rules (lifts up to 0.35)
# were mostly measuring that all RAID humans wrote 96+ word abstracts. M5a is
# kept for inspectability, not accuracy (held-out 0.531); M5b carries the
# verdict.
#
# Dropped on MAGE (lift < 0.05 or no valid cut): sent_cv_len, punct_dash_rate,
# punct_paren_open_rate, tpl_struct_complex_rate, lex_long_word_rate,
# pos_content_rate, pos_noun_rate, lex_booster_rate,
# tpl_nominalization_chain_rate.
RULES: tuple[tuple[str, str, float, float], ...] = (
    # Vocabulary reuse: AI repeats less in the mid-frequency band and spaces
    # its repeats more evenly.
    ("rep_dis_rate", "low", 0.1061, 0.4),
    ("rep_stdev_repeat_gap_log", "low", 2.9202, 0.4),
    # Punctuation: humans use more commas and punctuation overall.
    ("pos_punct_rate", "low", 0.0826, 0.4),
    ("punct_comma_rate", "low", 0.0338, 0.4),
    # Structure: AI prose nests deeper and builds bigger, bushier trees.
    ("parse_mean_depth", "high", 5.7273, 0.4),
    ("parse_mean_branching", "high", 1.0514, 0.4),
    ("parse_mean_nodes", "high", 15.0625, 0.4),
    ("parse_mean_subordinate", "high", 0.2083, 0.4),
    # Register: more verbs.
    ("pos_verb_rate", "high", 0.1697, 0.4),
)

# Logistic squashing of the signed rule total.
#
# RULE_BIAS shifts the decision point, so it does set accuracy. On MAGE the
# train-split grid picks 0.0: the rules fire about as often as not, so the
# signed total is already centred. (The RAID-era +4.5 made "no evidence" read
# as AI, which is half of why short text scored as AI.)
#
# RULE_SCALE only sets how fast the score moves away from 0.5, so it cannot
# change the predicted labels; it is a calibration knob, chosen on Brier.
# Both come from calibrate_rules(); see --audit to re-derive them.
RULE_BIAS = 0.0
RULE_SCALE = 0.20


def _rule_fires(value: float, direction: str, threshold: float) -> bool:
    """Whether one rule's condition holds for a feature value."""
    if direction == "high":
        return value >= threshold
    if direction == "low":
        return value <= threshold
    raise ValueError(f"unknown rule direction: {direction!r}")


class RuleScorer:
    """M5a: hand-tuned threshold scorer over named M4 features.

    Each rule that fires adds its weight; each that does not subtracts it.
    The signed total is squashed into 0.0-1.0 by a logistic, so the output
    reads as a probability without any training step.
    """

    def __init__(self, rules=RULES, bias: float = RULE_BIAS,
                 scale: float = RULE_SCALE) -> None:
        unknown = [name for name, _, _, _ in rules if name not in FEATURE_NAMES]
        if unknown:
            raise ValueError(f"rules name unknown features: {unknown}")
        self.rules = tuple(rules)
        self.bias = bias
        self.scale = scale

    @property
    def max_total(self) -> float:
        """Total weight on the board, i.e. the score when every rule fires."""
        return sum(weight for _, _, _, weight in self.rules)

    def explain(self, features: dict[str, float]) -> dict:
        """Score one feature dict and report every rule's contribution.

        Returns {score, total, max_total, fired, missed}, where fired and
        missed hold (feature, value, threshold, contribution) rows.
        """
        total = 0.0
        fired: list[tuple[str, float, float, float]] = []
        missed: list[tuple[str, float, float, float]] = []
        for name, direction, threshold, weight in self.rules:
            value = features[name]
            if _rule_fires(value, direction, threshold):
                total += weight
                fired.append((name, value, threshold, weight))
            else:
                total -= weight
                missed.append((name, value, threshold, -weight))
        return {
            "score": 1.0 / (1.0 + math.exp(-self.scale * (total + self.bias))),
            "total": total,
            "max_total": self.max_total,
            "fired": fired,
            "missed": missed,
        }

    def score_features(self, features: dict[str, float]) -> float:
        """Probability in 0.0-1.0 that the text behind these features is AI."""
        return self.explain(features)["score"]

    def score(self, text: str) -> float:
        """Extract features from text and score it."""
        return self.score_features(extract_features(text))

    def predict(self, text: str, cutoff: float = 0.5) -> str:
        """Label one text as AI_LABEL or HUMAN_LABEL."""
        return AI_LABEL if self.score(text) >= cutoff else HUMAN_LABEL


# --------------------------------------------------------------------------
# M5b: logistic regression classifier
# --------------------------------------------------------------------------

class ModelScorer:
    """M5b: a saved probabilistic classifier over the M4 feature vector.

    The original fit/select_C helpers use standardised logistic regression.
    Train-only model selection can persist other scikit-learn pipelines with
    the same feature order and predict_proba interface.
    """

    def __init__(self, pipeline=None, feature_names=FEATURE_NAMES,
                 metadata: dict | None = None) -> None:
        self.pipeline = pipeline
        self.feature_names = tuple(feature_names)
        self.metadata = dict(metadata or {})

    # Chosen by 5-fold CV on the training split alone (never the test split),
    # over C in 0.01..100: on the MAGE sample train CV accuracy climbs
    # 0.594 -> 0.653 and plateaus at 10-30. Stronger regularisation underfits
    # here. Change it via fit(C=...) or select_C().
    DEFAULT_C = 30.0

    @staticmethod
    def _build_pipeline(max_iter: int = 2000, C: float = DEFAULT_C):
        """Standardiser plus logistic regression, built fresh each fit."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=max_iter, C=C)),
        ])

    def fit(self, vectors, labels, test_size: float = 0.25,
            random_state: int = 42, C: float | None = None,
            groups=None) -> dict:
        """Train on a split of (vectors, labels) and return a report.

        labels are the strings "ai" / "human"; they are mapped to 1 / 0 so the
        positive class is always AI, whatever order sklearn would infer.
        """
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )
        from sklearn.model_selection import cross_val_score

        C = self.DEFAULT_C if C is None else C
        X = np.asarray(vectors, dtype=float)
        y = np.asarray([1 if label == AI_LABEL else 0 for label in labels],
                       dtype=int)

        train_idx, test_idx = split_indices(
            list(labels), test_size, random_state, groups=groups)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = (np.asarray(groups)[train_idx]
                        if groups is not None else None)

        self.pipeline = self._build_pipeline(C=C)
        self.pipeline.fit(X_train, y_train)

        train_pred = self.pipeline.predict(X_train)
        test_pred = self.pipeline.predict(X_test)
        test_prob = self.pipeline.predict_proba(X_test)[:, 1]

        # 5-fold CV on the TRAINING split only. Running it over train+test
        # would leak the test rows into a number reported beside test accuracy.
        cv_splits = _cv_indices(X_train, y_train, train_groups, random_state)
        cv = cross_val_score(self._build_pipeline(C=C), X_train, y_train,
                             cv=cv_splits, scoring="accuracy", error_score="raise")

        return {
            "C": float(C),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features": X.shape[1],
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
            "cv_mean": float(cv.mean()),
            "cv_std": float(cv.std()),
            "cv_scores": cv.tolist(),
            "train_indices": list(map(int, train_idx)),
            "test_indices": list(map(int, test_idx)),
            "split_method": ("stratified_group" if groups is not None
                             else "stratified_rows"),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
            "report": classification_report(
                y_test, test_pred, target_names=[HUMAN_LABEL, AI_LABEL],
                digits=3, zero_division=0),
        }

    @classmethod
    def select_C(cls, vectors, labels, grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0,
                                             10.0, 30.0, 100.0),
                 test_size: float = 0.25, random_state: int = 42,
                 groups=None) -> dict:
        """Choose C by 5-fold CV on the training split only.

        The test split is never touched, so the accuracy fit() reports
        afterwards stays an honest held-out number. Returns
        {"best_C", "best_cv", "grid": [(C, mean, std), ...]}.
        """
        import numpy as np
        from sklearn.model_selection import cross_val_score

        X = np.asarray(vectors, dtype=float)
        y = np.asarray([1 if label == AI_LABEL else 0 for label in labels],
                       dtype=int)
        train_idx, _ = split_indices(list(labels), test_size, random_state,
                                     groups=groups)
        train_groups = (np.asarray(groups)[train_idx]
                        if groups is not None else None)
        cv_splits = _cv_indices(X[train_idx], y[train_idx], train_groups,
                               random_state)

        rows = []
        best = None
        for C in grid:
            cv = cross_val_score(cls._build_pipeline(C=C), X[train_idx],
                                 y[train_idx], cv=cv_splits, scoring="accuracy",
                                 error_score="raise")
            rows.append((float(C), float(cv.mean()), float(cv.std())))
            if best is None or cv.mean() > best[1]:
                best = (float(C), float(cv.mean()))
        return {"best_C": best[0], "best_cv": best[1], "grid": rows}

    def _require_fitted(self):
        if self.pipeline is None:
            raise RuntimeError("model is not fitted; call fit() or load()")
        return self.pipeline

    def score_features(self, features: dict[str, float]) -> float:
        """Probability in 0.0-1.0 that the text behind these features is AI."""
        import numpy as np
        from threadpoolctl import threadpool_limits

        pipeline = self._require_fitted()
        vector = np.asarray([[features[n] for n in self.feature_names]],
                            dtype=float)
        with threadpool_limits(limits=2):
            return float(pipeline.predict_proba(vector)[0, 1])

    def score(self, text: str) -> float:
        """Extract features from text and score it."""
        return self.score_features(extract_features(text))

    def predict(self, text: str, cutoff: float = 0.5) -> str:
        """Label one text as AI_LABEL or HUMAN_LABEL."""
        return AI_LABEL if self.score(text) >= cutoff else HUMAN_LABEL

    def top_coefficients(self, n: int = 15) -> list[tuple[str, float]]:
        """Largest-magnitude standardised coefficients, as (feature, weight).

        The pipeline standardises first, so these are directly comparable.
        """
        pipeline = self._require_fitted()
        classifier = pipeline.named_steps.get("logreg")
        if classifier is None or not hasattr(classifier, "coef_"):
            raise ValueError("this classifier has no signed linear feature coefficients")
        coefficients = classifier.coef_[0]
        if len(coefficients) != len(self.feature_names):
            raise ValueError("transformed linear coefficients do not map directly to M4 features")
        ranked = sorted(zip(self.feature_names, coefficients),
                        key=lambda pair: -abs(pair[1]))
        return [(name, float(weight)) for name, weight in ranked[:n]]

    def save(self, path: str = MODEL_PATH) -> str:
        """Persist the fitted pipeline and its feature order."""
        import joblib

        self._require_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = str(path) + ".tmp"
        joblib.dump(
            {"pipeline": self.pipeline, "feature_names": self.feature_names,
             "metadata": self.metadata}, temporary)
        os.replace(temporary, path)
        return path

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> ModelScorer:
        """Load a model saved by save(), checking the feature order matches."""
        import joblib

        bundle = joblib.load(path)
        names = tuple(bundle["feature_names"])
        if names != tuple(FEATURE_NAMES):
            raise ValueError(
                "saved model's feature order does not match M4's "
                f"FEATURE_NAMES ({len(names)} vs {len(FEATURE_NAMES)}); "
                "retrain with --train")
        metadata = bundle.get("metadata", {})
        if metadata.get("schema_version", 1) != 1:
            raise ValueError("unsupported saved model schema; retrain with --train")
        if metadata.get("tagger") and metadata["tagger"] != tagger_mode():
            raise ValueError(
                f"saved model requires {metadata['tagger']} POS tagging; "
                f"runtime uses {tagger_mode()}; restore the tagger or retrain")
        source_hash = metadata.get("feature_extractor_sha256")
        if source_hash and source_hash != feature_extractor_sha256():
            raise ValueError("feature extractor changed since training; retrain with --train")
        if "estimator_config" in metadata:
            _validate_estimator_config(bundle["pipeline"], metadata["estimator_config"])
        return cls(pipeline=bundle["pipeline"], feature_names=names,
                   metadata=metadata)


def _estimator_parameters(value):
    """Canonical constructor parameters, excluding learned state and repr addresses."""
    if hasattr(value, "get_params"):
        return {"class": f"{type(value).__module__}.{type(value).__qualname__}",
                "parameters": {name: _estimator_parameters(parameter)
                               for name, parameter in sorted(value.get_params(deep=False).items())}}
    if isinstance(value, dict):
        return {name: _estimator_parameters(parameter)
                for name, parameter in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_estimator_parameters(parameter) for parameter in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return _estimator_parameters(value.item())
    raise ValueError(f"unsupported estimator parameter type: {type(value).__name__}")


def _validate_estimator_config(pipeline, config: dict) -> None:
    """Reject an artifact whose estimator differs from its recorded recipe."""
    from m5_estimators import build_estimator

    if not isinstance(config, dict):
        raise ValueError("saved estimator configuration must be a dictionary")
    expected = build_estimator(config)
    if _estimator_parameters(pipeline) != _estimator_parameters(expected):
        raise ValueError("saved estimator does not match its recorded configuration; retrain")


# --------------------------------------------------------------------------
# Dataset plumbing
# --------------------------------------------------------------------------

def dataset_sha256(path: str = DATA_PATH) -> str:
    """Fingerprint the exact CSV used to train or evaluate a saved model."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_extractor_sha256() -> str:
    """Hash the source that defines the feature values, not only their names."""
    digest = hashlib.sha256()
    for filename in ("m1_lexer.py", "m2_symtab.py", "m3_parser.py", "m4_features.py"):
        digest.update(filename.encode("utf-8") + b"\0")
        with open(os.path.join(REPO_ROOT, "src", filename), "rb") as handle:
            digest.update(handle.read().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_versions() -> dict:
    """Exact versions needed to reproduce a training run."""
    import importlib.metadata
    import platform

    packages = {name: importlib.metadata.version(name)
                for name in ("scikit-learn", "nltk", "numpy", "joblib", "requests", "pyarrow")}
    return {"python": platform.python_version(), **packages}


def load_dataset_rows(path: str = DATA_PATH) -> list[dict]:
    """Read the CSV contract and reject ambiguous IDs or labels."""
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("training dataset is empty")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("training dataset contains duplicate row IDs")
    if any(row["label"] not in (AI_LABEL, HUMAN_LABEL) for row in rows):
        raise ValueError("training dataset labels must be 'ai' or 'human'")
    return rows


def dataset_metadata(path: str = DATA_PATH,
                     require_manifest: bool = True) -> dict:
    """Read M0's group assignments and verify they belong to this exact CSV."""
    rows = load_dataset_rows(path)
    result = {"dataset_sha256": dataset_sha256(path),
              "row_ids": [row["id"] for row in rows], "groups": None,
              "dataset_sources": {}}
    manifest_path = os.path.join(os.path.dirname(path), "train_manifest.json")
    if not os.path.exists(manifest_path):
        if require_manifest:
            raise ValueError("dataset manifest missing; regenerate with m0_dataset.py")
        return result
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("csv_sha256") != result["dataset_sha256"]:
        raise ValueError("dataset differs from its manifest; regenerate with m0_dataset.py")
    row_groups = manifest.get("row_groups", {})
    if set(row_groups) != set(result["row_ids"]):
        raise ValueError("dataset manifest group IDs do not match CSV rows")
    if any(not isinstance(group, str) or not group for group in row_groups.values()):
        raise ValueError("dataset manifest contains invalid group assignments")
    result.update(groups=[row_groups[row_id] for row_id in result["row_ids"]],
                  dataset_sources=manifest.get("sources", {}),
                  manifest_sha256=hashlib.sha256(json.dumps(
                      manifest, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")).hexdigest())
    return result


def load_dataset(path: str = DATA_PATH) -> tuple[list[dict], list[str]]:
    """Extract features for every row. Returns (feature_dicts, labels).

    Runs M1 -> M2 -> M3 once per row and reuses the artefacts, which is why
    this calls extract_features_from rather than extract_features.
    """
    features: list[dict] = []
    labels: list[str] = []
    for row in load_dataset_rows(path):
        tokens = tokenize(row["text"])
        features.append(extract_features_from(
            tokens, build_symtab(tokens), parse_tokens(tokens)))
        labels.append(row["label"])
    return features, labels


def evaluate_rules(features: list[dict], labels: list[str],
                   scorer: RuleScorer | None = None,
                   cutoff: float = 0.5) -> dict:
    """Accuracy of the rule scorer over a labelled feature set."""
    scorer = scorer or RuleScorer()
    scores = [scorer.score_features(f) for f in features]
    truth = [1 if label == AI_LABEL else 0 for label in labels]
    predictions = [1 if s >= cutoff else 0 for s in scores]

    correct = sum(1 for p, t in zip(predictions, truth) if p == t)
    true_pos = sum(1 for p, t in zip(predictions, truth) if p == 1 and t == 1)
    false_pos = sum(1 for p, t in zip(predictions, truth) if p == 1 and t == 0)
    false_neg = sum(1 for p, t in zip(predictions, truth) if p == 0 and t == 1)
    ai_scores = [s for s, t in zip(scores, truth) if t == 1]
    human_scores = [s for s, t in zip(scores, truth) if t == 0]

    precision = (true_pos / (true_pos + false_pos)
                 if (true_pos + false_pos) else 0.0)
    recall = (true_pos / (true_pos + false_neg)
              if (true_pos + false_neg) else 0.0)

    return {
        "n": len(truth),
        "accuracy": correct / len(truth) if truth else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)
               if (precision + recall) else 0.0),
        "mean_score_ai": sum(ai_scores) / len(ai_scores) if ai_scores else 0.0,
        "mean_score_human": (sum(human_scores) / len(human_scores)
                             if human_scores else 0.0),
        "scores": scores,
    }


def audit_rules(features: list[dict], labels: list[str],
                scorer: RuleScorer | None = None) -> list[dict]:
    """Per-rule hit rates by class, for threshold tuning.

    A useful rule fires much more often on one class than the other; `lift`
    is that gap. A rule with lift near zero is not earning its weight.
    """
    scorer = scorer or RuleScorer()
    ai = [f for f, label in zip(features, labels) if label == AI_LABEL]
    human = [f for f, label in zip(features, labels) if label != AI_LABEL]

    rows = []
    for name, direction, threshold, weight in scorer.rules:
        ai_hits = sum(1 for f in ai if _rule_fires(f[name], direction, threshold))
        human_hits = sum(
            1 for f in human if _rule_fires(f[name], direction, threshold))
        ai_rate = ai_hits / len(ai) if ai else 0.0
        human_rate = human_hits / len(human) if human else 0.0
        rows.append({
            "feature": name,
            "direction": direction,
            "threshold": threshold,
            "weight": weight,
            "ai_rate": ai_rate,
            "human_rate": human_rate,
            "lift": ai_rate - human_rate,
        })
    rows.sort(key=lambda row: -row["lift"])
    return rows


def calibrate_rules(features: list[dict], labels: list[str],
                    scorer: RuleScorer | None = None) -> dict:
    """Pick RULE_SCALE by Brier score, i.e. by how honest the number is.

    Accuracy cannot choose a scale: any scale > 0 is monotonic through the
    fixed 0.5 cutoff, so the predicted labels never change. What the scale does
    change is calibration, so this grids on Brier score (mean squared error
    against the 0/1 truth) and reports accuracy alongside for reference.

    Kept as a helper rather than run on import: RULE_SCALE is a checked-in
    constant so a given text always scores the same.
    """
    scorer = scorer or RuleScorer()
    truth = [1 if label == AI_LABEL else 0 for label in labels]

    # Bias first, because it moves the decision point and so does change
    # accuracy. Most texts fire fewer than half the rules, which leaves the
    # signed total negative for both classes; the bias re-centres it.
    bias_grid = []
    best_bias = None
    for bias in [b / 2 for b in range(-8, 21)]:
        candidate = RuleScorer(scorer.rules, bias, scorer.scale)
        result = evaluate_rules(features, labels, candidate)
        row = {"bias": bias, "accuracy": result["accuracy"],
               "f1": result["f1"], "precision": result["precision"],
               "recall": result["recall"]}
        bias_grid.append(row)
        if best_bias is None or row["accuracy"] > best_bias["accuracy"]:
            best_bias = row

    # Then scale, on Brier, holding that bias.
    grid = []
    best = None
    for scale in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.42, 0.50, 0.60, 0.80):
        candidate = RuleScorer(scorer.rules, best_bias["bias"], scale)
        result = evaluate_rules(features, labels, candidate)
        brier = _mean_squared(result["scores"], truth)
        row = {
            "scale": scale,
            "bias": best_bias["bias"],
            "accuracy": result["accuracy"],
            "brier": brier,
            "separation": result["mean_score_ai"] - result["mean_score_human"],
        }
        grid.append(row)
        if best is None or brier < best["brier"]:
            best = row
    return {"best": best, "grid": grid,
            "best_bias": best_bias, "bias_grid": bias_grid}


def _mean_squared(scores: list[float], truth: list[int]) -> float:
    """Brier score: mean squared error of probabilities against 0/1 truth."""
    if not truth:
        return 0.0
    return sum((s - t) ** 2 for s, t in zip(scores, truth)) / len(truth)


def split_indices(labels: list[str], test_size: float = 0.25,
                  random_state: int = 42,
                  groups=None) -> tuple[list[int], list[int]]:
    """Stratified train/test index split, shared by both scorers.

    Using one split for M5a and M5b keeps their accuracies comparable.
    """
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    if any(label not in (AI_LABEL, HUMAN_LABEL) for label in labels):
        raise ValueError("labels must be 'ai' or 'human'")
    stratify = [1 if label == AI_LABEL else 0 for label in labels]
    if groups is not None:
        if len(groups) != len(labels):
            raise ValueError("groups must have one assignment per row")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        n_splits = round(1 / test_size)
        if n_splits < 2 or not math.isclose(1 / n_splits, test_size):
            raise ValueError("grouped test_size must be reciprocal of an integer")
        if len(set(groups)) < n_splits:
            raise ValueError("too few independent groups for the held-out split")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=random_state)
        train, test = next(cv.split(list(range(len(labels))), stratify, groups))
        if any(len({labels[i] for i in part}) != 2 for part in (train, test)):
            raise ValueError("both classes must occur in train and held-out groups")
        return train.tolist(), test.tolist()
    return train_test_split(
        list(range(len(labels))), test_size=test_size,
        random_state=random_state, stratify=stratify)


def _cv_indices(X, y, groups=None, random_state: int = 42) -> list:
    """Training-only CV folds; related documents stay in one group."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if groups is None:
        return list(StratifiedKFold(n_splits=5).split(X, y))
    if len(set(groups)) < 5:
        raise ValueError("training CV requires at least five independent groups")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                    random_state=random_state)
    folds = list(splitter.split(X, y, groups))
    for train, validation in folds:
        if len(set(y[train])) != 2 or len(set(y[validation])) != 2:
            raise ValueError("both classes must occur in every grouped CV fold")
    return folds


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_rule_report(features: list[dict], labels: list[str],
                       test_idx: list[int] | None = None) -> None:
    scorer = RuleScorer()
    result = evaluate_rules(features, labels, scorer)
    print("== M5a frozen MAGE rule baseline (descriptive) ==")
    print(f"rules={len(scorer.rules)} max_total={scorer.max_total:.1f} "
          f"scale={scorer.scale} bias={scorer.bias}")
    print(f"in-sample  n={result['n']} accuracy={result['accuracy']:.3f} "
          f"precision={result['precision']:.3f} recall={result['recall']:.3f} "
          f"f1={result['f1']:.3f}")
    if test_idx is not None:
        held = evaluate_rules([features[i] for i in test_idx],
                              [labels[i] for i in test_idx], scorer)
        # Historical rule fitting used another MAGE split, so this number is
        # descriptive; only the newly trained model has this grouped holdout.
        print(f"test rows  n={held['n']} accuracy={held['accuracy']:.3f} "
              f"precision={held['precision']:.3f} "
              f"recall={held['recall']:.3f} f1={held['f1']:.3f}")
    truth = [1 if label == AI_LABEL else 0 for label in labels]
    print(f"mean score: human={result['mean_score_human']:.3f} "
          f"ai={result['mean_score_ai']:.3f} "
          f"separation={result['mean_score_ai']-result['mean_score_human']:+.3f} "
          f"brier={_mean_squared(result['scores'], truth):.3f}")


def _print_audit(features: list[dict], labels: list[str]) -> None:
    print("== M5a per-rule audit (sorted by lift) ==")
    print(f"{'feature':32} {'dir':5} {'thresh':>9} {'w':>4} "
          f"{'ai':>6} {'human':>6} {'lift':>7}")
    for row in audit_rules(features, labels):
        print(f"{row['feature']:32} {row['direction']:5} "
              f"{row['threshold']:9.4f} {row['weight']:4.1f} "
              f"{row['ai_rate']:6.3f} {row['human_rate']:6.3f} "
              f"{row['lift']:+7.3f}")
    calibration = calibrate_rules(features, labels)
    print("\nbias grid (bias moves the decision point, so it sets accuracy):")
    for row in calibration["bias_grid"]:
        if row["bias"] % 1 and row is not calibration["best_bias"]:
            continue  # thin the printout to whole steps plus the winner
        marker = " <- best" if row is calibration["best_bias"] else ""
        print(f"  bias={row['bias']:+5.1f} acc={row['accuracy']:.3f} "
              f"p={row['precision']:.3f} r={row['recall']:.3f} "
              f"f1={row['f1']:.3f}{marker}")
    print(f"\nscale grid at bias={calibration['best']['bias']:+.1f} "
          f"(accuracy is scale-invariant; pick on brier):")
    for row in calibration["grid"]:
        marker = " <- best" if row is calibration["best"] else ""
        print(f"  scale={row['scale']:.2f} acc={row['accuracy']:.3f} "
              f"brier={row['brier']:.4f} sep={row['separation']:+.3f}{marker}")


def _classification_metrics(labels, probabilities) -> dict:
    """Small JSON-safe metrics record, with AI always the positive class."""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    if not labels:
        return {"n": 0, "accuracy": None, "roc_auc": None}
    truth = [int(label == AI_LABEL) for label in labels]
    predictions = [int(probability >= 0.5) for probability in probabilities]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average="binary", zero_division=0)
    return {"n": len(labels), "accuracy": float(accuracy_score(truth, predictions)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "roc_auc": (float(roc_auc_score(truth, probabilities))
                        if len(set(truth)) == 2 else None),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist()}


def _heldout_metrics(model: ModelScorer, features: list[dict],
                     rows: list[dict], test_idx: list[int]) -> dict:
    """Metrics for every source and for the pipeline's 50-word abstention gate."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    rule_scorer = RuleScorer()
    if test_idx:
        vectors = np.asarray([[features[i][name] for name in model.feature_names]
                              for i in test_idx], dtype=float)
        with threadpool_limits(limits=2):
            probabilities = model.pipeline.predict_proba(vectors)[:, 1]
    else:
        probabilities = np.empty(0, dtype=float)
    predictions = dict(zip(test_idx, probabilities))
    rule_predictions = {i: rule_scorer.score_features(features[i]) for i in test_idx}

    def summarize(indices):
        scored = [i for i in indices if sum(any(c.isalpha() for c in word)
                  for word in rows[i]["text"].split()) >= 50]
        def metrics(subset):
            labels = [rows[i]["label"] for i in subset]
            return {"model": _classification_metrics(labels, [predictions[i] for i in subset]),
                    "rules": _classification_metrics(labels, [rule_predictions[i] for i in subset])}
        return {"n_test": len(indices), "n_scored": len(scored),
                "n_abstained": len(indices) - len(scored),
                "coverage": len(scored) / len(indices) if indices else 0.0,
                "all_rows": metrics(indices), "at_least_50_words": metrics(scored)}

    source_names = sorted({rows[i]["domain"].split(":", 1)[0] for i in test_idx})
    return {"overall": summarize(test_idx),
            "per_source": {source: summarize([i for i in test_idx
                           if rows[i]["domain"].split(":", 1)[0] == source])
                           for source in source_names},
            "rule_evaluation_note": "Frozen rules were tuned on the previous MAGE split; "
                                    "their metrics are descriptive, not an independent holdout."}


def _print_model_report(features: list[dict], labels: list[str],
                        save: bool = False, provenance: dict | None = None,
                        rows: list[dict] | None = None, select_c: bool = False) -> dict:
    provenance = provenance or {}
    groups = provenance.get("groups")
    vectors = [to_vector(f) for f in features]
    selection = ModelScorer.select_C(vectors, labels, groups=groups) if select_c else None
    C = selection["best_C"] if selection else ModelScorer.DEFAULT_C
    model = ModelScorer()
    report = model.fit(vectors, labels, groups=groups, C=C)
    print("== M5b logistic regression ==")
    print(f"train={report['n_train']} test={report['n_test']} "
          f"features={report['n_features']} C={report['C']:g} "
          f"split={report['split_method']}")
    print(f"train_accuracy={report['train_accuracy']:.3f} "
          f"test_accuracy={report['test_accuracy']:.3f} "
          f"test_roc_auc={report['test_roc_auc']:.3f}")
    print(f"5-fold CV accuracy on train={report['cv_mean']:.3f} "
          f"+-{report['cv_std']:.3f}")
    if selection:
        print(f"C selected using training groups only: {selection['best_C']:g}")
    print(f"confusion matrix [[hh, ha], [ah, aa]] = {report['confusion_matrix']}")
    print(report["report"])
    try:
        coefficients = model.top_coefficients(15)
    except ValueError as error:
        print(str(error))
    else:
        print("top coefficients (standardised; + pushes toward AI):")
        for name, weight in coefficients:
            print(f"  {name:32} {weight:+8.4f}")
    if save:
        from datetime import datetime, timezone
        from collections import Counter

        if rows is None or not provenance.get("dataset_sha256") or groups is None:
            raise ValueError("saving a training run requires verified dataset provenance")
        train_idx, test_idx = report["train_indices"], report["test_indices"]
        model.metadata = {
            "schema_version": 1, "dataset_sha256": provenance["dataset_sha256"],
            "manifest_sha256": provenance["manifest_sha256"],
            "dataset_sources": provenance["dataset_sources"],
            "train_ids": [rows[i]["id"] for i in train_idx],
            "test_ids": [rows[i]["id"] for i in test_idx],
            "train_groups": sorted({groups[i] for i in train_idx}),
            "test_groups": sorted({groups[i] for i in test_idx}),
            "tagger": tagger_mode(), "split_method": report["split_method"],
            "random_state": 42, "test_size": 0.25,
            "feature_names": list(FEATURE_NAMES),
            "feature_extractor_sha256": feature_extractor_sha256(),
            "versions": runtime_versions(),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        saved_report = {**model.metadata, "metrics": report, "c_selection": selection,
                        "test_source_counts": dict(Counter(rows[i]["domain"].split(":", 1)[0]
                                                           for i in test_idx)),
                        "evaluation": _heldout_metrics(model, features, rows, test_idx)}
        print(f"\nsaved -> {model.save()}")
        report_path = os.path.join(os.path.dirname(MODEL_PATH), "training_report.json")
        temporary = report_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(saved_report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, report_path)
        print(f"report -> {report_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    """Report both scorers on the training sample; see module docstring for flags."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--score" in argv:
        path = argv[argv.index("--score") + 1]
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        explanation = RuleScorer().explain(extract_features(text))
        print(f"M5a rule score = {explanation['score']:.3f} "
              f"(total {explanation['total']:+.1f} of "
              f"+-{explanation['max_total']:.1f})")
        print("fired:")
        for name, value, threshold, contribution in explanation["fired"]:
            print(f"  {name:32} {value:9.4f} vs {threshold:<9.4f} "
                  f"{contribution:+.1f}")
        if os.path.exists(MODEL_PATH):
            print(f"M5b model score = "
                  f"{ModelScorer.load().score(text):.3f}")
        else:
            print(f"M5b model not found at {MODEL_PATH}; run --train first.")
        return

    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"dataset not found: {DATA_PATH}\n"
                         f"regenerate it with: .venv\\Scripts\\python.exe "
                         f"src\\m0_dataset.py")

    print(f"loading {DATA_PATH} ...")
    provenance = dataset_metadata()
    rows = load_dataset_rows()
    features, labels = load_dataset()
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    print(f"{len(labels)} rows {counts}\n")

    train_idx, test_idx = split_indices(labels, groups=provenance["groups"])
    if "--audit" in argv:
        print("Rule audit and calibration use training groups only.")
        _print_audit([features[i] for i in train_idx], [labels[i] for i in train_idx])
        return

    _print_rule_report(features, labels, test_idx)
    print()
    _print_model_report(features, labels, save="--train" in argv,
                        provenance=provenance, rows=rows, select_c="--select-c" in argv)


if __name__ == "__main__":
    main()
