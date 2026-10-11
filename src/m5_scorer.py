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
    python src/m5_scorer.py --score FILE    score one text file with both
"""

from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m1_lexer import tokenize  # noqa: E402
from m2_symtab import build_symtab  # noqa: E402
from m3_parser import parse_tokens  # noqa: E402
from m4_features import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
    extract_features_from,
    to_vector,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "raid_clean.csv")
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
# hit rate) on the 760-row sample, subject to firing on 15-95% of AI rows so
# no rule is a near-constant. Weights follow the measured lift in tiers:
# >=0.30 -> 1.6, >=0.25 -> 1.4, >=0.20 -> 1.2, >=0.15 -> 0.9, >=0.10 -> 0.7,
# below that -> 0.4.
#
# Because these were fitted on the whole sample, rule accuracy measured on the
# same rows is in-sample and optimistic; the CLI reports a held-out figure on
# the same split M5b uses. Rerun `--audit` after changing the dataset: a rule
# whose lift has collapsed has stopped earning its weight.
#
# Two candidate rules were dropped rather than retuned. tpl_topic_sentence_rate
# and lex_transition_rate both came out with their best cut in the *opposite*
# direction to the hypothesis (AI using more topic sentences and connectives),
# at lift <= 0.10. Flipping a rule to chase a sign with no story behind it is
# curve-fitting, so they are out.
RULES: tuple[tuple[str, str, float, float], ...] = (
    # Vocabulary reuse: AI repeats less in the mid-frequency band and spaces
    # its repeats far more evenly.
    ("rep_dis_rate", "low", 0.1184, 1.6),
    ("rep_stdev_repeat_gap_log", "low", 3.2637, 1.6),
    # Pacing: uniform sentence lengths are the single clearest machine tell.
    ("sent_cv_len", "low", 0.2314, 1.6),
    # Punctuation: humans reach for dashes, commas and parentheses far more.
    ("punct_dash_rate", "low", 0.0108, 1.6),
    ("pos_punct_rate", "low", 0.0958, 1.4),
    ("punct_comma_rate", "low", 0.0242, 1.2),
    ("punct_paren_open_rate", "low", 0.0035, 1.2),
    # Structure: AI prose nests deeper and builds bigger, bushier trees.
    ("parse_mean_depth", "high", 9.2361, 1.4),
    ("parse_mean_branching", "high", 1.7335, 1.4),
    ("parse_mean_nodes", "high", 25.4773, 1.2),
    ("tpl_struct_complex_rate", "high", 0.1484, 0.9),
    ("parse_mean_subordinate", "high", 0.3038, 0.7),
    # Register: longer words, denser content, more nouns and verbs.
    ("lex_long_word_rate", "high", 0.4165, 1.2),
    ("pos_verb_rate", "high", 0.1435, 0.9),
    ("pos_content_rate", "high", 0.6256, 0.9),
    ("pos_noun_rate", "high", 0.3070, 0.9),
    # Boilerplate: weak as binary rules, kept at low weight because the
    # direction is consistent and theoretically motivated.
    ("lex_booster_rate", "high", 0.0068, 0.4),
    ("tpl_nominalization_chain_rate", "high", 0.0119, 0.4),
)

# Logistic squashing of the signed rule total.
#
# RULE_BIAS shifts the decision point, so it does set accuracy. It has to be
# positive: a text has to fire more than half the total weight to clear a
# zero-bias cutoff, and most texts fire fewer rules than that, which left the
# signed total negative for *both* classes (mean score human 0.08, ai 0.42)
# and the scorer predicting "human" far too often (recall 0.41). At +4.5 the
# boundary sits between the classes: accuracy 0.705 -> 0.774, recall -> 0.63.
#
# RULE_SCALE only sets how fast the score moves away from 0.5, so it cannot
# change the predicted labels; it is a calibration knob, chosen on Brier.
# Both come from calibrate_rules(); see --audit to re-derive them.
RULE_BIAS = 4.5
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
    """M5b: logistic regression over the M4 feature vector.

    Features are standardised before fitting, since the vector mixes rates in
    0-1 with log counts and raw depths. Wraps a scikit-learn pipeline; `fit`
    reports held-out accuracy and `save`/`load` persist it.
    """

    def __init__(self, pipeline=None, feature_names=FEATURE_NAMES) -> None:
        self.pipeline = pipeline
        self.feature_names = tuple(feature_names)

    # Chosen by 5-fold CV on the training split alone (never the test split),
    # over C in 0.01..100: train CV accuracy climbs 0.753 -> 0.867 and plateaus
    # at 30. Stronger regularisation underfits here; the 89 features genuinely
    # support a flexible boundary. Change it via fit(C=...) or select_C().
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
            random_state: int = 42, C: float | None = None) -> dict:
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
        from sklearn.model_selection import cross_val_score, train_test_split

        C = self.DEFAULT_C if C is None else C
        X = np.asarray(vectors, dtype=float)
        y = np.asarray([1 if label == AI_LABEL else 0 for label in labels],
                       dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

        self.pipeline = self._build_pipeline(C=C)
        self.pipeline.fit(X_train, y_train)

        train_pred = self.pipeline.predict(X_train)
        test_pred = self.pipeline.predict(X_test)
        test_prob = self.pipeline.predict_proba(X_test)[:, 1]

        # 5-fold CV on the TRAINING split only. Running it over train+test
        # would leak the test rows into a number reported beside test accuracy.
        cv = cross_val_score(self._build_pipeline(C=C), X_train, y_train, cv=5,
                             scoring="accuracy")

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
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
            "report": classification_report(
                y_test, test_pred, target_names=[HUMAN_LABEL, AI_LABEL],
                digits=3, zero_division=0),
        }

    @classmethod
    def select_C(cls, vectors, labels, grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0,
                                             10.0, 30.0, 100.0),
                 test_size: float = 0.25, random_state: int = 42) -> dict:
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
        train_idx, _ = split_indices(list(labels), test_size, random_state)

        rows = []
        best = None
        for C in grid:
            cv = cross_val_score(cls._build_pipeline(C=C), X[train_idx],
                                 y[train_idx], cv=5, scoring="accuracy")
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

        pipeline = self._require_fitted()
        vector = np.asarray([[features[n] for n in self.feature_names]],
                            dtype=float)
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
        coefficients = pipeline.named_steps["logreg"].coef_[0]
        ranked = sorted(zip(self.feature_names, coefficients),
                        key=lambda pair: -abs(pair[1]))
        return [(name, float(weight)) for name, weight in ranked[:n]]

    def save(self, path: str = MODEL_PATH) -> str:
        """Persist the fitted pipeline and its feature order."""
        import joblib

        self._require_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "feature_names": self.feature_names},
            path)
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
        return cls(pipeline=bundle["pipeline"], feature_names=names)


# --------------------------------------------------------------------------
# Dataset plumbing
# --------------------------------------------------------------------------

def load_dataset(path: str = DATA_PATH) -> tuple[list[dict], list[str]]:
    """Extract features for every row. Returns (feature_dicts, labels).

    Runs M1 -> M2 -> M3 once per row and reuses the artefacts, which is why
    this calls extract_features_from rather than extract_features.
    """
    features: list[dict] = []
    labels: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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
                  random_state: int = 42) -> tuple[list[int], list[int]]:
    """Stratified train/test index split, shared by both scorers.

    Using one split for M5a and M5b keeps their accuracies comparable.
    """
    from sklearn.model_selection import train_test_split

    stratify = [1 if label == AI_LABEL else 0 for label in labels]
    return train_test_split(
        list(range(len(labels))), test_size=test_size,
        random_state=random_state, stratify=stratify)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_rule_report(features: list[dict], labels: list[str],
                       test_idx: list[int] | None = None) -> None:
    scorer = RuleScorer()
    result = evaluate_rules(features, labels, scorer)
    print("== M5a rule-based scorer ==")
    print(f"rules={len(scorer.rules)} max_total={scorer.max_total:.1f} "
          f"scale={scorer.scale} bias={scorer.bias}")
    print(f"in-sample  n={result['n']} accuracy={result['accuracy']:.3f} "
          f"precision={result['precision']:.3f} recall={result['recall']:.3f} "
          f"f1={result['f1']:.3f}")
    if test_idx is not None:
        held = evaluate_rules([features[i] for i in test_idx],
                              [labels[i] for i in test_idx], scorer)
        # Thresholds were tuned on all rows, so this is the honest number to
        # compare against M5b; it shares M5b's split.
        print(f"held-out   n={held['n']} accuracy={held['accuracy']:.3f} "
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


def _print_model_report(features: list[dict], labels: list[str],
                        save: bool = False) -> None:
    model = ModelScorer()
    report = model.fit([to_vector(f) for f in features], labels)
    print("== M5b logistic regression ==")
    print(f"train={report['n_train']} test={report['n_test']} "
          f"features={report['n_features']} C={report['C']:g}")
    print(f"train_accuracy={report['train_accuracy']:.3f} "
          f"test_accuracy={report['test_accuracy']:.3f} "
          f"test_roc_auc={report['test_roc_auc']:.3f}")
    print(f"5-fold CV accuracy on train={report['cv_mean']:.3f} "
          f"+-{report['cv_std']:.3f}")
    print(f"confusion matrix [[hh, ha], [ah, aa]] = "
          f"{report['confusion_matrix']}")
    print(report["report"])
    print("top coefficients (standardised; + pushes toward AI):")
    for name, weight in model.top_coefficients(15):
        print(f"  {name:32} {weight:+8.4f}")
    if save:
        print(f"\nsaved -> {model.save()}")


def main(argv: list[str] | None = None) -> None:
    """Report both scorers on the RAID sample; see module docstring for flags."""
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
        print(f"dataset not found: {DATA_PATH}\n"
              f"regenerate it with: .venv\\Scripts\\python.exe "
              f"src\\m0_dataset.py")
        return

    print(f"loading {DATA_PATH} ...")
    features, labels = load_dataset()
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    print(f"{len(labels)} rows {counts}\n")

    if "--audit" in argv:
        _print_audit(features, labels)
        return

    _, test_idx = split_indices(labels)
    _print_rule_report(features, labels, test_idx)
    print()
    _print_model_report(features, labels, save="--train" in argv)


if __name__ == "__main__":
    main()
