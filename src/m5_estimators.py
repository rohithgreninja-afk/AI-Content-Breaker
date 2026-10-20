"""Explicit, reproducible sklearn candidates for the fixed M4 feature vector."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _keys(config: dict, expected: set[str]) -> None:
    if set(config) != expected:
        raise ValueError(f"estimator configuration keys must be {sorted(expected)}")


def _logreg(C: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("logreg", LogisticRegression(C=C, max_iter=2000))])


def _boosting(config: dict) -> HistGradientBoostingClassifier:
    _keys(config, {"max_iter", "learning_rate", "max_leaf_nodes",
                   "min_samples_leaf", "l2_regularization"})
    return HistGradientBoostingClassifier(
        **config, random_state=42, early_stopping=False, categorical_features=None)


def build_estimator(config: dict) -> Pipeline:
    """Build a fresh candidate; no fitting or data access happens here."""
    if not isinstance(config, dict):
        raise ValueError("estimator configuration must be an object")
    family = config.get("family")
    if family == "logreg":
        _keys(config, {"family", "C"})
        return _logreg(config["C"])
    if family == "hist_gradient_boosting":
        classifier = _boosting({key: value for key, value in config.items() if key != "family"})
    elif family == "extra_trees":
        _keys(config, {"family", "n_estimators", "min_samples_leaf", "max_features"})
        classifier = ExtraTreesClassifier(
            **{key: value for key, value in config.items() if key != "family"},
            random_state=42, n_jobs=2)
    elif family == "soft_vote":
        _keys(config, {"family", "logreg_C", "hgb", "weights"})
        classifier = VotingClassifier(
            estimators=[("linear", _logreg(config["logreg_C"])),
                        ("nonlinear", _boosting(config["hgb"]))],
            voting="soft", weights=config["weights"])
    else:
        raise ValueError(f"unsupported estimator family: {family!r}")
    return Pipeline([("classifier", classifier)])
