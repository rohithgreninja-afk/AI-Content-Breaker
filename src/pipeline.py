"""Pipeline: run AI Content Breaker front to back on one text.

    python src/pipeline.py FILE            analyse a text file
    python src/pipeline.py --text "..."    analyse a string
    python src/pipeline.py --test N        analyse row N (0-based) of the held-out
                                           test split, the rows M5b never trained on
    python src/pipeline.py --eval          run every test-split row and report
                                           accuracy for M5a and M5b
    python src/pipeline.py                 analyse a built-in sample

Every stage is a (name, run, show) row. run(ctx) reads earlier results from the
shared ctx dict and stores its own; show(ctx) prints them.

CORE_STAGES (M1 to M5) is the scoring path and never includes M6. M6 is the
next row in STAGES: it rescores fixed edits through CORE_STAGES, so a
perturbation cannot recurse into the harness.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m1_lexer import tokenize  # noqa: E402
from m2_symtab import build_symtab, top_words  # noqa: E402
from m3_parser import parse_tokens, render_tree  # noqa: E402
from m4_features import N_FEATURES, extract_features_from  # noqa: E402
from m5_scorer import (  # noqa: E402
    DATA_PATH,
    MODEL_PATH,
    ModelScorer,
    RuleScorer,
    dataset_metadata,
)

SAMPLE = (
    "The proposed method improves segmentation accuracy on noisy images. "
    "Although the results are promising, we note that the training cost "
    "remains high. Using a small graph, we show that the approach scales "
    "to larger inputs without retuning. We also compare against three "
    "baselines on two public benchmarks and report where each one fails."
)
# Below this many words both scorers abstain instead of giving a verdict.
# The features need text to measure: on the MAGE held-out split M5b scores
# 0.54 at 20-50 words (near chance) and 0.64+ above 50. (On the earlier RAID
# data, where every human text was 96+ words, short text was also called AI
# almost every time.)
MIN_WORDS = 50
SHOWN_FEATURES = (
    "parse_mean_depth", "parse_mean_branching", "sent_cv_len",
    "rep_word_entropy", "lex_type_token_ratio", "punct_dash_rate",
    "pos_punct_rate",
)
_MODEL: dict = {}


def _model() -> tuple[ModelScorer | None, str]:
    """Reload when training replaces the artifact in a long-running process."""
    signature = _model_signature()
    if "model" not in _MODEL or _MODEL.get("signature") != signature:
        if not os.path.exists(MODEL_PATH):
            _MODEL["model"] = (None, "no saved model; run m5_scorer.py --train")
        else:
            try:
                _MODEL["model"] = (ModelScorer.load(), "")
            except ValueError as error:
                _MODEL["model"] = (None, str(error))
        _MODEL["signature"] = signature
    return _MODEL["model"]


def _model_signature() -> tuple[int, int] | None:
    try:
        stat = os.stat(MODEL_PATH)
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


def _head(name: str) -> None:
    print(f"\n== {name} ==")


# ---- M1 -------------------------------------------------------------------
def run_m1(ctx: dict) -> None:
    ctx["tokens"] = tokenize(ctx["text"])


def show_m1(ctx: dict) -> None:
    kinds = Counter(token["type"] for token in ctx["tokens"])
    print(f"{len(ctx['tokens'])} tokens: {dict(kinds)}")


# ---- M2 -------------------------------------------------------------------
def run_m2(ctx: dict) -> None:
    ctx["sym_tab"] = build_symtab(ctx["tokens"])


def show_m2(ctx: dict) -> None:
    print(f"{len(ctx['sym_tab'])} unique words")
    print("top words:", top_words(ctx["sym_tab"], 5))


# ---- M3 -------------------------------------------------------------------
def run_m3(ctx: dict) -> None:
    ctx["meta"] = parse_tokens(ctx["tokens"])


def show_m3(ctx: dict) -> None:
    meta = ctx["meta"]
    print(f"tagger={meta['tagger']} sentences={meta['n_sentences']} "
          f"parse_success_rate={meta['parse_success_rate']:.3f} "
          f"mean_coverage={meta['mean_coverage']:.3f}")
    print(f"mean_depth={meta['mean_depth']:.2f} "
          f"mean_branching={meta['mean_branching']:.2f} "
          f"mean_nodes={meta['mean_nodes']:.1f}")
    print("structure:", {k: v for k, v in meta["structure_counts"].items() if v})
    print("templates:", {k: v for k, v in meta["template_counts"].items() if v})
    if meta["sentences"]:
        first = meta["sentences"][0]
        print(f"first sentence tags: {' '.join(first['tags'])}")
        if first["tree"]:
            print(render_tree(first["tree"], 1))
        else:
            print("  (no tree: parse failed)")


# ---- M4 -------------------------------------------------------------------
def run_m4(ctx: dict) -> None:
    ctx["features"] = extract_features_from(
        ctx["tokens"], ctx["sym_tab"], ctx["meta"])


def show_m4(ctx: dict) -> None:
    print(f"{N_FEATURES} features; selected:")
    for name in SHOWN_FEATURES:
        print(f"  {name:28} {ctx['features'][name]:10.4f}")


# ---- M5 -------------------------------------------------------------------
def _verdict(score: float | None, abstain: bool) -> str | None:
    """Return "abstain", "ai" or "human"; None when there is no score (no model)."""
    if score is None:
        return None
    if abstain:
        return "abstain"
    return "ai" if score >= 0.5 else "human"


def run_m5(ctx: dict) -> None:
    # Whitespace-separated pieces holding at least one letter, the same unit
    # M0 uses for its word limits. M1 tokens would count a hex hash or URL as
    # dozens of words and let junk through the gate.
    ctx["n_words"] = sum(
        1 for piece in ctx["text"].split() if any(c.isalpha() for c in piece))
    ctx["abstain"] = ctx["n_words"] < MIN_WORDS
    # Scores are still computed so the rule breakdown stays inspectable, but
    # below MIN_WORDS they are not a verdict; read ctx["verdicts"], not 0.5.
    ctx["rules"] = RuleScorer().explain(ctx["features"])
    model, note = _model()
    ctx["model_score"] = model.score_features(ctx["features"]) if model else None
    ctx["model_note"] = note
    ctx["verdicts"] = {
        "M5a": _verdict(ctx["rules"]["score"], ctx["abstain"]),
        "M5b": _verdict(ctx["model_score"], ctx["abstain"]),
    }


def show_m5(ctx: dict) -> None:
    if ctx["abstain"]:
        print(f"abstain: only {ctx['n_words']} words; both scorers need at "
              f"least {MIN_WORDS} words for a verdict")
        return
    rules = ctx["rules"]
    print(f"M5a rule score = {rules['score']:.3f} -> {ctx['verdicts']['M5a']} "
          f"(total {rules['total']:+.1f} of +-{rules['max_total']:.1f}, "
          f"{len(rules['fired'])} fired, {len(rules['missed'])} missed)")
    for name, value, threshold, weight in rules["fired"]:
        print(f"  fired  {name:30} {value:9.4f} vs {threshold:<9.4f} {weight:+.1f}")
    if ctx["model_score"] is None:
        print(f"M5b model: {ctx['model_note']}")
    else:
        print(f"M5b model score = {ctx['model_score']:.3f} -> "
              f"{ctx['verdicts']['M5b']}")


def run_m6(ctx: dict) -> None:
    """Measure verdict movement. Reuses the M1–M5 context already in ctx."""
    from m6_harness import evaluate

    ctx["robustness"] = evaluate(ctx["text"], baseline=ctx)


def show_m6(ctx: dict) -> None:
    from m6_harness import format_report

    print(format_report(ctx["robustness"]), end="")


CORE_STAGES = (
    ("M1 lexer", run_m1, show_m1),
    ("M2 symbol table", run_m2, show_m2),
    ("M3 parser + templates", run_m3, show_m3),
    ("M4 features", run_m4, show_m4),
    ("M5 scoring", run_m5, show_m5),
)
STAGES = [
    *CORE_STAGES,
    ("M6 robustness", run_m6, show_m6),
]


def run_pipeline(text: str, stages=None, show: bool = True) -> dict:
    """Run the stages in order over text and return the shared ctx dict."""
    ctx: dict = {"text": text}
    for name, run, display in (STAGES if stages is None else stages):
        run(ctx)
        if show:
            _head(name)
            display(ctx)
    return ctx


def _test_rows() -> list[dict]:
    """Use the saved model's actual holdout, after checking data provenance."""
    if not os.path.exists(MODEL_PATH):
        raise ValueError("no saved model; run src/m5_scorer.py --train first")
    model = ModelScorer.load()
    provenance = dataset_metadata(DATA_PATH)
    if model.metadata.get("dataset_sha256") != provenance["dataset_sha256"]:
        raise ValueError("saved model does not match the current dataset; "
                         "run src/m5_scorer.py --train before --eval or --test")
    if (model.metadata.get("manifest_sha256")
            != provenance.get("manifest_sha256")):
        raise ValueError("saved model does not match the dataset manifest; retrain")
    with open(DATA_PATH, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in rows}
    train_ids = model.metadata.get("train_ids", [])
    test_ids = model.metadata.get("test_ids", [])
    if (not train_ids or not test_ids or set(train_ids) & set(test_ids)
            or len(train_ids) != len(set(train_ids))
            or len(test_ids) != len(set(test_ids))
            or len(set(train_ids + test_ids)) != len(rows)
            or set(train_ids + test_ids) != set(by_id)):
        raise ValueError("saved model has invalid split provenance; retrain")
    groups = dict(zip(provenance["row_ids"], provenance["groups"]))
    if {groups[i] for i in train_ids} & {groups[i] for i in test_ids}:
        raise ValueError("saved model split contains related texts in both "
                         "training and test data; retrain")
    # The object whose provenance was checked must also supply the scores.
    _MODEL.update(model=(model, ""), signature=_model_signature())
    return [by_id[row_id] for row_id in test_ids]


def test_split_row(n: int) -> tuple[str, str]:
    """Row n (0-based) of M5's held-out test split, as (text, true_label)."""
    rows = _test_rows()
    if not 0 <= n < len(rows):
        raise SystemExit(f"--test N must be between 0 and {len(rows) - 1}")
    return rows[n]["text"], rows[n]["label"]


def _report(name: str, scores: list[float], truth: list[int]) -> None:
    if not truth:
        print(f"{name:10} no rows left to score")
        return
    pred = [1 if score >= 0.5 else 0 for score in scores]
    tp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 1)
    tn = len(truth) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"{name:10} accuracy={(tp + tn) / len(truth):.3f} "
          f"precision={precision:.3f} recall={recall:.3f}  "
          f"confusion [[hh, ha], [ah, aa]] = [[{tn}, {fp}], [{fn}, {tp}]]")


def evaluate_test_split() -> None:
    """Push every held-out test row through the pipeline and report accuracy.

    Rows under MIN_WORDS get no verdict, so they are counted and left out of
    the accuracy figures rather than scored as right or wrong.
    """
    # Validate before running any feature extraction or reporting accuracy.
    rows = _test_rows()
    truth: list[int] = []
    rule_scores: list[float] = []
    model_scores: list[float | None] = []
    abstained = {"ai": 0, "human": 0}
    for row in rows:
        ctx = run_pipeline(row["text"], stages=CORE_STAGES, show=False)
        if ctx["abstain"]:
            abstained[row["label"]] += 1
            continue
        truth.append(1 if row["label"] == "ai" else 0)
        rule_scores.append(ctx["rules"]["score"])
        model_scores.append(ctx["model_score"])
    n_ai = sum(1 for row in rows if row["label"] == "ai")
    print(f"held-out test split: {len(rows)} rows "
          f"({n_ai} ai, {len(rows) - n_ai} human)")
    print(f"abstained (under {MIN_WORDS} words): "
          f"{abstained['ai'] + abstained['human']} rows "
          f"({abstained['ai']} ai, {abstained['human']} human); "
          f"scoring the other {len(truth)}")
    _report("M5a rules", rule_scores, truth)
    print("M5a is a frozen MAGE comparison; its historical tuning rows may "
          "overlap this holdout.")
    model, note = _model()
    if model is None:
        print(f"M5b model: {note}")
    else:
        _report("M5b model", model_scores, truth)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--eval" in argv:
        try:
            evaluate_test_split()
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(str(error)) from error
        return
    if "--text" in argv:
        text = argv[argv.index("--text") + 1]
        source = "--text"
    elif "--test" in argv:
        n = int(argv[argv.index("--test") + 1])
        try:
            text, label = test_split_row(n)
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(str(error)) from error
        source = f"test split row {n} (true label: {label})"
    elif argv and not argv[0].startswith("--"):
        with open(argv[0], encoding="utf-8") as handle:
            text = handle.read()
        source = argv[0]
    else:
        text, source = SAMPLE, "built-in sample"
    print(f"input: {source} ({len(text)} characters)")
    run_pipeline(text)


if __name__ == "__main__":
    main()
