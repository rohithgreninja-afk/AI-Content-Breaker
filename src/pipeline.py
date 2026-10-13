"""Pipeline: run AI Content Breaker front to back on one text.

    python src/pipeline.py FILE            analyse a text file
    python src/pipeline.py --text "..."    analyse a string
    python src/pipeline.py --test N        analyse row N (0-189) of the held-out
                                           test split, the rows M5b never trained on
    python src/pipeline.py --eval          run every test-split row and report
                                           accuracy for M5a and M5b
    python src/pipeline.py                 analyse a built-in sample

Every stage is a (name, run, show) row. run(ctx) reads earlier results from the
shared ctx dict and stores its own; show(ctx) prints them.

Adding a stage later (M6, M7, ...):
    1. write run_m6(ctx) and show_m6(ctx)
    2. append ("M6 robustness", run_m6, show_m6) to STAGES
Nothing else changes. CORE_STAGES (M1 to M5) is the scoring path and never
includes M6, so a harness can call
    run_pipeline(perturbed_text, stages=CORE_STAGES, show=False)
to re-score perturbed text without recursing into itself.
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
    split_indices,
)

SAMPLE = (
    "The proposed method improves segmentation accuracy on noisy images. "
    "Although the results are promising, we note that the training cost "
    "remains high. Using a small graph, we show that the approach scales."
)
# Both scorers were tuned on RAID rows of 40+ characters; far shorter text
# scores unreliably.
MIN_WORDS_NOTE = 20
SHOWN_FEATURES = (
    "parse_mean_depth", "parse_mean_branching", "sent_cv_len",
    "rep_word_entropy", "lex_type_token_ratio", "punct_dash_rate",
    "pos_punct_rate",
)
_MODEL: dict = {}


def _model() -> tuple[ModelScorer | None, str]:
    """Load the saved M5b model once. Returns (model_or_None, note)."""
    if "model" not in _MODEL:
        if not os.path.exists(MODEL_PATH):
            _MODEL["model"] = (None, "no saved model; run m5_scorer.py --train")
        else:
            try:
                _MODEL["model"] = (ModelScorer.load(), "")
            except ValueError as error:
                _MODEL["model"] = (None, str(error))
    return _MODEL["model"]


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
def run_m5(ctx: dict) -> None:
    ctx["rules"] = RuleScorer().explain(ctx["features"])
    model, note = _model()
    ctx["model_score"] = model.score_features(ctx["features"]) if model else None
    ctx["model_note"] = note


def show_m5(ctx: dict) -> None:
    rules = ctx["rules"]
    n_words = sum(s["n_words"] for s in ctx["meta"]["sentences"])
    if n_words < MIN_WORDS_NOTE:
        print(f"note: only {n_words} words; scores are unreliable on text this short")
    verdict = "ai" if rules["score"] >= 0.5 else "human"
    print(f"M5a rule score = {rules['score']:.3f} -> {verdict} "
          f"(total {rules['total']:+.1f} of +-{rules['max_total']:.1f}, "
          f"{len(rules['fired'])} fired, {len(rules['missed'])} missed)")
    for name, value, threshold, weight in rules["fired"]:
        print(f"  fired  {name:30} {value:9.4f} vs {threshold:<9.4f} {weight:+.1f}")
    if ctx["model_score"] is None:
        print(f"M5b model: {ctx['model_note']}")
    else:
        verdict = "ai" if ctx["model_score"] >= 0.5 else "human"
        print(f"M5b model score = {ctx['model_score']:.3f} -> {verdict}")


CORE_STAGES = (
    ("M1 lexer", run_m1, show_m1),
    ("M2 symbol table", run_m2, show_m2),
    ("M3 parser + templates", run_m3, show_m3),
    ("M4 features", run_m4, show_m4),
    ("M5 scoring", run_m5, show_m5),
)
# Add M6 / M7 here, after M5, e.g. STAGES.append(("M6 robustness", run_m6, show_m6)).
STAGES = list(CORE_STAGES)


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
    """Rows of M5's held-out test split, in split order."""
    with open(DATA_PATH, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _, test_idx = split_indices([row["label"] for row in rows])
    return [rows[i] for i in test_idx]


def test_split_row(n: int) -> tuple[str, str]:
    """Row n (0-based) of M5's held-out test split, as (text, true_label)."""
    rows = _test_rows()
    if not 0 <= n < len(rows):
        raise SystemExit(f"--test N must be between 0 and {len(rows) - 1}")
    return rows[n]["text"], rows[n]["label"]


def _report(name: str, scores: list[float], truth: list[int]) -> None:
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
    """Push every held-out test row through the pipeline and report accuracy."""
    rows = _test_rows()
    truth = [1 if row["label"] == "ai" else 0 for row in rows]
    rule_scores: list[float] = []
    model_scores: list[float | None] = []
    for row in rows:
        ctx = run_pipeline(row["text"], stages=CORE_STAGES, show=False)
        rule_scores.append(ctx["rules"]["score"])
        model_scores.append(ctx["model_score"])
    print(f"held-out test split: {len(rows)} rows "
          f"({sum(truth)} ai, {len(rows) - sum(truth)} human)")
    _report("M5a rules", rule_scores, truth)
    if None in model_scores:
        print(f"M5b model: {ctx['model_note']}")
    else:
        _report("M5b model", model_scores, truth)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--eval" in argv:
        evaluate_test_split()
        return
    if "--text" in argv:
        text = argv[argv.index("--text") + 1]
        source = "--text"
    elif "--test" in argv:
        n = int(argv[argv.index("--test") + 1])
        text, label = test_split_row(n)
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
