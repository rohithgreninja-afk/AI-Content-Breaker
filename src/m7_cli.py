"""M7: one entry point that prints a verdict and the M1–M6 report.

Text or a file goes in. The existing pipeline does the analysis. This module
does not add a stage, and it does not blend the rule score into the model
score. M5a and M5b stay separate verdicts. A score is not a confidence.

    python src/m7_cli.py
    python src/m7_cli.py path\\to\\text.txt
    python src/m7_cli.py --text "..."
    python src/m7_cli.py --test N
    python src/m7_cli.py --eval
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline  # noqa: E402
from m4_features import N_FEATURES  # noqa: E402


def build_report(ctx: dict, source: str) -> dict:
    """Assemble the verdict from a finished M1–M6 context.

    rule_score is the M5a score. clf_score is the M5b score, or None when no
    model is loaded. The feature dict keeps M4's existing names and order.
    """
    return {
        "source": source,
        "n_characters": len(ctx["text"]),
        "n_words": ctx["n_words"],
        "abstain": ctx["abstain"],
        "min_words": pipeline.MIN_WORDS,
        "rule_score": ctx["rules"]["score"],
        "rule_verdict": ctx["verdicts"]["M5a"],
        "clf_score": ctx["model_score"],
        "model_verdict": ctx["verdicts"]["M5b"],
        "features": dict(ctx["features"]),
        "robustness": ctx["robustness"],
    }


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_report(report: dict, ctx: dict) -> str:
    """Verdict first, then the same stage transcript the pipeline prints."""
    model_verdict = report["model_verdict"]
    if model_verdict is None:
        model_verdict = "unavailable"
    lines = [
        f"M5a rule verdict = {report['rule_verdict']}  "
        f"rule_score={_score(report['rule_score'])}",
        f"M5b model verdict = {model_verdict}  "
        f"clf_score={_score(report['clf_score'])}",
    ]
    if report["abstain"]:
        lines.append(
            f"abstain: only {report['n_words']} words; both scorers need at "
            f"least {report['min_words']} words for a verdict"
        )
    lines.append(
        f"source: {report['source']} "
        f"({report['n_characters']} characters, {report['n_words']} words)"
    )
    if len(report["features"]) != N_FEATURES:
        raise ValueError(
            f"report has {len(report['features'])} features; M4 defines {N_FEATURES}"
        )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        for name, _run, display in pipeline.STAGES:
            pipeline._head(name)
            display(ctx)
    return "\n".join(lines) + "\n" + buffer.getvalue()


def _read_inputs(argv: list[str]) -> tuple[str, str]:
    """Same input order as pipeline.main: --text, then a path, then the sample."""
    if "--text" in argv:
        text = argv[argv.index("--text") + 1]
        return text, "--text"
    if argv and not argv[0].startswith("--"):
        with open(argv[0], encoding="utf-8") as handle:
            return handle.read(), argv[0]
    return pipeline.SAMPLE, "built-in sample"


def main(argv: list[str] | None = None) -> None:
    """Print the verdict and stage report, or forward a holdout check."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--eval" in argv or "--test" in argv:
        pipeline.main(argv)
        return
    text, source = _read_inputs(argv)
    ctx = pipeline.run_pipeline(text, show=False)
    print(format_report(build_report(ctx, source), ctx), end="")


if __name__ == "__main__":
    main()
