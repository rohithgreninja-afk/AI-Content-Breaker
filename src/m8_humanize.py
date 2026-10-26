"""M8: attribute-directed translation.

M5a names the synthesized attributes that sit on the AI side of a frozen
rule. This pass applies rewrites that lower the detector score, then runs
M1–M5 again. A human verdict does not stop it. Calling it again on the
returned text continues. It is an entry point, not a pipeline stage.

    python src/m8_humanize.py
    python src/m8_humanize.py path\\to\\text.txt
    python src/m8_humanize.py --text "..."
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6_harness import (  # noqa: E402
    COORDINATORS,
    SYNONYMS,
    _match_case,
    _scan,
    _sentences,
    _splice,
    _split_coordination,
)
from pipeline import CORE_STAGES, MIN_WORDS, SAMPLE, run_pipeline  # noqa: E402

# One rewrite per call. The next call continues from the returned text.
MAX_PASSES = 1
# A leading word is the subject side; these open a later clause.
SUBORDINATORS = frozenset({"because", "although", "while", "which", "if"})
# M5a can fire this rule, and no local rewrite lowers the verb share
# without deleting a verb.
UNSUPPORTED = "pos_verb_rate"

PURPOSE = (
    "Applies the rewrite for a fired M5a rule, then scores the passage again. "
    "M5b is reported and does not choose the rewrite."
)


def _score(text: str) -> dict:
    """M1–M5 only, so a translation pass cannot call M6 or itself."""
    return run_pipeline(text, stages=CORE_STAGES, show=False)


def _word_count(tokens: list[dict]) -> int:
    return sum(1 for token in tokens if token["type"] == "WORD")


def _longest(sentences: list[list[dict]]) -> list[list[dict]]:
    """Longer sentences first. Equal lengths keep source order."""
    return sorted(sentences, key=_word_count, reverse=True)


def rewrite_structure(text: str) -> tuple[str, bool]:
    """Split the longest coordinated sentence into two sentences."""
    best: str | None = None
    best_words = -1
    for sentence in _sentences(_scan(text)):
        edited = _split_coordination(text, sentence)
        if edited is None:
            continue
        words = _word_count(sentence)
        if words > best_words:
            best = edited
            best_words = words
    if best is None or best == text:
        return text, False
    return best, True


def _split_before(text: str, token: dict) -> str | None:
    """Turn the gap before token into a sentence boundary and capitalize it."""
    start = token["start"]
    cursor = start - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    if cursor < 0:
        return None
    insert_at = cursor if text[cursor] == "," else cursor + 1
    edited = _splice(text, insert_at, start, ". ")
    word_at = insert_at + 2
    if word_at < len(edited) and edited[word_at].isalpha():
        edited = edited[:word_at] + edited[word_at].upper() + edited[word_at + 1:]
    return edited if edited != text else None


def rewrite_subordinate(text: str) -> tuple[str, bool]:
    """Split the longest sentence before one non-initial subordinator."""
    for sentence in _longest(_sentences(_scan(text))):
        words = [token for token in sentence if token["type"] == "WORD"]
        for index, token in enumerate(words):
            if index == 0 or index == len(words) - 1:
                continue
            if token["value"].lower() not in SUBORDINATORS:
                continue
            edited = _split_before(text, token)
            if edited is not None:
                return edited, True
    return text, False


def rewrite_commas(text: str) -> tuple[str, bool]:
    """Insert a comma before every bare coordinator that has a word before it."""
    positions: list[int] = []
    for sentence in _sentences(_scan(text)):
        words = [token for token in sentence if token["type"] == "WORD"]
        for index, token in enumerate(words):
            if index == 0 or token["value"].lower() not in COORDINATORS:
                continue
            cursor = token["start"] - 1
            while cursor >= 0 and text[cursor].isspace():
                cursor -= 1
            if cursor < 0 or not text[cursor].isalnum():
                continue
            positions.append(cursor + 1)
    if not positions:
        return text, False
    edited = text
    for position in reversed(positions):
        edited = _splice(edited, position, position, ",")
    return (edited, True) if edited != text else (text, False)


def _word_rows(text: str) -> tuple[dict[str, int], dict[str, int], list[dict]]:
    """Lowercased counts, first stream position, and the word tokens."""
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    words: list[dict] = []
    for token in _scan(text):
        if token["type"] != "WORD":
            continue
        words.append(token)
        key = token["value"].lower()
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, token["start"])
    return counts, first, words


def _replace_key(text: str, words: list[dict], source: str, replacement: str,
                 limit: int | None = None) -> str:
    """Replace source with replacement. limit counts from the end of the text."""
    hits = [token for token in words if token["value"].lower() == source]
    if limit is not None:
        hits = hits[-limit:]
    edited = text
    for token in reversed(hits):
        edited = _splice(
            edited, token["start"], token["end"],
            _match_case(token["value"], replacement),
        )
    return edited


def rewrite_doubleton(text: str) -> tuple[str, bool]:
    """Collapse one listed synonym pair onto the more frequent word.

    A tie keeps the word that appears first. One pair per pass, in the
    synonym-table order.
    """
    counts, first, words = _word_rows(text)
    for source, target in SYNONYMS.items():
        if counts.get(source, 0) < 1 or counts.get(target, 0) < 1:
            continue
        if counts[source] > counts[target]:
            keep, drop = source, target
        elif counts[target] > counts[source]:
            keep, drop = target, source
        elif first[source] <= first[target]:
            keep, drop = source, target
        else:
            keep, drop = target, source
        edited = _replace_key(text, words, drop, keep)
        if edited != text:
            return edited, True
    return text, False


def rewrite_repeat_gap(text: str) -> tuple[str, bool]:
    """Replace the last copy of one word that already occurs three times.

    The most repeated listed word wins. Ties go to the earliest word.
    """
    counts, first, words = _word_rows(text)
    candidates = [
        (count, -first[key], key)
        for key, count in counts.items()
        if count >= 3 and key in SYNONYMS
    ]
    if not candidates:
        return text, False
    _count, _start, key = max(candidates)
    edited = _replace_key(text, words, key, SYNONYMS[key], limit=1)
    return (edited, True) if edited != text else (text, False)


# Fixed schedule. Structure runs before punctuation and lexicon because a
# split moves the later sites. Attributes are the M5a names for that family.
FAMILIES = (
    ("structure", "Split coordinated sentence",
     ("parse_mean_depth", "parse_mean_nodes", "parse_mean_branching"),
     rewrite_structure),
    ("subordinate", "Split at subordinator",
     ("parse_mean_subordinate",),
     rewrite_subordinate),
    ("punctuation", "Insert commas before coordinators",
     ("punct_comma_rate", "pos_punct_rate"),
     rewrite_commas),
    ("doubleton", "Collapse synonym pair",
     ("rep_dis_rate",),
     rewrite_doubleton),
    ("repeat_gap", "Replace a later repeat",
     ("rep_stdev_repeat_gap_log",),
     rewrite_repeat_gap),
)


def _passed(ctx: dict) -> bool:
    """Both verdicts human. M5a alone decides when no model is loaded."""
    if ctx["abstain"] or ctx["verdicts"]["M5a"] != "human":
        return False
    if ctx["model_score"] is None:
        return True
    return ctx["verdicts"]["M5b"] == "human"


def _side(ctx: dict) -> dict:
    return {
        "n_words": ctx["n_words"],
        "abstain": ctx["abstain"],
        "rule_score": ctx["rules"]["score"],
        "model_score": ctx["model_score"],
        "rule_verdict": ctx["verdicts"]["M5a"],
        "model_verdict": ctx["verdicts"]["M5b"],
        "fired": [row[0] for row in ctx["rules"]["fired"]],
    }


def _tracked(ctx: dict) -> float | None:
    """The AI score a later press has to push down. M5b when a model is loaded."""
    if ctx["model_score"] is not None:
        return ctx["model_score"]
    return ctx["rules"]["score"]


def _candidate(text: str, ctx: dict, ident: str, name: str,
               chosen: str, rewrite) -> dict | None:
    """One rewrite, kept only when it lowers the tracked AI score."""
    rewritten, applied = rewrite(text)
    if not applied or rewritten == text:
        return None
    nxt = _score(rewritten)
    if nxt["abstain"] or nxt["n_words"] < MIN_WORDS:
        return None
    old = _tracked(ctx)
    new = _tracked(nxt)
    if old is None or new is None or not new < old:
        return None
    model_delta = None
    if ctx["model_score"] is not None and nxt["model_score"] is not None:
        model_delta = nxt["model_score"] - ctx["model_score"]
    return {
        "family": ident,
        "name": name,
        "attribute": chosen,
        "feature_before": ctx["features"][chosen],
        "feature_after": nxt["features"][chosen],
        "feature_delta": nxt["features"][chosen] - ctx["features"][chosen],
        "rule_score": nxt["rules"]["score"],
        "rule_delta": nxt["rules"]["score"] - ctx["rules"]["score"],
        "model_score": nxt["model_score"],
        "model_delta": model_delta,
        "rule_verdict": nxt["verdicts"]["M5a"],
        "model_verdict": nxt["verdicts"]["M5b"],
        "text": rewritten,
        "_ctx": nxt,
    }


def _one_pass(text: str, ctx: dict) -> dict | None:
    """Apply the first rewrite that lowers the score.

    Fired M5a rules are tried first. A rule that is no longer fired can
    still run, so another press can keep lowering the score after both
    verdicts already read human.
    """
    fired = {row[0] for row in ctx["rules"]["fired"]}
    ranked = [spec for spec in FAMILIES if any(attr in fired for attr in spec[2])]
    rest = [spec for spec in FAMILIES if spec not in ranked]
    for ident, name, attributes, rewrite in (*ranked, *rest):
        chosen = next((attr for attr in attributes if attr in fired), attributes[0])
        step = _candidate(text, ctx, ident, name, chosen, rewrite)
        if step is not None:
            return step
    return None


def humanize(text: str) -> dict:
    """Lower the detector score by one rewrite.

    A human verdict does not stop the pass. Another call on the returned
    text applies the next rewrite that still lowers the score. Under
    MIN_WORDS the text is unchanged and neither verdict is claimed.
    """
    ctx = _score(text)
    before = _side(ctx)
    if ctx["abstain"]:
        return _finish(text, before, before, [], False, "abstain", ctx)
    started_human = _passed(ctx)

    step = _one_pass(text, ctx)
    steps: list[dict] = []
    if step is not None:
        ctx = step.pop("_ctx")
        text = step["text"]
        steps.append(step)
    more = _one_pass(text, ctx) is not None
    passed = _passed(ctx)
    if not steps:
        reason = "already_human" if started_human else "no_rewrite"
    elif more:
        reason = "budget"
    elif passed:
        reason = "passed"
    else:
        reason = "no_rewrite"
    return _finish(text, before, _side(ctx), steps, passed, reason, ctx)


def _finish(text: str, before: dict, after: dict, steps: list[dict],
            passed: bool, reason: str, ctx: dict) -> dict:
    fired = set(after["fired"])
    unsupported = [UNSUPPORTED] if (
        not passed and reason != "abstain" and UNSUPPORTED in fired
    ) else []
    return {
        "purpose": PURPOSE,
        "text": text,
        "passed": passed,
        "reason": reason,
        "unsupported": unsupported,
        "min_words": MIN_WORDS,
        "max_passes": MAX_PASSES,
        "before": before,
        "after": after,
        "steps": steps,
        "model_note": ctx.get("model_note", ""),
    }


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _label(value: str | None) -> str:
    return "unavailable" if value is None else value


def format_report(report: dict) -> str:
    """Before verdicts, each translation pass, then the rewritten passage."""
    before = report["before"]
    after = report["after"]
    lines = [
        "M8 attribute-directed translation",
        f"M5a rule verdict = {_label(before['rule_verdict'])}  "
        f"rule_score={_num(before['rule_score'])}",
        f"M5b model verdict = {_label(before['model_verdict'])}  "
        f"clf_score={_num(before['model_score'])}",
    ]
    if report["reason"] == "abstain":
        lines.append(
            f"abstain: only {before['n_words']} words; both scorers need at "
            f"least {report['min_words']} words for a verdict"
        )
        lines.append("no rewrite applied")
        return "\n".join(lines) + "\n"
    if not report["steps"]:
        lines.append("no rewrite applied")
    for index, step in enumerate(report["steps"], 1):
        lines.append(
            f"pass {index}  {step['attribute']}  {step['name']}  "
            f"{step['feature_before']:.4f} -> {step['feature_after']:.4f}  "
            f"M5a {_num(step['rule_score'])} {_label(step['rule_verdict'])}  "
            f"M5b {_num(step['model_score'])} {_label(step['model_verdict'])}"
        )
    lines.append(f"result: {report['reason']}")
    if report["unsupported"]:
        lines.append("no rewrite for: " + ", ".join(report["unsupported"]))
    lines.append(
        f"M5a after = {_label(after['rule_verdict'])}  "
        f"rule_score={_num(after['rule_score'])}"
    )
    lines.append(
        f"M5b after = {_label(after['model_verdict'])}  "
        f"clf_score={_num(after['model_score'])}"
    )
    lines.append("")
    lines.append(report["text"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Print the translation report for --text, a file, or the built-in sample."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--text" in argv:
        text = argv[argv.index("--text") + 1]
    elif argv and not argv[0].startswith("--"):
        with open(argv[0], encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = SAMPLE
    print(format_report(humanize(text)), end="")


if __name__ == "__main__":
    main()
