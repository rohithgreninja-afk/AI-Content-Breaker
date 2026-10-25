"""M6: robustness harness for this detector.

The harness measures how far our own verdict moves when a passage is edited
in five small, fixed ways. It is a robustness-evaluation tool, not an evasion
tool: the edits are not searched, tuned, or chosen to flip a label.

    python src/m6_harness.py path\\to\\text.txt

Each edit is tied to one existing M4 feature. A compound run applies all five
in order. Runs that cannot apply are left out of the fragility fraction.
Ordinary analysis stays on M1–M5; this module calls CORE_STAGES itself.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m1_lexer import longest_match  # noqa: E402
from m3_parser import (  # noqa: E402
    START_TEMPLATES,
    build_dfa,
    run_dfa,
    tag_sentence,
)
from m4_features import FUNCTION_WORDS, TRANSITION_WORDS  # noqa: E402
from pipeline import CORE_STAGES, MIN_WORDS, run_pipeline  # noqa: E402

# One fixed substitute per word. Replacements are not function words, so a
# repeated content word becomes a new type instead of disappearing.
SYNONYMS = {
    "method": "technique",
    "approach": "procedure",
    "result": "outcome",
    "results": "outcomes",
    "system": "setup",
    "model": "design",
    "data": "records",
    "analysis": "review",
    "paper": "article",
    "study": "inquiry",
    "problem": "issue",
    "solution": "remedy",
    "performance": "output",
    "accuracy": "correctness",
    "image": "picture",
    "images": "pictures",
    "graph": "diagram",
    "cost": "expense",
    "training": "practice",
    "input": "entry",
    "inputs": "entries",
    "baseline": "reference",
    "baselines": "references",
    "benchmark": "trial",
    "benchmarks": "trials",
    "error": "mistake",
    "errors": "mistakes",
    "feature": "signal",
    "features": "signals",
    "sentence": "line",
    "word": "term",
    "words": "terms",
    "text": "passage",
    "question": "query",
    "example": "instance",
    "examples": "instances",
    "value": "amount",
    "group": "set",
    "groups": "sets",
    "effect": "impact",
    "effects": "impacts",
    "process": "sequence",
    "report": "account",
    "test": "check",
    "tests": "checks",
}

# Plain wording for each transition word. None of the substitutes are
# themselves in TRANSITION_WORDS, so the lexical transition rate can fall.
PLAIN_TRANSITIONS = {
    "however": "but",
    "therefore": "so",
    "moreover": "and",
    "furthermore": "and",
    "additionally": "and",
    "consequently": "so",
    "nevertheless": "yet",
    "nonetheless": "yet",
    "thus": "so",
    "hence": "so",
    "accordingly": "so",
    "similarly": "also",
    "likewise": "also",
    "conversely": "instead",
    "meanwhile": "then",
    "subsequently": "later",
    "ultimately": "last",
    "overall": "generally",
    "notably": "especially",
    "specifically": "exactly",
    "particularly": "especially",
    "significantly": "greatly",
    "importantly": "mainly",
    "essentially": "basically",
    "fundamentally": "mainly",
    "arguably": "perhaps",
    "indeed": "truly",
    "firstly": "first",
    "secondly": "second",
    "thirdly": "third",
    "finally": "last",
    "lastly": "last",
}

COORDINATORS = frozenset({"and", "but", "or"})
# A sentence at or under this many words can be joined to the next one.
SHORT_SENTENCE = 12
# Inserted into a sentence-initial template so the DFA prefix no longer matches.
NEUTRAL_WORD = "now"

_START_DFA = build_dfa(START_TEMPLATES)

PURPOSE = (
    "Measures how this detector's own verdict moves after fixed edits. "
    "This is a robustness check, not an evasion tool."
)


def _scan(text: str) -> list[dict]:
    """Lexer tokens with character spans. idx stays the stream position."""
    tokens: list[dict] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        picked = longest_match(text, index)
        if picked is None:
            index += 1
            continue
        token_type, value = picked
        tokens.append({
            "type": token_type,
            "value": value,
            "idx": len(tokens),
            "start": index,
            "end": index + len(value),
        })
        index += len(value)
    return tokens


def _splice(text: str, start: int, end: int, new: str) -> str:
    return text[:start] + new + text[end:]


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _sentences(tokens: list[dict]) -> list[list[dict]]:
    """Same grouping as M3: split after a sentence end, drop wordless groups."""
    sentences: list[list[dict]] = []
    current: list[dict] = []
    for token in tokens:
        current.append(token)
        if token["type"] == "SENTENCE_END":
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return [
        sentence for sentence in sentences
        if any(token["type"] in ("WORD", "NUMBER") for token in sentence)
    ]


def edit_synonym(text: str) -> tuple[str, bool]:
    """Replace later repeats of one content word with a fixed synonym.

    Function words and transition words are left alone. The most repeated
    eligible word wins; ties go to the earliest one.
    """
    grouped: dict[str, list[dict]] = {}
    for token in _scan(text):
        if token["type"] != "WORD":
            continue
        key = token["value"].lower()
        if key in FUNCTION_WORDS or key in TRANSITION_WORDS or key not in SYNONYMS:
            continue
        grouped.setdefault(key, []).append(token)
    candidates = [
        (len(spans), -spans[0]["start"], key, spans)
        for key, spans in grouped.items() if len(spans) >= 2
    ]
    if not candidates:
        return text, False
    _count, _start, key, spans = max(candidates)
    edited = text
    for token in reversed(spans[1:]):
        edited = _splice(
            edited, token["start"], token["end"],
            _match_case(token["value"], SYNONYMS[key]),
        )
    return (edited, True) if edited != text else (text, False)


def edit_transitions(text: str) -> tuple[str, bool]:
    """Replace every listed transition word with plain wording."""
    hits = [
        token for token in _scan(text)
        if token["type"] == "WORD" and token["value"].lower() in PLAIN_TRANSITIONS
    ]
    if not hits:
        return text, False
    edited = text
    for token in reversed(hits):
        edited = _splice(
            edited, token["start"], token["end"],
            _match_case(token["value"], PLAIN_TRANSITIONS[token["value"].lower()]),
        )
    return (edited, True) if edited != text else (text, False)


def _split_coordination(text: str, sentence: list[dict]) -> str | None:
    """Split one sentence at its first coordinating conjunction, if any."""
    words = [token for token in sentence if token["type"] == "WORD"]
    if len(words) < 3:
        return None
    for index, token in enumerate(words):
        if index == 0 or index == len(words) - 1:
            continue
        if token["value"].lower() not in COORDINATORS:
            continue
        start = token["start"]
        cursor = start - 1
        while cursor >= 0 and text[cursor].isspace():
            cursor -= 1
        if cursor >= 0 and text[cursor] == ",":
            start = cursor
        elif cursor + 1 < token["start"]:
            start = cursor + 1
        nxt = words[index + 1]
        edited = _splice(text, start, nxt["start"], ". ")
        # ". " is two characters, so the moved word now starts at start + 2.
        word_at = start + 2
        if word_at < len(edited) and edited[word_at].isalpha():
            edited = edited[:word_at] + edited[word_at].upper() + edited[word_at + 1:]
        return edited if edited != text else None
    return None


def _join_short_sentences(text: str, sentences: list[list[dict]]) -> str | None:
    """Join the first pair of short sentences with 'and'."""
    for previous, following in zip(sentences, sentences[1:]):
        if previous[-1]["type"] != "SENTENCE_END":
            continue
        prev_words = [token for token in previous if token["type"] == "WORD"]
        next_words = [token for token in following if token["type"] == "WORD"]
        if not prev_words or not next_words:
            continue
        if len(prev_words) > SHORT_SENTENCE or len(next_words) > SHORT_SENTENCE:
            continue
        end = previous[-1]
        nxt = next_words[0]
        edited = _splice(text, end["start"], nxt["start"], " and ")
        word_at = end["start"] + len(" and ")
        if word_at < len(edited) and edited[word_at].isalpha():
            edited = edited[:word_at] + edited[word_at].lower() + edited[word_at + 1:]
        return edited if edited != text else None
    return None


def edit_coordination(text: str) -> tuple[str, bool]:
    """Split a coordinated sentence, or else join two short ones.

    The rewrite uses the lexer only: a coordinator with words on both sides
    becomes a sentence boundary; otherwise two short sentences are joined.
    """
    sentences = _sentences(_scan(text))
    for sentence in sentences:
        edited = _split_coordination(text, sentence)
        if edited is not None:
            return edited, True
    edited = _join_short_sentences(text, sentences)
    if edited is not None:
        return edited, True
    return text, False


def edit_template(text: str) -> tuple[str, bool]:
    """Insert one neutral word into the first sentence-initial DFA template."""
    for sentence in _sentences(_scan(text)):
        tagged = tag_sentence(sentence)
        tags = [token["coarse"] for token in tagged]
        if run_dfa(_START_DFA, tags, 0) is None:
            continue
        first = next(token for token in sentence if token["type"] in ("WORD", "NUMBER"))
        if first["end"] < len(text) and text[first["end"]].isspace():
            piece = NEUTRAL_WORD + " "
        else:
            piece = " " + NEUTRAL_WORD
        edited = _splice(text, first["end"], first["end"], piece)
        return (edited, True) if edited != text else (text, False)
    return text, False


def edit_bigram(text: str) -> tuple[str, bool]:
    """Swap the first adjacent pair of distinct content words."""
    tokens = _scan(text)
    for previous, following in zip(tokens, tokens[1:]):
        if previous["type"] != "WORD" or following["type"] != "WORD":
            continue
        left = previous["value"]
        right = following["value"]
        if left.lower() in FUNCTION_WORDS or right.lower() in FUNCTION_WORDS:
            continue
        if left.lower() == right.lower():
            continue
        edited = _splice(text, following["start"], following["end"], left)
        edited = _splice(edited, previous["start"], previous["end"], right)
        return (edited, True) if edited != text else (text, False)
    return text, False


def edit_compound(text: str) -> tuple[str, bool]:
    """Apply P1 through P5 in that order to one copy."""
    edited = text
    applied = False
    for edit in (edit_synonym, edit_transitions, edit_coordination,
                 edit_template, edit_bigram):
        edited, did = edit(edited)
        applied = applied or did
    if not applied or edited == text:
        return text, False
    return edited, True


EDITS = (
    ("P1", "Synonym substitution", "lex_type_token_ratio", edit_synonym),
    ("P2", "Transition replacement", "lex_transition_rate", edit_transitions),
    ("P3", "Coordination rewrite", "parse_stdev_depth", edit_coordination),
    ("P4", "Template interruption", "tpl_score", edit_template),
    ("P5", "Adjacent swap", "rep_word_entropy", edit_bigram),
    ("C", "Compound P1–P5", "", edit_compound),
)

# Fixed explanations. These are not searched for, and they do not change the edit.
UNAPPLIED_REASONS = {
    "P1": "no repeated listed word",
    "P2": "no listed transition",
    "P3": "no coordinator or short pair",
    "P4": "no sentence-initial template",
    "P5": "no adjacent content pair",
    "C": "Not applied",
}


def _score(text: str) -> dict:
    return run_pipeline(text, stages=CORE_STAGES, show=False)


def _blank_run(spec, applied: bool, before: float | None) -> dict:
    ident, name, target, _edit = spec
    return {
        "id": ident,
        "name": name,
        "target": target,
        "applied": applied,
        "reason": None if applied else UNAPPLIED_REASONS[ident],
        "before": before,
        "after": None,
        "delta": None,
        "rule_score": None,
        "model_score": None,
        "rule_delta": None,
        "model_delta": None,
        "rule_flipped": None,
        "model_flipped": None,
    }


def _fragility(runs: list[dict], flip_key: str) -> tuple[float | None, int, int]:
    """Fraction of applied runs whose verdict differs. Unapplied runs are out."""
    considered = [run for run in runs if run["applied"] and run[flip_key] is not None]
    if not considered:
        return None, 0, 0
    flips = sum(1 for run in considered if run[flip_key])
    return flips / len(considered), flips, len(considered)


def _fragility_record(runs: list[dict]) -> dict:
    """M5a and M5b fragility for one set of runs."""
    rule_rate, rule_flips, rule_applied = _fragility(runs, "rule_flipped")
    model_rate, model_flips, model_applied = _fragility(runs, "model_flipped")
    return {
        "M5a": rule_rate,
        "M5b": model_rate,
        "M5a_flips": rule_flips,
        "M5a_applied": rule_applied,
        "M5b_flips": model_flips,
        "M5b_applied": model_applied,
    }


def evaluate(text: str, baseline: dict | None = None) -> dict:
    """Score the original text and each applicable edit.

    Below MIN_WORDS the baseline abstains. Edits are still detected, but no
    variant is scored and no flip is claimed. Pass an existing M1–M5 context
    as baseline when this runs as a pipeline stage; perturbations are still
    scored with CORE_STAGES, so M6 does not call itself.
    """
    if baseline is None:
        baseline = _score(text)
    features = baseline["features"]
    rule_score = baseline["rules"]["score"]
    model_score = baseline["model_score"]
    rule_verdict = baseline["verdicts"]["M5a"]
    model_verdict = baseline["verdicts"]["M5b"]
    runs: list[dict] = []

    for spec in EDITS:
        _ident, _name, target, edit = spec
        edited, applied = edit(text)
        before = features.get(target) if target else None
        if baseline["abstain"] or not applied:
            runs.append(_blank_run(spec, applied, before))
            continue
        scored = _score(edited)
        after = scored["features"].get(target) if target else None
        run_rule = scored["rules"]["score"]
        run_model = scored["model_score"]
        runs.append({
            "id": spec[0],
            "name": spec[1],
            "target": target,
            "applied": True,
            "reason": None,
            "before": before,
            "after": after,
            "delta": None if after is None or before is None else after - before,
            "rule_score": run_rule,
            "model_score": run_model,
            "rule_delta": run_rule - rule_score,
            "model_delta": None if run_model is None or model_score is None else run_model - model_score,
            "rule_flipped": rule_verdict != scored["verdicts"]["M5a"],
            "model_flipped": (
                None if model_verdict is None or scored["verdicts"]["M5b"] is None
                else model_verdict != scored["verdicts"]["M5b"]
            ),
        })

    fragility = _fragility_record(runs)
    fragility["singles"] = _fragility_record(
        [run for run in runs if run["id"] != "C"])
    return {
        "abstain": baseline["abstain"],
        "n_words": baseline["n_words"],
        "min_words": MIN_WORDS,
        "purpose": PURPOSE,
        "baseline": {
            "rule_score": rule_score,
            "model_score": model_score,
            "rule_verdict": rule_verdict,
            "model_verdict": model_verdict,
        },
        "runs": runs,
        "fragility": fragility,
    }


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _yn(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "Y" if value else "N"


def format_report(report: dict) -> str:
    """Render the report. Callers print it, so stage capture can see the text."""
    lines = [
        report["purpose"],
        f"words={report['n_words']} abstain={report['abstain']}",
    ]
    base = report["baseline"]
    lines.append(
        f"baseline M5a={base['rule_score']:.3f} {base['rule_verdict']}  "
        f"M5b={_num(base['model_score'])} {base['model_verdict']}"
    )
    for run in report["runs"]:
        if not run["applied"]:
            state = run.get("reason") or "not applied"
        elif report["abstain"]:
            state = "not scored (baseline abstains)"
        else:
            target = run["target"] or "scores only"
            state = (f"{target} {_num(run['before'])} -> {_num(run['after'])} "
                     f"delta {_delta(run['delta'])}")
        lines.append(
            f"{run['id']:3} {run['name']:24} {state}  "
            f"M5a change={_delta(run['rule_delta'])}  "
            f"M5b change={_delta(run['model_delta'])}  "
            f"M5a flip={_yn(run['rule_flipped'])}  "
            f"M5b flip={_yn(run['model_flipped'])}"
        )
    fragility = report["fragility"]
    singles = fragility["singles"]
    lines.append(
        f"fragility P1–P5 M5a={_num(singles['M5a'])} "
        f"({singles['M5a_flips']}/{singles['M5a_applied']})  "
        f"M5b={_num(singles['M5b'])} "
        f"({singles['M5b_flips']}/{singles['M5b_applied']})"
    )
    lines.append(
        f"fragility with compound M5a={_num(fragility['M5a'])} "
        f"({fragility['M5a_flips']}/{fragility['M5a_applied']})  "
        f"M5b={_num(fragility['M5b'])} "
        f"({fragility['M5b_flips']}/{fragility['M5b_applied']})"
    )
    return "\n".join(lines) + "\n"


def _print_report(report: dict) -> None:
    print(format_report(report), end="")


def main(argv: list[str] | None = None) -> None:
    """Read one text file and print the robustness report."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python src/m6_harness.py path\\to\\text.txt")
        print(PURPOSE)
        return
    try:
        with open(argv[0], encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise SystemExit(f"could not read {argv[0]}: {error}") from error
    _print_report(evaluate(text))


if __name__ == "__main__":
    main()
