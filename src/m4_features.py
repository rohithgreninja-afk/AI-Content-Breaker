"""M4: feature extraction over the M1 / M2 / M3 outputs.

One text in, a flat dict of floats out:

    extract_features(text) -> dict[str, float]

The key order is fixed and published as FEATURE_NAMES, so M5 can turn the dict
into a vector without guessing. Every text yields every key (0.0 when the
signal is absent), which keeps the design matrix rectangular.

Feature groups:
    lex_*    lexical: token mix, word length, vocabulary richness (M1 + M2)
    rep_*    repetition: hapax rate, top-word mass, local reuse (M2)
    sent_*   sentence length statistics, including burstiness (M1 + M3)
    pos_*    coarse POS distribution as per-token shares (M3)
    parse_*  parse-tree depth, branching, subordination, success (M3a)
    tpl_*    template DFA hit rates and structural mix (M3b)
    punct_*  punctuation habits (M1)

Design notes:
- Everything is a rate, share, or mean, so features stay comparable across
  texts of different lengths. Raw counts appear only as log1p sizes.
- All derived values are guarded against empty input: no division by zero and
  no NaN, because scikit-learn rejects NaN at fit time.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m1_lexer import tokenize  # noqa: E402
from m2_symtab import build_symtab  # noqa: E402
from m3_parser import (  # noqa: E402
    COARSE_TAGS,
    STRUCTURE_CLASSES,
    TEMPLATE_NAMES,
    parse_tokens,
)

# Function words are a style signal that survives topic change, so their rate
# is tracked on its own rather than being folded into the POS shares.
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "as", "that", "this", "these", "those", "it",
    "its", "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "do", "does", "did", "not", "no", "so", "than", "then", "there",
    "their", "they", "we", "our", "you", "your", "he", "she", "his", "her",
    "i", "me", "my", "him", "them", "us", "who", "which", "what", "when",
    "where", "while", "because", "about", "into", "over", "under", "between",
    "through", "during", "against", "without", "within", "can", "could",
    "may", "might", "must", "shall", "should", "will", "would",
})

# Discourse connectives. Counted lexically, which is independent of the M3b
# DFA (that one matches tag shapes, not specific words).
TRANSITION_WORDS = frozenset({
    "however", "therefore", "moreover", "furthermore", "additionally",
    "consequently", "nevertheless", "nonetheless", "thus", "hence",
    "accordingly", "similarly", "likewise", "conversely", "meanwhile",
    "subsequently", "ultimately", "overall", "notably", "specifically",
    "particularly", "significantly", "importantly", "essentially",
    "fundamentally", "arguably", "indeed", "firstly", "secondly", "thirdly",
    "finally", "lastly",
})

# Hedges and boosters: AI prose tends to lean on both.
HEDGE_WORDS = frozenset({
    "may", "might", "could", "perhaps", "possibly", "potentially", "likely",
    "generally", "typically", "often", "sometimes", "usually", "relatively",
    "somewhat", "arguably", "seemingly", "apparently", "roughly", "about",
    "approximately", "tend", "tends", "tended", "suggest", "suggests",
})
BOOSTER_WORDS = frozenset({
    "very", "extremely", "highly", "significantly", "substantially",
    "considerably", "remarkably", "crucial", "critical", "essential", "vital",
    "key", "major", "profound", "immense", "vast", "numerous", "myriad",
    "comprehensive", "robust", "seamless", "innovative", "cutting", "leverage",
    "leveraging", "delve", "realm", "landscape", "tapestry", "testament",
})

PUNCT_OF_INTEREST = (",", ";", ":", "-", "'", '"', "(", ")")
# Published key for each tracked mark, since the mark itself is a poor dict key.
PUNCT_KEYS = {
    ",": "comma", ";": "semicolon", ":": "colon", "-": "dash",
    "'": "apostrophe", '"': "quote", "(": "paren_open", ")": "paren_close",
}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _stdev(values) -> float:
    """Population standard deviation; 0.0 for fewer than two values."""
    values = list(values)
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _ratio(numerator: float, denominator: float) -> float:
    """Safe division: an empty denominator yields 0.0, never NaN."""
    return numerator / denominator if denominator else 0.0


def lexical_features(tokens: list[dict], sym_tab: dict[str, dict]) -> dict:
    """Token mix, word length, and vocabulary richness (M1 + M2)."""
    words = [t for t in tokens if t["type"] == "WORD"]
    lengths = [len(t["value"]) for t in words]
    n_words = len(words)
    n_types = len(sym_tab)
    lowered = [t["value"].lower() for t in words]

    return {
        "lex_n_tokens_log": math.log1p(len(tokens)),
        "lex_n_words_log": math.log1p(n_words),
        "lex_type_token_ratio": _ratio(n_types, n_words),
        # Root TTR (Guiraud) is far less length-sensitive than plain TTR.
        "lex_root_ttr": _ratio(n_types, math.sqrt(n_words)) if n_words else 0.0,
        "lex_mean_word_len": _mean(lengths),
        "lex_stdev_word_len": _stdev(lengths),
        "lex_long_word_rate": _ratio(sum(1 for n in lengths if n >= 7), n_words),
        "lex_short_word_rate": _ratio(sum(1 for n in lengths if n <= 3), n_words),
        "lex_number_rate": _ratio(
            sum(1 for t in tokens if t["type"] == "NUMBER"), len(tokens)),
        "lex_function_word_rate": _ratio(
            sum(1 for w in lowered if w in FUNCTION_WORDS), n_words),
        "lex_transition_rate": _ratio(
            sum(1 for w in lowered if w in TRANSITION_WORDS), n_words),
        "lex_hedge_rate": _ratio(
            sum(1 for w in lowered if w in HEDGE_WORDS), n_words),
        "lex_booster_rate": _ratio(
            sum(1 for w in lowered if w in BOOSTER_WORDS), n_words),
        "lex_upper_rate": _ratio(
            sum(1 for t in words if t["value"][:1].isupper()), n_words),
    }


def repetition_features(tokens: list[dict], sym_tab: dict[str, dict]) -> dict:
    """How vocabulary is reused, from the M2 symbol table and its positions."""
    n_words = sum(1 for t in tokens if t["type"] == "WORD")
    counts = sorted((entry["count"] for entry in sym_tab.values()), reverse=True)
    n_types = len(counts)

    # Mean gap between repeats of the same word, in token positions. Tight
    # gaps mean a word is being reused in a small window.
    gaps: list[float] = []
    for entry in sym_tab.values():
        positions = entry["positions"]
        if len(positions) > 1:
            gaps.extend(b - a for a, b in zip(positions, positions[1:]))

    # Shannon entropy of the word distribution, in bits. Lower entropy means
    # a more predictable, more repetitive vocabulary.
    entropy = 0.0
    if n_words:
        for count in counts:
            p = count / n_words
            entropy -= p * math.log2(p)

    return {
        "rep_hapax_rate": _ratio(sum(1 for c in counts if c == 1), n_types),
        "rep_dis_rate": _ratio(sum(1 for c in counts if c == 2), n_types),
        "rep_top1_share": _ratio(counts[0] if counts else 0, n_words),
        "rep_top5_share": _ratio(sum(counts[:5]), n_words),
        "rep_top20_share": _ratio(sum(counts[:20]), n_words),
        "rep_max_count_log": math.log1p(counts[0] if counts else 0),
        "rep_mean_repeat_gap_log": math.log1p(_mean(gaps)),
        "rep_stdev_repeat_gap_log": math.log1p(_stdev(gaps)),
        "rep_word_entropy": entropy,
        # Entropy as a share of the maximum for this many types, which
        # separates "repetitive" from "simply short".
        "rep_entropy_ratio": _ratio(
            entropy, math.log2(n_types)) if n_types > 1 else 0.0,
    }


def sentence_features(meta: dict) -> dict:
    """Sentence length statistics, including burstiness (M1 + M3)."""
    lengths = [s["n_words"] for s in meta["sentences"]]
    mean_len = _mean(lengths)
    stdev_len = _stdev(lengths)
    # Consecutive differences: uniform pacing is a strong machine tell.
    deltas = [abs(b - a) for a, b in zip(lengths, lengths[1:])]

    return {
        "sent_count_log": math.log1p(len(lengths)),
        "sent_mean_len": mean_len,
        "sent_stdev_len": stdev_len,
        # Coefficient of variation: length-independent spread.
        "sent_cv_len": _ratio(stdev_len, mean_len),
        "sent_min_len": float(min(lengths)) if lengths else 0.0,
        "sent_max_len": float(max(lengths)) if lengths else 0.0,
        "sent_range_len": float(max(lengths) - min(lengths)) if lengths else 0.0,
        "sent_mean_abs_delta": _mean(deltas),
        "sent_short_rate": _ratio(sum(1 for n in lengths if n <= 8), len(lengths)),
        "sent_long_rate": _ratio(sum(1 for n in lengths if n >= 30), len(lengths)),
    }


def pos_features(meta: dict) -> dict:
    """Coarse POS distribution as per-token shares (M3)."""
    counts = meta["pos_counts"]
    total = sum(counts.values())
    features = {
        f"pos_{tag.lower()}_rate": _ratio(counts.get(tag, 0), total)
        for tag in COARSE_TAGS
    }
    # A few ratios that are more stable than the raw shares.
    nouns = counts.get("NOUN", 0)
    verbs = counts.get("VERB", 0)
    features["pos_noun_verb_ratio"] = _ratio(nouns, verbs)
    features["pos_adj_noun_ratio"] = _ratio(counts.get("ADJ", 0), nouns)
    features["pos_adv_verb_ratio"] = _ratio(counts.get("ADV", 0), verbs)
    # Nominal vs verbal weight: formal, machine-like prose skews nominal.
    content = nouns + verbs + counts.get("ADJ", 0) + counts.get("ADV", 0)
    features["pos_content_rate"] = _ratio(content, total)
    return features


def parse_features(meta: dict) -> dict:
    """Parse-tree shape and success, from the M3a CFG parser."""
    sentences = meta["sentences"]
    depths = [s["depth"] for s in sentences]
    return {
        "parse_success_rate": meta["parse_success_rate"],
        "parse_mean_coverage": meta["mean_coverage"],
        "parse_mean_depth": meta["mean_depth"],
        "parse_max_depth": float(meta["max_depth"]),
        "parse_stdev_depth": _stdev(depths),
        "parse_mean_branching": meta["mean_branching"],
        "parse_mean_nodes": meta["mean_nodes"],
        "parse_mean_clauses": meta["mean_clauses"],
        "parse_mean_pps": meta["mean_pps"],
        "parse_mean_subordinate": meta["mean_subordinate"],
        # Tree size per word: how much structure the parser found per token.
        "parse_nodes_per_word": _ratio(
            sum(s["nodes"] for s in sentences),
            sum(s["n_words"] for s in sentences)),
    }


def template_features(meta: dict) -> dict:
    """Template DFA hit rates and structural mix, from M3b."""
    total = meta["n_sentences"]
    features = {"tpl_score": meta["template_score"]}
    for name in TEMPLATE_NAMES:
        features[f"tpl_{name}_rate"] = _ratio(
            meta["template_counts"][name], total)
    for cls in STRUCTURE_CLASSES:
        features[f"tpl_struct_{cls.lower()}_rate"] = _ratio(
            meta["structure_counts"][cls], total)
    # How many distinct templates the text leans on at all. A text that reuses
    # one frame reads differently from one that mixes several.
    features["tpl_distinct"] = float(
        sum(1 for name in TEMPLATE_NAMES if meta["template_counts"][name]))
    return features


def punctuation_features(tokens: list[dict]) -> dict:
    """Punctuation habits (M1)."""
    total = len(tokens)
    marks = [t["value"] for t in tokens if t["type"] == "PUNCT"]
    enders = [t["value"] for t in tokens if t["type"] == "SENTENCE_END"]
    features = {
        f"punct_{PUNCT_KEYS[mark]}_rate": _ratio(marks.count(mark), total)
        for mark in PUNCT_OF_INTEREST
    }
    features["punct_rate"] = _ratio(len(marks), total)
    features["punct_exclaim_rate"] = _ratio(enders.count("!"), total)
    features["punct_question_rate"] = _ratio(enders.count("?"), total)
    return features


def extract_features_from(tokens: list[dict], sym_tab: dict[str, dict],
                          meta: dict) -> dict[str, float]:
    """Assemble the feature dict from already-computed M1/M2/M3 outputs.

    Use this when the caller already has the three artefacts; extract_features
    is the one-shot wrapper.
    """
    features: dict[str, float] = {}
    features.update(lexical_features(tokens, sym_tab))
    features.update(repetition_features(tokens, sym_tab))
    features.update(sentence_features(meta))
    features.update(pos_features(meta))
    features.update(parse_features(meta))
    features.update(template_features(meta))
    features.update(punctuation_features(tokens))
    # Floats throughout: M5 feeds these straight into numpy.
    return {key: float(value) for key, value in features.items()}


def extract_features(text: str) -> dict[str, float]:
    """Run the whole front end over one text and return its feature dict."""
    tokens = tokenize(text)
    sym_tab = build_symtab(tokens)
    meta = parse_tokens(tokens)
    return extract_features_from(tokens, sym_tab, meta)


# Canonical feature order, derived once from a probe text so the names can
# never drift out of step with the functions above.
FEATURE_NAMES: tuple[str, ...] = tuple(
    extract_features("A probe sentence, for ordering. It has two sentences.")
)

N_FEATURES = len(FEATURE_NAMES)


def to_vector(features: dict[str, float]) -> list[float]:
    """Flatten a feature dict into FEATURE_NAMES order."""
    return [features[name] for name in FEATURE_NAMES]


def main() -> None:
    """Print the feature vector for one human and one AI row, side by side."""
    import csv

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "data", "raid_clean.csv")
    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    human = next(r for r in rows if r["label"] == "human")
    ai = next(r for r in rows if r["label"] == "ai")
    left = extract_features(human["text"])
    right = extract_features(ai["text"])

    print(f"{N_FEATURES} features")
    print(f"human id={human['id'][:8]} domain={human['domain']}")
    print(f"ai    id={ai['id'][:8]} domain={ai['domain']}")
    print(f"\n{'feature':34} {'human':>10} {'ai':>10}")
    for name in FEATURE_NAMES:
        print(f"{name:34} {left[name]:10.4f} {right[name]:10.4f}")


if __name__ == "__main__":
    main()
