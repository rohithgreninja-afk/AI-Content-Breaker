"""Smoke tests for the M4 feature extractor."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m1_lexer import tokenize  # noqa: E402
from m2_symtab import build_symtab  # noqa: E402
from m3_parser import parse_tokens  # noqa: E402
from m4_features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    extract_features,
    extract_features_from,
    to_vector,
)

SAMPLE = (
    "The cat sat on the mat. It was a warm afternoon, and the dog slept. "
    "However, the kitchen door stayed open all day."
)


class TestContract(unittest.TestCase):
    """The shape M5 depends on: fixed width, fixed order, all floats."""

    def test_every_text_yields_every_name(self) -> None:
        for text in (SAMPLE, "Short.", "", "   ", "!!!", "42."):
            features = extract_features(text)
            self.assertEqual(set(features), set(FEATURE_NAMES), repr(text))
            self.assertEqual(len(features), N_FEATURES, repr(text))

    def test_names_are_unique_and_ordered(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertEqual(list(extract_features(SAMPLE)), list(FEATURE_NAMES))

    def test_all_values_are_finite_floats(self) -> None:
        for text in (SAMPLE, "", "   ", "...", "a", "1.5 2.5", "Café!"):
            for name, value in extract_features(text).items():
                self.assertIsInstance(value, float, f"{name} on {text!r}")
                self.assertFalse(math.isnan(value), f"{name} NaN on {text!r}")
                self.assertFalse(math.isinf(value), f"{name} inf on {text!r}")

    def test_to_vector_matches_name_order(self) -> None:
        features = extract_features(SAMPLE)
        vector = to_vector(features)
        self.assertEqual(len(vector), N_FEATURES)
        self.assertEqual(vector, [features[n] for n in FEATURE_NAMES])

    def test_extraction_is_deterministic(self) -> None:
        self.assertEqual(extract_features(SAMPLE), extract_features(SAMPLE))

    def test_from_precomputed_matches_one_shot(self) -> None:
        """extract_features_from must agree with the wrapper."""
        tokens = tokenize(SAMPLE)
        sym_tab = build_symtab(tokens)
        meta = parse_tokens(tokens)
        self.assertEqual(extract_features_from(tokens, sym_tab, meta),
                         extract_features(SAMPLE))


class TestEmptyInput(unittest.TestCase):
    """Degenerate input must not crash or produce NaN."""

    def test_empty_text_is_all_zero(self) -> None:
        features = extract_features("")
        self.assertEqual(set(features.values()), {0.0})

    def test_whitespace_and_punct_only(self) -> None:
        for text in ("   ", "... !!!", ",,,"):
            features = extract_features(text)
            self.assertEqual(len(features), N_FEATURES)
            self.assertTrue(all(math.isfinite(v) for v in features.values()))

    def test_single_word_has_no_stdev(self) -> None:
        features = extract_features("Word.")
        self.assertEqual(features["lex_stdev_word_len"], 0.0)
        self.assertEqual(features["sent_stdev_len"], 0.0)
        self.assertEqual(features["sent_mean_abs_delta"], 0.0)


class TestFeatureMeaning(unittest.TestCase):
    """Spot checks that individual features measure what they claim."""

    def test_rates_are_bounded(self) -> None:
        features = extract_features(SAMPLE)
        bounded = [n for n in FEATURE_NAMES
                   if n.endswith("_rate") or n.endswith("_share")
                   or n in ("lex_type_token_ratio", "parse_success_rate",
                            "parse_mean_coverage", "tpl_score",
                            "rep_entropy_ratio")]
        self.assertGreater(len(bounded), 20)
        for name in bounded:
            self.assertGreaterEqual(features[name], 0.0, name)
            self.assertLessEqual(features[name], 1.0, name)

    def test_pos_rates_sum_to_one(self) -> None:
        features = extract_features(SAMPLE)
        total = sum(features[n] for n in FEATURE_NAMES
                    if n.startswith("pos_") and n.endswith("_rate")
                    and n != "pos_content_rate")
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_structure_rates_sum_to_one(self) -> None:
        features = extract_features(SAMPLE)
        total = sum(features[n] for n in FEATURE_NAMES
                    if n.startswith("tpl_struct_"))
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_repetition_lowers_type_token_ratio(self) -> None:
        varied = extract_features(
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa.")
        repeated = extract_features(
            "Alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha.")
        self.assertGreater(varied["lex_type_token_ratio"],
                           repeated["lex_type_token_ratio"])
        self.assertGreater(varied["rep_hapax_rate"], repeated["rep_hapax_rate"])
        self.assertGreater(varied["rep_word_entropy"],
                           repeated["rep_word_entropy"])
        self.assertGreater(repeated["rep_top1_share"], varied["rep_top1_share"])

    def test_uniform_sentences_have_lower_length_spread(self) -> None:
        uniform = extract_features("Aa bb cc dd. Ee ff gg hh. Ii jj kk ll.")
        varied = extract_features("Aa. Bb cc dd ee ff gg hh ii jj kk ll mm. No.")
        self.assertLess(uniform["sent_cv_len"], varied["sent_cv_len"])
        self.assertLess(uniform["sent_mean_abs_delta"],
                        varied["sent_mean_abs_delta"])

    def test_transition_words_are_counted(self) -> None:
        plain = extract_features("The cat sat down.")
        marked = extract_features("However, the cat sat down.")
        self.assertEqual(plain["lex_transition_rate"], 0.0)
        self.assertGreater(marked["lex_transition_rate"], 0.0)

    def test_punctuation_is_counted_by_mark(self) -> None:
        features = extract_features("One, two; three: four. Five!")
        self.assertGreater(features["punct_comma_rate"], 0.0)
        self.assertGreater(features["punct_semicolon_rate"], 0.0)
        self.assertGreater(features["punct_colon_rate"], 0.0)
        self.assertGreater(features["punct_exclaim_rate"], 0.0)
        self.assertEqual(features["punct_question_rate"], 0.0)

    def test_sentence_count_tracks_terminators(self) -> None:
        one = extract_features("Just one sentence here.")
        three = extract_features("One here. Two here. Three here.")
        self.assertLess(one["sent_count_log"], three["sent_count_log"])

    def test_longer_words_raise_word_length(self) -> None:
        short = extract_features("The cat ate a bug.")
        long_ = extract_features("Comprehensive methodologies demonstrate "
                                 "substantial improvements.")
        self.assertLess(short["lex_mean_word_len"], long_["lex_mean_word_len"])
        self.assertLess(short["lex_long_word_rate"], long_["lex_long_word_rate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
