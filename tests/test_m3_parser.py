"""Smoke tests for M3: sentence split, tagging, CFG parser, template DFA."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m1_lexer import tokenize  # noqa: E402
from m3_parser import (  # noqa: E402
    ANY_TEMPLATES,
    START_TEMPLATES,
    TEMPLATE_NAMES,
    build_dfa,
    coarse_tag,
    count_labels,
    match_templates,
    parse_tags,
    parse_text,
    run_dfa,
    split_sentences,
    structure_class,
    tag_sentence,
    tree_branching,
    tree_depth,
    tree_nodes,
)


class TestSentenceSplit(unittest.TestCase):
    """Segmentation of the M1 token stream into sentences."""

    def test_terminator_stays_with_its_sentence(self) -> None:
        sentences = split_sentences(tokenize("The cat sat. The dog ran!"))
        self.assertEqual(len(sentences), 2)
        self.assertEqual([t["value"] for t in sentences[0]],
                         ["The", "cat", "sat", "."])
        self.assertEqual(sentences[1][-1]["value"], "!")

    def test_unterminated_tail_is_a_sentence(self) -> None:
        sentences = split_sentences(tokenize("One. Two"))
        self.assertEqual(len(sentences), 2)
        self.assertEqual([t["value"] for t in sentences[1]], ["Two"])

    def test_wordless_groups_are_dropped(self) -> None:
        # M1 emits three SENTENCE_END tokens for "...", so the two trailing
        # groups hold no WORD/NUMBER and must not count as sentences.
        sentences = split_sentences(tokenize("Wait... go"))
        self.assertEqual(len(sentences), 2)
        self.assertEqual([t["value"] for t in sentences[0]],
                         ["Wait", "."])

    def test_empty_stream(self) -> None:
        self.assertEqual(split_sentences([]), [])
        self.assertEqual(split_sentences(tokenize("   ")), [])


class TestTagging(unittest.TestCase):
    """Penn -> coarse mapping and per-sentence tagging."""

    def test_coarse_mapping(self) -> None:
        self.assertEqual(coarse_tag("NNS"), "NOUN")
        self.assertEqual(coarse_tag("VBZ"), "VERB")
        self.assertEqual(coarse_tag("MD"), "MODAL")
        self.assertEqual(coarse_tag("PRP$"), "DET")   # possessive fills DET
        self.assertEqual(coarse_tag("PRP"), "PRON")
        self.assertEqual(coarse_tag("IN"), "PREP")
        self.assertEqual(coarse_tag("TO"), "PART")
        self.assertEqual(coarse_tag("CD"), "NUM")
        self.assertEqual(coarse_tag(","), "PUNCT")    # nltk tags punct as self
        self.assertEqual(coarse_tag("``"), "PUNCT")
        self.assertEqual(coarse_tag("ZZZ"), "X")

    def test_punct_and_end_are_tagged_without_the_tagger(self) -> None:
        tagged = tag_sentence(tokenize("Yes, really.")[:])
        self.assertEqual([t["coarse"] for t in tagged][-2:], ["ADV", "END"])
        self.assertEqual(tagged[1]["coarse"], "PUNCT")
        # idx values are carried through from M1 untouched.
        self.assertEqual([t["idx"] for t in tagged], [0, 1, 2, 3])

    def test_tagger_covers_every_token(self) -> None:
        sentence = split_sentences(tokenize("We tested 3 models, carefully."))[0]
        tagged = tag_sentence(sentence)
        self.assertEqual(len(tagged), len(sentence))
        self.assertTrue(all(t["coarse"] for t in tagged))


class TestCfgParser(unittest.TestCase):
    """Recursive-descent parsing over coarse tag sequences."""

    @staticmethod
    def _parse(tags: list[str], penns: list[str] | None = None):
        """Parse a tag sequence. Without penns, no verb chaining is allowed."""
        return parse_tags(tags, [tag.lower() for tag in tags], penns)

    def test_simple_clause_parses_fully(self) -> None:
        tree, consumed = self._parse(["DET", "NOUN", "VERB"])
        self.assertIsNotNone(tree)
        self.assertEqual(consumed, 3)
        self.assertEqual(tree["label"], "Sentence")
        self.assertEqual(count_labels(tree, "Clause"), 1)
        self.assertEqual(count_labels(tree, "NP"), 1)

    def test_pp_attachment(self) -> None:
        tags = ["DET", "NOUN", "VERB", "PREP", "DET", "NOUN"]
        tree, consumed = self._parse(tags)
        self.assertEqual(consumed, len(tags))
        self.assertEqual(count_labels(tree, "PP"), 1)
        # A PP is a phrase, not a clause, so the sentence stays SIMPLE.
        self.assertEqual(structure_class(tree, True), "SIMPLE")

    def test_subordinate_clause_is_complex(self) -> None:
        # "Although the results improve, we note the cost."
        tags = ["PREP", "DET", "NOUN", "VERB", "PUNCT",
                "PRON", "VERB", "DET", "NOUN"]
        tree, consumed = self._parse(tags)
        self.assertEqual(consumed, len(tags))
        self.assertEqual(count_labels(tree, "SubClause"), 1)
        self.assertEqual(structure_class(tree, True), "COMPLEX")

    def test_comma_stops_a_vp_running_into_the_next_clause(self) -> None:
        """The comma must end the first VP, not be absorbed as a complement."""
        tags = ["PRON", "VERB", "DET", "NOUN", "PUNCT", "PRON", "VERB"]
        tree, consumed = self._parse(tags)
        self.assertEqual(consumed, len(tags))
        # Two separate main clauses, joined by the comma.
        self.assertEqual(
            sum(1 for c in tree["children"] if c["label"] == "Clause"), 2)

    def test_relative_clause_is_complex(self) -> None:
        # "The method that works is fast."
        tags = ["DET", "NOUN", "DET", "VERB", "VERB", "ADJ"]
        tree, consumed = self._parse(tags)
        self.assertEqual(consumed, len(tags))
        self.assertEqual(count_labels(tree, "Relative"), 1)
        self.assertEqual(structure_class(tree, True), "COMPLEX")

    def test_leading_det_is_not_a_subordinator(self) -> None:
        """A sentence-initial DET must start the subject, not a SubClause."""
        tree, consumed = self._parse(["DET", "NOUN", "VERB"])
        self.assertIsNotNone(tree)
        self.assertEqual(consumed, 3)
        self.assertEqual(count_labels(tree, "SubClause"), 0)
        self.assertEqual(structure_class(tree, True), "SIMPLE")

    def test_compound_clause_via_conj(self) -> None:
        tags = ["PRON", "VERB", "CONJ", "PRON", "VERB"]
        tree, consumed = self._parse(tags)
        self.assertEqual(consumed, len(tags))
        self.assertEqual(count_labels(tree, "Clause"), 2)
        self.assertEqual(structure_class(tree, True), "COMPOUND")

    def test_modal_and_auxiliary_chain(self) -> None:
        # "it can be shown to hold": the chain continues through VB/VBN.
        tags = ["PRON", "MODAL", "VERB", "VERB", "PART", "VERB"]
        penns = ["PRP", "MD", "VB", "VBN", "TO", "VB"]
        tree, consumed = self._parse(tags, penns)
        self.assertEqual(consumed, len(tags))
        self.assertEqual(count_labels(tree, "Clause"), 1)

    def test_finite_verbs_do_not_chain(self) -> None:
        """"the method that works is fast": VBZ VBZ is two predicates.

        Without the Penn gate the relative clause ate the main verb and the
        whole sentence failed to parse.
        """
        tags = ["DET", "NOUN", "DET", "VERB", "VERB", "ADJ"]
        penns = ["DT", "NN", "WDT", "VBZ", "VBZ", "JJ"]
        tree, consumed = self._parse(tags, penns)
        self.assertEqual(consumed, len(tags))
        self.assertEqual(count_labels(tree, "Relative"), 1)
        self.assertEqual(structure_class(tree, True), "COMPLEX")

    def test_noun_compound_is_one_np(self) -> None:
        tags = ["DET", "NOUN", "NOUN", "NOUN", "VERB"]
        tree, consumed = self._parse(tags)
        self.assertEqual(consumed, len(tags))
        self.assertEqual(count_labels(tree, "NP"), 1)

    def test_verbless_input_fails_to_parse(self) -> None:
        tree, consumed = self._parse(["DET", "ADJ", "NOUN"])
        self.assertIsNone(tree)
        self.assertEqual(consumed, 0)
        self.assertEqual(structure_class(tree, False), "FRAGMENT")

    def test_trailing_junk_is_a_partial_parse(self) -> None:
        tags = ["PRON", "VERB", "CONJ"]
        tree, consumed = self._parse(tags)
        self.assertIsNotNone(tree)
        # The dangling CONJ is rolled back, not consumed.
        self.assertEqual(consumed, 2)
        self.assertEqual(structure_class(tree, False), "FRAGMENT")

    def test_empty_input(self) -> None:
        self.assertEqual(self._parse([]), (None, 0))

    def test_tree_metrics(self) -> None:
        leaf = {"label": "NOUN", "value": "cat"}
        self.assertEqual(tree_depth(leaf), 1)
        self.assertEqual(tree_nodes(leaf), 1)
        self.assertEqual(tree_branching(leaf), 0.0)

        node = {"label": "NP", "children": [leaf, leaf]}
        self.assertEqual(tree_depth(node), 2)
        self.assertEqual(tree_nodes(node), 3)
        self.assertEqual(tree_branching(node), 2.0)


class TestTemplateDfa(unittest.TestCase):
    """The deterministic prefix automaton over coarse tag sequences."""

    def test_build_dfa_shares_prefixes(self) -> None:
        dfa = build_dfa((("a", ("NOUN", "VERB")), ("b", ("NOUN", "ADJ"))))
        # start, after-NOUN (shared), after-VERB, after-ADJ = 4 states.
        # Without prefix sharing the two patterns would need 5.
        self.assertEqual(len(dfa["transitions"]), 4)
        self.assertEqual(len(dfa["transitions"][0]), 1)
        shared = dfa["transitions"][0]["NOUN"]
        self.assertEqual(set(dfa["transitions"][shared]), {"VERB", "ADJ"})
        self.assertEqual(set(dfa["accepting"].values()), {"a", "b"})

    def test_run_dfa_longest_match_wins(self) -> None:
        dfa = build_dfa((("short", ("DET", "NOUN")),
                         ("long", ("DET", "NOUN", "VERB"))))
        self.assertEqual(run_dfa(dfa, ["DET", "NOUN"]), "short")
        self.assertEqual(run_dfa(dfa, ["DET", "NOUN", "VERB"]), "long")
        self.assertIsNone(run_dfa(dfa, ["NOUN", "VERB"]))

    def test_run_dfa_respects_offset(self) -> None:
        dfa = build_dfa((("hit", ("DET", "NOUN")),))
        self.assertIsNone(run_dfa(dfa, ["ADV", "DET", "NOUN"], 0))
        self.assertEqual(run_dfa(dfa, ["ADV", "DET", "NOUN"], 1), "hit")

    def test_start_templates_are_anchored(self) -> None:
        # "Furthermore ," at the start -> discourse_opener.
        self.assertIn("discourse_opener", match_templates(
            ["ADV", "PUNCT", "DET", "NOUN", "VERB", "END"]))
        # Same pair mid-sentence must not fire the anchored pattern.
        self.assertNotIn("discourse_opener", match_templates(
            ["DET", "NOUN", "VERB", "ADV", "PUNCT", "END"]))

    def test_any_templates_match_mid_sentence(self) -> None:
        # "... it is important to note ..." -> hedged_assertion.
        tags = ["DET", "NOUN", "VERB", "PRON", "VERB", "ADJ", "PART", "VERB"]
        self.assertIn("hedged_assertion", match_templates(tags))

    def test_hits_are_deduped_and_named(self) -> None:
        tags = ["PREP", "NOUN", "PUNCT", "DET", "ADV", "ADJ", "NOUN",
                "VERB", "DET", "ADV", "ADJ", "NOUN", "END"]
        hits = match_templates(tags)
        self.assertEqual(len(hits), len(set(hits)))
        self.assertIn("prep_phrase_opener", hits)
        self.assertIn("superlative_frame", hits)

    def test_no_templates_on_plain_sentence(self) -> None:
        self.assertEqual(match_templates(["NOUN", "VERB", "END"]), [])

    def test_template_names_cover_both_pattern_sets(self) -> None:
        declared = {name for name, _ in START_TEMPLATES}
        declared |= {name for name, _ in ANY_TEMPLATES}
        self.assertEqual(set(TEMPLATE_NAMES), declared)


class TestParseText(unittest.TestCase):
    """The public parse_text contract that M4 consumes."""

    def test_metadata_shape(self) -> None:
        meta = parse_text("The cat sat on the mat. The dog ran quickly.")
        self.assertEqual(meta["n_sentences"], 2)
        self.assertEqual(len(meta["sentences"]), 2)
        self.assertIn(meta["tagger"], ("nltk", "fallback"))
        for key in ("parse_success_rate", "mean_coverage", "mean_depth",
                    "mean_branching", "mean_nodes", "mean_clauses",
                    "mean_pps", "template_score"):
            self.assertIsInstance(meta[key], float, key)
        self.assertIsInstance(meta["max_depth"], int)
        # Every declared name is present so M4's vector is fixed-width.
        self.assertEqual(set(meta["template_counts"]), set(TEMPLATE_NAMES))
        self.assertEqual(set(meta["structure_counts"]),
                         {"SIMPLE", "COMPOUND", "COMPLEX", "FRAGMENT"})
        self.assertEqual(sum(meta["structure_counts"].values()), 2)

    def test_rates_are_bounded(self) -> None:
        meta = parse_text("It is important to note that there are many "
                          "factors. In conclusion, the main benefit is clear.")
        for key in ("parse_success_rate", "mean_coverage", "template_score"):
            self.assertGreaterEqual(meta[key], 0.0, key)
            self.assertLessEqual(meta[key], 1.0, key)
        self.assertGreater(meta["template_score"], 0.0)

    def test_pos_counts_total_matches_tokens(self) -> None:
        text = "We ran 2 tests, then stopped."
        meta = parse_text(text)
        counted = sum(meta["pos_counts"].values())
        tagged_total = sum(s["n_tokens"] for s in meta["sentences"])
        self.assertEqual(counted, tagged_total)

    def test_empty_text(self) -> None:
        meta = parse_text("")
        self.assertEqual(meta["n_sentences"], 0)
        self.assertEqual(meta["sentences"], [])
        self.assertEqual(meta["parse_success_rate"], 0.0)
        self.assertEqual(meta["template_score"], 0.0)
        self.assertEqual(meta["max_depth"], 0)
        self.assertEqual(sum(meta["template_counts"].values()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
