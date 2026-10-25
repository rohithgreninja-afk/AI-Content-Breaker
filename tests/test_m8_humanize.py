"""M8 rewrites from fired M5a rules. It does not search M5b."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import m8_humanize
import pipeline
from m5_scorer import MODEL_PATH


FEATURES = {
    "parse_mean_depth": 8.0,
    "parse_mean_nodes": 20.0,
    "parse_mean_branching": 1.4,
    "parse_mean_subordinate": 0.4,
    "punct_comma_rate": 0.01,
    "pos_punct_rate": 0.02,
    "rep_dis_rate": 0.05,
    "rep_stdev_repeat_gap_log": 1.0,
    "pos_verb_rate": 0.3,
}


def _ctx(verdict="ai", abstain=False, fired=("punct_comma_rate",), words=80):
    """A pipeline context with only the fields M8 reads."""
    label = "abstain" if abstain else verdict
    score = 0.2 if verdict == "human" else 0.8
    return {
        "n_words": 10 if abstain else words,
        "abstain": abstain,
        "text": "unused",
        "rules": {
            "score": score,
            "fired": [(name, 1.0, 0.5, 0.4) for name in fired],
        },
        "model_score": None if verdict == "missing" else score,
        "model_note": "",
        "verdicts": {
            "M5a": label if verdict != "missing" else ("abstain" if abstain else "human"),
            "M5b": None if verdict == "missing" else label,
        },
        "features": dict(FEATURES),
    }


class TestRewrites(unittest.TestCase):
    def test_structure_splits_the_longest_coordinated_sentence(self):
        text = (
            "Cats and dogs play. The long method and the long result stay "
            "stable across the full study today."
        )
        edited, applied = m8_humanize.rewrite_structure(text)
        self.assertTrue(applied)
        self.assertIn("Cats and dogs play.", edited)
        self.assertIn("The long method.", edited)
        self.assertIn("The long result", edited)

    def test_subordinator_starts_the_next_sentence(self):
        edited, applied = m8_humanize.rewrite_subordinate(
            "Go now. We left because the rain started hard.")
        self.assertTrue(applied)
        self.assertIn("We left.", edited)
        self.assertIn("Because the rain started hard.", edited)

    def test_commas_land_before_bare_coordinators(self):
        edited, applied = m8_humanize.rewrite_commas("Cats and dogs and birds play.")
        self.assertTrue(applied)
        self.assertEqual(edited, "Cats, and dogs, and birds play.")

    def test_synonym_pair_collapses_onto_the_earlier_word(self):
        edited, applied = m8_humanize.rewrite_doubleton(
            "The method works. The technique works.")
        self.assertTrue(applied)
        self.assertNotIn("technique", edited.lower())
        self.assertEqual(edited.lower().count("method"), 2)

    def test_later_repeat_becomes_its_synonym(self):
        edited, applied = m8_humanize.rewrite_repeat_gap(
            "The method one. The method two. The method three.")
        self.assertTrue(applied)
        self.assertEqual(edited.lower().count("method"), 2)
        self.assertIn("technique", edited.lower())

    def test_punctuation_only_does_not_crash(self):
        report = m8_humanize.humanize("... !!!")
        self.assertEqual(report["reason"], "abstain")
        self.assertEqual(report["text"], "... !!!")
        self.assertFalse(report["passed"])
        self.assertEqual(report["steps"], [])


class TestSchedule(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        with patch.object(m8_humanize, "_score", return_value=_ctx(abstain=True)):
            report = m8_humanize.humanize("too short")
        self.assertEqual(report["text"], "too short")
        self.assertEqual(report["reason"], "abstain")
        self.assertFalse(report["passed"])

    def test_already_human_does_no_edits(self):
        with patch.object(m8_humanize, "_score", return_value=_ctx("human")):
            report = m8_humanize.humanize("word " * 60)
        self.assertEqual(report["reason"], "already_human")
        self.assertEqual(report["steps"], [])
        self.assertTrue(report["passed"])
        self.assertTrue(report["text"].startswith("word"))

    def test_missing_model_passes_on_the_rule_verdict(self):
        with patch.object(m8_humanize, "_score", return_value=_ctx("missing")):
            report = m8_humanize.humanize("word " * 60)
        self.assertTrue(report["passed"])
        self.assertIsNone(report["before"]["model_score"])

    def test_a_pass_names_an_attribute_that_was_fired(self):
        calls = {"n": 0}

        def score(_text):
            calls["n"] += 1
            if calls["n"] == 1:
                return _ctx("ai", fired=("punct_comma_rate",))
            return _ctx("human", fired=())

        def commas(text):
            return text + ",", True

        families = ((
            "punctuation", "Insert commas before coordinators",
            ("punct_comma_rate", "pos_punct_rate"), commas,
        ),)
        with patch.object(m8_humanize, "_score", side_effect=score), \
                patch.object(m8_humanize, "FAMILIES", families):
            report = m8_humanize.humanize("word " * 60)
        self.assertEqual(report["reason"], "passed")
        self.assertEqual(report["steps"][0]["attribute"], "punct_comma_rate")
        self.assertTrue(report["steps"][0]["text"].endswith(","))

    def test_verb_rate_has_no_rewrite(self):
        with patch.object(m8_humanize, "_score", return_value=_ctx("ai", fired=("pos_verb_rate",))):
            report = m8_humanize.humanize("word " * 60)
        self.assertEqual(report["reason"], "no_rewrite")
        self.assertEqual(report["unsupported"], ["pos_verb_rate"])
        self.assertEqual(report["steps"], [])
        self.assertFalse(report["passed"])

    def test_a_rewrite_below_the_word_gate_is_dropped(self):
        def score(text):
            return _ctx("ai", abstain=len(text.split()) < 50, fired=("punct_comma_rate",))

        def shrink(_text):
            return "too short", True

        families = (("punctuation", "Insert commas", ("punct_comma_rate",), shrink),)
        original = "word " * 60
        with patch.object(m8_humanize, "_score", side_effect=score), \
                patch.object(m8_humanize, "FAMILIES", families):
            report = m8_humanize.humanize(original)
        self.assertEqual(report["text"], original)
        self.assertEqual(report["reason"], "no_rewrite")

    def test_budget_stops_at_six_passes(self):
        def commas(text):
            return text + ",", True

        families = (("punctuation", "Insert commas", ("punct_comma_rate",), commas),)
        with patch.object(m8_humanize, "_score", return_value=_ctx("ai")), \
                patch.object(m8_humanize, "FAMILIES", families):
            report = m8_humanize.humanize("word " * 60)
        self.assertEqual(len(report["steps"]), m8_humanize.MAX_PASSES)
        self.assertEqual(report["reason"], "budget")
        self.assertFalse(report["passed"])

    def test_module_is_not_a_pipeline_stage(self):
        names = [name for name, _run, _show in pipeline.STAGES]
        self.assertEqual(len(pipeline.CORE_STAGES), 5)
        self.assertEqual(len(names), 6)
        self.assertFalse(any(name.startswith("M8") for name in names))


class TestLiveSample(unittest.TestCase):
    def test_sample_steps_follow_fired_rules_and_keep_the_gate(self):
        before_model = Path(MODEL_PATH).read_bytes()
        report = m8_humanize.humanize(pipeline.SAMPLE)
        self.assertEqual(Path(MODEL_PATH).read_bytes(), before_model)
        self.assertGreaterEqual(report["after"]["n_words"], pipeline.MIN_WORDS)
        ctx = pipeline.run_pipeline(pipeline.SAMPLE, stages=pipeline.CORE_STAGES, show=False)
        current = pipeline.SAMPLE
        for step in report["steps"]:
            fired = {row[0] for row in ctx["rules"]["fired"]}
            self.assertIn(step["attribute"], fired)
            self.assertNotEqual(step["text"], current)
            current = step["text"]
            ctx = pipeline.run_pipeline(current, stages=pipeline.CORE_STAGES, show=False)
            self.assertGreaterEqual(ctx["n_words"], pipeline.MIN_WORDS)
        self.assertEqual(report["text"], current)
        if report["passed"]:
            self.assertEqual(ctx["verdicts"]["M5a"], "human")
            self.assertEqual(ctx["verdicts"]["M5b"], "human")
        rendered = m8_humanize.format_report(report)
        self.assertIn("M8 attribute-directed translation", rendered)
        self.assertIn(f"result: {report['reason']}", rendered)


if __name__ == "__main__":
    unittest.main()
