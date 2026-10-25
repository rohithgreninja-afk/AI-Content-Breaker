"""M6 reports fragility of this detector. It does not search for a flip."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import m6_harness
import pipeline
from m5_scorer import MODEL_PATH

RICH = (
    "Furthermore, the proposed method improves segmentation accuracy on noisy "
    "images and the same method remains stable across public benchmarks. "
    "The results show that the training cost stays high while those benchmarks "
    "stay fair for every group. We compare the approach against three baselines "
    "and report where each one fails in practice today. A small graph shows "
    "that the system scales to larger inputs without retuning every parameter. "
    "This study notes the effect on performance and the change in error rates "
    "across repeated cases during the full evaluation period."
)


class TestEdits(unittest.TestCase):
    def test_synonym_replaces_a_later_repeat_only(self):
        edited, applied = m6_harness.edit_synonym(
            "The method helps. The method lasts.")
        self.assertTrue(applied)
        self.assertEqual(edited, "The method helps. The technique lasts.")

    def test_transition_words_become_plain_wording(self):
        edited, applied = m6_harness.edit_transitions("However the cat sat.")
        self.assertTrue(applied)
        self.assertEqual(edited, "But the cat sat.")
        self.assertNotIn("however", edited.lower())

    def test_coordination_splits_at_a_conjunction(self):
        edited, applied = m6_harness.edit_coordination(
            "The cat sat and the dog slept.")
        self.assertTrue(applied)
        self.assertEqual(edited, "The cat sat. The dog slept.")
        edited, applied = m6_harness.edit_coordination(
            "The cat sat, and the dog slept.")
        self.assertTrue(applied)
        self.assertEqual(edited, "The cat sat. The dog slept.")

    def test_coordination_joins_two_short_sentences(self):
        edited, applied = m6_harness.edit_coordination("Cats sleep. Dogs run.")
        self.assertTrue(applied)
        self.assertEqual(edited, "Cats sleep and dogs run.")

    def test_template_inserts_one_neutral_word(self):
        edited, applied = m6_harness.edit_template(
            "Furthermore, the main benefit is clear.")
        self.assertTrue(applied)
        self.assertEqual(edited, "Furthermore now, the main benefit is clear.")

    def test_bigram_swaps_one_adjacent_content_pair(self):
        edited, applied = m6_harness.edit_bigram("The noisy images remain.")
        self.assertTrue(applied)
        self.assertEqual(edited, "The images noisy remain.")

    def test_inapplicable_edits_leave_the_text_unchanged(self):
        text = "Hello."
        for edit in (m6_harness.edit_synonym, m6_harness.edit_transitions,
                     m6_harness.edit_coordination, m6_harness.edit_template,
                     m6_harness.edit_bigram, m6_harness.edit_compound):
            edited, applied = edit(text)
            self.assertFalse(applied)
            self.assertEqual(edited, text)

    def test_each_applicable_edit_changes_its_text(self):
        samples = {
            m6_harness.edit_synonym: "The method helps the method.",
            m6_harness.edit_transitions: "Therefore cats sleep.",
            m6_harness.edit_coordination: "Cats sleep. Dogs run.",
            m6_harness.edit_template: "Furthermore, cats sleep well.",
            m6_harness.edit_bigram: "The noisy images remain here.",
        }
        for edit, text in samples.items():
            edited, applied = edit(text)
            self.assertTrue(applied, edit.__name__)
            self.assertNotEqual(edited, text)


class TestReport(unittest.TestCase):
    def test_short_text_abstains_and_claims_no_flip(self):
        report = m6_harness.evaluate("A short sentence.")
        self.assertTrue(report["abstain"])
        self.assertLess(report["n_words"], pipeline.MIN_WORDS)
        self.assertIsNone(report["fragility"]["M5a"])
        self.assertIsNone(report["fragility"]["M5b"])
        for run in report["runs"]:
            self.assertIsNone(run["rule_flipped"])
            self.assertIsNone(run["model_flipped"])

    def test_unapplied_runs_are_outside_the_denominator(self):
        report = m6_harness.evaluate(" ".join(["hello"] * 60) + ".")
        self.assertFalse(report["abstain"])
        self.assertTrue(all(not run["applied"] for run in report["runs"]))
        self.assertIsNone(report["fragility"]["M5a"])
        self.assertEqual(report["fragility"]["M5a_applied"], 0)
        self.assertIsNone(report["fragility"]["M5b"])

    def test_same_input_gives_the_same_report(self):
        text = " ".join(["hello"] * 60) + "."
        self._assert_same_report(m6_harness.evaluate(text), m6_harness.evaluate(text))

    def _assert_same_report(self, left, right):
        """Edits match exactly. Scores may differ in the last float digits."""
        self.assertEqual(left.keys(), right.keys())
        for key in ("abstain", "n_words", "min_words", "purpose"):
            self.assertEqual(left[key], right[key])
        for side in ("rule_verdict", "model_verdict"):
            self.assertEqual(left["baseline"][side], right["baseline"][side])
        self.assertAlmostEqual(left["baseline"]["rule_score"], right["baseline"]["rule_score"], places=5)
        self._almost(left["baseline"]["model_score"], right["baseline"]["model_score"])
        self.assertEqual(len(left["runs"]), len(right["runs"]))
        for one, two in zip(left["runs"], right["runs"]):
            for key in ("id", "name", "target", "applied", "rule_flipped", "model_flipped"):
                self.assertEqual(one[key], two[key])
            for key in ("before", "after", "delta", "rule_score", "model_score",
                        "rule_delta", "model_delta"):
                self._almost(one[key], two[key])
        self.assertEqual(left["fragility"], right["fragility"])

    def _almost(self, left, right):
        if left is None or right is None:
            self.assertIsNone(left)
            self.assertIsNone(right)
        else:
            self.assertAlmostEqual(left, right, places=5)

    def test_rich_passage_stays_inside_the_fragility_range(self):
        report = m6_harness.evaluate(RICH)
        self.assertGreaterEqual(report["n_words"], 50)
        self.assertFalse(report["abstain"])
        applied = [run for run in report["runs"] if run["applied"]]
        self.assertGreaterEqual(len(applied), 1)
        for key, flips_key, applied_key in (
            ("M5a", "M5a_flips", "M5a_applied"),
            ("M5b", "M5b_flips", "M5b_applied"),
        ):
            rate = report["fragility"][key]
            counted = report["fragility"][applied_key]
            if counted == 0:
                self.assertIsNone(rate)
            else:
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 1.0)
                self.assertEqual(
                    report["fragility"][flips_key] / counted, rate)

    def test_scoring_core_excludes_the_harness(self):
        self.assertEqual(
            [name.split()[0] for name, _run, _show in pipeline.CORE_STAGES],
            ["M1", "M2", "M3", "M4", "M5"],
        )
        self.assertEqual(
            [name.split()[0] for name, _run, _show in pipeline.STAGES],
            ["M1", "M2", "M3", "M4", "M5", "M6"],
        )

    def test_saved_model_file_is_not_written(self):
        existed = os.path.exists(MODEL_PATH)
        before = os.stat(MODEL_PATH) if existed else None
        m6_harness.evaluate(" ".join(["hello"] * 60) + ".")
        self.assertEqual(os.path.exists(MODEL_PATH), existed)
        if before is not None:
            after = os.stat(MODEL_PATH)
            self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
            self.assertEqual(before.st_size, after.st_size)


if __name__ == "__main__":
    unittest.main()
