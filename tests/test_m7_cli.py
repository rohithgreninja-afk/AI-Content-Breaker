"""M7 prints both verdicts and the M1–M6 report. It is not a pipeline stage."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import m7_cli
import pipeline
from m4_features import N_FEATURES
from m5_scorer import MODEL_PATH


class TestReport(unittest.TestCase):
    def test_sample_report_leads_with_both_verdicts(self):
        ctx = pipeline.run_pipeline(pipeline.SAMPLE, show=False)
        report = m7_cli.build_report(ctx, "built-in sample")
        self.assertFalse(report["abstain"])
        self.assertIn(report["rule_verdict"], ("ai", "human"))
        self.assertIn(report["model_verdict"], ("ai", "human"))
        self.assertNotIn("confidence", report)
        self.assertNotIn("label", report)
        self.assertEqual(len(report["features"]), N_FEATURES)
        self.assertEqual(len(report["robustness"]["runs"]), 6)
        printed = m7_cli.format_report(report, ctx)
        lines = printed.splitlines()
        self.assertTrue(lines[0].startswith("M5a rule verdict = "))
        self.assertTrue(lines[1].startswith("M5b model verdict = "))
        header, _break, _body = printed.partition("\n== M1 lexer ==")
        self.assertNotIn("flip", header.lower())
        for name, _run, _show in pipeline.STAGES:
            self.assertIn(f"== {name} ==", printed)

    def test_short_text_abstains_and_claims_no_flip(self):
        ctx = pipeline.run_pipeline("A short sentence.", show=False)
        report = m7_cli.build_report(ctx, "--text")
        self.assertTrue(report["abstain"])
        self.assertEqual(report["rule_verdict"], "abstain")
        self.assertEqual(report["model_verdict"], "abstain")
        self.assertIsNone(report["robustness"]["fragility"]["M5a"])
        self.assertIsNone(report["robustness"]["fragility"]["M5b"])
        printed = m7_cli.format_report(report, ctx)
        self.assertIn("abstain: only", printed.split("== M1 lexer ==")[0])

    def test_m7_is_outside_both_stage_lists(self):
        self.assertEqual(
            [name.split()[0] for name, _run, _show in pipeline.CORE_STAGES],
            ["M1", "M2", "M3", "M4", "M5"],
        )
        self.assertEqual(
            [name.split()[0] for name, _run, _show in pipeline.STAGES],
            ["M1", "M2", "M3", "M4", "M5", "M6"],
        )
        for stages in (pipeline.CORE_STAGES, pipeline.STAGES):
            self.assertFalse(any(name.startswith("M7") for name, _run, _show in stages))

    def test_saved_model_file_is_not_written(self):
        existed = os.path.exists(MODEL_PATH)
        before = os.stat(MODEL_PATH) if existed else None
        ctx = pipeline.run_pipeline("A short sentence.", show=False)
        m7_cli.format_report(m7_cli.build_report(ctx, "--text"), ctx)
        self.assertEqual(os.path.exists(MODEL_PATH), existed)
        if before is not None:
            after = os.stat(MODEL_PATH)
            self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
            self.assertEqual(before.st_size, after.st_size)


if __name__ == "__main__":
    unittest.main()
