"""HTTP contract, CLI fidelity and frozen-model recovery regression checks."""

from __future__ import annotations

from contextlib import redirect_stdout
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline
import web_api
import web_training
from m5_scorer import ModelScorer


class TestAnalysisContract(unittest.TestCase):
    def test_full_cli_transcript_preserves_input_context_and_every_stage(self):
        result = web_api.analyze_text(pipeline.SAMPLE)
        output = io.StringIO()
        with redirect_stdout(output):
            pipeline.main(["--text", pipeline.SAMPLE])
        expected = output.getvalue().replace("input: --text (", "input: web text (", 1)
        self.assertEqual(result["console_output"], expected)
        self.assertEqual([stage["id"] for stage in result["stages"]],
                         ["M1", "M2", "M3", "M4", "M5", "M6"])
        self.assertIn("robustness", result)
        self.assertEqual(len(result["robustness"]["runs"]), 6)
        self.assertEqual(len(result["features"]["all"]), 89)
        self.assertIsInstance(result["parser"]["first_tree"], str)
        self.assertEqual(result["symbols"]["top_words"],
                         pipeline.top_words(pipeline.run_pipeline(pipeline.SAMPLE, show=False)["sym_tab"], 5))

    def test_short_text_abstains_in_both_scorers(self):
        result = web_api.analyze_text("A short sentence.")
        self.assertTrue(result["console_output"].startswith("input: web text (17 characters)\n"))
        self.assertTrue(result["summary"]["abstain"])
        self.assertEqual(result["summary"]["model_verdict"], "abstain")
        self.assertEqual(result["summary"]["rule_verdict"], "abstain")

    def test_formatter_does_not_replace_process_stdout(self):
        stdout = sys.stdout
        observed = []

        def formatter(context):
            observed.append(sys.stdout)
            print("stage output")

        captured = web_api.stage_output(formatter, {})
        self.assertEqual(captured, "stage output\n")
        self.assertEqual(observed, [stdout])


class TestModelState(unittest.TestCase):
    def test_check_loads_artifact_and_reports_ready(self):
        state = web_api.ApplicationState()
        result = state.status()
        self.assertTrue(result["ready"])
        self.assertEqual(result["model"]["n_features"], 89)
        self.assertEqual(state.start_training()[0], 200)

    def test_corrupt_artifact_is_unavailable_and_does_not_crash(self):
        state = web_api.ApplicationState()
        with patch.object(web_api.ModelScorer, "load", side_effect=EOFError("corrupt model")):
            result = state.status()
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("corrupt model", result["message"])
        pipeline._MODEL.clear()

    def test_training_cannot_start_duplicate_concurrent_jobs(self):
        state = web_api.ApplicationState()
        with patch.object(state, "_load_model"), patch.object(web_api.threading, "Thread") as worker:
            state.model_error = "missing"
            first = state.start_training()
            second = state.start_training()
        self.assertEqual(first[0], 202)
        self.assertEqual(second[0], 202)
        self.assertEqual(worker.call_count, 1)
        worker.return_value.start.assert_called_once()

    def test_failed_training_is_visible_and_retryable(self):
        state = web_api.ApplicationState()
        with patch.object(state, "_load_model"), patch.object(state, "_command", side_effect=RuntimeError("dataset failed")):
            state._train()
            result = state.status()
            self.assertFalse(result["training"]["running"])
            self.assertEqual(result["training"]["error"], "dataset failed")
            with patch.object(web_api.threading, "Thread") as worker:
                self.assertEqual(state.start_training()[0], 202)
                worker.return_value.start.assert_called_once()

    def test_existing_csv_without_manifest_regenerates_dataset_before_training(self):
        state = web_api.ApplicationState()
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "train_clean.csv"
            dataset.write_text("id,label,text\n", encoding="utf-8")
            with patch.object(web_api, "DATA_PATH", str(dataset)), \
                    patch.object(state, "status", return_value={"ready": False}), \
                    patch.object(state, "_load_model"), patch.object(state, "_command") as command:
                state._train()
        self.assertEqual(command.call_args_list[0].args[0], ["src/m0_dataset.py", "--parquet"])
        self.assertEqual(command.call_args_list[1].args[0], ["src/web_training.py"])

    def test_shutdown_terminates_owned_training_subprocess(self):
        state = web_api.ApplicationState()
        started = threading.Event()
        errors = []

        def run_command():
            try:
                state._command(["-c", "import time; print('started', flush=True); time.sleep(60)"], "test")
            except RuntimeError as error:
                errors.append(str(error))

        with patch.object(state, "_log", side_effect=lambda line: started.set() if line == "started" else None):
            worker = threading.Thread(target=run_command, daemon=True)
            worker.start()
            try:
                self.assertTrue(started.wait(timeout=5), "Training subprocess did not start")
                process = state.training_process
                state.stop_training()
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                self.assertIsNotNone(process.returncode)
                self.assertTrue(errors)
                self.assertEqual(state.start_training()[0], 503)
            finally:
                state.stop_training()


class TestHTTPContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = web_api.create_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        pipeline._MODEL.clear()

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=15)
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        connection.request(method, path, json.dumps(body) if body is not None else "{}", request_headers)
        response = connection.getresponse()
        data = json.loads(response.read())
        status = response.status
        connection.close()
        return status, data

    def test_real_model_status_and_analysis(self):
        status, model = self.request("GET", "/api/model")
        self.assertEqual(status, 200)
        self.assertTrue(model["ready"])
        status, result = self.request("POST", "/api/analyze", {"text": pipeline.SAMPLE})
        self.assertEqual(status, 200)
        self.assertIsInstance(result["summary"]["model_score"], float)

    def test_valid_sample_and_unknown_endpoint(self):
        self.assertEqual(self.request("GET", "/api/sample"), (200, {"text": pipeline.SAMPLE}))
        self.assertEqual(self.request("GET", "/api/no-such-endpoint")[0], 404)

    def test_invalid_and_oversized_text(self):
        for body in ({}, {"text": 42}, {"text": "   "}, ["text"]):
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", "/api/analyze", body)[0], 400)
                self.assertEqual(self.request("POST", "/api/robustness", body)[0], 400)
        self.assertEqual(self.request("POST", "/api/analyze", {"text": "a" * 50_001})[0], 413)
        self.assertEqual(self.request("POST", "/api/robustness", {"text": "a" * 50_001})[0], 413)

    def test_robustness_abstains_without_adding_a_stage(self):
        status, result = self.request("POST", "/api/robustness", {"text": "A short sentence."})
        self.assertEqual(status, 200)
        self.assertTrue(result["abstain"])
        self.assertIsNone(result["fragility"]["M5a"])
        self.assertIsNone(result["fragility"]["M5b"])
        self.assertEqual(len(result["runs"]), 6)
        analyzed = self.request("POST", "/api/analyze", {"text": pipeline.SAMPLE})[1]
        self.assertEqual([stage["id"] for stage in analyzed["stages"]],
                         ["M1", "M2", "M3", "M4", "M5", "M6"])
        self.assertIn("fragility", analyzed["robustness"])

    def test_cross_site_and_dns_rebinding_requests_are_rejected(self):
        self.assertEqual(self.request("POST", "/api/model/train", {}, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.request("GET", "/api/model", headers={"Host": "evil.example:8000"})[0], 403)
        self.assertEqual(self.request("GET", "/api/model", headers={"Origin": "http://localhost:5173"})[0], 200)

    def test_wrong_content_type(self):
        self.assertEqual(self.request("POST", "/api/analyze", {}, {"Content-Type": "text/plain"})[0], 415)


@unittest.skipUnless(Path(web_api.DATA_PATH).exists(), "Training recovery needs the downloaded pinned dataset")
class TestRecovery(unittest.TestCase):
    def test_changed_training_identity_fails_before_fitting(self):
        with patch.object(web_training, "runtime_versions", return_value={"python": "wrong"}), \
                patch.object(web_training, "build_estimator", wraps=web_training.build_estimator) as estimator:
            with self.assertRaisesRegex(ValueError, "Training identity changed"):
                web_training.refit_committed_model("unused.joblib")
            # validate_protocol builds candidate recipes, but no estimator is fit.
            self.assertEqual(estimator.call_count, 0)

    def test_missing_model_refits_frozen_recipe_to_separate_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "recovered.joblib")
            with redirect_stdout(io.StringIO()):
                web_training.refit_committed_model(path)
            restored = ModelScorer.load(path)
            shipped = ModelScorer.load()
            self.assertEqual(restored.metadata, shipped.metadata)
            for text in (pipeline.SAMPLE, "The cat slept by the window."):
                self.assertAlmostEqual(restored.score(text), shipped.score(text), places=12)


if __name__ == "__main__":
    unittest.main()
