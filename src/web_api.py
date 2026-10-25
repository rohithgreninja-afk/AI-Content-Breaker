"""Local JSON API for the React interface; uses the existing analysis pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import FunctionType
from urllib.parse import urlsplit

import m6_harness
import m8_humanize
import pipeline
from m3_parser import render_tree, tagger_mode
from m5_scorer import DATA_PATH, MODEL_PATH, REPO_ROOT, ModelScorer

MAX_CHARACTERS = 50_000
MAX_BODY_BYTES = 210_000
ROOT = Path(REPO_ROOT)


def stage_output(display, context: dict) -> str:
    """Capture one CLI formatter without redirecting process-wide stdout."""
    buffer = io.StringIO()
    namespace = dict(display.__globals__, print=partial(print, file=buffer))
    formatter = FunctionType(display.__code__, namespace, display.__name__,
                             display.__defaults__, display.__closure__)
    formatter(context)
    return buffer.getvalue()


def analyze_text(text: str) -> dict:
    """Return every CLI stage, its structured data and the original transcript."""
    started = time.perf_counter()
    context = pipeline.run_pipeline(text, show=False)
    stages = [{"id": name.split()[0], "name": name.split(" ", 1)[1],
               "output": stage_output(display, context)}
              for name, _, display in pipeline.STAGES]
    meta = context["meta"]
    first = meta["sentences"][0] if meta["sentences"] else None
    tree = render_tree(first["tree"], 1) if first and first["tree"] else ""
    return {
        "summary": {
            "n_words": context["n_words"], "n_tokens": len(context["tokens"]),
            "n_sentences": meta["n_sentences"], "n_unique_words": len(context["sym_tab"]),
            "abstain": context["abstain"], "min_words": pipeline.MIN_WORDS,
            "model_score": context["model_score"], "rule_score": context["rules"]["score"],
            "model_verdict": context["verdicts"]["M5b"],
            "rule_verdict": context["verdicts"]["M5a"],
            "model_note": context["model_note"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        },
        "lexer": {"counts": dict(Counter(token["type"] for token in context["tokens"])),
                  "tokens": context["tokens"]},
        "symbols": {"unique_count": len(context["sym_tab"]),
                    "top_words": pipeline.top_words(context["sym_tab"], 5)},
        "parser": {**meta, "first_tree": tree},
        "features": {"all": context["features"],
                     "selected": {name: context["features"][name] for name in pipeline.SHOWN_FEATURES}},
        "rules": context["rules"], "robustness": context["robustness"], "stages": stages,
        "console_output": f"input: web text ({len(text)} characters)\n" + "".join(
            f"\n== {stage['id']} {stage['name']} ==\n{stage['output']}" for stage in stages),
    }


def robustness_report(text: str) -> dict:
    """Score fixed edits of this passage. Does not change ordinary analysis."""
    return m6_harness.evaluate(text)


def humanize_text(text: str) -> dict:
    """Rewrite from fired M5a rules. Does not change ordinary analysis."""
    return m8_humanize.humanize(text)


class ApplicationState:
    """Coordinate model loading, one training job and serialized inference."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.signature = object()
        self.model = None
        self.model_error = ""
        self.training = {"running": False, "phase": "idle", "logs": [], "error": None}
        self.stopping = threading.Event()
        self.training_process = None

    def _load_model(self) -> None:
        signature = pipeline._model_signature()
        # Tagger installation can make an otherwise unchanged artifact usable.
        signature = (signature, tagger_mode())
        if signature == self.signature:
            return
        self.signature, self.model, self.model_error = signature, None, ""
        try:
            if not Path(MODEL_PATH).exists():
                self.model_error = "No trained model found. Check & train model to create it."
            else:
                self.model = ModelScorer.load()
        except Exception as error:
            self.model_error = f"The saved model could not be loaded: {error}"
        pipeline._MODEL.update(model=(self.model, self.model_error),
                               signature=pipeline._model_signature())

    def status(self) -> dict:
        """Check that the saved model can actually load in this runtime."""
        with self.lock:
            self._load_model()
            ready = self.model is not None
            status = ("training" if self.training["running"] else
                      "ready" if ready else "error" if self.training["error"] else
                      "unavailable" if Path(MODEL_PATH).exists() else "missing")
            message = ("Training is running. You can keep this page open."
                       if self.training["running"] else "Model trained and ready to analyze."
                       if ready else self.training["error"] or self.model_error)
            model_info = None
            if self.model:
                metadata = self.model.metadata
                config = metadata.get("estimator_config", {})
                family = config.get("family", "logreg")
                accuracy = None
                try:
                    report = json.loads((ROOT / "models" / "training_report.json").read_text(encoding="utf-8"))
                    if all(report.get(key) == value for key, value in metadata.items()):
                        accuracy = report["evaluation"]["overall"]["at_least_50_words"]["model"]["accuracy"]
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                model_info = {"name": "ExtraTrees" if family == "extra_trees" else family.replace("_", " ").title(),
                              "family": family, "n_features": len(self.model.feature_names),
                              "n_train": len(metadata.get("train_ids", [])),
                              "trained_at": metadata.get("trained_at_utc"), "accuracy": accuracy}
            return {"ready": ready, "status": status, "message": message, "model": model_info,
                    "training": {**self.training, "logs": list(self.training["logs"])},
                    "min_words": pipeline.MIN_WORDS, "max_characters": MAX_CHARACTERS}

    def start_training(self) -> tuple[int, dict]:
        """Train only when necessary, and never start duplicate concurrent jobs."""
        with self.lock:
            if self.stopping.is_set():
                return 503, {"error": "The server is shutting down. Restart npm run dev before training."}
            status = self.status()
            if status["ready"]:
                return 200, status
            if not self.training["running"]:
                self.training = {"running": True, "phase": "preparing", "logs": [], "error": None}
                threading.Thread(target=self._train, daemon=True, name="model-training").start()
            return 202, self.status()

    def _log(self, line: str) -> None:
        with self.lock:
            self.training["logs"] = (self.training["logs"] + [line])[-100:]

    def _command(self, arguments: list[str], phase: str) -> None:
        with self.lock:
            if self.stopping.is_set():
                raise RuntimeError("Training stopped because the server is shutting down.")
            self.training["phase"] = phase
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            process = subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace",
                                       creationflags=creationflags,
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.training_process = process
        self._log(phase)
        try:
            with process:
                for line in process.stdout:
                    self._log(line.rstrip())
                if process.wait() != 0:
                    raise RuntimeError(f"{phase} failed. Review the training log and README setup instructions.")
        finally:
            with self.lock:
                if self.training_process is process:
                    self.training_process = None

    def stop_training(self) -> None:
        """Stop the owned training subprocess before the API exits normally."""
        with self.lock:
            self.stopping.set()
            process = self.training_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            except ProcessLookupError:
                pass

    def _train(self) -> None:
        try:
            if tagger_mode() != "nltk":
                self._command(["-m", "nltk.downloader", "averaged_perceptron_tagger_eng"],
                              "Installing the English POS tagger")
                # tagger_mode caches its availability; use the fresh subprocess
                # for fitting, and refresh the parser's cached tagger below.
                import m3_parser
                m3_parser._TAGGER_MODE = None
            with self.lock:
                self.signature = object()
            if not self.status()["ready"]:
                dataset = Path(DATA_PATH)
                manifest = dataset.parent / "train_manifest.json"
                if not dataset.exists() or not manifest.exists():
                    self._command(["src/m0_dataset.py", "--parquet"], "Fetching the pinned training datasets")
                self._command(["src/web_training.py"], "Training the committed classifier")
            with self.lock:
                self.signature = object()
                self._load_model()
                if not self.model:
                    raise RuntimeError(self.model_error)
                self.training.update(running=False, phase="complete", error=None)
            self._log("Model trained and ready to analyze.")
        except Exception as error:
            with self.lock:
                self.training.update(running=False, phase="failed", error=str(error))
            self._log(str(error))


class APIHandler(BaseHTTPRequestHandler):
    """Serve local-only JSON endpoints behind Vite's /api development proxy."""

    server_version = "ContentBreaker/1.0"

    def _respond(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=True, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _local_request(self) -> bool:
        try:
            host = urlsplit("http://" + self.headers.get("Host", ""))
            valid = (host.hostname in ("localhost", "127.0.0.1")
                     and host.port == self.server.server_port and not host.username)
            origin_header = self.headers.get("Origin")
            if origin_header:
                origin = urlsplit(origin_header)
                valid = (valid and origin.scheme == "http" and
                         origin.hostname in ("localhost", "127.0.0.1") and
                         origin.port in (5173, 4173, self.server.server_port) and
                         not origin.username and not origin.path and not origin.query and not origin.fragment)
        except ValueError:
            valid = False
        if not valid:
            self._respond(403, {"error": "This API accepts requests from the local application only."})
        return valid

    def _json_body(self) -> dict | None:
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
            self._respond(415, {"error": "Send the request as application/json."})
            return None
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY_BYTES:
            self._respond(413 if length > MAX_BODY_BYTES else 400,
                          {"error": "The request is too large or has no valid Content-Length."})
            return None
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Expected an object")
            return data
        except (UnicodeError, ValueError):
            self._respond(400, {"error": "The request must contain a valid JSON object."})
            return None

    def do_GET(self) -> None:
        """Read model readiness or a built-in sample without mutating files."""
        if not self._local_request():
            return
        if self.path == "/api/model":
            self._respond(200, self.server.state.status())
        elif self.path == "/api/sample":
            self._respond(200, {"text": pipeline.SAMPLE})
        else:
            self._respond(404, {"error": "Endpoint not found."})

    def _submitted_text(self, data: dict) -> str | None:
        """Validate a text field. Responds with an error and returns None."""
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            self._respond(400, {"error": "Enter some text to analyze."})
            return None
        if len(text) > MAX_CHARACTERS:
            self._respond(413, {"error": f"Use no more than {MAX_CHARACTERS:,} characters."})
            return None
        return text

    def do_POST(self) -> None:
        """Start a training job, analyze text, measure robustness, or humanize."""
        if not self._local_request():
            return
        data = self._json_body()
        if data is None:
            return
        if self.path == "/api/model/train":
            status, response = self.server.state.start_training()
            self._respond(status, response)
        elif self.path in ("/api/analyze", "/api/robustness", "/api/humanize"):
            text = self._submitted_text(data)
            if text is None:
                return
            try:
                with self.server.state.inference_lock:
                    self.server.state.status()
                    if self.path == "/api/analyze":
                        result = analyze_text(text)
                    elif self.path == "/api/robustness":
                        result = robustness_report(text)
                    else:
                        result = humanize_text(text)
                self._respond(200, result)
            except Exception as error:
                self.log_error("Analysis failed: %s", error)
                self._respond(500, {"error": "Analysis could not finish. Check model status and try again."})
        else:
            self._respond(404, {"error": "Endpoint not found."})


def create_server(port: int = 8000, state: ApplicationState | None = None) -> ThreadingHTTPServer:
    """Create a loopback-only HTTP server; port 0 is useful in integration tests."""
    server = ThreadingHTTPServer(("127.0.0.1", port), APIHandler)
    server.state = state if state is not None else ApplicationState()
    return server


def main() -> None:
    """Run the backend used by npm run dev."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = create_server(args.port)

    def stop_on_signal(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    print(f"AI Content Breaker API: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.state.stop_training()
        server.server_close()


if __name__ == "__main__":
    main()
