import json
import sqlite3
from pathlib import Path

from src.config import settings
from src.database import database
from src.dataset.loader import load_problems
from src.llm import pipeline
from src.llm.ollama_client import OllamaProviderError


def test_ollama_provider_flows_through_evaluators_and_temp_database(monkeypatch, tmp_path):
    class FakeOllamaProvider:
        model = "qwen3:4b"

        def generate(self, prompt):
            assert prompt == (
                "You are an expert Python programmer.\n"
                "Solve this programming problem.\n"
                "Return ONLY the complete Python code needed to solve it.\n"
                "Do not explain your reasoning.\n"
                "Do not include tags.\n"
                "Programming problem:\n"
                "Write a Python function named is_prime that returns True if an integer is prime and False otherwise.\n"
            )
            return {
                "text": "```python\ndef is_prime(n): return n > 1\n```\nA short explanation.",
                "response_time": 0.2,
                "input_tokens": 20,
                "output_tokens": 12,
                "total_tokens": 32,
                "provider_metadata": {
                    "provider": "ollama",
                    "model": "qwen3:4b",
                    "model_digest": "fixture-digest",
                },
            }

    monkeypatch.setattr(settings, "CODE_EXECUTION_BACKEND", "safe")
    monkeypatch.setattr(pipeline, "OllamaProvider", FakeOllamaProvider)
    task = load_problems(
        Path(__file__).resolve().parents[1] / "data" / "sample" / "problems.csv",
        "sample",
    )[0]

    result = pipeline.run_experiment([task], provider="ollama")[0]

    assert result["provider"] == "ollama"
    assert result["model"] == "qwen3:4b"
    assert result["model_name"] == "qwen3:4b"
    assert result["metadata"]["provider_metadata"]["model_digest"] == "fixture-digest"
    assert result["correctness_status"] == "executor_unavailable"
    assert result["correctness_score"] is None
    assert {
        "correctness_score",
        "completeness_score",
        "security_score",
        "maintainability_score",
        "explainability_score",
        "usability_score",
        "efficiency_score",
        "robustness_score",
        "robustness_checks",
    } <= set(result)

    db_path = tmp_path / "ollama-smoke.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    database.save_results([result])
    with sqlite3.connect(db_path) as connection:
        model, model_name, metadata = connection.execute(
            "SELECT model, model_name, evaluation_metadata FROM evaluations"
        ).fetchone()

    assert (model, model_name) == ("qwen3:4b", "qwen3:4b")
    saved_metadata = json.loads(metadata)
    assert saved_metadata["provider"] == "ollama"
    assert saved_metadata["provider_metadata"]["model_digest"] == "fixture-digest"


def test_ollama_pipeline_supports_multiple_explicit_model_identities(monkeypatch):
    call_order = []

    class FakeOllamaProvider:
        def __init__(self, model):
            self.model = model

        def generate(self, prompt):
            call_order.append((self.model, prompt))
            return {
                "text": "def add(a, b): return a + b",
                "response_time": 1.0,
                "output_tokens": 8,
                "provider_metadata": {"provider": "ollama", "done_reason": "stop", "completion_status": "complete"},
            }

    monkeypatch.setattr(pipeline, "OllamaProvider", FakeOllamaProvider)
    monkeypatch.setattr(
        pipeline,
        "evaluate_response",
        lambda problem, text, metadata: {
            "correctness_score": None,
            "completeness_score": 75.0,
            "security_score": 90.0,
            "maintainability_score": 80.0,
            "explainability_score": 70.0,
            "usability_score": 85.0,
            "efficiency_score": None,
            "robustness_score": 100.0,
            "correctness_status": "executor_unavailable",
        },
    )
    tasks = [
        {"task_id": task_id, "problem_id": task_id, "dataset": "sample", "prompt": f"task {task_id}"}
        for task_id in ("P001", "P002")
    ]

    results = pipeline.run_experiment(
        tasks, provider="ollama", ollama_models=["qwen3:4b", "qwen3.5:latest"]
    )

    assert len(results) == 4
    assert [model for model, _ in call_order] == [
        "qwen3:4b", "qwen3:4b", "qwen3.5:latest", "qwen3.5:latest"
    ]
    assert {(row["provider"], row["model"], row["model_name"]) for row in results} == {
        ("ollama", "qwen3:4b", "qwen3:4b"),
        ("ollama", "qwen3.5:latest", "qwen3.5:latest"),
    }
    assert {row["task_id"] for row in results} == {"P001", "P002"}
    assert len({row["run_id"] for row in results}) == 1
    assert all(row["correctness_score"] is None and row["overall_score"] is None for row in results)


def test_ollama_provider_failure_retains_only_sanitized_diagnostic(monkeypatch):
    class FailingOllamaProvider:
        model = "qwen3.5:latest"

        def generate(self, prompt):
            raise OllamaProviderError(
                "Local Ollama request to /api/generate failed with HTTP 500."
            )

    monkeypatch.setattr(pipeline, "OllamaProvider", lambda model: FailingOllamaProvider())
    task = {"task_id": "P001", "problem_id": "P001", "dataset": "sample", "prompt": "fixture"}

    result = pipeline.run_experiment(
        [task], provider="ollama", ollama_models=["qwen3.5:latest"]
    )[0]

    assert result["provider_error"] == "OllamaProviderError"
    assert result["metadata"]["provider_error_message"] == (
        "Local Ollama request to /api/generate failed with HTTP 500."
    )
    assert result["response_text"] is None
    assert result["correctness_score"] is None
    assert result["overall_score"] is None
