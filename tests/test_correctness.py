import json
import importlib
import sys
from types import ModuleType
from pathlib import Path

import pandas as pd
import pytest

from src.dataset.loader import _mbpp_test_list, load_problems
from src.evaluation.correctness import run_tests
from src.evaluation.executor import ExecutionResult
from src.evaluation.runner import evaluate_response
from src.config import settings
import src.evaluation.executor as executor_module


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(autouse=True)
def fixed_fixture_executor(monkeypatch):
    # These tests use only deterministic, finite fixtures; the host backend is
    # explicitly selected here and is never the application default.
    monkeypatch.setattr(settings, "CODE_EXECUTION_BACKEND", "host-dev")


def test_sample_adapter_parses_json_booleans_as_cases():
    task = load_problems(DATA_DIR / "sample" / "problems.csv", "sample")[0]
    assert task["tests"]["kind"] == "sample"
    assert task["tests"]["cases"][0] == {"expression": "is_prime(2)", "expected": True}
    assert task["task_id"] == task["problem_id"] == "P001"


def test_sample_cases_pass_and_report_individual_outcomes():
    tests = {"kind": "sample", "cases": [
        {"expression": "is_prime(2)", "expected": True},
        {"expression": "is_prime(9)", "expected": False},
    ]}
    code = "def is_prime(n):\n    return n > 1 and all(n % i for i in range(2, n))\n"
    result = run_tests(code, tests)
    assert result["correctness_status"] == "passed"
    assert (result["passed_tests"], result["failed_tests"], result["total_tests"]) == (2, 0, 2)
    assert result["correctness_score"] == 100.0
    assert [case["status"] for case in result["test_results"]] == ["passed", "passed"]


def test_sample_failing_case_produces_partial_score_and_failure_detail():
    tests = {"kind": "sample", "cases": [
        {"expression": "add(2, 3)", "expected": 5},
        {"expression": "add(2, 4)", "expected": 7},
    ]}
    result = run_tests("def add(a, b):\n    return a + b\n", tests)
    assert result["correctness_status"] == "failed"
    assert (result["passed_tests"], result["failed_tests"], result["total_tests"]) == (1, 1, 2)
    assert result["correctness_score"] == 50.0
    assert result["test_results"][1]["error_type"] == "assertion_failure"


def test_invalid_sample_specification_has_no_score():
    result = run_tests("def f(): return True", {"kind": "sample", "cases": []})
    assert result["correctness_status"] == "invalid_tests"
    assert result["correctness_score"] is None
    assert result["total_tests"] == 0


def test_generated_syntax_error_is_classified():
    result = run_tests("def broken(:", {"kind": "sample", "cases": [
        {"expression": "broken()", "expected": True},
    ]})
    assert result["correctness_status"] == "syntax_error"
    assert result["correctness_score"] == 0.0
    assert result["test_results"][0]["error_type"] == "syntax_error"



def test_safe_unavailable_keeps_correctness_unavailable_for_syntax_error(monkeypatch):
    monkeypatch.setattr(settings, "CODE_EXECUTION_BACKEND", "safe")
    result = run_tests("def broken(:", {"kind": "sample", "cases": [
        {"expression": "broken()", "expected": True},
    ]})
    assert result["correctness_status"] == "executor_unavailable"
    assert result["execution_status"] == "executor_unavailable"
    assert result["correctness_score"] is None
    assert result["passed_tests"] is None
    assert result["failed_tests"] is None
    assert result["correctness_errors"][0]["error_type"] == "syntax_error"
def test_runtime_error_is_classified_per_case():
    result = run_tests(
        "def boom():\n    raise RuntimeError('fixture failure')\n",
        {"kind": "sample", "cases": [{"expression": "boom()", "expected": 1}]},
    )
    assert result["correctness_status"] == "runtime_error"
    assert result["test_results"][0]["error_type"] == "runtime_error"
    assert "fixture failure" in result["test_results"][0]["error"]


def test_timeout_is_classified_with_short_controlled_fixture():
    result = run_tests(
        "def hang():\n    while True:\n        pass\n",
        {"kind": "sample", "cases": [{"expression": "hang()", "expected": True}]},
        timeout=0.3,
    )
    assert result["correctness_status"] == "timeout"
    assert result["timeout"] is True
    assert result["correctness_score"] == 0.0


def test_humaneval_check_is_invoked_and_each_assertion_is_recorded():
    task = load_problems(DATA_DIR / "humaneval_dataset.csv", "humaneval")[0]
    code = """def has_close_elements(numbers, threshold):
    for i, value in enumerate(numbers):
        for other in numbers[i + 1:]:
            if abs(value - other) < threshold:
                return True
    return False
"""
    result = run_tests(code, task["tests"], task["entry_point"])
    assert result["correctness_status"] == "passed"
    assert result["passed_tests"] == result["total_tests"] == 7


def test_humaneval_wrong_candidate_does_not_false_pass():
    task = load_problems(DATA_DIR / "humaneval_dataset.csv", "humaneval")[0]
    result = run_tests(
        "def has_close_elements(numbers, threshold):\n    return False\n",
        task["tests"], task["entry_point"],
    )
    assert result["correctness_status"] == "failed"
    assert (result["passed_tests"], result["failed_tests"], result["total_tests"]) == (3, 4, 7)


def test_humaneval_completion_uses_prompt_prefix_and_imports():
    task = load_problems(DATA_DIR / "humaneval_dataset.csv", "humaneval")[0]
    completion = """    for i, value in enumerate(numbers):
        for other in numbers[i + 1:]:
            if abs(value - other) < threshold:
                return True
    return False
"""
    result = run_tests(
        completion,
        task["tests"],
        task["entry_point"],
        prompt=task["prompt"],
    )
    assert result["correctness_status"] == "passed"
    assert result["passed_tests"] == result["total_tests"] == 7


def test_mbpp_csv_list_preserves_adjacent_assertion_literals():
    df = pd.read_csv(DATA_DIR / "mbpp_dataset.csv")
    first = df.iloc[0]
    primary = _mbpp_test_list(first["test_list"])
    challenge = _mbpp_test_list(first["challenge_test_list"])
    assert len(primary) == 3
    assert len(challenge) == 2
    assert all(item.lstrip().startswith("assert ") for item in primary + challenge)


def test_mbpp_first_csv_task_runs_primary_setup_and_challenge_tests():
    task = load_problems(DATA_DIR / "mbpp_dataset.csv", "mbpp")[0]
    code = """def remove_Occ(s, ch):
    for i, value in enumerate(s):
        if value == ch:
            s = s[:i] + s[i + 1:]
            break
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ch:
            s = s[:i] + s[i + 1:]
            break
    return s
"""
    result = run_tests(code, task["tests"], task["entry_point"])
    assert result["correctness_status"] == "passed"
    assert result["passed_tests"] == result["total_tests"] == 5


def test_mbpp_setup_code_is_available_to_solution_and_assertions():
    spec = {"kind": "assertions", "setup_code": "OFFSET = 4\n", "cases": [
        {"source": "primary", "code": "assert add_offset(3) == 7"},
    ]}
    result = run_tests(
        "def add_offset(value):\n    return value + OFFSET\n",
        spec,
    )
    assert result["correctness_status"] == "passed"
    assert result["passed_tests"] == result["total_tests"] == 1


def test_combined_dataset_keeps_its_mbpp_assertion_tests():
    task = next(
        item for item in load_problems(DATA_DIR / "programming_dataset.csv", "programming")
        if item["dataset"] == "mbpp"
    )
    assert task["tests"]["kind"] == "assertions"
    assert len(task["tests"]["cases"]) == 3
    assert {item["source"] for item in task["tests"]["cases"]} == {"primary"}


def test_combined_dataset_builder_retains_mbpp_setup_and_challenge(monkeypatch, tmp_path):
    datasets_stub = ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "datasets", datasets_stub)
    data_loader = importlib.import_module("src.data_loader")

    humaneval = pd.DataFrame([{
        "task_id": "HumanEval/fixture",
        "prompt": "def solve(x):\n",
        "canonical_solution": "return x",
        "test": "def check(candidate):\n    assert candidate(1) == 1",
        "entry_point": "solve",
    }])
    mbpp = pd.DataFrame([{
        "task_id": 1,
        "text": "Return x plus an offset",
        "code": "def solve(x): return x + OFFSET",
        "test_list": ["assert solve(1) == 3"],
        "test_setup_code": "OFFSET = 2",
        "challenge_test_list": ["assert solve(2) == 4"],
    }])
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_loader, "load_humaneval", lambda: humaneval)
    monkeypatch.setattr(data_loader, "load_mbpp", lambda: mbpp)

    combined = data_loader.create_common_dataset()
    row = combined[combined["dataset"] == "MBPP"].iloc[0]
    assert row["test_setup_code"] == "OFFSET = 2"
    assert row["challenge_test_list"] == "assert solve(2) == 4"
    combined_tasks = load_problems(tmp_path / "programming_dataset.csv", "programming")
    mbpp_task = next(task for task in combined_tasks if task["dataset"] == "mbpp")
    assert mbpp_task["tests"]["setup_code"] == "OFFSET = 2"
    assert [case["source"] for case in mbpp_task["tests"]["cases"]] == ["primary", "challenge"]


def test_result_contract_includes_counts_status_and_errors():
    result = run_tests(
        "def f(): return 1",
        {"kind": "sample", "cases": [{"expression": "f()", "expected": 2}]},
    )
    assert set(("correctness_status", "passed", "passed_tests", "failed_tests", "total_tests",
                "correctness_score", "test_results", "correctness_errors", "timeout",
                "execution_status", "execution_error", "executor_backend", "executor_metadata")) <= set(result)


def test_safe_default_blocks_without_launching_subprocess(monkeypatch):
    monkeypatch.setattr(settings, "CODE_EXECUTION_BACKEND", "safe")
    def forbidden(*args, **kwargs):
        raise AssertionError("safe default must not execute on the host")
    monkeypatch.setattr(executor_module.subprocess, "run", forbidden)

    result = run_tests(
        "def f(): return 1",
        {"kind": "sample", "cases": [{"expression": "f()", "expected": 1}]},
    )

    assert result["correctness_status"] == "executor_unavailable"
    assert result["execution_status"] == "executor_unavailable"
    assert result["correctness_score"] is None
    assert result["passed_tests"] is None
    assert result["total_tests"] == 1
    assert result["executor_backend"] == "safe-unavailable"


@pytest.mark.parametrize("payload", [
    "import socket; socket.create_connection(('127.0.0.1', 1), timeout=0.1)",
    "open('C:/nonexistent-controlled-fixture', 'w').write('x')",
    "import subprocess; subprocess.Popen(['cmd', '/c', 'exit'])",
    "while True: pass",
    "print('x' * 100000000)",
])
def test_untrusted_security_fixtures_are_blocked_without_execution(monkeypatch, payload):
    monkeypatch.setattr(settings, "CODE_EXECUTION_BACKEND", "safe")
    monkeypatch.setattr(
        executor_module.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("payload was launched")),
    )
    result = run_tests(
        payload,
        {"kind": "sample", "cases": [{"expression": "True", "expected": True}]},
    )
    assert result["execution_status"] == "executor_unavailable"
    assert result["correctness_score"] is None


def test_host_development_backend_reports_no_isolation():
    result = run_tests(
        "def f(): return 1",
        {"kind": "sample", "cases": [{"expression": "f()", "expected": 1}]},
    )
    assert result["executor_backend"] == "host-dev-unsafe"
    assert result["executor_metadata"]["isolation"] == "none"


def test_policy_rejection_is_an_infrastructure_outcome_not_a_code_failure():
    class RejectingExecutor:
        name = "fixture-policy-backend"
        def execute(self, harness_path, timeout):
            return ExecutionResult("policy_rejected", self.name, error="fixture policy")

    result = run_tests(
        "def f(): return 1",
        {"kind": "sample", "cases": [{"expression": "f()", "expected": 1}]},
        executor=RejectingExecutor(),
    )
    assert result["correctness_status"] == "policy_rejected"
    assert result["execution_status"] == "policy_rejected"
    assert result["correctness_score"] is None


def test_runner_preserves_percentage_score_from_test_evidence():
    task = {
        "prompt": "Add two numbers",
        "tests": {"kind": "sample", "cases": [
            {"expression": "add(1, 2)", "expected": 3},
            {"expression": "add(1, 3)", "expected": 5},
        ]},
        "required_features": [],
        "task_id": "fixture-score",
        "dataset": "sample",
    }
    result = evaluate_response(task, "```python\ndef add(a, b): return a + b\n```")
    assert result["passed_tests"] == 1
    assert result["total_tests"] == 2
    assert result["correctness_score"] == 50.0
    json.dumps(result)
