"""Dataset adapters for the canonical correctness task consumed by the runner."""

import ast
import io
import json
import math
import tokenize
from pathlib import Path

import pandas as pd


def _text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _sample_spec(value):
    try:
        cases = json.loads(_text(value))
        if not isinstance(cases, list) or not cases:
            raise ValueError("sample tests must be a non-empty JSON list")
        normalized = []
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("expression"), str) or "expected" not in case:
                raise ValueError("each sample case needs expression and expected fields")
            normalized.append({"expression": case["expression"], "expected": case["expected"]})
        return {"kind": "sample", "cases": normalized}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"kind": "invalid", "error": f"Invalid sample test specification: {exc}"}


def _mbpp_test_list(value):
    """Read Python-list CSV serialization, including adjacent literals in old files."""
    raw = _text(value).strip()
    if not raw:
        return []
    if not (raw.startswith("[") and raw.endswith("]")):
        # The combined dataset stores MBPP assertions as newline-separated text.
        return [line.strip() for line in raw.splitlines() if line.strip()]

    try:
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("test list must contain strings")
        if len(parsed) > 1:
            return parsed
        # Some existing CSV rows omit commas between adjacent string literals.
        # Python concatenates those into one value, so recover each literal token.
        tokens = tokenize.generate_tokens(io.StringIO(raw).readline)
        literals = [ast.literal_eval(tok.string) for tok in tokens if tok.type == tokenize.STRING]
        return literals if len(literals) > 1 else parsed
    except (SyntaxError, ValueError, tokenize.TokenError) as exc:
        raise ValueError(f"Invalid MBPP test_list: {exc}") from exc


def _mbpp_spec(assertions, setup_code="", challenge_assertions=None):
    cases = [
        {"source": "primary", "code": assertion}
        for assertion in assertions
    ]
    cases.extend(
        {"source": "challenge", "code": assertion}
        for assertion in (challenge_assertions or [])
    )
    if not cases:
        return {"kind": "invalid", "error": "MBPP task has no assertion tests"}
    return {"kind": "assertions", "setup_code": setup_code, "cases": cases}


def _canonical_task(task_id, prompt, dataset, entry_point, test_spec, metadata=None):
    task_id = str(task_id)
    return {
        # problem_id remains as a compatibility alias for existing pipeline/UI code.
        "task_id": task_id,
        "problem_id": task_id,
        "dataset": dataset,
        "prompt": _text(prompt),
        "entry_point": _text(entry_point) or None,
        "tests": test_spec,
        "metadata": metadata or {},
        # Existing non-correctness evaluators still consume this field.
        "required_features": json.dumps([]),
    }


def _load_sample(df):
    required = {"problem_id", "prompt", "tests", "required_features"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    records = []
    for _, row in df.iterrows():
        task = _canonical_task(
            row["problem_id"], row["prompt"], "sample", None,
            _sample_spec(row["tests"]), {"source": "data/sample/problems.csv"},
        )
        task["required_features"] = _text(row["required_features"])
        records.append(task)
    return records


def _load_humaneval_row(row, dataset="humaneval"):
    return _canonical_task(
        row["task_id"], row["prompt"], "humaneval", row.get("entry_point"),
        {"kind": "humaneval", "source": _text(row["test"])},
        {"source_dataset": dataset},
    )


def _load_mbpp_row(row, dataset="mbpp"):
    try:
        assertions = _mbpp_test_list(row.get("test_list", row.get("test", "")))
        challenge = _mbpp_test_list(row.get("challenge_test_list", ""))
        test_spec = _mbpp_spec(assertions, _text(row.get("test_setup_code", "")), challenge)
    except ValueError as exc:
        test_spec = {"kind": "invalid", "error": str(exc)}
    return _canonical_task(
        row["task_id"], row.get("text", row.get("prompt", "")), "mbpp", None,
        test_spec, {"source_dataset": dataset},
    )


def load_problems(path: Path, dataset: str = "sample"):
    """Load raw CSV rows into the common task shape used by correctness evaluation."""
    df = pd.read_csv(path)

    if dataset == "sample":
        return _load_sample(df)

    if dataset == "humaneval":
        required = {"task_id", "prompt", "test"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"HumanEval dataset is missing columns: {sorted(missing)}")
        return [_load_humaneval_row(row) for row in df.to_dict("records")]

    if dataset == "mbpp":
        required = {"task_id", "text", "test_list"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"MBPP dataset is missing columns: {sorted(missing)}")
        return [_load_mbpp_row(row) for row in df.to_dict("records")]

    if dataset == "programming":
        required = {"task_id", "dataset", "prompt", "test"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Programming dataset is missing columns: {sorted(missing)}")
        records = []
        for row in df.to_dict("records"):
            source = _text(row["dataset"]).lower()
            if source == "humaneval":
                records.append(_load_humaneval_row(row, "programming_dataset.csv"))
            elif source == "mbpp":
                # The combined file has no setup/challenge columns; the source
                # builder already flattened primary assertion strings into test.
                records.append(_load_mbpp_row(row, "programming_dataset.csv"))
            else:
                records.append(_canonical_task(
                    row["task_id"], row["prompt"], source or "programming", row.get("entry_point"),
                    {"kind": "invalid", "error": f"Unsupported combined dataset label: {source}"},
                ))
        return records

    raise ValueError(f"Unknown dataset: {dataset}")
