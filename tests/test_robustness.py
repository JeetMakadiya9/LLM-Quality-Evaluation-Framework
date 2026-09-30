from src.evaluation.robustness import evaluate_robustness
from src.scoring.scoring_engine import DIMENSION_WEIGHTS, calculate_scores


def test_complete_code_response_has_auditable_checks():
    text = "def add(a, b): return a + b"
    result = evaluate_robustness(text, completion_metadata={"done_reason": "stop", "completion_status": "complete", "truncated": False})
    assert result["robustness_score"] == 100.0
    assert all(value is True for value in result["robustness_checks"].values())


def test_truncated_response_scores_lower_than_complete_response():
    text = "def add(a, b): return a + b"
    complete = evaluate_robustness(text, completion_metadata={"done_reason": "stop", "completion_status": "complete", "truncated": False})
    partial = evaluate_robustness(text, completion_metadata={"done_reason": "length", "completion_status": "partial", "truncated": True})
    assert partial["robustness_score"] < complete["robustness_score"]
    assert partial["robustness_status"] == "partial"
    assert partial["robustness_checks"]["generation_completed_without_truncation"] is False


def test_provider_failure_is_recorded_without_overall_score():
    result = evaluate_robustness(None, provider_error="RuntimeError")
    assert result["robustness_score"] == 0.0
    assert result["robustness_status"] == "provider_error"
    row = {name: 100.0 for name in DIMENSION_WEIGHTS}; row.update(result); row["correctness_score"] = None
    assert calculate_scores(row)["overall_score"] is None


def test_empty_and_malformed_outputs_lower_reliability_score():
    empty = evaluate_robustness("")
    malformed = evaluate_robustness("```python\ndef add(a, b): return a + b")
    assert empty["robustness_checks"]["response_non_empty"] is False
    assert malformed["robustness_checks"]["response_structure_valid"] is False
    assert empty["robustness_score"] < 100.0
    assert malformed["robustness_score"] < 100.0


def test_inline_thinking_removed_response_parses_as_valid():
    text = "def add(a, b): return a + b"
    result = evaluate_robustness(text, completion_metadata={"done_reason": "stop", "inline_thinking_removed": True})
    assert result["robustness_checks"]["output_parsing_succeeded"] is True
    assert result["robustness_score"] == 100.0


def test_all_eight_dimensions_and_weights_participate():
    assert len(DIMENSION_WEIGHTS) == 8 and sum(DIMENSION_WEIGHTS.values()) == 1.0
    assert calculate_scores({key: 100 for key in DIMENSION_WEIGHTS})["overall_score"] == 100.0
def test_unknown_completion_metadata_is_not_counted_as_complete_or_failed():
    result = evaluate_robustness("def add(a, b): return a + b", completion_metadata={"completion_status": "unknown", "truncated": False})
    assert result["robustness_checks"]["generation_completed_without_truncation"] is None