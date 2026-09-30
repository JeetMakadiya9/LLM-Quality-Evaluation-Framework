import json

from src.evaluation.completeness import completeness_score

FEATURES = [
    {"name": "positive case", "keywords": ["n > 1"]},
    {"name": "divisibility", "keywords": ["%"]},
]


def test_complete_response_uses_feature_coverage_score():
    result = completeness_score("def is_prime(n):\n    return n > 1 and n % 2\n", json.dumps(FEATURES), {"done_reason": "stop"})
    assert result["completeness_score"] == 100.0
    assert result["completeness_evidence"]["completion_status"] == "complete"
    assert result["completeness_evidence"]["detected_features"] == ["positive case", "divisibility"]


def test_truncated_response_applies_transparent_multiplier():
    result = completeness_score("def is_prime(n): return n > 1 and n % 2", json.dumps(FEATURES), {"done_reason": "length"})
    assert result["completeness_score"] == 75.0
    assert result["completeness_evidence"]["truncated"] is True
    assert result["completeness_evidence"]["truncation_multiplier"] == 0.75


def test_missing_features_reduce_content_coverage_before_truncation():
    result = completeness_score("def is_prime(n): return n > 1", json.dumps(FEATURES), {"done_reason": "stop"})
    assert result["completeness_score"] == 50.0
    assert result["missing_features"] == "divisibility"


def test_all_features_present_still_penalized_when_truncated():
    result = completeness_score("n > 1 and n % 2", json.dumps(FEATURES), {"completion_status": "partial", "truncated": True})
    assert result["completeness_evidence"]["content_completeness"] == 100.0
    assert result["completeness_score"] == 75.0


def test_empty_response_or_missing_rubric_is_unavailable():
    empty = completeness_score("", json.dumps(FEATURES), {"done_reason": "stop"})
    no_rubric = completeness_score("def f(): return 1", "[]", {"done_reason": "stop"})
    assert empty["completeness_score"] is None
    assert no_rubric["completeness_score"] is None
    assert empty["completeness_evidence"]["status"] == "unavailable"