from src.scoring.scoring_engine import calculate_scores

def test_score_range():
    row = {
        "correctness_score": 100,
        "completeness_score": 100,
        "security_score": 100,
        "maintainability_score": 100,
        "explainability_score": 100,
        "usability_score": 100,
        "efficiency_score": 100,
        "robustness_score": 100,
        "response_time": 1,
        "total_tokens": 50,
    }
    result = calculate_scores(row)
    assert result["overall_score"] == 100


def test_unavailable_correctness_does_not_produce_an_overall_score():
    row = {
        "correctness_score": None,
        "completeness_score": 100,
        "security_score": 100,
        "maintainability_score": 100,
        "explainability_score": 100,
        "usability_score": 100,
        "efficiency_score": 100,
        "robustness_score": 100,
    }
    result = calculate_scores(row)
    assert result["overall_score"] is None
