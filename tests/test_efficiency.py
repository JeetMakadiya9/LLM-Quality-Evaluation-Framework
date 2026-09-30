from src.evaluation.efficiency import efficiency_evidence, normalize_efficiency


def test_generation_telemetry_uses_ollama_generation_duration():
    evidence = efficiency_evidence(12.0, 120, 6_000_000_000)
    assert evidence["response_time_seconds"] == 12.0
    assert evidence["generated_tokens"] == 120
    assert evidence["generation_time_seconds"] == 6.0
    assert evidence["tokens_per_second"] == 20.0
    assert evidence["throughput_basis"] == "provider_generation_duration"


def test_missing_and_zero_telemetry_are_not_fabricated():
    missing = efficiency_evidence(None, None, None)
    zero = efficiency_evidence(0, 0, 0)
    rows = [{"efficiency_evidence": missing}, {"efficiency_evidence": zero}]
    normalize_efficiency(rows)
    assert missing["tokens_per_second"] is None
    assert all(row["efficiency_score"] is None for row in rows)


def test_relative_normalization_rewards_faster_and_higher_throughput():
    rows = [
        {"efficiency_evidence": {"response_time_seconds": 10.0, "tokens_per_second": 20.0}},
        {"efficiency_evidence": {"response_time_seconds": 20.0, "tokens_per_second": 10.0}},
    ]
    normalize_efficiency(rows)
    assert rows[0]["efficiency_score"] == 100.0
    assert rows[1]["efficiency_score"] == 0.0
    assert rows[0]["efficiency_evidence"]["normalized_components"]["latency_score"] > rows[1]["efficiency_evidence"]["normalized_components"]["latency_score"]
    assert rows[0]["efficiency_evidence"]["normalized_components"]["throughput_score"] > rows[1]["efficiency_evidence"]["normalized_components"]["throughput_score"]


def test_time_only_fallback_is_labeled_and_normalized():
    rows = [
        {"efficiency_evidence": efficiency_evidence(2.0, None)},
        {"efficiency_evidence": efficiency_evidence(4.0, None)},
    ]
    normalize_efficiency(rows)
    assert rows[0]["efficiency_score"] == 100.0
    assert rows[1]["efficiency_score"] == 0.0
    assert rows[0]["efficiency_evidence"]["throughput_basis"] == "wall_clock_response_time_fallback"
    assert rows[0]["efficiency_evidence"]["tokens_per_second"] is None