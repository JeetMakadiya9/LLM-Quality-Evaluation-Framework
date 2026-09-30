import pandas as pd

from src.analysis import analysis


def test_summary_keeps_reused_model_labels_separate_by_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "RESULTS_DIR", tmp_path)
    frame = pd.DataFrame([
        {"provider": None, "model": "Model_A", "correctness_score": 20.0},
        {"provider": "mock", "model": "Model_A", "correctness_score": 100.0},
        {"provider": "ollama", "model": "Ollama", "correctness_score": 50.0},
    ])

    summary = analysis.generate_summary_csv(frame)

    assert set(summary.index) == {
        "LEGACY / Model_A (legacy)", "DEMO / mock / Model_A", "ollama / Ollama"
    }
    assert summary.loc["LEGACY / Model_A (legacy)", "correctness_score"] == 20.0
    assert summary.loc["DEMO / mock / Model_A", "correctness_score"] == 100.0


def test_comparison_uses_provider_and_real_model_identity_and_separates_demo_rows():
    from src.analysis.analysis import DIMENSIONS, build_model_comparison, dimension_comparison_frame

    scores = {dimension: 80.0 for dimension in DIMENSIONS}
    rows = [
        {**scores, "provider": "ollama", "model": "Ollama", "model_name": "ollama/qwen3:4b", "task_id": "P001", "run_id": "run-1", "overall_score": None,
         "provider_status": "completed", "evaluation_metadata": '{"provider_metadata":{"completion_status":"complete","done_reason":"stop"},"efficiency_evidence":{"generation_time_seconds":2.0,"tokens_per_second":10.0}}'},
        {**scores, "provider": "ollama", "model": "qwen3.5:latest", "model_name": "qwen3.5:latest", "task_id": "P001", "run_id": "run-1", "overall_score": None,
         "provider_status": "completed", "evaluation_metadata": '{"provider_metadata":{"completion_status":"partial","done_reason":"length"},"efficiency_evidence":{"generation_time_seconds":4.0,"tokens_per_second":5.0}}'},
        {**scores, "provider": "mock", "model": "Model_A", "model_name": "gpt-fixture", "task_id": "P001", "run_id": "run-1", "overall_score": 80.0,
         "provider_status": "completed", "evaluation_metadata": "{}"},
    ]
    summary = build_model_comparison(pd.DataFrame(rows))
    assert set(summary["comparison_label"]) == {
        "ollama / qwen3:4b", "ollama / qwen3.5:latest", "DEMO / mock / Model_A"
    }
    first = summary.set_index("comparison_label").loc["ollama / qwen3:4b"]
    second = summary.set_index("comparison_label").loc["ollama / qwen3.5:latest"]
    assert first["task_count"] == 1 and first["complete_responses"] == 1
    assert second["partial_responses"] == 1
    assert first["correctness_score"] == 80.0
    assert pd.isna(first["overall_score"])
    assert first["efficiency_score_available_tasks"] == 1
    assert first["generation_time_seconds"] == 2.0
    assert first["tokens_per_second"] == 10.0
    comparison = dimension_comparison_frame(summary)
    assert len(comparison) == 9
    assert "Robustness / Reliability" in comparison["dimension"].tolist()
    assert pd.isna(comparison.loc[comparison["dimension"] == "Overall Score", "ollama / qwen3:4b"].iloc[0])


def test_ollama_model_name_prefix_is_removed_from_visible_identity():
    frame = pd.DataFrame([{"provider": "ollama", "model": "Ollama", "model_name": "ollama/qwen3:4b"}])
    prepared = analysis._comparison_frame(frame)
    assert prepared.iloc[0]["model_identity"] == "qwen3:4b"
    assert prepared.iloc[0]["comparison_label"] == "ollama / qwen3:4b"


def test_comparison_overall_is_unavailable_when_any_task_score_is_missing():
    from src.analysis.analysis import DIMENSIONS, build_model_comparison

    rows = []
    for task, correctness, overall in (("P001", 100.0, 80.0), ("P002", None, None)):
        row = {dimension: 80.0 for dimension in DIMENSIONS}
        row.update(provider="ollama", model="qwen3:4b", model_name="qwen3:4b", task_id=task, overall_score=overall)
        row["correctness_score"] = correctness
        rows.append(row)
    summary = build_model_comparison(pd.DataFrame(rows)).iloc[0]
    assert summary["overall_score_available_tasks"] == 1
    assert pd.isna(summary["overall_score"])


def test_analysis_keeps_the_same_model_separate_across_runs():
    from src.analysis.analysis import build_model_comparison

    frame = pd.DataFrame([
        {"provider": "ollama", "model": "qwen3:4b", "model_name": "qwen3:4b", "run_id": "run-old", "task_id": "P001", "correctness_score": 20.0},
        {"provider": "ollama", "model": "qwen3:4b", "model_name": "qwen3:4b", "run_id": "run-new", "task_id": "P001", "correctness_score": 90.0},
    ])
    summary = build_model_comparison(frame)
    assert set(summary["comparison_label"]) == {
        "run-old / ollama / qwen3:4b", "run-new / ollama / qwen3:4b"
    }

