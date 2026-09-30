from src.config import settings
from src.llm import pipeline

def test_ollama_initialization_failure_is_recorded_without_fake_scores(monkeypatch):
    class Unavailable:
        def __init__(self): raise ConnectionError("local fixture endpoint unavailable")
    monkeypatch.setattr(settings,"OLLAMA_MODEL","qwen3:4b")
    monkeypatch.setattr(pipeline,"OLLAMA_MODEL","qwen3:4b")
    monkeypatch.setattr(pipeline,"OllamaProvider",Unavailable)
    task={"problem_id":"P001","task_id":"P001","dataset":"sample","prompt":"write function","tests":{},"required_features":"[]"}
    result=pipeline.run_experiment([task],provider="ollama")[0]
    assert result["provider_status"]=="error"
    assert result["provider_error"]=="ConnectionError"
    assert result["robustness_score"]==0.0
    assert result["correctness_score"] is None and result["overall_score"] is None
    assert result["response_time"] is None and result["total_tokens"] is None
