import json

import pytest

from src.config import settings
from src.llm import ollama_client


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_ollama_provider_uses_local_api_and_returns_pipeline_contract(monkeypatch):
    calls = []
    responses = [
        {"version": "0.34.4"},
        {"models": [{"name": "qwen3:4b", "digest": "fixture-digest"}]},
        {
            "model": "qwen3:4b",
            "response": "def add(a, b): return a + b",
            "thinking": "internal reasoning must not be used as the answer",
            "prompt_eval_count": 12,
            "eval_count": 8,
            "total_duration": 1_000_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_duration": 200_000_000,
            "eval_duration": 600_000_000,
            "done_reason": "stop",
        },
    ]

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, json.loads(request.data) if request.data else None, timeout))
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(ollama_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(settings, "OLLAMA_NUM_CTX", 4096)
    monkeypatch.setattr(settings, "OLLAMA_TEMPERATURE", 0)
    monkeypatch.setattr(settings, "OLLAMA_SEED", 42)
    monkeypatch.setattr(settings, "OLLAMA_NUM_PREDICT", 1024)
    monkeypatch.setattr(settings, "OLLAMA_THINK", False)

    provider = ollama_client.OllamaProvider()
    result = provider.generate("solve this fixture")

    assert result["text"] == "def add(a, b): return a + b"
    assert result["response_time"] >= 0
    assert (result["input_tokens"], result["output_tokens"], result["total_tokens"]) == (12, 8, 20)
    assert result["provider_metadata"]["provider"] == "ollama"
    assert result["provider_metadata"]["model_digest"] == "fixture-digest"
    assert result["provider_metadata"]["completion_status"] == "complete"
    assert result["provider_metadata"]["truncated"] is False
    assert result["provider_metadata"]["thinking_present"] is True
    assert result["provider_metadata"]["thinking_field_present"] is True
    assert result["provider_metadata"]["think_requested"] is False
    assert result["provider_metadata"]["settings_requested"] == {
        "num_ctx": 4096,
        "temperature": 0,
        "seed": 42,
        "num_predict": 1024,
    }
    assert calls[-1][0] == "http://127.0.0.1:11434/api/generate"
    assert calls[-1][1]["stream"] is False
    assert calls[-1][1]["think"] is False
    assert calls[-1][1]["options"]["num_predict"] == 1024


def test_ollama_provider_removes_inline_thinking_from_response(monkeypatch):
    provider = object.__new__(ollama_client.OllamaProvider)
    provider.model = "qwen3:4b"
    provider.model_digest = "fixture-digest"
    provider.ollama_version = "fixture-version"
    provider.options = {"num_predict": 1024}
    monkeypatch.setattr(provider, "_request_json", lambda endpoint, payload: {
        "response": "<think>private reasoning is omitted</think>\ndef add(a, b): return a + b",
        "done": True,
        "done_reason": "stop",
        "eval_count": 12,
    })

    result = provider.generate("fixture prompt")

    assert result["text"] == "def add(a, b): return a + b"
    assert "private reasoning" not in result["text"]
    assert result["provider_metadata"]["thinking_present"] is True
    assert result["provider_metadata"]["thinking_field_present"] is False
    assert result["provider_metadata"]["inline_thinking_removed"] is True


def test_ollama_provider_rejects_unclosed_inline_thinking(monkeypatch):
    provider = object.__new__(ollama_client.OllamaProvider)
    provider.model = "qwen3:4b"
    provider.model_digest = "fixture-digest"
    provider.ollama_version = "fixture-version"
    provider.options = {"num_predict": 1024}
    monkeypatch.setattr(provider, "_request_json", lambda endpoint, payload: {
        "response": "<think>incomplete reasoning", "done_reason": "stop"
    })

    with pytest.raises(ollama_client.OllamaProviderError, match="unclosed thinking block"):
        provider.generate("fixture prompt")


def test_ollama_provider_requires_model_to_be_installed(monkeypatch):
    responses = [{"version": "0.34.4"}, {"models": []}]
    monkeypatch.setattr(
        ollama_client,
        "urlopen",
        lambda request, timeout: FakeResponse(responses.pop(0)),
    )

    with pytest.raises(ollama_client.OllamaProviderError, match="not available"):
        ollama_client.OllamaProvider()


def test_ollama_provider_rejects_empty_final_response(monkeypatch):
    provider = object.__new__(ollama_client.OllamaProvider)
    provider.model = "qwen3:4b"
    provider.model_digest = "fixture-digest"
    provider.ollama_version = "fixture-version"
    provider.options = {"num_predict": 256}
    monkeypatch.setattr(provider, "_request_json", lambda endpoint, payload: {
        "response": "", "thinking": "reasoning exhausted the token cap"
    })

    with pytest.raises(ollama_client.OllamaProviderError, match="no final response text"):
        provider.generate("fixture prompt")


def test_ollama_provider_returns_nonempty_truncated_response_as_partial(monkeypatch):
    provider = object.__new__(ollama_client.OllamaProvider)
    provider.model = "qwen3:4b"
    provider.model_digest = "fixture-digest"
    provider.ollama_version = "fixture-version"
    provider.options = {"num_predict": 1024}
    monkeypatch.setattr(provider, "_request_json", lambda endpoint, payload: {
        "model": "qwen3:4b",
        "response": "def add(a, b): return a + b",
        "done": True,
        "done_reason": "length",
        "eval_count": 1024,
        "total_duration": 2_000_000_000,
        "eval_duration": 1_000_000_000,
    })

    result = provider.generate("fixture prompt")

    assert result["text"] == "def add(a, b): return a + b"
    assert result["response_time"] >= 0
    assert result["output_tokens"] == 1024
    assert result["provider_metadata"]["done_reason"] == "length"
    assert result["provider_metadata"]["completion_status"] == "partial"
    assert result["provider_metadata"]["truncated"] is True
    assert result["provider_metadata"]["timing_ns"]["total_duration"] == 2_000_000_000

def test_ollama_provider_rejects_api_failure(monkeypatch):
    def fail_request(endpoint, payload):
        raise ollama_client.OllamaProviderError("request failed")

    provider = object.__new__(ollama_client.OllamaProvider)
    provider.model = "qwen3:4b"
    provider.model_digest = "fixture-digest"
    provider.ollama_version = "fixture-version"
    provider.options = {"num_predict": 1024}
    monkeypatch.setattr(provider, "_request_json", fail_request)

    with pytest.raises(ollama_client.OllamaProviderError, match="request failed"):
        provider.generate("fixture prompt")