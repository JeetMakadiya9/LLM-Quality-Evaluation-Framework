"""Minimal client for the local Ollama generation API."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import settings


class OllamaProviderError(RuntimeError):
    """Raised when the local Ollama API cannot serve a request."""


def _final_response_text(value):
    """Return user-facing response text without inline Qwen thinking blocks."""
    if not isinstance(value, str):
        return ""
    text = value
    while "<think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start + len("<think>"))
        if end < 0:
            raise OllamaProviderError("Ollama response contains an unclosed thinking block.")
        text = text[:start] + text[end + len("</think>"):]
    # Some Qwen renderers emit a closing marker without an opening marker.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


class OllamaProvider:
    def __init__(self, host=None, model=None, timeout=None):
        self.host = (host or settings.OLLAMA_HOST).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS
        self.options = {
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "temperature": settings.OLLAMA_TEMPERATURE,
            "seed": settings.OLLAMA_SEED,
            "num_predict": settings.OLLAMA_NUM_PREDICT,
        }

        version = self._request_json("/api/version")
        self.ollama_version = version.get("version")
        tags = self._request_json("/api/tags")
        model_info = next(
            (item for item in tags.get("models", []) if item.get("name") == self.model),
            None,
        )
        if model_info is None:
            raise OllamaProviderError(
                f"Ollama model {self.model!r} is not available in the local model list."
            )
        self.model_digest = model_info.get("digest")

    def _request_json(self, endpoint, payload=None):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(f"{self.host}{endpoint}", data=data, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise OllamaProviderError(
                f"Local Ollama request to {endpoint} failed with HTTP {exc.code}."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaProviderError(
                f"Local Ollama request to {endpoint} could not be completed."
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaProviderError(
                f"Local Ollama returned invalid JSON from {endpoint}."
            ) from exc

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": settings.OLLAMA_THINK,
            "options": self.options,
        }
        start = time.perf_counter()
        response = self._request_json("/api/generate", payload)
        elapsed = time.perf_counter() - start

        raw_text = response.get("response")
        text = _final_response_text(raw_text)
        done_reason = response.get("done_reason")
        if not text:
            raise OllamaProviderError(
                "Ollama returned no final response text; the generation may have "
                "reached its token limit while producing reasoning."
            )

        input_tokens = response.get("prompt_eval_count")
        output_tokens = response.get("eval_count")
        total_tokens = (
            input_tokens + output_tokens
            if isinstance(input_tokens, int) and isinstance(output_tokens, int)
            else None
        )

        return {
            "text": text,
            # The pipeline and OpenAI provider use wall-clock seconds.
            "response_time": round(elapsed, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "provider_metadata": {
                "provider": "ollama",
                "model": response.get("model", self.model),
                "model_digest": self.model_digest,
                "ollama_version": self.ollama_version,
                "settings_requested": dict(self.options),
                "think_requested": settings.OLLAMA_THINK,
                "thinking_field_present": "thinking" in response,
                "thinking_present": bool(response.get("thinking")) or (
                    isinstance(raw_text, str) and "<think>" in raw_text.lower()
                ),
                "inline_thinking_removed": isinstance(raw_text, str) and (
                    "<think>" in raw_text.lower() or "</think>" in raw_text.lower()
                ),
                "settings_observation": (
                    "The request values are retained as requested. Ollama's final `response` "
                    "field is used; any separate `thinking` field is not included."
                ),
                "timing_ns": {
                    "total_duration": response.get("total_duration"),
                    "load_duration": response.get("load_duration"),
                    "prompt_eval_duration": response.get("prompt_eval_duration"),
                    "generation_duration": response.get("eval_duration"),
                },
                "done_reason": done_reason,
                "completion_status": "partial" if done_reason == "length" else "complete" if done_reason == "stop" else "unknown",
                "truncated": done_reason == "length",
            },
        }
