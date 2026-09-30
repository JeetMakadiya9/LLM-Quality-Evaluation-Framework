import time
import os
from dotenv import load_dotenv
from openai import OpenAI # type: ignore
load_dotenv()
from src.config.settings import OPENAI_API_KEY
class OpenAIProvider:
    def __init__(self, model: str):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def generate(self, prompt: str):
        start = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        elapsed = time.perf_counter() - start
        text = getattr(response, "output_text", "") or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        return {
            "text": text,
            "response_time": round(elapsed, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
