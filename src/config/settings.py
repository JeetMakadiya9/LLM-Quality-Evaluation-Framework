from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
DB_PATH = ROOT_DIR / "evaluation.db"

load_dotenv(ROOT_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LIVE_MODE = os.getenv("LIVE_MODE", "0") == "1"

# local-test uses a normal subprocess with predefined project tests. It is not a sandbox.
CODE_EXECUTION_BACKEND = os.getenv("CODE_EXECUTION_BACKEND", "local-test").strip().lower()
CODE_EXECUTION_TIMEOUT_SECONDS = float(os.getenv("CODE_EXECUTION_TIMEOUT_SECONDS", "5"))

# Local Ollama provider settings. The provider uses Ollama's HTTP API directly,
# so no additional Python client dependency is required.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_SEED = int(os.getenv("OLLAMA_SEED", "42"))
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").strip().lower() == "true"
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))

MODELS = {
    "Model_A": os.getenv("MODEL_A", "gpt-5.6-luna"),
    "Model_B": os.getenv("MODEL_B", "gpt-5.6-terra"),
    "Model_C": os.getenv("MODEL_C", "gpt-5.6-sol"),
}

PROMPT_TEMPLATE = """You are an expert Python programmer.

Solve the programming problem below.

Requirements:
1. Return a complete Python solution.
2. Explain the approach after the code.
3. Handle reasonable edge cases.
4. Do not use external packages unless explicitly allowed.

Programming problem:
{prompt}
"""
