from src.config.settings import PROMPT_TEMPLATE

OLLAMA_PROMPT_TEMPLATE = """You are an expert Python programmer.
Solve this programming problem.
Return ONLY the complete Python code needed to solve it.
Do not explain your reasoning.
Do not include tags.
Programming problem:
{prompt}
"""

def build_prompt(problem_text: str, provider: str | None = None) -> str:
    """Build a provider-appropriate prompt without changing the task text."""
    template = OLLAMA_PROMPT_TEMPLATE if provider == "ollama" else PROMPT_TEMPLATE
    return template.format(prompt=problem_text)