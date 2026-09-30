import re

def extract_code(text: str) -> str:
    matches = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[0].strip()

    # Fallback: use the whole response.
    return text.strip()

def extract_explanation(text: str) -> str:
    matches = re.findall(r"```(?:python|py)?\s*.*?```(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    return matches[0].strip() if matches else ""
