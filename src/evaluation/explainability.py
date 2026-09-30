import re

def explainability_score(explanation: str, code: str):
    if not explanation.strip():
        return 0.0
    score = 50.0
    words = len(explanation.split())
    if words >= 15:
        score += 20
    if any(term in explanation.lower() for term in ["function", "input", "return", "loop", "condition"]):
        score += 20
    if "```" not in explanation and len(code) > 0:
        score += 10
    return min(100.0, score)
