def detect_error_handling(code: str) -> bool:
    """
    Detect whether generated code contains basic error handling.
    """
    if not code:
        return False

    error_handling_patterns = [
        "try:",
        "except",
        "finally:",
        "raise ",
        "assert ",
    ]

    return any(pattern in code for pattern in error_handling_patterns)


def calculate_usability_score(
    code: str,
    explanation: str,
    has_error_handling: bool
) -> float:
    """
    Calculate usability score from generated code,
    explanation quality, and error handling.
    """

    score = 50.0

    # Code exists
    if code and code.strip():
        score += 15

    # Explanation exists
    if explanation and explanation.strip():
        score += 15

    # Error handling exists
    if has_error_handling:
        score += 10

    # Reasonable code length
    if code and len(code.splitlines()) <= 100:
        score += 10

    return min(100.0, score)


def default_usability_score(response_text: str):
    """
    Development baseline usability score.
    Replace/augment this with actual questionnaire data
    for the final research experiment.
    """

    words = len(response_text.split())
    score = 60.0

    if "```" in response_text:
        score += 15

    if words >= 20:
        score += 15

    if words <= 400:
        score += 10

    return min(100.0, score)