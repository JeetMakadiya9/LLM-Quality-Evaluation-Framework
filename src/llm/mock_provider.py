import time

RESPONSES = {
    "prime": """```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```

The function rejects values below 2 and checks divisors up to the square root.""",
    "palindrome": """```python
def is_palindrome(s):
    return s == s[::-1]
```

The function compares the string with its reversed version.""",
    "factorial": """```python
def factorial(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

The function validates the input and computes the factorial iteratively.""",
    "maximum": """```python
def find_max(numbers):
    if not numbers:
        raise ValueError("numbers cannot be empty")
    return max(numbers)
```

The function rejects an empty list and returns the maximum value.""",
    "duplicate": """```python
def has_duplicates(items):
    return len(items) != len(set(items))
```

The function compares the list length with the number of unique values.""",
}

def _choose_response(prompt):
    lower = prompt.lower()
    for key, value in RESPONSES.items():
        if key in lower:
            return value
    return """```python
def solve(value):
    return value
```

This baseline implementation returns the supplied value."""

def generate(prompt, model_name):
    start = time.perf_counter()
    text = _choose_response(prompt)
    elapsed = time.perf_counter() - start
    input_words = len(prompt.split())
    output_words = len(text.split())
    return {
        "text": text,
        "response_time": round(elapsed, 6),
        "input_tokens": input_words,
        "output_tokens": output_words,
        "total_tokens": input_words + output_words,
        "provider_metadata": {
            "provider": "mock", "model": model_name,
            "token_count_method": "whitespace word-count proxy",
            "response_time_unit": "seconds",
        },
    }
