"""Raw response telemetry and run-relative efficiency normalization."""


def efficiency_evidence(response_time, generated_tokens, generation_duration_ns=None):
    """Build raw evidence; prefer Ollama generation duration for throughput."""
    time_value = float(response_time) if response_time is not None and float(response_time) > 0 else None
    token_value = int(generated_tokens) if generated_tokens is not None else None
    duration_seconds = None
    basis = "unavailable"
    if isinstance(generation_duration_ns, (int, float)) and generation_duration_ns > 0:
        duration_seconds = float(generation_duration_ns) / 1_000_000_000
        basis = "provider_generation_duration"
    elif time_value is not None and time_value > 0:
        duration_seconds = time_value
        basis = "wall_clock_response_time_fallback"

    tokens_per_second = None
    if token_value is not None and token_value >= 0 and duration_seconds is not None and duration_seconds > 0:
        tokens_per_second = round(token_value / duration_seconds, 4)
    return {
        "response_time_seconds": time_value,
        "generated_tokens": token_value,
        "generation_time_seconds": round(duration_seconds, 6) if basis == "provider_generation_duration" else None,
        "tokens_per_second": tokens_per_second,
        "throughput_basis": basis,
        "generation_duration_ns": generation_duration_ns,
    }


def _relative_score(value, values, higher_is_better):
    low, high = min(values), max(values)
    if high == low:
        return 100.0  # all observations are tied for best within this comparison group
    fraction = (value - low) / (high - low)
    return round(100.0 * (fraction if higher_is_better else 1.0 - fraction), 2)


def normalize_efficiency(records):
    """Attach 0-100 run/model-relative scores to mutable result records.

    Latency contributes a lower-is-better min-max component; token throughput
    contributes a higher-is-better component. Available components are averaged.
    Equal values receive 100 for that component because all are tied for best.
    """
    evidences = [row.get("efficiency_evidence") or {} for row in records]
    times = [e["response_time_seconds"] for e in evidences if isinstance(e.get("response_time_seconds"), (int, float))]
    rates = [e["tokens_per_second"] for e in evidences if isinstance(e.get("tokens_per_second"), (int, float))]
    for row, evidence in zip(records, evidences):
        components = {}
        time_value = evidence.get("response_time_seconds")
        rate = evidence.get("tokens_per_second")
        if isinstance(time_value, (int, float)) and times:
            components["latency_score"] = _relative_score(time_value, times, higher_is_better=False)
        if isinstance(rate, (int, float)) and rates:
            components["throughput_score"] = _relative_score(rate, rates, higher_is_better=True)
        score = round(sum(components.values()) / len(components), 2) if components else None
        evidence.update({
            "normalized_components": components,
            "normalization_scope": "same run and provider/model group",
            "normalization_formula": "mean of available min-max components: lower latency is better; higher tokens/second is better",
            "efficiency_score": score,
        })
        row["efficiency_evidence"] = evidence
        row["efficiency_score"] = score
    return records