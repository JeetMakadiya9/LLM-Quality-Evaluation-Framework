"""Lightweight operational response-reliability proxy, not statistical reliability."""

import re

from src.evaluation.extractor import extract_code

_CODE_MARKERS = re.compile(r"(?m)^\s*(?:async\s+def|def|class)\s+|^\s*return\b|^\s*import\s+|^\s*from\s+\S+\s+import\s+")


def evaluate_robustness(response_text, provider_error=None, evaluator_error=None,
                        completion_metadata=None, parse_error=None):
    metadata = completion_metadata if isinstance(completion_metadata, dict) else {}
    text = response_text if isinstance(response_text, str) else ""
    non_empty = bool(text.strip())
    code = extract_code(text) if non_empty else ""
    fence_count = text.count("```")
    balanced_fences = fence_count % 2 == 0
    code_detected = bool(code.strip()) and bool(_CODE_MARKERS.search(code))
    residual_thinking = "<think>" in text.lower() or "</think>" in text.lower()
    structure_valid = code_detected and balanced_fences

    done_reason = metadata.get("done_reason")
    completion_status = metadata.get("completion_status")
    truncated = bool(metadata.get("truncated")) or done_reason == "length" or completion_status == "partial"
    completion_known = (
        done_reason in {"stop", "length"}
        or completion_status in {"complete", "partial"}
    )
    if completion_known:
        generation_complete = not truncated and (
            done_reason == "stop" or completion_status == "complete"
        )
    else:
        generation_complete = None

    checks = {
        "provider_request_succeeded": provider_error is None,
        "response_non_empty": non_empty,
        "response_structure_valid": structure_valid,
        "generation_completed_without_truncation": generation_complete,
        "output_parsing_succeeded": non_empty and parse_error is None and not residual_thinking and evaluator_error is None,
    }
    applicable = [value for value in checks.values() if value is not None]
    score = round(100.0 * sum(value is True for value in applicable) / len(applicable), 2) if applicable else None
    status = "provider_error" if provider_error is not None else "evaluator_error" if evaluator_error is not None else "partial" if truncated else "evaluated"
    evidence = {
        "done_reason": done_reason,
        "completion_status": completion_status or "unknown",
        "truncated": truncated,
        "checks_passed": sum(value is True for value in applicable),
        "checks_applicable": len(applicable),
        "formula": "100 * passed applicable checks / applicable checks",
        "scope": "operational response proxy; not statistical reliability",
    }
    return {
        "robustness_score": score,
        "robustness_status": status,
        "robustness_checks": checks,
        "robustness_evidence": evidence,
    }