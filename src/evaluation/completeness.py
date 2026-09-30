"""Feature-based completeness with an explicit completion/truncation adjustment."""

import json

_TRUNCATION_FACTOR = 0.75


def _load_features(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("required features must be a JSON list")
    features = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("each required feature must have a name")
        keywords = item.get("keywords", [])
        if not isinstance(keywords, list) or not all(isinstance(word, str) for word in keywords):
            raise ValueError("feature keywords must be a list of strings")
        features.append({"name": item["name"], "keywords": keywords})
    return features


def completeness_score(code: str, required_features_json: str, completion_metadata=None):
    """Return an operational proxy score and evidence for feature coverage.

    Truncation applies a 0.75 multiplier to content coverage; it does not erase
    the feature evidence or force every partial answer to zero.
    """
    metadata = completion_metadata if isinstance(completion_metadata, dict) else {}
    done_reason = metadata.get("done_reason")
    completion_status = metadata.get("completion_status")
    truncated = bool(metadata.get("truncated")) or done_reason == "length" or completion_status == "partial"
    if not completion_status:
        completion_status = "partial" if truncated else "complete" if done_reason == "stop" else "unknown"

    try:
        features = _load_features(required_features_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        features = []
        spec_error = f"{type(exc).__name__}: {exc}"
    else:
        spec_error = None

    text = code if isinstance(code, str) else ""
    lower = text.lower()
    detected = []
    missing = []
    for feature in features:
        found = bool(feature["keywords"]) and any(word.lower() in lower for word in feature["keywords"])
        (detected if found else missing).append(feature["name"])

    if not text.strip() or not features or spec_error:
        score = None
        content_score = None
        adjustment = None
        status = "unavailable"
    else:
        content_score = 100.0 * len(detected) / len(features)
        adjustment = _TRUNCATION_FACTOR if truncated else 1.0
        score = round(content_score * adjustment, 2)
        status = "evaluated"

    evidence = {
        "status": status,
        "expected_features": features,
        "detected_features": detected,
        "missing_features": missing,
        "expected_feature_count": len(features),
        "detected_feature_count": len(detected),
        "content_completeness": round(content_score, 2) if content_score is not None else None,
        "completion_status": completion_status,
        "done_reason": done_reason,
        "truncated": truncated,
        "truncation_multiplier": adjustment,
        "formula": "content coverage = detected / expected * 100; truncated response score = content coverage * 0.75",
        "score": score,
        "specification_error": spec_error,
    }
    return {
        "completeness_score": score,
        "missing_features": ", ".join(missing) if features else ("No rubric supplied" if not spec_error else spec_error),
        "completeness_evidence": evidence,
    }