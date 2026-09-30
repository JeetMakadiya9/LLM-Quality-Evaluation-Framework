from src.evaluation.usability import calculate_usability_score, detect_error_handling
DIMENSION_WEIGHTS={"correctness_score":.20,"completeness_score":.15,"security_score":.15,"maintainability_score":.10,"explainability_score":.10,"usability_score":.10,"efficiency_score":.10,"robustness_score":.10}
def calculate_scores(row):
    if row.get("usability_score") is None and row.get("evaluation_status")!="provider_error":
        row["usability_score"]=calculate_usability_score(row.get("generated_code", ""),row.get("explanation", ""),detect_error_handling(row.get("generated_code", "")))
    values=[row.get(k) for k in DIMENSION_WEIGHTS]
    overall=round(sum(float(row[k])*w for k,w in DIMENSION_WEIGHTS.items()),2) if all(v is not None for v in values) else None
    result={k:row.get(k) for k in DIMENSION_WEIGHTS}
    result.update(overall_score=overall,score_method="Weighted mean across eight available dimensions; missing dimensions yield no overall score.")
    return result
