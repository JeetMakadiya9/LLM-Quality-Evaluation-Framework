from src.evaluation.extractor import extract_code, extract_explanation
from src.evaluation.correctness import run_tests
from src.evaluation.security import security_score
from src.evaluation.maintainability import maintainability_score
from src.evaluation.completeness import completeness_score
from src.evaluation.explainability import explainability_score
from src.evaluation.usability import default_usability_score
from src.evaluation.robustness import evaluate_robustness

def evaluate_response(problem, response_text, provider_metadata=None):
    text = response_text or ""
    provider_metadata = provider_metadata if isinstance(provider_metadata, dict) else {}
    code = extract_code(text)
    explanation = extract_explanation(text)
    result = {"generated_code": code, "explanation": explanation}
    result.update(run_tests(code, problem.get("tests"), entry_point=problem.get("entry_point"), prompt=problem.get("prompt")))
    result.update(security_score(code))
    result.update(maintainability_score(code))
    result.update(completeness_score(code, problem.get("required_features", "[]"), provider_metadata))
    result["explainability_score"] = explainability_score(explanation, code)
    result["usability_score"] = default_usability_score(text)
    result.update(evaluate_robustness(text, completion_metadata=provider_metadata))
    result.update(task_id=problem.get("task_id", problem.get("problem_id")), dataset=problem.get("dataset"), entry_point=problem.get("entry_point"), metadata=problem.get("metadata", {}))
    return result