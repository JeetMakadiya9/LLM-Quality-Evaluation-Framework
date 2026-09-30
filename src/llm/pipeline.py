from uuid import uuid4
from src.config.settings import MODELS, OLLAMA_MODEL
from src.llm.prompt_generator import build_prompt
from src.llm.mock_provider import generate as mock_generate
from src.llm.openai_client import OpenAIProvider
from src.llm.ollama_client import OllamaProvider, OllamaProviderError
from src.evaluation.runner import evaluate_response
from src.evaluation.robustness import evaluate_robustness
from src.evaluation.efficiency import efficiency_evidence, normalize_efficiency
from src.scoring.scoring_engine import calculate_scores, DIMENSION_WEIGHTS

def _provider_failure(problem, provider_name, model_name, run_id, error):
    failure_name = type(error).__name__
    metadata = {"provider": provider_name}
    if isinstance(error, OllamaProviderError):
        # OllamaProviderError messages are deliberately sanitized by the client:
        # they contain endpoint/status or parser category, never response bodies.
        metadata["provider_error_message"] = str(error)
    result = {key: None for key in DIMENSION_WEIGHTS}
    result.update({
        "problem_id": problem.get("problem_id", problem.get("task_id")),
        "task_id": problem.get("task_id", problem.get("problem_id")),
        "dataset": problem.get("dataset"),
        "model": model_name if provider_name == "ollama" else model_name,
        "model_name": model_name, "provider": provider_name,
        "provider_status": "error", "provider_error": failure_name,
        "evaluation_status": "provider_error", "evaluation_error": failure_name,
        "response_text": None, "generated_code": None, "explanation": None,
        "prompt": build_prompt(problem["prompt"], provider=provider_name), "response_time": None,
        "input_tokens": None, "output_tokens": None, "total_tokens": None,
        "overall_score": None, "score_method": "Unavailable because provider returned no response.",
        "run_id": run_id, "metadata": metadata,
    })
    result.update(evaluate_robustness(None, provider_error=failure_name))
    return result

def run_experiment(problems, live=False, provider=None, ollama_models=None):
    results=[]; providers={}; run_id=str(uuid4())
    selected=provider or ("openai" if live else "mock")
    if selected not in {"openai","mock","ollama"}:
        raise ValueError("provider must be one of: openai, mock, ollama")
    if selected=="openai":
        for label,model in MODELS.items(): providers[label]=OpenAIProvider(model)
        model_names=MODELS
    elif selected=="ollama":
        # An explicit list permits a fair, same-run comparison. Existing callers
        # retain the configured single-model behavior when the option is omitted.
        if ollama_models is None:
            requested=[None]
        else:
            requested=list(dict.fromkeys(name.strip() for name in ollama_models if isinstance(name,str) and name.strip()))
            if not requested:
                raise ValueError("ollama_models must contain at least one model name")
        model_names={}
        provider_errors={}
        for requested_model in requested:
            fallback_name=requested_model or OLLAMA_MODEL
            try:
                local=OllamaProvider() if requested_model is None else OllamaProvider(model=requested_model)
                model_name=local.model
                providers[model_name]=local
                model_names[model_name]=model_name
            except Exception as exc:
                model_names[fallback_name]=fallback_name
                provider_errors[fallback_name]=exc
    else:
        model_names=MODELS

    # Keep requests sequential by model so small-memory local runtimes do not
    # repeatedly switch models between adjacent task prompts.
    for label,model_name in model_names.items():
        for problem in problems:
            prompt=build_prompt(problem["prompt"], provider=selected)
            if selected=="ollama" and model_name in provider_errors:
                results.append(_provider_failure(problem,selected,model_name,run_id,provider_errors[model_name]))
                continue
            try:
                raw=mock_generate(prompt,model_name) if selected=="mock" else providers[label].generate(prompt)
                if not isinstance(raw,dict) or not isinstance(raw.get("text"),str):
                    raise ValueError("provider returned no text response")
            except Exception as exc:
                results.append(_provider_failure(problem,selected,model_name,run_id,exc))
                continue
            provider_metadata = raw.get("provider_metadata") or {}
            evaluation=evaluate_response(problem,raw["text"],provider_metadata)
            evaluation.update({"model":label,"model_name":model_name,"provider":selected,
                "provider_status":"completed","evaluation_status":"completed",
                "problem_id":problem.get("problem_id",problem.get("task_id")),
                "task_id":problem.get("task_id",problem.get("problem_id")),
                "dataset":problem.get("dataset"),
                "entry_point":problem.get("entry_point"),
                "prompt":prompt,"response_text":raw["text"],
                "response_time":raw.get("response_time"),"input_tokens":raw.get("input_tokens"),
                "output_tokens":raw.get("output_tokens"),"total_tokens":raw.get("total_tokens"),
                "run_id":run_id})
            timings = provider_metadata.get("timing_ns") or {}
            evaluation["efficiency_evidence"] = efficiency_evidence(
                raw.get("response_time"), raw.get("output_tokens"), timings.get("generation_duration")
            )
            metadata=dict(evaluation.get("metadata") or {}); metadata["provider"]=selected
            if provider_metadata: metadata["provider_metadata"]=provider_metadata
            metadata["completeness_evidence"] = evaluation.get("completeness_evidence")
            metadata["efficiency_evidence"] = evaluation["efficiency_evidence"]
            metadata["robustness_evidence"] = evaluation.get("robustness_evidence")
            evaluation["metadata"]=metadata
            results.append(evaluation)

    groups={}
    for row in results:
        if row.get("provider_status") != "completed":
            continue
        key=(row.get("run_id"),row.get("provider"),row.get("model"))
        groups.setdefault(key,[]).append(row)
    for group in groups.values():
        normalize_efficiency(group)
        for row in group:
            row["metadata"]["efficiency_evidence"] = row["efficiency_evidence"]
            row.update(calculate_scores(row))
    return results
