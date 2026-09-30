# LLM Quality Evaluation Framework

**A Multi-Dimensional Quality Evaluation Framework for LLM-Based Intelligent Programming Assistants**

The application loads programming tasks, obtains responses from mock, optional OpenAI, or local Ollama providers, evaluates each response, stores evidence in SQLite, creates a model-summary CSV and charts, and presents them in Streamlit.

## Architecture

`Dataset → prompt/provider → response and code extraction → eight evaluators → weighted score → SQLite → CSV/charts → dashboard`

## The eight dimensions

1. **Correctness:** dataset-specific predefined tests and per-test evidence; passed tests / total tests × 100. The `local-test` backend invokes the existing harness through a local Python subprocess with `shell=False`, a temporary working directory, captured output, and a short timeout.
2. **Completeness:** required-feature keyword coverage, adjusted when a provider reports a truncated response.
3. **Security:** Bandit static-analysis findings with severity penalties; this is not runtime sandboxing.
4. **Maintainability:** Radon maintainability index with a labeled heuristic fallback.
5. **Explainability:** deterministic explanation presence/length/term proxy, not semantic judgment.
6. **Usability:** response structure and length proxy, not a user study.
7. **Response Efficiency:** raw wall-clock latency, generated token count, and tokens/second where timing telemetry permits. Mock “token” values may be whitespace word-count proxies.
8. **Robustness / Reliability:** provider success, non-empty response, code structure, completion/truncation, and response parsing checks. This is an operational proxy, not statistical reliability.

The overall score is a weighted mean with weights `0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10`, respectively. It is unavailable if any dimension is unavailable.

### Operational metric formulas

- **Completeness:** `content coverage = detected required features / expected required features × 100`. If the provider reports `done_reason=length`, the final score is `content coverage × 0.75`. The evidence stores expected/detected/missing features, completion status, and truncation.
- **Response efficiency:** within each run and provider/model group, `latency score = 100 × (max latency − observed latency) / (max − min)` and `throughput score = 100 × (observed tokens/second − min) / (max − min)`. The efficiency score is the mean of available components. If a component has no variation, tied observations receive 100 for that component. Ollama throughput uses its generation-duration telemetry; wall-clock time is a labeled fallback. These scores are relative to that comparison group and are not cross-run or industry performance claims.
- **Robustness / Reliability:** `100 × passed applicable checks / applicable checks`. The checks cover provider request success, non-empty output, recognizable code structure, completed generation without truncation when completion metadata exists, and successful output parsing. Unknown completion metadata is not treated as a pass or failure.

These are deterministic operational proxies. They are not scientifically validated percentages, and run-relative efficiency scores should only be interpreted within their stated run/model group.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run an evaluation

Mock provider (no API key required):

```powershell
python main.py --provider mock --limit 2
```

Optional local Ollama (must already be running as the current user):

```powershell
python main.py --provider ollama --limit 2
```

The Ollama request defaults to `qwen3:4b`, `think=false`, temperature `0`, seed `42`, context `4096`, and `num_predict=1024`. The provider uses Ollama's final `response` field and removes inline thinking blocks. `done_reason=length` is retained as partial-completion metadata; it is not treated as a complete response.

To evaluate multiple already-installed Ollama models against the same selected task set in one run, repeat `--ollama-model`:

```powershell
python main.py --provider ollama --ollama-model qwen3:4b --ollama-model qwen3.5:latest --limit 2
```

Only use model names already present in `ollama list`; this command does not download models. Results retain provider, exact model name, run ID, and task ID. The dashboard compares identities only within the selected run. A multi-model experiment has not been performed by adding this option. On the currently installed machine, `qwen3.5:latest` is larger than the reported 4 GiB VRAM; its runtime suitability has not been verified. Cloud-tagged models are not local inference models.

Optional OpenAI provider (requires `OPENAI_API_KEY` in `.env`):

```powershell
python main.py --provider openai --limit 2
```

Show CLI options with `python main.py --help`. Open the dashboard with:

```powershell
streamlit run dashboard/app.py
```

Records are appended to `evaluation.db`; each row has a SQLite evaluation ID and each experiment run has a `run_id`. CSV summaries and charts are written under `results/`.

## Limitations

Correctness uses a controlled local Python subprocess to execute predefined benchmark tests. This avoids external sandbox infrastructure and is intended for controlled academic evaluation. It is not a security sandbox for arbitrary untrusted code. The explicit `safe` backend still fails closed when no isolated executor is available; `host-dev` remains a compatibility alias for the ordinary subprocess mode. Bundled sample outputs are demonstrations, not benchmark findings. Full HumanEval/MBPP results, human evaluation, and statistical analysis are not implied. The heuristic dimensions require validation before research claims.
