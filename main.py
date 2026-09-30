import argparse
from src.config.settings import DATA_DIR
from src.dataset.loader import load_problems
from src.llm.pipeline import run_experiment
from src.llm.ollama_client import OllamaProviderError
from src.database.database import init_db, save_results
from src.analysis.analysis import load_results_df, generate_summary_csv, generate_all_charts

def main():
    parser=argparse.ArgumentParser(description="Evaluate programming-assistant responses across eight dimensions.")
    parser.add_argument("--sample",action="store_true",help="Use the deterministic mock provider.")
    parser.add_argument("--live",action="store_true",help="Use the optional configured OpenAI provider.")
    parser.add_argument("--provider",choices=["mock","openai","ollama"],default=None)
    parser.add_argument("--ollama-model",action="append",default=None,metavar="NAME",
        help="Ollama model to include (repeat for a same-run comparison; defaults to OLLAMA_MODEL).")
    parser.add_argument("--dataset",choices=["sample","humaneval","mbpp","programming"],default="sample")
    parser.add_argument("--limit",type=int,default=None)
    args=parser.parse_args(); provider=args.provider or ("openai" if args.live else "mock")
    if args.ollama_model and provider!="ollama":
        parser.error("--ollama-model can only be used with --provider ollama")
    paths={"sample":DATA_DIR/"sample"/"problems.csv","humaneval":DATA_DIR/"humaneval_dataset.csv","mbpp":DATA_DIR/"mbpp_dataset.csv","programming":DATA_DIR/"programming_dataset.csv"}
    tasks=load_problems(paths[args.dataset],args.dataset)
    if args.limit is not None: tasks=tasks[:max(0,args.limit)]
    try: results=run_experiment(tasks,live=args.live,provider=provider,ollama_models=args.ollama_model)
    except OllamaProviderError as exc:
        print(f"Ollama unavailable: {exc} Start Ollama as a normal user or use --provider mock.")
        return 2
    if provider=="ollama" and any(row.get("provider_status")=="error" for row in results):
        print("Ollama unavailable. Failure records will be saved without scores for unevaluated dimensions.")
    init_db(); save_results(results); df=load_results_df(); generate_summary_csv(df); generate_all_charts(df)
    print(f"Completed {len(results)} evaluation records with provider '{provider}'.")
    print("Historical SQLite rows are preserved; each invocation has a run_id.")
    print("Run: streamlit run dashboard/app.py")
    return 0
if __name__=="__main__": raise SystemExit(main())
