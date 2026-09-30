import json
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config.settings import DB_PATH, RESULTS_DIR
DIMENSIONS=["correctness_score","completeness_score","security_score","maintainability_score","explainability_score","usability_score","efficiency_score","robustness_score"]

def _comparison_frame(df):
    """Build explicit identities without presenting mock labels as real models."""
    work=df.copy()
    if "provider" not in work:
        work["provider"]="unknown"
    work["provider"]=work["provider"].fillna("unknown").replace("", "unknown")
    if "model" not in work: work["model"]="unknown"
    if "model_name" not in work: work["model_name"]=None
    identity=work["model_name"].where(work["model_name"].notna(),work["model"])
    identity=identity.fillna("unknown").astype(str)
    ollama=work["provider"].eq("ollama")
    identity.loc[ollama]=identity.loc[ollama].str.replace(r"^ollama/", "", regex=True)
    mock=work["provider"].eq("mock")
    identity.loc[mock]=work.loc[mock,"model"].fillna("mock fixture").astype(str)
    unknown=work["provider"].eq("unknown")
    identity.loc[unknown]=work.loc[unknown,"model"].fillna("legacy record").astype(str)+" (legacy)"
    work["model_identity"]=identity
    work["provider_category"]="real provider"
    work.loc[mock,"provider_category"]="demo / mock"
    work.loc[unknown,"provider_category"]="legacy / unknown"
    labels=work["provider"].astype(str)+" / "+work["model_identity"]
    labels.loc[mock]="DEMO / mock / "+work.loc[mock,"model_identity"]
    labels.loc[unknown]="LEGACY / "+work.loc[unknown,"model_identity"]
    work["comparison_label"]=labels
    return work

def _with_provider_evidence(df):
    work=df.copy()
    def values(raw):
        try:
            metadata=json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata={}
        provider=metadata.get("provider_metadata",{})
        efficiency=metadata.get("efficiency_evidence",{})
        return pd.Series({
            "completion_status":provider.get("completion_status"),
            "done_reason":provider.get("done_reason"),
            "truncated":provider.get("truncated"),
            "generation_time_seconds":efficiency.get("generation_time_seconds"),
            "tokens_per_second":efficiency.get("tokens_per_second"),
            "efficiency_evidence":efficiency,
        })
    if "evaluation_metadata" in work:
        derived=work["evaluation_metadata"].apply(values)
        for column in derived:
            work[column]=derived[column]
    return work

def _scope_labels_to_run(work):
    """Keep otherwise identical model labels separate across experiment runs."""
    scoped=work.copy()
    if "run_id" not in scoped:
        scoped["run_id"]=None
    scoped["run_scope"]=scoped["run_id"].fillna("legacy (no run ID)").astype(str)
    if scoped["run_scope"].nunique()>1:
        scoped["comparison_label"]=scoped["run_scope"]+" / "+scoped["comparison_label"].astype(str)
    return scoped

def build_model_comparison(df):
    """Aggregate one selected run by provider and actual model identity.

    Means ignore unavailable scores; the accompanying available-task counts make
    the denominator visible rather than converting missing values to zero.
    """
    work=_scope_labels_to_run(_with_provider_evidence(_comparison_frame(df)))
    metrics=DIMENSIONS+["overall_score","response_time","output_tokens","generation_time_seconds","tokens_per_second","efficiency_score"]
    for column in metrics:
        if column not in work: work[column]=None
        work[column]=pd.to_numeric(work[column],errors="coerce")
    rows=[]
    for label,group in work.groupby("comparison_label",dropna=False,sort=True):
        first=group.iloc[0]
        row={"comparison_label":label,"provider":first["provider"],"provider_category":first["provider_category"],"model_identity":first["model_identity"],"task_count":len(group)}
        statuses=group.get("completion_status",pd.Series(index=group.index,dtype=object))
        row["complete_responses"]=int(statuses.eq("complete").sum())
        row["partial_responses"]=int((statuses.eq("partial")|statuses.eq("length")).sum())
        row["provider_failures"]=int(group.get("provider_status",pd.Series(index=group.index,dtype=object)).eq("error").sum())
        for column in metrics:
            row[column]=group[column].mean()
            row[f"{column}_available_tasks"]=int(group[column].notna().sum())
        if row["overall_score_available_tasks"] != len(group):
            row["overall_score"]=None
        rows.append(row)
    return pd.DataFrame(rows)

def dimension_comparison_frame(summary):
    """Return all eight dimension rows plus overall score, preserving N/A."""
    scores=DIMENSIONS+["overall_score"]
    if summary.empty:
        return pd.DataFrame(columns=["dimension"])
    indexed=summary.set_index("comparison_label")
    result=indexed[scores].transpose().reset_index(names="dimension")
    names={
        "correctness_score":"Correctness","completeness_score":"Completeness",
        "security_score":"Security","maintainability_score":"Maintainability",
        "explainability_score":"Explainability","usability_score":"Usability",
        "efficiency_score":"Response Efficiency","robustness_score":"Robustness / Reliability",
        "overall_score":"Overall Score",
    }
    result["dimension"]=result["dimension"].map(names)
    return result

def load_results_df():
    con=sqlite3.connect(DB_PATH)
    try: return pd.read_sql_query("SELECT * FROM evaluations",con)
    finally: con.close()
def generate_summary_csv(df):
    RESULTS_DIR.mkdir(exist_ok=True)
    comparison=build_model_comparison(df).set_index("comparison_label")
    columns=[c for c in DIMENSIONS+["overall_score","response_time","output_tokens","total_tokens","generation_time_seconds","tokens_per_second","task_count","complete_responses","partial_responses","provider_failures"] if c in comparison]
    columns += [f"{dimension}_available_tasks" for dimension in DIMENSIONS+["overall_score"] if f"{dimension}_available_tasks" in comparison]
    out=comparison[columns].round(2)
    out["evaluation_count"]=comparison["task_count"]
    path=RESULTS_DIR/"tables"/"model_summary.csv"; path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(path)
    return out
def generate_all_charts(df):
    dest=RESULTS_DIR/"figures"; dest.mkdir(parents=True,exist_ok=True)
    dimensions=[c for c in DIMENSIONS if c in df]
    work=_scope_labels_to_run(_with_provider_evidence(_comparison_frame(df)))
    numeric=dimensions+[c for c in ("overall_score","response_time","total_tokens","output_tokens","tokens_per_second") if c in work]
    for column in numeric: work[column]=pd.to_numeric(work[column],errors="coerce")
    ax=work.groupby("comparison_label")[dimensions].mean().plot(kind="bar",figsize=(12,6))
    ax.set_title("Eight-Dimension Programming Assistant Evaluation"); ax.set_ylabel("Score"); ax.set_ylim(0,100)
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(dest/"dimension_comparison.png",dpi=160); plt.close()
    if "overall_score" in work:
        overall=work.groupby("comparison_label")["overall_score"].mean().dropna().sort_values(ascending=False)
        fig,ax=plt.subplots(figsize=(8,5))
        if overall.empty: ax.text(.5,.5,"Overall score unavailable for these runs",ha="center",va="center",transform=ax.transAxes)
        else: overall.plot(kind="bar",ax=ax)
        ax.set_title("Average Overall Score"); ax.set_ylabel("Score"); ax.set_ylim(0,100)
        plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(dest/"overall_score.png",dpi=160); plt.close(fig)
    if {"response_time","tokens_per_second"} <= set(work.columns):
        data=work.dropna(subset=["response_time","tokens_per_second"]); fig,ax=plt.subplots(figsize=(8,5))
        for model,group in data.groupby("comparison_label"): ax.scatter(group["response_time"],group["tokens_per_second"],label=model)
        if data.empty: ax.text(.5,.5,"Latency / generation throughput telemetry unavailable",ha="center",va="center",transform=ax.transAxes)
        else: ax.legend()
        ax.set_title("Response Latency and Generation Throughput"); ax.set_xlabel("Wall-clock response time (seconds; lower is better)"); ax.set_ylabel("Generated tokens / second (higher is better)")
        fig.tight_layout(); fig.savefig(dest/"efficiency_quality.png",dpi=160); plt.close(fig)
