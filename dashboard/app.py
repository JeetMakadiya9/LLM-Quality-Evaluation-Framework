"""Streamlit dashboard for real, demo, and historical evaluation records."""

import html
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.analysis import (  # noqa: E402
    DIMENSIONS,
    _comparison_frame,
    _with_provider_evidence,
    build_model_comparison,
    dimension_comparison_frame,
)
from src.config import settings  # noqa: E402


DIMENSION_LABELS = {
    "correctness_score": "Correctness",
    "completeness_score": "Completeness",
    "security_score": "Security",
    "maintainability_score": "Maintainability",
    "explainability_score": "Explainability",
    "usability_score": "Usability",
    "efficiency_score": "Response Efficiency",
    "robustness_score": "Robustness / Reliability",
    "overall_score": "Overall Score",
}
NAVIGATION = [
    "Overview",
    "Model Comparison",
    "8-Dimension Analysis",
    "Response Efficiency",
    "Task Results",
    "Evidence & Limitations",
    "Reports",
    "Settings",
]
COLORS = {
    "blue": "#4477f5",
    "purple": "#8b5cf6",
    "green": "#17a673",
    "orange": "#ef9b37",
    "red": "#e45b68",
    "navy": "#101a2d",
    "muted": "#718096",
}


def _load_results(path):
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query("SELECT * FROM evaluations", connection)


def _parse_json(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _display_score(value, digits=2):
    return "N/A" if pd.isna(value) else f"{float(value):.{digits}f}"


def _prepare_data(frame):
    data = frame.copy()
    numeric = DIMENSIONS + ["overall_score", "response_time", "input_tokens", "output_tokens", "total_tokens"]
    for column in numeric:
        if column not in data:
            data[column] = None
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = _with_provider_evidence(_comparison_frame(data))
    if "run_id" not in data:
        data["run_id"] = None
    if "id" not in data:
        data["id"] = range(len(data))
    data["run_group"] = data["run_id"].fillna("legacy (no run ID)").astype(str)
    for column in ("provider_status", "evaluation_status", "provider_error", "evaluation_error", "execution_status", "correctness_status", "executor_backend"):
        if column not in data:
            data[column] = None
    if "evaluation_metadata" not in data:
        data["evaluation_metadata"] = None
    if "task_id" not in data:
        data["task_id"] = data.get("problem_id")
    elif "problem_id" in data:
        data["task_id"] = data["task_id"].where(data["task_id"].notna(), data["problem_id"])
    data["response_status"] = data.apply(_response_status, axis=1)
    return data


def _response_status(row):
    if row.get("provider_status") == "error" or row.get("evaluation_status") == "provider_error":
        return "Failed"
    if row.get("truncated") is True or row.get("done_reason") == "length" or row.get("completion_status") == "partial":
        return "Partial"
    if row.get("done_reason") == "stop" or row.get("completion_status") == "complete":
        return "Complete"
    return "Unknown"


def _overall_unavailable_reason(rows):
    missing = [column for column in DIMENSIONS if rows[column].isna().any()]
    if not missing:
        return None
    labels = ", ".join(DIMENSION_LABELS[column] for column in missing)
    if rows["correctness_score"].isna().any():
        other_missing = [DIMENSION_LABELS[column] for column in missing if column != "correctness_score"]
        detail = " Other unavailable dimensions: " + ", ".join(other_missing) + "." if other_missing else ""
        return "Overall Score unavailable because Correctness could not be evaluated for this run; all eight dimensions are required." + detail
    return "Overall Score unavailable because required dimensions are missing for at least one selected result: " + labels + "."


def _run_kind(rows):
    providers = set(rows["provider"].fillna("unknown").astype(str))
    if providers == {"mock"}:
        return "DEMO / MOCK"
    if "unknown" in providers:
        return "LEGACY / UNKNOWN"
    return "REAL EXPERIMENT"


def _run_option_label(run_id, frame):
    rows = frame[frame["run_group"] == run_id]
    return f"{_run_kind(rows)}  ·  {run_id}"


def _format_table_scores(frame, columns):
    view = frame.copy()
    for column in columns:
        if column in view:
            view[column] = view[column].map(_display_score)
    return view


def _kpi(label, value, caption, accent="blue", icon="•"):
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_caption = html.escape(str(caption))
    st.markdown(
        f'<div class="kpi-card accent-{accent}">'
        f'<div class="kpi-head"><div class="kpi-label">{safe_label}</div>'
        f'<span class="kpi-icon">{html.escape(str(icon))}</span></div>'
        f'<div class="kpi-value">{safe_value}</div>'
        f'<div class="kpi-caption">{safe_caption}</div></div>',
        unsafe_allow_html=True,
    )


def _empty_state(title, detail):
    st.markdown(
        '<div class="empty-state"><div class="empty-icon">—</div>'
        f'<div class="empty-title">{html.escape(title)}</div>'
        f'<div class="empty-copy">{html.escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def _comparison_table(rows):
    summary = build_model_comparison(rows)
    if summary.empty:
        return summary, pd.DataFrame(columns=["Dimension"])
    wide = dimension_comparison_frame(summary)
    summary_by_name = summary.set_index("comparison_label")
    for model in summary["comparison_label"]:
        model_summary = summary_by_name.loc[model]
        for index, label in enumerate(wide["dimension"]):
            source = next((key for key, value in DIMENSION_LABELS.items() if value == label), None)
            score = wide.loc[index, model]
            count = int(model_summary.get(f"{source}_available_tasks", 0)) if source else 0
            total = int(model_summary["task_count"])
            if pd.isna(score):
                wide.loc[index, model] = "N/A"
            elif count < total:
                wide.loc[index, model] = f"{float(score):.2f}  ·  {count}/{total}"
            else:
                wide.loc[index, model] = f"{float(score):.2f}"
    return summary, wide.rename(columns={"dimension": "Dimension"})


def _render_comparison_table(frame, table_class=""):
    """Render the compact comparison as a light, readable HTML table."""
    if frame.empty:
        st.info("No comparison data is available for this selection.")
        return
    columns = list(frame.columns)
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f'<div class="comparison-wrap {html.escape(table_class)}"><table class="comparison-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _task_table_frame(rows):
    columns = [
        ("task_id", "Task ID"),
        ("model_identity", "Model"),
        ("response_status", "Status"),
        ("correctness_score", DIMENSION_LABELS["correctness_score"]),
        ("completeness_score", DIMENSION_LABELS["completeness_score"]),
        ("security_score", DIMENSION_LABELS["security_score"]),
        ("maintainability_score", DIMENSION_LABELS["maintainability_score"]),
        ("explainability_score", DIMENSION_LABELS["explainability_score"]),
        ("usability_score", DIMENSION_LABELS["usability_score"]),
        ("efficiency_score", DIMENSION_LABELS["efficiency_score"]),
        ("robustness_score", DIMENSION_LABELS["robustness_score"]),
        ("overall_score", DIMENSION_LABELS["overall_score"]),
        ("response_time", "Latency (s)"),
        ("output_tokens", "Tokens"),
        ("tokens_per_second", "Tokens/s"),
    ]
    table = rows[[key for key, _ in columns if key in rows]].copy()
    table = table.rename(columns={key: label for key, label in columns})
    score_columns = [label for key, label in columns if key.endswith("_score")]
    table = _format_table_scores(table, score_columns)
    for column, digits in (("Latency (s)", 2), ("Tokens/s", 2)):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce").map(
                lambda value: "N/A" if pd.isna(value) else f"{float(value):.{digits}f}"
            )
    if "Tokens" in table:
        table["Tokens"] = pd.to_numeric(table["Tokens"], errors="coerce").map(
            lambda value: "N/A" if pd.isna(value) else str(int(value))
        )
    return table


def _task_ids_caption(values):
    task_ids = sorted({str(value) for value in values if pd.notna(value)})
    if not task_ids:
        return "no task IDs"
    if len(task_ids) == 1:
        return task_ids[0]
    parts = [re.fullmatch(r"([^\d]*)(\d+)", task_id) for task_id in task_ids]
    if all(parts) and len({part.group(1) for part in parts}) == 1 and len({len(part.group(2)) for part in parts}) == 1:
        numbers = [int(part.group(2)) for part in parts]
        if all(right == left + 1 for left, right in zip(numbers, numbers[1:])):
            return f"{task_ids[0]}–{task_ids[-1]}"
    return ", ".join(task_ids)


def _style_efficiency_plot(figure):
    """Apply high-contrast labels to the dedicated efficiency charts."""
    figure.update_layout(
        template="plotly_white",
        font=dict(size=12, color="#17233b"),
        legend=dict(font=dict(size=11, color="#17233b")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=65, r=20, t=28, b=55),
    )
    figure.update_xaxes(
        showgrid=False,
        tickfont=dict(size=11, color="#17233b"),
        title_font=dict(size=12, color="#17233b"),
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#d8e0ec",
        tickfont=dict(size=11, color="#17233b"),
        title_font=dict(size=12, color="#17233b"),
    )
    figure.update_traces(
        textfont=dict(size=11, color="#17233b"),
        textposition="outside",
        cliponaxis=False,
    )
    return figure


def _efficiency_figure(rows):
    metrics = [
        ("Latency", "response_time", "seconds"),
        ("Throughput", "tokens_per_second", "tokens/s"),
        ("Run-relative efficiency", "efficiency_score", "score / 100"),
    ]
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=.13,
        subplot_titles=[label for label, _, _ in metrics],
    )
    task_ids = rows["task_id"].fillna("Unknown").astype(str)
    model_column = next((column for column in ("model_identity", "model_name", "model") if column in rows), None)
    model_ids = rows[model_column].fillna("Unknown model").astype(str) if model_column else pd.Series("Model", index=rows.index)
    model_names = sorted(model_ids.unique())
    palette = [COLORS["blue"], COLORS["purple"], COLORS["green"], COLORS["orange"], COLORS["red"]]
    model_colors = {name: palette[index % len(palette)] for index, name in enumerate(model_names)}

    for row_index, (label, source, unit) in enumerate(metrics, start=1):
        for model_name in model_names:
            selected = model_ids.eq(model_name)
            values = pd.to_numeric(rows.loc[selected, source], errors="coerce")
            valid = values.notna()
            if not valid.any():
                continue
            task_values = task_ids.loc[selected][valid]
            shown_values = values[valid]
            figure.add_trace(
                go.Bar(
                    x=task_values,
                    y=shown_values,
                    name=model_name,
                    legendgroup=model_name,
                    offsetgroup=model_name,
                    marker_color=model_colors[model_name],
                    showlegend=row_index == 1,
                    text=[f"{value:.1f}" for value in shown_values],
                    textposition="outside",
                    textfont=dict(size=10, color="#17233b"),
                    cliponaxis=False,
                    hovertemplate=f"{model_name}<br>%{{x}}: %{{y:.2f}} {unit}<extra></extra>",
                ),
                row=row_index,
                col=1,
            )
    figure.update_layout(
        template="plotly_white", height=360,
        margin=dict(l=58, r=12, t=66, b=34), bargap=.28,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#17233b"),
        barmode="group",
        legend=dict(
            orientation="h", x=0, y=1.19, xanchor="left", yanchor="bottom",
            font=dict(size=10, color="#17233b"),
        ),
    )
    figure.update_annotations(font=dict(size=11, color="#17233b"))
    figure.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#17233b"))
    figure.update_xaxes(title_text="Task ID", title_font=dict(size=10, color="#17233b"), row=3, col=1)
    figure.update_yaxes(
        showgrid=True, gridcolor="#d8e0ec", zeroline=False,
        tickfont=dict(size=10, color="#17233b"),
        title_font=dict(size=10, color="#17233b"),
    )
    for row_index, (_, _, unit) in enumerate(metrics, start=1):
        figure.update_yaxes(title_text=unit, row=row_index, col=1)
    return figure


def _style_page():
    st.markdown(
        """
        <style>
        :root { --navy:#101a2d; --blue:#4477f5; --purple:#8b5cf6; --green:#17a673; --orange:#ef9b37; --red:#e45b68; }
        .stApp { background:#f4f6fb; color:#17233b; }
        [data-testid="stHeader"], [data-testid="stToolbar"] { display:none; }
        [data-testid="stMainBlockContainer"] { width:100%; max-width:1800px; padding:.35rem 1.2rem .8rem !important; }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap:.42rem; }
        section[data-testid="stSidebar"] { width:245px !important; min-width:245px !important; background:linear-gradient(180deg,#101a2d 0%,#17233b 100%); border-right:1px solid #23324d; }
        section[data-testid="stSidebar"] > div { width:245px !important; }
        section[data-testid="stSidebar"] * { color:#eaf0fb; }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.45rem .62rem; border-radius:9px; margin:.08rem 0; position:relative; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div > label > div > div:first-child { display:none; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] [data-testid="stMarkdownContainer"] p { display:flex; align-items:center; gap:8px; margin:0; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(1) [data-testid="stMarkdownContainer"] p:before { content:"⌂"; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(2) [data-testid="stMarkdownContainer"] p:before { content:"▤"; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(3) [data-testid="stMarkdownContainer"] p:before { content:"◉"; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(4) [data-testid="stMarkdownContainer"] p:before { content:"◷"; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(5) [data-testid="stMarkdownContainer"] p:before { content:"☷"; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(6) [data-testid="stMarkdownContainer"] p:before { content:"ⓘ"; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(7) [data-testid="stMarkdownContainer"] p:before { content:"▣"; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(8) [data-testid="stMarkdownContainer"] p:before { content:"⚙"; }
        section[data-testid="stSidebar"] [data-testid="stRadioGroup"] [data-testid="stMarkdownContainer"] p:before { flex:0 0 16px; display:inline-flex; justify-content:center; color:#a9c8f3; font-size:.88rem; font-weight:700; }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#22314d; }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:#174d9e; box-shadow:inset 3px 0 #68a4ff; }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * { color:#fff; }
        section[data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color:#62a0ff; }
        .brand { display:flex; gap:10px; align-items:center; padding:.55rem .15rem .9rem; border-bottom:1px solid #2b3a56; margin-bottom:.7rem; }
        .brand-icon { display:flex; width:36px; height:36px; border-radius:11px; align-items:center; justify-content:center; background:linear-gradient(135deg,#1e7cf4,#49b7ee); color:#fff; font-size:1.18rem; font-weight:700; }
        .brand-name { font-size:1.02rem; font-weight:750; line-height:1.17; color:#fff; }
        .brand-sub { font-size:.72rem; color:#aebbd1; margin-top:3px; }
        .nav-caption { font-size:.68rem; letter-spacing:.12em; color:#8fa1bc; text-transform:uppercase; margin:.5rem 0 .45rem; }
        .sidebar-run { border:1px solid #33435f; border-radius:11px; background:rgba(255,255,255,.055); padding:.7rem; margin-top:1rem; }
        .sidebar-run-title { color:#fff; font-weight:700; font-size:.86rem; margin-bottom:.55rem; }
        .sidebar-run-line { color:#bac6d9; font-size:.72rem; line-height:1.55; overflow-wrap:anywhere; }
        .sidebar-run-line b { color:#eef3fc; font-weight:650; }
        .page-title { display:flex; align-items:center; gap:8px; font-size:1.72rem; font-weight:780; letter-spacing:-.035em; line-height:1.1; color:#14213a; }
        .page-title-icon { color:#276be5; font-size:1.3rem; line-height:1; }
        .page-subtitle { color:#718096; font-size:.78rem; margin:.22rem 0 .65rem; }
        .section-title { color:#17233b; font-size:1.04rem; font-weight:720; letter-spacing:-.015em; margin:.15rem 0 .65rem; }
        .section-hint { color:#8491a5; font-size:.75rem; margin-top:-.45rem; margin-bottom:.65rem; }
        .kpi-card { position:relative; height:91px; box-sizing:border-box; background:#fff; border:1px solid #e6eaf2; border-radius:10px; padding:9px 9px 7px 12px; box-shadow:0 3px 10px rgba(24,39,75,.04); overflow:hidden; margin-bottom:.05rem; }
        .kpi-card:before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--blue); }
        .kpi-card.accent-purple:before { background:var(--purple); } .kpi-card.accent-green:before { background:var(--green); }
        .kpi-card.accent-orange:before { background:var(--orange); } .kpi-card.accent-red:before { background:var(--red); }
        .kpi-head { display:flex; align-items:center; justify-content:space-between; gap:3px; }
        .kpi-label { color:#728096; font-size:.62rem; font-weight:680; letter-spacing:.025em; text-transform:uppercase; line-height:1.15; }
        .kpi-icon { flex:0 0 23px; width:23px; height:23px; display:flex; align-items:center; justify-content:center; border-radius:7px; background:#edf4ff; color:#4477f5; font-size:.9rem; font-weight:700; }
        .accent-green .kpi-icon { color:#15936a; background:#e6f7ef; } .accent-orange .kpi-icon { color:#d98016; background:#fff3df; }
        .accent-red .kpi-icon { color:#d9485b; background:#fff0f2; } .accent-purple .kpi-icon { color:#8054e8; background:#f1edff; }
        .kpi-value { color:#192640; font-size:1.32rem; line-height:1.08; font-weight:760; margin:.2rem 0 .1rem; }
        .kpi-caption { color:#8995a8; font-size:.64rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        [data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:#e5ebf4; border-radius:11px; padding:.1rem .35rem; box-shadow:0 3px 11px rgba(24,39,75,.035); }
        .overview-card-title { color:#17233b; font-size:.91rem; font-weight:730; letter-spacing:-.012em; margin:.15rem 0 .38rem; }
        .overview-card-note { color:#8491a5; font-size:.66rem; line-height:1.4; margin:.1rem 0 .2rem; }
        .overall-mini { display:flex; align-items:center; gap:8px; background:#f8faff; border:1px solid #e7ecf5; border-radius:8px; padding:7px 9px; margin-top:.35rem; }
        .overall-mini-label { color:#536178; font-size:.7rem; font-weight:700; }
        .overall-mini-value { color:#26344e; font-size:.9rem; font-weight:760; margin-left:auto; }
        .overall-mini-note { color:#8793a6; font-size:.62rem; }
        .panel { background:#fff; border:1px solid #e6eaf2; border-radius:14px; padding:16px 18px; box-shadow:0 4px 14px rgba(24,39,75,.04); }
        .empty-state { min-height:185px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; border:1px dashed #d7deea; border-radius:10px; background:linear-gradient(140deg,#fff,#f9faff); padding:12px; }
        .empty-icon { color:#9aa9c1; font-size:1.7rem; margin-bottom:.45rem; }
        .empty-title { color:#24324b; font-size:.98rem; font-weight:700; }
        .empty-copy { color:#7d899c; font-size:.78rem; max-width:340px; margin-top:.35rem; }
        .status-pill { display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:5px 10px; font-size:.69rem; font-weight:750; letter-spacing:.045em; }
        .status-real { color:#176d4b; background:#e7f6ef; border:1px solid #ccebdc; }
        .status-demo { color:#6650b3; background:#f0ecff; border:1px solid #ded5ff; }
        .status-legacy { color:#6b7280; background:#eff1f4; border:1px solid #e0e4ea; }
        .status-mini-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; margin-top:-.3rem; }
        .status-mini { background:#fff; border:1px solid #e6eaf2; border-radius:8px; padding:6px 8px; }
        .status-mini-label { color:#718096; font-size:.6rem; font-weight:700; letter-spacing:.035em; text-transform:uppercase; }
        .status-mini-value { color:#192640; font-size:.98rem; font-weight:760; line-height:1.15; margin-top:2px; }
        .status-mini-pct { color:#8995a8; font-size:.61rem; margin-left:4px; font-weight:500; }
        .comparison-wrap { width:100%; overflow-x:auto; border:1px solid #e4e9f1; border-radius:10px; background:#fff; box-shadow:0 2px 8px rgba(24,39,75,.025); }
        table.comparison-table { width:100%; border-collapse:collapse; color:#25324a; font-size:.72rem; background:#fff; }
        .comparison-table th { text-align:left; padding:7px 8px; background:#f0f5fc; color:#536178; font-size:.6rem; font-weight:750; letter-spacing:.025em; border-bottom:1px solid #e4e9f1; white-space:nowrap; }
        .comparison-table td { padding:6px 8px; border-bottom:1px solid #edf0f5; white-space:nowrap; }
        .comparison-table tbody tr:last-child td { border-bottom:0; }
        .comparison-table tbody tr:hover { background:#f8faff; }
        .comparison-table td:first-child { color:#34435c; font-weight:620; }
        .overview-comparison .comparison-table { font-size:.65rem; }
        .overview-comparison .comparison-table th { font-size:.53rem; padding:5px 6px; }
        .overview-comparison .comparison-table td { padding:2px 6px; line-height:1.12; }
        .task-results-table .comparison-table { font-size:.65rem; }
        .task-results-table .comparison-table th { font-size:.55rem; padding:6px 5px; }
        .task-results-table .comparison-table td { padding:5px; }
        .stSelectbox [data-baseweb="select"] > div, .stMultiSelect [data-baseweb="select"] > div { border-radius:9px; border-color:#dfe5ee; background:#fff; }
        [data-testid="stDataFrame"] { border:1px solid #e6eaf2; border-radius:10px; overflow:hidden; }
        div[data-testid="stAlert"] { border-radius:11px; }
        div[data-testid="stExpander"] { background:#fff; border:1px solid #e6eaf2; border-radius:11px; }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] { background:#17233b; border:1px solid #33435f; border-radius:11px; }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] summary { color:#eaf0fb; }
        h1,h2,h3 { letter-spacing:-.025em; }
        @media (max-width: 1100px) { .kpi-label { font-size:.55rem; } .kpi-icon { display:none; } }
        @media (max-width: 850px) { section[data-testid="stSidebar"] { width:230px !important; min-width:230px !important; } section[data-testid="stSidebar"] > div { width:230px !important; } [data-testid="stMainBlockContainer"] { padding:.7rem .8rem 1rem !important; } .page-title { font-size:1.45rem; } .kpi-card { height:88px; padding:8px; } .kpi-value { font-size:1.2rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="LLM Quality Evaluation", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
_style_page()

if not Path(settings.DB_PATH).exists():
    st.warning("No evaluation database was found. Run `python main.py --provider mock --limit 2` first.")
    st.stop()

raw = _load_results(settings.DB_PATH)
if raw.empty:
    st.warning("The database contains no evaluation records.")
    st.stop()

all_data = _prepare_data(raw)
st.sidebar.markdown(
    '<div class="brand"><div class="brand-icon">Ψ</div><div><div class="brand-name">LLM Quality</div><div class="brand-sub">Evaluation Framework</div></div></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown('<div class="nav-caption">Workspace</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigation",
    NAVIGATION,
    label_visibility="collapsed",
    key="dashboard_navigation",
)

run_order = all_data.groupby("run_group")["id"].max().sort_values(ascending=False).index.tolist()
header, run_col, provider_col, model_col = st.columns([2.0, 1.05, .85, 1.2], gap="small")
with run_col:
    selected_run = st.selectbox(
        "Run",
        run_order,
        index=0,
        format_func=lambda value: _run_option_label(value, all_data),
        key="global_run_filter",
    )
run_data = all_data[all_data["run_group"] == selected_run].copy()
provider_options = sorted(run_data["provider"].fillna("unknown").astype(str).unique().tolist())
default_provider_index = provider_options.index("ollama") if "ollama" in provider_options else 0
with provider_col:
    selected_provider = st.selectbox("Provider", provider_options, index=default_provider_index, key="global_provider_filter")
provider_data = run_data[run_data["provider"].fillna("unknown").astype(str) == selected_provider].copy()
model_options = sorted(provider_data["comparison_label"].unique().tolist())
with model_col:
    selected_models = st.multiselect("Model(s)", model_options, default=model_options, key="global_model_filter")
view_data = provider_data[provider_data["comparison_label"].isin(selected_models)].copy()
if view_data.empty:
    st.info("Select at least one model to view this run.")
    st.stop()

kind = _run_kind(run_data)
badge_class = "status-real" if kind == "REAL EXPERIMENT" else "status-demo" if kind == "DEMO / MOCK" else "status-legacy"
unique_tasks = view_data["task_id"].dropna().astype(str).nunique()
summary, comparison_table = _comparison_table(view_data)
model_identities = sorted(view_data["model_identity"].astype(str).unique().tolist())
status_counts = view_data["response_status"].value_counts()
complete_count = int(status_counts.get("Complete", 0))
partial_count = int(status_counts.get("Partial", 0))
failed_count = int(status_counts.get("Failed", 0))
unknown_count = int(status_counts.get("Unknown", 0))
provider_failures = int(view_data["provider_status"].eq("error").sum())
if "provider_status" not in view_data or view_data["provider_status"].isna().all():
    provider_failures = int(view_data["evaluation_status"].eq("provider_error").sum())
efficiency_mean = view_data["efficiency_score"].mean() if len(model_identities) == 1 else None
overall_mean = summary["overall_score"].mean() if not summary.empty and summary["overall_score"].notna().all() else None
overall_reason = _overall_unavailable_reason(view_data)

with header:
    if page == "Overview":
        st.markdown(f'<div class="page-title"><span class="page-title-icon">▥</span>{html.escape(page)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Comprehensive evaluation of programming assistants across 8 quality dimensions</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="page-title">{html.escape(page)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-subtitle">{html.escape(kind)} · selected run {html.escape(str(selected_run))}</div>', unsafe_allow_html=True)

st.markdown(
    f'<span class="status-pill {badge_class}">{html.escape(kind)}</span> '
    f'<span style="color:#718096;font-size:.78rem">Run {html.escape(str(selected_run))} · '
    f'{html.escape(selected_provider)} · {len(model_identities)} model(s) · {unique_tasks} task(s)</span>',
    unsafe_allow_html=True,
)

# Compact current-run context at the bottom of the sidebar.
run_providers = ", ".join(sorted(run_data["provider"].fillna("unknown").astype(str).unique()))
run_models = ", ".join(sorted(run_data["model_identity"].astype(str).unique()))
run_task_count = int(run_data["task_id"].dropna().astype(str).nunique())
run_status = "Failed" if int(run_data["response_status"].eq("Failed").sum()) == len(run_data) else "Partial" if int(run_data["response_status"].eq("Partial").sum()) or int(run_data["response_status"].eq("Unknown").sum()) else "Complete"
st.sidebar.markdown(
    '<div class="sidebar-run"><div class="sidebar-run-title">Current Run</div>'
    f'<div class="sidebar-run-line"><b>ID</b> · {html.escape(str(selected_run))}</div>'
    f'<div class="sidebar-run-line"><b>Provider</b> · {html.escape(run_providers)}</div>'
    f'<div class="sidebar-run-line"><b>Models</b> · {html.escape(run_models)}</div>'
    f'<div class="sidebar-run-line"><b>Tasks</b> · {run_task_count}</div>'
    f'<div class="sidebar-run-line"><b>Status</b> · {html.escape(run_status)}</div></div>',
    unsafe_allow_html=True,
)
with st.sidebar.expander("Quick access"):
    st.caption("Run a new experiment from the project CLI:")
    st.code("python main.py --provider mock --limit 2", language="powershell")
    st.caption("Reports: use the Reports page. Database: " + str(settings.DB_PATH))


if page == "Overview":
    classified_total = complete_count + partial_count + failed_count
    complete_pct = 100 * complete_count / classified_total if classified_total else 0
    partial_pct = 100 * partial_count / classified_total if classified_total else 0
    failed_pct = 100 * failed_count / classified_total if classified_total else 0
    kpis = [
        ("Models Evaluated", str(len(model_identities)), ", ".join(model_identities), "blue", "♙"),
        ("Tasks Evaluated", str(unique_tasks), _task_ids_caption(view_data["task_id"]), "purple", "▤"),
        ("Complete Responses", str(complete_count), f"{complete_pct:.0f}%", "green", "✓"),
        ("Partial Responses", str(partial_count), f"{partial_pct:.0f}%", "orange", "!"),
        ("Provider Failures", str(provider_failures), f"{failed_pct:.0f}%", "red" if provider_failures else "blue", "×"),
        ("Overall Score", _display_score(overall_mean), "Correctness unavailable" if overall_mean is None and view_data["correctness_score"].isna().any() else "eight dimensions required", "purple", "★"),
        ("Response Efficiency", _display_score(efficiency_mean), "avg, run-relative" if efficiency_mean is not None else "run-relative", "blue", "◷"),
    ]
    for col, (label, value, caption, accent, icon) in zip(st.columns(7, gap="small"), kpis):
        with col:
            _kpi(label, value, caption, accent, icon)
    if overall_reason and page != "Overview":
        st.info(overall_reason)
    if unknown_count:
        st.caption(f"{unknown_count} result(s) have no verified completion status and are excluded from the response-status chart.")

    first_left, first_right = st.columns([1.04, 1], gap="small")
    with first_left:
        with st.container(border=True):
            st.markdown('<div class="overview-card-title">Overall Score by Model</div>', unsafe_allow_html=True)
            valid_overall = summary.dropna(subset=["overall_score"])
            if valid_overall.empty:
                _empty_state("Overall Score Unavailable", "Correctness could not be evaluated for this run. Overall Score requires all eight dimensions.")
            else:
                overall_chart = px.bar(
                    valid_overall, x="model_identity", y="overall_score", color="model_identity",
                    text_auto=".1f", range_y=[0, 100],
                    color_discrete_sequence=[COLORS["blue"], COLORS["purple"], COLORS["green"]],
                )
                overall_chart.update_layout(
                    template="plotly_white", height=258, margin=dict(l=8, r=8, t=8, b=8),
                    showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(overall_chart, use_container_width=True)
    with first_right:
        with st.container(border=True):
            st.markdown('<div class="overview-card-title">Model Comparison <span class="overview-card-note">(Average Scores)</span></div>', unsafe_allow_html=True)
            if kind == "REAL EXPERIMENT" and len(model_identities) == 1:
                st.markdown('<div class="overview-card-note">1 real model available in this selected run.</div>', unsafe_allow_html=True)
            _render_comparison_table(comparison_table, "overview-comparison")

    radar_col, efficiency_col, status_col = st.columns([1.05, 1.05, .95], gap="small")
    with radar_col:
        with st.container(border=True):
            st.markdown('<div class="overview-card-title">8-Dimension Comparison</div>', unsafe_allow_html=True)
            labels = [DIMENSION_LABELS[column] for column in DIMENSIONS]
            radar = go.Figure()
            for model_index, (_, model) in enumerate(summary.iterrows()):
                values = [None if pd.isna(model[column]) else float(model[column]) for column in DIMENSIONS]
                if any(value is not None for value in values):
                    radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]], theta=labels + [labels[0]],
                        fill="toself", name=str(model["model_identity"]), opacity=.68,
                        line=dict(color=[COLORS["blue"], COLORS["purple"], COLORS["green"]][model_index % 3]),
                    ))
            radar.update_layout(
                template="plotly_white", height=238, margin=dict(l=22, r=22, t=22, b=18),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=8, color="#536178"),
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=7), gridcolor="#e7ebf2"), angularaxis=dict(gridcolor="#e7ebf2")),
                legend=dict(orientation="h", y=1.08, x=0, font=dict(size=8)),
            )
            if radar.data:
                st.plotly_chart(radar, use_container_width=True)
            else:
                _empty_state("Dimension profile unavailable", "No dimension scores are available for this selection.")
            if view_data["correctness_score"].isna().any():
                st.markdown('<div class="overview-card-note">Correctness: N/A for this run.</div>', unsafe_allow_html=True)
    with efficiency_col:
        with st.container(border=True):
            st.markdown('<div class="overview-card-title">Response Efficiency <span class="overview-card-note">(Run-relative)</span></div>', unsafe_allow_html=True)
            st.plotly_chart(_efficiency_figure(view_data), use_container_width=True)
            st.markdown('<div class="overview-card-note">Efficiency is normalized within this run; it is not an absolute or industry benchmark.</div>', unsafe_allow_html=True)
    with status_col:
        with st.container(border=True):
            st.markdown('<div class="overview-card-title">Response Status Distribution</div>', unsafe_allow_html=True)
            if classified_total:
                status_labels = ["Complete", "Partial", "Failed"]
                status_values = [complete_count, partial_count, failed_count]
                status_chart = go.Figure(go.Pie(
                    labels=status_labels, values=status_values, hole=.68,
                    marker=dict(colors=[COLORS["green"], COLORS["orange"], COLORS["red"]]),
                    textinfo="none", sort=False, showlegend=False,
                    hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
                ))
                status_chart.update_layout(
                    template="plotly_white", height=158, margin=dict(l=4, r=4, t=4, b=2),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    annotations=[dict(text=f"{complete_pct:.0f}%<br><span style='font-size:9px'>Complete</span>", x=.5, y=.5, showarrow=False, font=dict(size=17, color="#20304a"))],
                )
                st.plotly_chart(status_chart, use_container_width=True)
                st.markdown(
                    '<div class="status-mini-grid">'
                    f'<div class="status-mini"><div class="status-mini-label">Complete</div><div class="status-mini-value">{complete_count}<span class="status-mini-pct">{complete_pct:.0f}%</span></div></div>'
                    f'<div class="status-mini"><div class="status-mini-label">Partial</div><div class="status-mini-value">{partial_count}<span class="status-mini-pct">{partial_pct:.0f}%</span></div></div>'
                    f'<div class="status-mini"><div class="status-mini-label">Failed</div><div class="status-mini-value">{failed_count}<span class="status-mini-pct">{failed_pct:.0f}%</span></div></div>'
                    '</div>', unsafe_allow_html=True,
                )
            else:
                _empty_state("No classified responses", "No complete, partial, or failed status is available.")
            st.markdown(
                '<div class="overall-mini"><span class="overall-mini-label">Overall Score</span>'
                f'<span class="overall-mini-value">{html.escape(_display_score(overall_mean))}</span>'
                '<span class="overall-mini-note">Correctness unavailable</span></div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown('<div class="overview-card-title">Task Results</div>', unsafe_allow_html=True)
        task_table = _task_table_frame(view_data.sort_values(["task_id", "model_identity"]))
        _render_comparison_table(task_table, "task-results-table")

elif page == "Model Comparison":
    st.markdown('<div class="section-title">Dimension and Overall Comparison</div>', unsafe_allow_html=True)
    if kind == "REAL EXPERIMENT" and len(model_identities) == 1:
        st.info("1 real model available in this selected run. The dashboard is ready to compare additional models when they have actual results in this same run.")
    _render_comparison_table(comparison_table)
    st.caption("The table reports actual scores and measured-task denominators. No ranking is inferred.")
    valid = summary.dropna(subset=["overall_score"])
    if valid.empty:
        _empty_state("Overall Score unavailable", overall_reason or "All eight dimensions must be available before an Overall Score is calculated.")
    else:
        st.plotly_chart(px.bar(valid, x="model_identity", y="overall_score", color="model_identity", text_auto=".1f", range_y=[0, 100]), use_container_width=True)

elif page == "8-Dimension Analysis":
    st.markdown('<div class="section-title">8-Dimension Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hint">Unavailable dimensions are omitted from the chart and remain visible as N/A in the table.</div>', unsafe_allow_html=True)
    radar, table_col = st.columns([1.12, 1], gap="large")
    with radar:
        figure = go.Figure()
        labels = [DIMENSION_LABELS[column] for column in DIMENSIONS]
        for _, model in summary.iterrows():
            available = [(DIMENSION_LABELS[column], model[column]) for column in DIMENSIONS if pd.notna(model[column])]
            if available:
                theta = [entry[0] for entry in available]
                values = [float(entry[1]) for entry in available]
                figure.add_trace(go.Scatterpolar(r=values + [values[0]], theta=theta + [theta[0]], fill="toself", name=str(model["model_identity"]), opacity=.72))
        figure.update_layout(
            template="plotly_white", height=390, margin=dict(l=35, r=35, t=35, b=25),
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e7ebf2"), angularaxis=dict(gridcolor="#e7ebf2")),
            legend=dict(orientation="h", y=-.08),
        )
        if figure.data:
            st.plotly_chart(figure, use_container_width=True)
        else:
            _empty_state("Dimension profile unavailable", "There are no measured dimension scores to plot.")
    with table_col:
        _render_comparison_table(comparison_table)
    st.caption("The radar chart plots only available dimension values; the table retains all eight dimensions and Overall Score.")

elif page == "Response Efficiency":
    st.markdown('<div class="section-title">Response Efficiency</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hint">Efficiency scores are normalized within the selected run and are not an absolute or industry benchmark.</div>', unsafe_allow_html=True)
    efficiency_columns = [
        column for column in ("task_id", "model_identity", "response_time", "output_tokens", "generation_time_seconds", "tokens_per_second", "efficiency_score")
        if column in view_data.columns
    ]
    efficiency = view_data[efficiency_columns].copy().rename(columns={
        "task_id": "Task ID", "model_identity": "Model", "response_time": "Latency (s)",
        "output_tokens": "Generated tokens", "generation_time_seconds": "Generation duration (s)",
        "tokens_per_second": "Tokens/sec", "efficiency_score": "Run-relative score",
    })
    metric_cols = st.columns(4, gap="medium")
    for col, label, source, digits in zip(
        metric_cols,
        ("Mean Latency", "Mean Generated Tokens", "Mean Generation Duration", "Mean Tokens/sec"),
        ("response_time", "output_tokens", "generation_time_seconds", "tokens_per_second"),
        (2, 0, 2, 2),
    ):
        values = pd.to_numeric(view_data[source], errors="coerce")
        col.metric(label, _display_score(values.mean(), digits), "selected records")
    display_efficiency = efficiency.copy()
    for column in ("Latency (s)", "Generated tokens", "Generation duration (s)", "Tokens/sec", "Run-relative score"):
        display_efficiency[column] = display_efficiency[column].map(lambda value: "N/A" if pd.isna(value) else f"{float(value):.0f}" if column == "Generated tokens" else f"{float(value):.2f}")
    st.dataframe(display_efficiency, hide_index=True, use_container_width=True)
    if view_data["response_time"].notna().any():
        latency = view_data[["task_id", "model_identity", "response_time"]].rename(columns={"task_id": "Task ID", "model_identity": "Model", "response_time": "Latency (seconds)"})
        latency_figure = px.bar(
            latency, x="Task ID", y="Latency (seconds)", color="Model", barmode="group",
            text_auto=".1f", color_discrete_sequence=[COLORS["blue"], COLORS["purple"], COLORS["green"]],
        )
        st.plotly_chart(_style_efficiency_plot(latency_figure), use_container_width=True)
    else:
        _empty_state("Latency unavailable", "No response-time telemetry is stored for the selected results.")
    if view_data["tokens_per_second"].notna().any():
        throughput = view_data[["task_id", "model_identity", "tokens_per_second"]].rename(columns={"task_id": "Task ID", "model_identity": "Model", "tokens_per_second": "Generated tokens/sec"})
        throughput_figure = px.bar(
            throughput, x="Task ID", y="Generated tokens/sec", color="Model", barmode="group",
            text_auto=".1f", color_discrete_sequence=[COLORS["purple"], COLORS["blue"], COLORS["green"]],
        )
        st.plotly_chart(_style_efficiency_plot(throughput_figure), use_container_width=True)
    else:
        _empty_state("Generation throughput unavailable", "Token-rate telemetry was not stored for the selected results.")

elif page == "Task Results":
    st.markdown('<div class="section-title">Task-Level Results</div>', unsafe_allow_html=True)
    task_filter, status_filter, search_col = st.columns([1.2, 1.2, 1.5])
    task_values = sorted(view_data["task_id"].dropna().astype(str).unique().tolist())
    with task_filter:
        selected_tasks = st.multiselect("Task ID", task_values, default=task_values)
    status_values = [value for value in ("Complete", "Partial", "Failed", "Unknown") if value in view_data["response_status"].unique()]
    with status_filter:
        selected_statuses = st.multiselect("Status", status_values, default=status_values)
    with search_col:
        search = st.text_input("Search task or model", placeholder="e.g. P001")
    filtered = view_data[
        view_data["task_id"].astype(str).isin(selected_tasks)
        & view_data["response_status"].isin(selected_statuses)
    ].copy()
    if search.strip():
        query = search.strip().casefold()
        filtered = filtered[filtered.apply(lambda row: query in str(row.get("task_id", "")).casefold() or query in str(row.get("model_identity", "")).casefold(), axis=1)]
    score_tab, telemetry_tab = st.tabs(["Scores", "Latency & Tokens"])
    with score_tab:
        score_columns = ["task_id", "model_identity", "response_status", *DIMENSIONS, "overall_score"]
        score_view = filtered[[column for column in score_columns if column in filtered]].copy()
        score_view = score_view.rename(columns={"task_id": "Task ID", "model_identity": "Model", "response_status": "Status", **DIMENSION_LABELS})
        score_view = _format_table_scores(score_view, [DIMENSION_LABELS[column] for column in DIMENSIONS] + ["Overall Score"])
        st.dataframe(score_view, hide_index=True, use_container_width=True, height=390)
    with telemetry_tab:
        telemetry_columns = ["task_id", "model_identity", "response_status", "response_time", "output_tokens", "generation_time_seconds", "tokens_per_second", "efficiency_score"]
        telemetry_view = filtered[[column for column in telemetry_columns if column in filtered]].copy().rename(columns={
            "task_id": "Task ID", "model_identity": "Model", "response_status": "Status", "response_time": "Latency (s)",
            "output_tokens": "Tokens", "generation_time_seconds": "Generation duration (s)", "tokens_per_second": "Tokens/sec", "efficiency_score": "Run-relative efficiency",
        })
        telemetry_view = _format_table_scores(telemetry_view, ["Latency (s)", "Generation duration (s)", "Tokens/sec", "Run-relative efficiency"])
        if "Tokens" in telemetry_view:
            telemetry_view["Tokens"] = telemetry_view["Tokens"].map(lambda value: "N/A" if pd.isna(value) else str(int(value)))
        st.dataframe(telemetry_view, hide_index=True, use_container_width=True, height=390)
    if filtered.empty:
        st.info("No task records match these filters.")
    else:
        with st.expander("Inspect one result and its stored response"):
            result_options = [f"{row['model_identity']} · {row['task_id']}" for _, row in filtered.iterrows()]
            selected_result = st.selectbox("Result", result_options, key="task_inspect_result")
            row = filtered.iloc[result_options.index(selected_result)]
            st.text_area("Response text (display only; never executed)", str(row.get("response_text") or ""), height=150, disabled=True)
            st.text_area("Extracted code (display only; never executed)", str(row.get("generated_code") or ""), height=150, disabled=True)

elif page == "Evidence & Limitations":
    st.markdown('<div class="section-title">Evidence & Limitations</div>', unsafe_allow_html=True)
    evidence_columns = [
        "provider", "model_name", "run_id", "task_id", "response_status", "completion_status", "done_reason", "truncated",
        "response_time", "output_tokens", "generation_time_seconds", "tokens_per_second", "correctness_status",
        "execution_status", "executor_backend", "evaluation_status", "provider_error", "evaluation_error",
    ]
    evidence = view_data[[column for column in evidence_columns if column in view_data]].copy().rename(columns={
        "provider": "Provider", "model_name": "Model", "run_id": "Run ID", "task_id": "Task ID", "response_status": "Response status",
        "completion_status": "Completion status", "done_reason": "done_reason", "truncated": "Truncated", "response_time": "Latency (s)",
        "output_tokens": "Generated tokens", "generation_time_seconds": "Generation duration (s)", "tokens_per_second": "Tokens/sec",
        "correctness_status": "Correctness availability", "execution_status": "Execution status", "executor_backend": "Executor",
    })
    st.dataframe(evidence, hide_index=True, use_container_width=True)
    if overall_reason:
        st.warning(overall_reason)
    st.markdown(
        "**Correctness.** In the preserved real Qwen run `ef8da533-c3a5-4da5-ab80-4024f07c6be5`, generated code was not executed because the configured execution approach was not considered safe for arbitrary generated code. Correctness is unavailable for those records. Other runs show their record-level execution status above.\n\n"
        "**Overall Score.** The scoring rule requires all eight dimensions. When Correctness or another required dimension is unavailable, Overall Score is N/A.\n\n"
        "**Response Efficiency.** Scores use run-relative normalization and are not absolute or industry benchmarks.\n\n"
        "**Other dimensions.** Completeness, Security, Maintainability, Explainability, Usability, and Robustness / Reliability are operational proxies, not validated human measures. Static security analysis is not runtime isolation.\n\n"
        "Mock data is labeled DEMO / MOCK. Rows without provider identity are labeled as legacy/unknown and are not attributed to a real model."
    )
    with st.expander("Selected evaluation metadata"):
        evidence_options = [f"{row['model_identity']} · {row['task_id']}" for _, row in view_data.iterrows()]
        selected_evidence = st.selectbox("Record", evidence_options, key="evidence_record")
        row = view_data.iloc[evidence_options.index(selected_evidence)]
        st.json(_parse_json(row.get("evaluation_metadata")))

elif page == "Reports":
    st.markdown('<div class="section-title">Generated Reports</div>', unsafe_allow_html=True)
    st.caption("These are existing project artifacts. CSV summaries and charts may aggregate multiple runs; use the run filter above for dashboard views.")
    report_root = settings.RESULTS_DIR
    table_file = report_root / "tables" / "model_summary.csv"
    if table_file.exists():
        st.write(f"Model summary CSV · modified {pd.Timestamp(table_file.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}")
        st.download_button("Download model_summary.csv", table_file.read_bytes(), file_name="model_summary.csv", mime="text/csv")
    else:
        st.info("No generated summary CSV is present yet. Run an evaluation through the CLI to create reports.")
    figures = sorted((report_root / "figures").glob("*.png")) if (report_root / "figures").exists() else []
    if figures:
        st.subheader("Existing chart files")
        for figure in figures:
            st.write(f"{figure.name} · modified {pd.Timestamp(figure.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}")
            st.download_button(f"Download {figure.name}", figure.read_bytes(), file_name=figure.name, mime="image/png", key=f"download-{figure.name}")
    st.info("To generate or update reports, use the existing CLI: `python main.py --provider mock --limit 2`. This dashboard does not start experiments.")

elif page == "Settings":
    st.markdown('<div class="section-title">Read-Only Project Settings</div>', unsafe_allow_html=True)
    st.caption("Values shown here come from current project configuration. Secret values are not displayed.")
    settings_table = pd.DataFrame([
        ("Default Ollama model", settings.OLLAMA_MODEL),
        ("Ollama endpoint", settings.OLLAMA_HOST),
        ("Requested context", settings.OLLAMA_NUM_CTX),
        ("Generation token limit", settings.OLLAMA_NUM_PREDICT),
        ("Temperature", settings.OLLAMA_TEMPERATURE),
        ("Seed", settings.OLLAMA_SEED),
        ("Code execution backend", settings.CODE_EXECUTION_BACKEND),
        ("Execution timeout (seconds)", settings.CODE_EXECUTION_TIMEOUT_SECONDS),
        ("Evaluation database", str(settings.DB_PATH)),
    ], columns=["Setting", "Current value"])
    st.dataframe(settings_table, hide_index=True, use_container_width=True)
    st.warning("The configured local-test executor uses a normal subprocess and is not a security sandbox. Safe execution remains unavailable unless a verified isolated executor is configured.")
