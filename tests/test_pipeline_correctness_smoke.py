import json
import sqlite3
from pathlib import Path

from src.analysis import analysis
from src.database import database
from src.dataset.loader import load_problems
from src.llm.pipeline import run_experiment
from src.config import settings


def test_sample_csv_to_mock_pipeline_and_sqlite(tmp_path, monkeypatch):
    # Checked-in mock responses run through the configured controlled local test backend.
    monkeypatch.setattr(settings, "CODE_EXECUTION_BACKEND", "local-test")
    project_root = Path(__file__).resolve().parents[1]
    task = load_problems(project_root / "data" / "sample" / "problems.csv", "sample")[0]
    db_path = tmp_path / "smoke.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(analysis, "DB_PATH", db_path)
    monkeypatch.setattr(analysis, "RESULTS_DIR", tmp_path / "results")

    database.init_db()
    results = run_experiment([task], live=False)
    database.save_results(results)
    stored = analysis.load_results_df()
    summary = analysis.generate_summary_csv(stored)

    assert len(results) == len(stored) == 3
    assert all(result["correctness_status"] == "passed" for result in results)
    assert all(result["passed_tests"] == result["total_tests"] == 4 for result in results)
    assert all(result["correctness_score"] == 100.0 for result in results)
    assert all(result["executor_backend"] == "local-test" for result in results)
    assert all(result["overall_score"] is not None for result in results)
    assert all(result["robustness_score"] is not None for result in results)
    assert summary["robustness_score"].notna().all()
    assert summary["overall_score"].notna().all()
    assert set(stored["task_id"]) == {"P001"}
    assert set(stored["dataset"]) == {"sample"}
    assert set(stored["correctness_status"]) == {"passed"}
    assert set(stored["passed_tests"]) == {4}
    assert set(stored["total_tests"]) == {4}
    assert set(stored["correctness_score"]) == {100.0}
    assert summary["correctness_score"].eq(100.0).all()

    with sqlite3.connect(db_path) as con:
        serialized = con.execute("SELECT correctness_results FROM evaluations LIMIT 1").fetchone()[0]
    assert [case["status"] for case in json.loads(serialized)] == ["passed"] * 4
