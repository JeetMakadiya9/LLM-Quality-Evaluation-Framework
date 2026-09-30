import json
import sqlite3

from src.database import database


def test_database_persists_correctness_evidence_and_migrates(tmp_path, monkeypatch):
    db_path = tmp_path / "correctness.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    database.save_results([{
        "task_id": "P001",
        "problem_id": "P001",
        "dataset": "sample",
        "entry_point": "is_prime",
        "model": "Mock",
        "correctness_status": "failed",
        "passed": False,
        "passed_tests": 1,
        "failed_tests": 1,
        "total_tests": 2,
        "correctness_score": 50.0,
        "test_results": [{"index": 1, "status": "passed"}, {"index": 2, "status": "failed"}],
        "correctness_errors": [{"index": 2, "error_type": "assertion_failure"}],
        "timeout": False,
        "execution_status": "completed",
        "executor_backend": "host-dev-unsafe",
        "executor_metadata": {"isolation": "none"},
        "execution_error": None,
        "metadata": {"source": "fixture"},
    }])

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT task_id, dataset, entry_point, correctness_status, passed_tests, "
            "failed_tests, total_tests, correctness_score, correctness_results, "
            "correctness_errors, correctness_timeout, evaluation_metadata, tests_passed, tests_failed, "
            "execution_status, executor_backend, executor_metadata, execution_error "
            "FROM evaluations"
        ).fetchone()

    assert row[:8] == ("P001", "sample", "is_prime", "failed", 1, 1, 2, 50.0)
    assert json.loads(row[8])[1]["status"] == "failed"
    assert json.loads(row[9])[0]["error_type"] == "assertion_failure"
    assert row[10] == 0
    assert json.loads(row[11]) == {"source": "fixture"}
    assert (row[12], row[13]) == (1, 1)
    assert row[14:16] == ("completed", "host-dev-unsafe")
    assert json.loads(row[16]) == {"isolation": "none"}
    assert row[17] is None


def test_database_migrates_existing_schema_without_deleting_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE evaluations (id INTEGER PRIMARY KEY, model TEXT, correctness_score REAL)"
        )
        con.execute("INSERT INTO evaluations (model) VALUES ('legacy')")
    monkeypatch.setattr(database, "DB_PATH", db_path)

    database.init_db()

    with sqlite3.connect(db_path) as con:
        count = con.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        columns = {row[1] for row in con.execute("PRAGMA table_info(evaluations)")}
    assert count == 1
    assert {"task_id", "correctness_results", "correctness_score", "execution_status",
            "executor_backend", "executor_metadata", "execution_error"} <= columns


def test_database_preserves_unavailable_execution_without_zero_score(tmp_path, monkeypatch):
    db_path = tmp_path / "unavailable.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    database.save_results([{
        "task_id": "fixture",
        "correctness_status": "executor_unavailable",
        "execution_status": "executor_unavailable",
        "executor_backend": "safe-unavailable",
        "executor_metadata": {"isolation": "unavailable"},
        "correctness_score": None,
        "passed_tests": None,
        "failed_tests": None,
        "total_tests": 2,
        "correctness_errors": [{"error_type": "executor_unavailable"}],
    }])
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT correctness_status, execution_status, correctness_score, passed_tests, "
            "total_tests, executor_backend, executor_metadata FROM evaluations"
        ).fetchone()
    assert row[:6] == ("executor_unavailable", "executor_unavailable", None, None, 2, "safe-unavailable")
    assert json.loads(row[6]) == {"isolation": "unavailable"}


def test_database_persists_robustness_and_appends_runs(tmp_path,monkeypatch):
    db=tmp_path/"robustness.db"; monkeypatch.setattr(database,"DB_PATH",db); database.init_db()
    row={"model":"Mock","provider":"mock","problem_id":"P001","task_id":"P001","run_id":"run-fixture","robustness_score":83.33,"robustness_status":"evaluated","robustness_checks":{"response_non_empty":True},"provider_status":"completed","evaluation_status":"completed"}
    database.save_results([row]); database.save_results([row])
    with sqlite3.connect(db) as con: rows=con.execute("SELECT provider,run_id,robustness_score,robustness_checks FROM evaluations").fetchall()
    assert len(rows)==2 and rows[0][:3]==("mock","run-fixture",83.33)
    assert json.loads(rows[0][3])=={"response_non_empty":True}
