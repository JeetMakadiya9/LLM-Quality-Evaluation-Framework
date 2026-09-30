import json
import sqlite3
from contextlib import closing
from src.config.settings import DB_PATH

ADDITIONAL_COLUMNS={
 "task_id":"TEXT","dataset":"TEXT","entry_point":"TEXT","correctness_status":"TEXT",
 "passed_tests":"INTEGER","failed_tests":"INTEGER","total_tests":"INTEGER","correctness_results":"TEXT",
 "correctness_errors":"TEXT","correctness_timeout":"INTEGER","execution_status":"TEXT","executor_backend":"TEXT",
 "executor_metadata":"TEXT","execution_error":"TEXT","evaluation_metadata":"TEXT","provider":"TEXT","run_id":"TEXT",
 "robustness_score":"REAL","robustness_status":"TEXT","robustness_checks":"TEXT","provider_status":"TEXT",
 "provider_error":"TEXT","evaluation_status":"TEXT","evaluation_error":"TEXT","security_status":"TEXT"
}

def get_connection(): return sqlite3.connect(DB_PATH)

def init_db():
    with closing(get_connection()) as con, con:
        con.execute("""CREATE TABLE IF NOT EXISTS evaluations(
          id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT, model_name TEXT, problem_id TEXT, prompt TEXT,
          response_text TEXT, generated_code TEXT, explanation TEXT, tests_passed INTEGER, tests_failed INTEGER,
          correctness_score REAL, completeness_score REAL, security_score REAL, maintainability_score REAL,
          explainability_score REAL, usability_score REAL, efficiency_score REAL, robustness_score REAL,
          overall_score REAL, response_time REAL, input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,
          security_issue_count INTEGER, missing_features TEXT, test_errors TEXT, security_findings TEXT,
          maintainability_raw TEXT, score_method TEXT, task_id TEXT, dataset TEXT, entry_point TEXT,
          correctness_status TEXT, passed_tests INTEGER, failed_tests INTEGER, total_tests INTEGER,
          correctness_results TEXT, correctness_errors TEXT, correctness_timeout INTEGER, execution_status TEXT,
          executor_backend TEXT, executor_metadata TEXT, execution_error TEXT, evaluation_metadata TEXT,
          provider TEXT, run_id TEXT, robustness_status TEXT, robustness_checks TEXT, provider_status TEXT,
          provider_error TEXT, evaluation_status TEXT, evaluation_error TEXT, security_status TEXT)""")
        existing={row[1] for row in con.execute("PRAGMA table_info(evaluations)")}
        for name,kind in ADDITIONAL_COLUMNS.items():
            if name not in existing: con.execute(f"ALTER TABLE evaluations ADD COLUMN {name} {kind}")

def save_results(results):
    """Append records without deleting earlier runs; id and run_id provide traceability."""
    columns=["model","model_name","problem_id","prompt","response_text","generated_code","explanation",
      "tests_passed","tests_failed","correctness_score","completeness_score","security_score","maintainability_score",
      "explainability_score","usability_score","efficiency_score","robustness_score","overall_score","response_time",
      "input_tokens","output_tokens","total_tokens","security_issue_count","missing_features","test_errors",
      "security_findings","maintainability_raw","score_method","task_id","dataset","entry_point","correctness_status",
      "passed_tests","failed_tests","total_tests","correctness_results","correctness_errors","correctness_timeout",
      "evaluation_metadata","execution_status","executor_backend","executor_metadata","execution_error","provider","run_id",
      "robustness_status","robustness_checks","provider_status","provider_error","evaluation_status","evaluation_error","security_status"]
    with closing(get_connection()) as con, con:
        for result in results:
            row=dict(result); row.setdefault("tests_passed",row.get("passed_tests")); row.setdefault("tests_failed",row.get("failed_tests"))
            row["test_errors"]=json.dumps(row.get("correctness_errors",[]),ensure_ascii=False)
            row["correctness_results"]=json.dumps(row.get("test_results",[]),ensure_ascii=False)
            row["correctness_errors"]=json.dumps(row.get("correctness_errors",[]),ensure_ascii=False)
            row["correctness_timeout"]=int(bool(row.get("timeout",False)))
            row["evaluation_metadata"]=json.dumps(row.get("metadata",{}),ensure_ascii=False)
            row["executor_metadata"]=json.dumps(row.get("executor_metadata",{}),ensure_ascii=False)
            row["robustness_checks"]=json.dumps(row.get("robustness_checks",{}),ensure_ascii=False)
            con.execute(f"INSERT INTO evaluations ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",[row.get(c) for c in columns])
