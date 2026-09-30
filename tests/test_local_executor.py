from types import SimpleNamespace
import sys
from src.config import settings
from src.evaluation.correctness import run_tests
from src.evaluation.executor import ControlledLocalExecutor, select_executor
import src.evaluation.executor as executor_module

def test_local_test_selects_controlled_executor_by_default(monkeypatch):
    monkeypatch.setattr(settings,"CODE_EXECUTION_BACKEND","local-test")
    assert isinstance(select_executor(),ControlledLocalExecutor)

def test_known_sample_solution_passes_and_wrong_solution_gets_percentage(monkeypatch):
    monkeypatch.setattr(settings,"CODE_EXECUTION_BACKEND","local-test")
    tests={"kind":"sample","cases":[
      {"expression":"is_prime(2)","expected":True},
      {"expression":"is_prime(9)","expected":False},
      {"expression":"is_prime(17)","expected":True},
      {"expression":"is_prime(1)","expected":False}]}
    correct="def is_prime(n):\n    return n > 1 and all(n % i for i in range(2,n))\n"
    passed=run_tests(correct,tests)
    assert (passed["passed_tests"],passed["total_tests"],passed["correctness_score"])==(4,4,100.0)
    assert passed["executor_backend"]=="local-test"
    incorrect=run_tests("def is_prime(n): return n > 1\n",tests)
    assert (incorrect["passed_tests"],incorrect["total_tests"],incorrect["correctness_score"])==(3,4,75.0)

def test_configured_local_execution_timeout(monkeypatch):
    monkeypatch.setattr(settings,"CODE_EXECUTION_BACKEND","local-test")
    monkeypatch.setattr(settings,"CODE_EXECUTION_TIMEOUT_SECONDS",0.25)
    result=run_tests("def hang():\n    while True: pass\n",{"kind":"sample","cases":[{"expression":"hang()","expected":True}]})
    assert result["correctness_status"]=="timeout"
    assert result["timeout"] is True and result["correctness_score"]==0.0

def test_executor_invocation_is_shell_free_with_temp_cwd_and_explicit_python(tmp_path,monkeypatch):
    calls={}
    def fake_run(args,**kwargs):
        calls["args"]=args; calls.update(kwargs)
        return SimpleNamespace(stdout="",stderr="",returncode=0)
    monkeypatch.setattr(executor_module.subprocess,"run",fake_run)
    harness=tmp_path/"correctness_harness.py"
    result=ControlledLocalExecutor().execute(harness,3)
    assert result.execution_status=="completed"
    assert calls["args"]==[sys.executable,str(harness.resolve())]
    assert calls["executable"]==sys.executable
    assert calls["shell"] is False
    assert calls["cwd"]==str(tmp_path.resolve())
    assert calls["timeout"]==3
    assert calls["capture_output"] is True
    assert calls["text"] is True
    assert result.metadata["isolation"]=="none"
