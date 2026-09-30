"""Correctness harnesses for canonical sample, HumanEval, and MBPP tasks.

Result contract: ``status`` distinguishes pass/fail, generated-code errors,
timeouts, invalid task specifications, and infrastructure errors. ``test_results``
contains one record per supported case/assertion. ``correctness_score`` is the
percentage of required cases that passed, or ``None`` when no valid test suite or
no trustworthy execution result exists.
"""

import ast
import json
import tempfile
from pathlib import Path
from src.evaluation.executor import select_executor
from src.config import settings


def _coerce_spec(tests, entry_point=None):
    if isinstance(tests, dict):
        return tests
    if isinstance(tests, list) and all(isinstance(item, str) for item in tests):
        return {"kind": "assertions", "setup_code": "", "cases": [
            {"source": "primary", "code": item} for item in tests
        ]}
    if isinstance(tests, str):
        try:
            parsed = json.loads(tests)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list) and all(
            isinstance(item, dict) and "expression" in item and "expected" in item
            for item in parsed
        ):
            return {"kind": "sample", "cases": parsed}
        if entry_point and "def check(" in tests:
            return {"kind": "humaneval", "source": tests}
    return {"kind": "invalid", "error": "Tests must use the canonical dataset adapter format"}


def _human_eval_source(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"HumanEval test source has invalid syntax: {exc}") from exc

    checks = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check"]
    if len(checks) != 1 or isinstance(checks[0], ast.AsyncFunctionDef):
        raise ValueError("HumanEval test source must define exactly one synchronous check(candidate)")
    check = checks[0]
    if len(check.args.args) != 1 or check.args.vararg or check.args.kwarg:
        raise ValueError("HumanEval check must accept exactly one candidate argument")

    assertions = [node for node in ast.walk(check) if isinstance(node, ast.Assert)]
    if not assertions:
        raise ValueError("HumanEval check function contains no assertions")

    class InstrumentCheckAssertions(ast.NodeTransformer):
        def __init__(self):
            self.index = 0
            self.labels = []

        def visit_Assert(self, node):
            self.index += 1
            expression = ast.get_source_segment(source, node.test) or ast.unparse(node.test)
            label = f"assertion {self.index}: {expression[:180]}"
            self.labels.append(label)
            empty_args = ast.arguments(
                posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
            )
            message = node.msg if node.msg is not None else ast.Constant(value=None)
            call = ast.Call(
                func=ast.Name(id="__nlp_eval_record_assert__", ctx=ast.Load()),
                args=[
                    ast.Lambda(args=empty_args, body=node.test),
                    ast.Constant(value=label),
                    ast.Lambda(args=ast.arguments(
                        posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
                    ), body=message),
                ],
                keywords=[],
            )
            return ast.copy_location(ast.Expr(value=call), node)

    transformer = InstrumentCheckAssertions()
    transformer.visit(check)
    if transformer.index != len(assertions):
        raise ValueError("Unable to instrument every HumanEval assertion")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), transformer.labels


def _prepare(tests, entry_point):
    spec = _coerce_spec(tests, entry_point)
    kind = spec.get("kind")
    if kind == "invalid":
        return None, [], str(spec.get("error", "Invalid test specification"))

    if kind == "sample":
        cases = spec.get("cases")
        if not isinstance(cases, list) or not cases:
            return None, [], "Sample test specification must contain at least one case"
        labels = []
        for index, case in enumerate(cases, 1):
            if not isinstance(case, dict) or not isinstance(case.get("expression"), str) or "expected" not in case:
                return None, [], f"Sample case {index} needs expression and expected fields"
            try:
                ast.parse(case["expression"], mode="eval")
            except SyntaxError as exc:
                return None, [], f"Sample case {index} has invalid expression syntax: {exc}"
            labels.append(f"case {index}: {case['expression'][:180]}")
        return {**spec, "cases": cases}, labels, None

    if kind == "humaneval":
        if not isinstance(entry_point, str) or not entry_point.isidentifier():
            return None, [], "HumanEval task has no valid entry_point"
        try:
            source, labels = _human_eval_source(str(spec.get("source", "")))
        except ValueError as exc:
            return None, [], str(exc)
        return {**spec, "source": source}, labels, None

    if kind == "assertions":
        cases = spec.get("cases")
        if not isinstance(cases, list) or not cases:
            return None, [], "Assertion test specification must contain at least one case"
        setup_code = spec.get("setup_code", "") or ""
        try:
            compile(setup_code, "<test_setup>", "exec")
        except (SyntaxError, TypeError) as exc:
            return None, [], f"Test setup has invalid syntax: {exc}"
        normalized = []
        labels = []
        for index, case in enumerate(cases, 1):
            if isinstance(case, str):
                case = {"source": "primary", "code": case}
            if not isinstance(case, dict) or not isinstance(case.get("code"), str):
                return None, [], f"Assertion case {index} must contain code"
            try:
                tree = ast.parse(case["code"], mode="exec")
                if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
                    raise ValueError("each MBPP case must contain exactly one assert statement")
            except (SyntaxError, ValueError) as exc:
                return None, [], f"Assertion case {index} is invalid: {exc}"
            normalized.append({"source": case.get("source", "primary"), "code": case["code"]})
            labels.append(f"{case.get('source', 'primary')} case {index}: {case['code'][:180]}")
        return {**spec, "setup_code": setup_code, "cases": normalized}, labels, None

    return None, [], f"Unsupported test specification kind: {kind!r}"


def _human_eval_execution_code(code, prompt, entry_point):
    """Preserve HumanEval prompt imports and accept either full code or completion body."""
    imports = []
    for line in (prompt or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line[:1].isspace():
            break
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
            continue
        if stripped.startswith("def ") or stripped.startswith("async def "):
            break
        break

    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    defines_entry = tree is not None and any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in tree.body
    )
    if defines_entry:
        return "\n".join(imports + [code])

    # HumanEval's canonical prompt is a function prefix; a completion-only
    # response must be evaluated after that prefix, as in the benchmark harness.
    candidate = f"{(prompt or '').rstrip()}\n{code}"
    try:
        candidate_tree = ast.parse(candidate)
        candidate_defines_entry = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
            for node in candidate_tree.body
        )
    except SyntaxError:
        candidate_defines_entry = False
    if candidate_defines_entry:
        return candidate
    return "\n".join(imports + [code])


def _result(status, passed, passed_tests, failed_tests, total_tests, score,
            test_results=None, errors=None, stdout="", stderr="", timeout=False,
            execution_status=None, executor_backend=None, executor_metadata=None,
            execution_error=None):
    return {
        "correctness_status": status,
        "passed": passed,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "total_tests": total_tests,
        "correctness_score": score,
        "test_results": test_results or [],
        "correctness_errors": errors or [],
        "stdout": stdout,
        "stderr": stderr,
        "timeout": timeout,
        "execution_status": execution_status or status,
        "execution_error": execution_error,
        "executor_backend": executor_backend,
        "executor_metadata": executor_metadata or {},
    }


def _failed_cases(labels, error_type, message):
    return [
        {"index": index, "label": label, "status": "error", "error_type": error_type, "error": message}
        for index, label in enumerate(labels, 1)
    ]


def _harness_source(code, spec, entry_point, labels, result_path):
    payload = json.dumps({"code": code, "spec": spec, "entry_point": entry_point, "labels": labels})
    return f'''import json as __json
__payload = __json.loads({payload!r})
__result_path = {str(result_path)!r}
__seen = set()

def __emit(item):
    __seen.add(item["index"])
    with open(__result_path, "a", encoding="utf-8") as __out:
        __out.write(__json.dumps(item, ensure_ascii=False) + "\\n")
        __out.flush()

def __exception_text(exc):
    return f"{{type(exc).__name__}}: {{exc}}"

def __record_assert(condition, label, message):
    index = next(i for i, name in enumerate(__payload["labels"], 1) if name == label)
    try:
        ok = bool(condition())
        if ok:
            detail = None
        else:
            detail = message()
            if detail is None:
                detail = "assertion evaluated to false"
        __emit({{"index": index, "label": label, "status": "passed" if ok else "failed",
                "error_type": None if ok else "assertion_failure", "error": None if ok else str(detail)}})
    except BaseException as exc:
        __emit({{"index": index, "label": label, "status": "error", "error_type": "runtime_error",
                "error": __exception_text(exc)}})

try:
    __namespace = {{}}
    exec(compile(__payload["code"], "<generated_solution>", "exec"), __namespace, __namespace)
    __spec = __payload["spec"]
    if __spec["kind"] == "sample":
        for __index, __case in enumerate(__spec["cases"], 1):
            __label = __payload["labels"][__index - 1]
            try:
                __actual = eval(__case["expression"], __namespace, __namespace)
                __ok = __actual == __case["expected"]
                __ok = bool(__ok)
                __detail = None if __ok else f"expected {{__case['expected']!r}}, got {{__actual!r}}"
                __emit({{"index": __index, "label": __label, "status": "passed" if __ok else "failed",
                        "error_type": None if __ok else "assertion_failure", "error": __detail}})
            except BaseException as __exc:
                __emit({{"index": __index, "label": __label, "status": "error", "error_type": "runtime_error",
                        "error": __exception_text(__exc)}})
    elif __spec["kind"] == "assertions":
        exec(compile(__spec["setup_code"], "<test_setup>", "exec"), __namespace, __namespace)
        for __index, __case in enumerate(__spec["cases"], 1):
            __label = __payload["labels"][__index - 1]
            try:
                exec(compile(__case["code"], f"<mbpp_case_{{__index}}>", "exec"), __namespace, __namespace)
                __emit({{"index": __index, "label": __label, "status": "passed", "error_type": None, "error": None}})
            except AssertionError as __exc:
                __emit({{"index": __index, "label": __label, "status": "failed", "error_type": "assertion_failure",
                        "error": __exception_text(__exc) or "assertion failed"}})
            except BaseException as __exc:
                __emit({{"index": __index, "label": __label, "status": "error", "error_type": "runtime_error",
                        "error": __exception_text(__exc)}})
    elif __spec["kind"] == "humaneval":
        __candidate = __namespace.get(__payload["entry_point"])
        if not callable(__candidate):
            raise RuntimeError(f"Generated solution does not define callable entry point {{__payload['entry_point']!r}}")
        __test_namespace = {{"__builtins__": __builtins__, "__nlp_eval_record_assert__": __record_assert}}
        exec(compile(__spec["source"], "<humaneval_test>", "exec"), __test_namespace, __test_namespace)
        __test_namespace["check"](__candidate)
    else:
        raise RuntimeError("Unexpected canonical test kind")
except BaseException as __exc:
    __kind = "runtime_error"
    for __index, __label in enumerate(__payload["labels"], 1):
        if __index not in __seen:
            __emit({{"index": __index, "label": __label, "status": "error", "error_type": __kind,
                    "error": __exception_text(__exc)}})
'''


def run_tests(code, tests, entry_point=None, timeout=None, prompt=None, executor=None):
    """Execute canonical test cases and return per-case evidence.

    The default executor fails closed unless a genuine isolated backend is
    configured. ``host-dev`` must be explicitly selected and is not a sandbox.
    """
    spec, labels, invalid_reason = _prepare(tests, entry_point)
    if invalid_reason:
        return _result("invalid_tests", None, None, None, 0, None,
                       errors=[{"error_type": "invalid_tests", "error": invalid_reason}])

    total = len(labels)
    execution_code = code
    if spec["kind"] == "humaneval":
        execution_code = _human_eval_execution_code(code, prompt, entry_point)
    try:
        compile(execution_code, "<generated_solution>", "exec")
    except (SyntaxError, TypeError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        active_executor = executor or select_executor()
        if getattr(active_executor, "name", "") == "safe-unavailable":
            return _result(
                "executor_unavailable", None, None, None, total, None,
                errors=[{"error_type": "syntax_error", "error": message}],
                execution_status="executor_unavailable", executor_backend=active_executor.name,
                executor_metadata={"isolation": "unavailable"},
                execution_error="No verified isolated execution backend is installed or configured.",
            )
        outcomes = _failed_cases(labels, "syntax_error", message)
        return _result("syntax_error", False, 0, total, total, 0.0, outcomes,
                       [item for item in outcomes])

    if timeout is None:
        timeout = getattr(settings, "CODE_EXECUTION_TIMEOUT_SECONDS", 5)
    executor = executor or select_executor()
    with tempfile.TemporaryDirectory(prefix="llm_correctness_") as directory:
        result_path = Path(directory) / "test_results.jsonl"
        harness_path = Path(directory) / "correctness_harness.py"
        harness_path.write_text(
            _harness_source(execution_code, spec, entry_point, labels, result_path), encoding="utf-8"
        )
        execution = executor.execute(harness_path, timeout)
        timed_out = execution.execution_status == "timeout"
        stdout, stderr, returncode = execution.stdout, execution.stderr, execution.returncode
        if execution.execution_status == "executor_unavailable":
            return _result(
                "executor_unavailable", None, None, None, total, None,
                errors=[{"error_type": "executor_unavailable", "error": execution.error}],
                execution_status="executor_unavailable", executor_backend=execution.executor_backend,
                executor_metadata=execution.metadata, execution_error=execution.error,
            )
        if execution.execution_status == "policy_rejected":
            return _result(
                "policy_rejected", None, None, None, total, None,
                errors=[{"error_type": "policy_rejected", "error": execution.error}],
                execution_status="policy_rejected", executor_backend=execution.executor_backend,
                executor_metadata=execution.metadata, execution_error=execution.error,
            )
        if execution.execution_status == "infrastructure_error":
            return _result(
                "infrastructure_error", None, None, None, total, None,
                errors=[{"error_type": "infrastructure_error", "error": execution.error}],
                stdout=stdout, stderr=stderr, execution_status="infrastructure_error",
                executor_backend=execution.executor_backend, executor_metadata=execution.metadata,
                execution_error=execution.error,
            )

        outcomes = []
        if result_path.exists():
            try:
                for line in result_path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    if isinstance(item, dict) and isinstance(item.get("index"), int):
                        outcomes.append(item)
            except (OSError, json.JSONDecodeError) as exc:
                error = f"Invalid harness result file: {exc}"
                return _result("infrastructure_error", None, None, None, total, None,
                               errors=[{"error_type": "infrastructure_error", "error": error}],
                               stdout=stdout, stderr=stderr, execution_status="infrastructure_error",
                               executor_backend=execution.executor_backend,
                               executor_metadata=execution.metadata, execution_error=error)

        recorded = {item["index"] for item in outcomes}
        if timed_out:
            for index, label in enumerate(labels, 1):
                if index not in recorded:
                    outcomes.append({"index": index, "label": label, "status": "timeout",
                                     "error_type": "timeout", "error": f"Execution exceeded {timeout} seconds"})
        elif returncode not in (0, None) and not outcomes:
            return _result("infrastructure_error", None, None, None, total, None,
                           errors=[{"error_type": "infrastructure_error", "error": f"Harness exited with code {returncode}"}],
                           stdout=stdout, stderr=stderr, execution_status="infrastructure_error",
                           executor_backend=execution.executor_backend,
                           executor_metadata=execution.metadata,
                           execution_error=f"Harness exited with code {returncode}")
        elif len(recorded) < total:
            for index, label in enumerate(labels, 1):
                if index not in recorded:
                    outcomes.append({"index": index, "label": label, "status": "error",
                                     "error_type": "runtime_error", "error": "Harness did not produce an outcome"})

        outcomes.sort(key=lambda item: item["index"])
        if len(outcomes) != total:
            return _result("infrastructure_error", None, None, None, total, None, outcomes,
                           [{"error_type": "infrastructure_error", "error": "Incomplete or duplicate test outcomes"}],
                           stdout, stderr, timed_out, "infrastructure_error", execution.executor_backend,
                           execution.metadata, "Incomplete or duplicate test outcomes")

        passed_count = sum(item.get("status") == "passed" for item in outcomes)
        failed_count = total - passed_count
        score = round(100.0 * passed_count / total, 2) if total else None
        errors = [item for item in outcomes if item.get("status") != "passed"]
        if timed_out:
            status = "timeout"
        elif any(item.get("error_type") == "runtime_error" for item in errors):
            status = "runtime_error"
        elif failed_count:
            status = "failed"
        else:
            status = "passed"
        return _result(status, status == "passed", passed_count, failed_count, total, score,
                       outcomes, errors, stdout, stderr, timed_out, execution.execution_status,
                       execution.executor_backend, execution.metadata, execution.error)
