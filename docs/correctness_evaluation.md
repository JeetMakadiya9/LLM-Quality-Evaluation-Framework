# Correctness Evaluation Contract

## Task representation

`src.dataset.loader.load_problems()` adapts the selected CSV into a task record
containing `task_id` (also exposed as the existing `problem_id` alias),
`dataset`, `prompt`, `entry_point`, `tests`, and `metadata`. The `tests` value is
a canonical specification rather than a raw CSV cell:

- `sample`: JSON expression/expected-value cases.
- `humaneval`: benchmark test source and the task entry point.
- `assertions`: MBPP setup source and individual primary/challenge assertions.
- `invalid`: an adapter-detected malformed or unsupported specification.

The direct MBPP CSV adapter reads `test_setup_code` and
`challenge_test_list`. The existing checked-in `programming_dataset.csv` does
not contain those columns, so its MBPP rows can only use the primary assertions
present in `test`. The dataset builder now retains both fields in future combined
exports; the checked-in dataset was not changed.

## Result contract

`src.evaluation.correctness.run_tests()` returns:

| Field | Meaning |
|---|---|
| `correctness_status` | `passed`, `failed`, `syntax_error`, `runtime_error`, `timeout`, `invalid_tests`, `executor_unavailable`, or `infrastructure_error`. |
| `passed` | `True` only when every valid case passed; `False` for generated-code/test execution failures; `None` when no valid score can be established. |
| `passed_tests` / `failed_tests` / `total_tests` | Counts over the canonical test cases. Errors and timeouts count as not-passed cases; infrastructure/invalid-spec results use `None` counts where no execution result is trustworthy. |
| `correctness_score` | `100 × passed_tests / total_tests`, rounded to two decimals. `None` when there are no valid tests or execution infrastructure prevented a trustworthy result. |
| `test_results` | Ordered per-case status, label, and error details. |
| `correctness_errors` | Non-passing test outcomes or evaluator/infrastructure error details. |
| `timeout`, `stdout`, `stderr` | Process timeout flag and captured output. |
| `execution_status` | Backend outcome: `completed`, `timeout`, `executor_unavailable`, or `infrastructure_error`. |
| `execution_error` | Infrastructure/backend detail when execution did not complete normally. |
| `executor_backend` / `executor_metadata` | Backend identity and reported controls/configuration. |

For HumanEval, the adapter instruments assertions inside the dataset's
`check(candidate)` function and invokes it with the generated entry-point
callable. Prompt imports are retained; a completion-only function body is
combined with the HumanEval prompt prefix, while a complete function solution
is executed with the prompt's imports. Assertion failures are recorded and evaluation continues to collect
case-level evidence; this means later assertions run even after an earlier
failure, so results are assertion-level measurements rather than an exact
replay of Python's default fail-fast assertion behavior.

## Execution architecture and security status

`run_tests()` builds the dataset-specific correctness harness and submits its
temporary file to the `ExecutionBackend` abstraction in
`src/evaluation/executor.py`. The default `CODE_EXECUTION_BACKEND=safe` uses a
fail-closed unavailable backend because this machine has no verified usable
container, VM, Windows Sandbox, or WSL distribution. It returns
`executor_unavailable`, leaves the correctness score unset, and does not launch
the harness. There is no fallback to host execution.

`CODE_EXECUTION_BACKEND=host-dev` explicitly selects
`HostDevelopmentExecutor`. This starts an ordinary local Python subprocess in
a temporary directory with a timeout and cleanup. It is **unsafe development
execution**, not a sandbox. It does not isolate processes, files, network,
privileges, CPU, memory, or process count; captured output is not bounded. Use
it only with controlled, deterministic fixtures on a disposable development
machine. Never use it for arbitrary model output.

No network or filesystem containment is implemented or verified. The safe
backend is currently unavailable, so network/filesystem/process security
fixtures are blocked before execution and do not constitute successful
containment tests. Timeout applies only to the development host subprocess and
does not prove descendant-process cleanup or other resource bounds.

To use a safe benchmark executor, install/configure a genuine isolation backend
that provides a private filesystem, denies network by default, applies
resource/process limits, and has reproducible setup. Then implement and verify
that backend behind `ExecutionBackend`; do not change the default to host mode.
Windows optional-feature inspection required elevation and was not available to
this user, so Windows Sandbox/Hyper-V feature state remains unverified.

Audit evidence on the development machine (Windows 10.0.26200.9457): `docker`,
`podman`, `WindowsSandbox.exe`, and the `Get-VM` command were not found; common
Docker and Podman executable paths and `WindowsSandbox.exe` were absent. `wsl`
exists, but `wsl --list --quiet` reported that WSL is not installed. Windows
optional-feature state could not be queried because the command required
elevation. `hcsdiag.exe` exists, but this alone does not establish a usable
container runtime. Hyper-V service/query evidence was unavailable to the
current user. AppContainer and Windows Job Object containment have not been
implemented or verified; a Job Object alone would not supply filesystem and
network isolation.

For controlled fixture development only, set `CODE_EXECUTION_BACKEND=host-dev`
in the process environment. The application default remains `safe`.

## Persistence

SQLite keeps the existing score columns and adds task/dataset/entry-point,
status/count, serialized per-test result/error, timeout, and metadata columns.
`init_db()` adds missing columns to an existing table without dropping its
rows. The existing `save_results()` behavior still replaces the current
evaluation rows on each run.
