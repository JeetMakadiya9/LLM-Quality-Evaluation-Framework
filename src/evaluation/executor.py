"""Correctness execution backends.

``local-test`` runs only the generated solution plus the existing project harness
in an ordinary local Python subprocess. It is controlled execution, not a sandbox.
"""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Protocol
from src.config import settings

@dataclass
class ExecutionResult:
    execution_status: str
    executor_backend: str
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    error: str | None = None
    timeout: bool = False
    metadata: dict | None = None

class ExecutionBackend(Protocol):
    name: str
    def execute(self, harness_path, timeout) -> ExecutionResult: ...

class SafeExecutor:
    """Fail-closed placeholder for a genuine isolation backend."""
    name = "safe-unavailable"
    def execute(self, harness_path, timeout):
        return ExecutionResult(
            execution_status="executor_unavailable", executor_backend=self.name,
            error="No verified isolated execution backend is installed or configured.",
            metadata={"isolation": "unavailable"},
        )

class ControlledLocalExecutor:
    """Run the predefined evaluation harness locally; this is not a sandbox."""
    name = "local-test"

    def execute(self, harness_path, timeout):
        harness_path = Path(harness_path).resolve()
        metadata = {
            "isolation": "none",
            "execution_mode": "controlled-local-test",
            "shell": False,
            "working_directory": "temporary harness directory",
            "timeout_seconds": timeout,
            "executable": sys.executable,
        }
        try:
            process = subprocess.run(
                [sys.executable, str(harness_path)],
                executable=sys.executable,
                shell=False,
                cwd=str(harness_path.parent),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecutionResult(
                execution_status="completed", executor_backend=self.name,
                stdout=process.stdout, stderr=process.stderr,
                returncode=process.returncode, metadata=metadata,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return ExecutionResult(
                execution_status="timeout", executor_backend=self.name,
                stdout=stdout, stderr=stderr,
                error=f"Execution exceeded {timeout} seconds", timeout=True, metadata=metadata,
            )
        except OSError as exc:
            return ExecutionResult(
                execution_status="infrastructure_error", executor_backend=self.name,
                error=f"{type(exc).__name__}: {exc}", metadata=metadata,
            )

class HostDevelopmentExecutor(ControlledLocalExecutor):
    """Compatibility alias for the explicitly unsafe former host-dev mode."""
    name = "host-dev-unsafe"

def select_executor():
    """Select the configured backend without implicit fallback."""
    backend = getattr(settings, "CODE_EXECUTION_BACKEND", "local-test")
    if backend in {"local-test", "controlled-local"}:
        return ControlledLocalExecutor()
    if backend == "host-dev":
        return HostDevelopmentExecutor()
    return SafeExecutor()
