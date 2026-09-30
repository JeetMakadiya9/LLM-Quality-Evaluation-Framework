import subprocess
import sys
import tempfile
from pathlib import Path

def maintainability_score(code: str):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "solution.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "radon", "mi", "-s", str(path)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = proc.stdout.strip()
            # Typical output contains: "solution.py - A (94.00)"
            import re
            match = re.search(r"\(([0-9.]+)\)", output)
            if match:
                mi = float(match.group(1))
                return {"maintainability_score": round(mi, 2), "maintainability_raw": output}
        except Exception:
            pass

    # Lightweight fallback if Radon cannot parse/run.
    lines = [x for x in code.splitlines() if x.strip()]
    comments = sum(1 for x in lines if x.strip().startswith("#"))
    length_penalty = min(30, max(0, len(lines) - 20))
    score = max(0, 85 + min(10, comments) - length_penalty)
    return {"maintainability_score": float(score), "maintainability_raw": "fallback"}
