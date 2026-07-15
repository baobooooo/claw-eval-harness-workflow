from __future__ import annotations

from pathlib import Path
from typing import Any

from harness_eval.io import run_cmd


def get_patch(workspace: Path) -> str:
    if not (workspace / ".git").exists():
        return ""
    run_cmd(["git", "add", "-N", "."], cwd=workspace, timeout=120, check=False)
    res = run_cmd(["git", "diff", "--binary"], cwd=workspace, timeout=120, check=False)
    run_cmd(["git", "reset", "--quiet"], cwd=workspace, timeout=120, check=False)
    return res.stdout


def status_short(workspace: Path) -> str:
    if not (workspace / ".git").exists():
        return ""
    return run_cmd(["git", "status", "--short"], cwd=workspace, timeout=60, check=False).stdout


def validate_patch(workspace: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"enabled": bool(cfg.get("validate_generated_patch", True)), "ok": True}
    if not result["enabled"] or not (workspace / ".git").exists():
        return result
    diff_check = run_cmd(["git", "diff", "--check"], cwd=workspace, timeout=120, check=False)
    result["git_diff_check"] = {"returncode": diff_check.returncode, "stdout": diff_check.stdout[-4000:], "stderr": diff_check.stderr[-4000:]}
    if diff_check.returncode != 0:
        result["ok"] = False
    names = run_cmd(["git", "diff", "--name-only", "--diff-filter=ACMRT"], cwd=workspace, timeout=120, check=False)
    changed = [line.strip() for line in names.stdout.splitlines() if line.strip()]
    result["changed_files"] = changed
    py_files = [p for p in changed if p.endswith(".py") and (workspace / p).exists()]
    if py_files and cfg.get("py_compile_changed_python", True):
        import sys

        pyc = run_cmd([sys.executable, "-m", "py_compile", *py_files], cwd=workspace, timeout=120, check=False)
        result["python_compile"] = {"returncode": pyc.returncode, "stdout": pyc.stdout[-4000:], "stderr": pyc.stderr[-4000:]}
        if pyc.returncode != 0:
            result["ok"] = False
    return result
