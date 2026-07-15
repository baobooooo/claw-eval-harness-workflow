import subprocess
from pathlib import Path

from swecodex_harness.run_codex_on_swebench import _build_codex_retry_prompt, _validate_worktree_patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_validate_worktree_patch_rejects_syntax_invalid_python(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Patch Validation")
    _git(repo, "config", "user.email", "patch-validation@example.com")
    target = repo / "module.py"
    target.write_text("def ok():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "module.py")
    _git(repo, "commit", "-m", "baseline")

    target.write_text("def broken(:\n    return 1\n", encoding="utf-8")

    result = _validate_worktree_patch(
        repo,
        {"agent": {"validate_generated_patch": True, "py_compile_changed_python": True}},
    )

    assert result["enabled"] is True
    assert result["ok"] is False
    assert result["git_diff_check"]["returncode"] == 0
    assert result["python_compile"]["returncode"] != 0
    assert "SyntaxError" in result["python_compile"]["stderr"]


def test_zero_patch_status_is_not_labeled_ok():
    runner = (PROJECT_ROOT / "src/swecodex_harness/run_codex_on_swebench.py").read_text()

    assert "zero_patch_git_worktree" in runner
    assert "ok_zero_patch_git_worktree" not in runner


def test_codex_retry_prompt_requires_worktree_edit():
    prompt = _build_codex_retry_prompt("Base task", "git diff --binary was empty", 2)

    assert "Base task" in prompt
    assert "git diff --binary was empty" in prompt
    assert "make the source edit with a shell command" in prompt
    assert "git diff --stat" in prompt
    assert "do not use `sed -i`" in prompt
    assert "splitlines(keepends=True)" in prompt
    assert "Do not use `python -c`" in prompt


def test_stage1_config_allows_codex_generation_retry():
    config = (PROJECT_ROOT / "configs/project.yaml").read_text()

    assert "codex_generation_attempts: 4" in config
