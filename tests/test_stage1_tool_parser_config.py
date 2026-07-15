from pathlib import Path

import yaml

from swecodex_harness.run_codex_on_swebench import _codex_cmd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_config_is_portable_after_refactor():
    project_cfg = yaml.safe_load((PROJECT_ROOT / "configs/project.yaml").read_text())

    assert project_cfg["project"]["root"] == "."
    assert project_cfg["project"]["data_root"] == "data"
    assert project_cfg["project"]["runs_root"] == "runs"
    assert "codex" not in project_cfg


def test_model_registry_contains_required_profiles():
    profiles = yaml.safe_load((PROJECT_ROOT / "configs/models/models.yaml").read_text())["models"]

    for name in [
        "local_serverllm_qwen3_30b_moe",
        "local_sparsevllm_qwen3_30b_moe",
        "deepseek_v4_pro",
        "kimi_code_latest_stable",
        "minimax_m3",
        "mimo_v2_5_pro",
    ]:
        assert name in profiles
        assert profiles[name]["model"]
        assert profiles[name]["base_url"]


def test_harness_configs_are_separated_from_benchmark_configs():
    codex_cfg = yaml.safe_load((PROJECT_ROOT / "configs/harnesses/codex.yaml").read_text())
    openclaw_cfg = yaml.safe_load((PROJECT_ROOT / "configs/benchmarks/openclaw_claw_eval.yaml").read_text())

    assert codex_cfg["harness"]["name"] == "codex"
    assert openclaw_cfg["benchmark"]["name"] == "openclaw"
    assert "command_template" not in openclaw_cfg["benchmark"]


def test_stage1_launcher_uses_configured_tool_parser():
    launcher = (PROJECT_ROOT / "scripts/start_stage1_vllm.sh").read_text()

    assert 'cfg_value vllm.tool_call_parser "$STAGE1_CONFIG"' in launcher
    assert '--tool-call-parser "$TOOL_CALL_PARSER"' in launcher
    assert "--tool-call-parser qwen3_coder" not in launcher
    assert "--tool-call-parser qwen3_xml" not in launcher


def test_stage1_smoke_uses_configured_codex_base_url():
    smoke = (PROJECT_ROOT / "scripts/run_stage1_smoke.sh").read_text()

    assert 'BASE_URL="${BASE_URL:-$(cfg_value codex.base_url "$STAGE1_CONFIG")}"' in smoke
    assert "--base-url \"$BASE_URL\"" in smoke
    assert "--base-url http://127.0.0.1:8000/v1" not in smoke


def test_swebench_prompt_discourages_repeated_brittle_sed_edits():
    prompt = (PROJECT_ROOT / "configs/prompts/codex_swebench_prompt.md").read_text()

    assert "Reading and editing files inside this checkout does not require escalated permissions" in prompt
    assert "Before your final answer, inspect the actual patch with `git diff --check`" in prompt
    assert "python -m py_compile" in prompt
    assert "Before your final answer, you must leave at least one source-file change in the working tree" in prompt
    assert "Your first action must be a shell command" in prompt
    assert "/think" not in prompt
    assert "Do not try to use `apply_patch`" in prompt
    assert "Never request escalated permissions or command approval" in prompt
    assert "Use shell commands to inspect and edit files" in prompt
    assert "Prefer a short Python heredoc for file edits" in prompt
    assert "never use `sed -i` for source changes" in prompt
    assert "splitlines(keepends=True)" in prompt
    assert "out` list" in prompt
    assert "Path(...).read_text()" in prompt
    assert "text.replace(old, new, 1)" in prompt
    assert "triple-quoted strings" in prompt
    assert "never insert executable statements into a multi-line function signature" in prompt
    assert "Do not use `python -c` for source edits" in prompt
    assert "python - <<'PY'" in prompt
    assert "Never combine `python - <<'PY'` with shell input redirection" in prompt
    assert "Every Python edit script must track whether it changed the file" in prompt
    assert "raise SystemExit(\"edit target not found\")" in prompt
    assert "Do not add any second heredoc marker" in prompt
    assert "let me run" in prompt
    assert "Do not repeat a failed shell command unchanged" in prompt
    assert "inspect numbered context with `nl -ba`" in prompt
    assert "line-list Python heredoc edit" in prompt
    assert "{{FAIL_TO_PASS}}" not in prompt
    assert "{{PASS_TO_PASS}}" not in prompt
    assert "{{TEST_PATCH}}" not in prompt
    assert "FAIL_TO_PASS" not in prompt
    assert "PASS_TO_PASS" not in prompt


def test_codex_command_uses_supported_exec_flags_only(tmp_path):
    cfg = {
        "codex": {
            "executable": "codex",
            "extra_args": ["--json"],
            "sandbox_mode": "workspace-write",
            "approval_policy": "never",
        },
        "model": {"served_model_name": "qwen3-30b-a3b"},
    }

    cmd = _codex_cmd(cfg, tmp_path, "do work", tmp_path / "final.txt")

    assert "--sandbox" in cmd
    assert "--ask-for-approval" not in cmd
