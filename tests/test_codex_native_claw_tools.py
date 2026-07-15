from __future__ import annotations

import json
from pathlib import Path

from harness_eval.harnesses.codex import CodexHarness
from harness_eval.harnesses.native_claw import native_claw_tools_enabled
from harness_eval.types import BenchmarkTask, ModelProfile


def _model() -> ModelProfile:
    return ModelProfile(
        name="deepseek_test",
        provider="deepseek",
        model="deepseek-test",
        base_url="http://127.0.0.1:9/v1",
        api_key_env="TEST_API_KEY",
        protocol="openai_chat",
    )


def _task(tmp_path: Path) -> BenchmarkTask:
    agent = tmp_path / "agent"
    agent.mkdir(parents=True)
    return BenchmarkTask(
        benchmark="openclaw",
        task_id="T_codex_proxy_mode",
        row={},
        prompt="Fetch the evidence with the official tool and answer.",
        workspace=tmp_path / "workspace",
        output_dir=tmp_path / "out",
        metadata={
            "live_tool_bridge": True,
            "live_tool_bridge_requested": True,
            "agent_workspace": str(agent),
            "claw_tool_bridge_url": "http://127.0.0.1:9999",
            "claw_sandbox_url": "http://127.0.0.1:8080",
            "claw_sandbox_mode": "official_docker",
            "allowed_tools": ["Bash", "web_fetch"],
            "allowed_tool_specs": [
                {"name": "Bash", "description": "Executes bash in the sandbox.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
                {"name": "web_fetch", "description": "Fetch full webpage content for a URL.", "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
            ],
            "timeout_seconds": 60,
            "max_turns": 5,
        },
    )


def test_live_bridge_no_longer_auto_bypasses_codex_cli(tmp_path):
    task = _task(tmp_path)

    assert native_claw_tools_enabled(task, {}) is False
    assert native_claw_tools_enabled(task, {"native_claw_tools": True}) is True


def test_codex_dry_run_uses_cli_with_model_tool_proxy(tmp_path):
    task = _task(tmp_path)
    harness = CodexHarness(
        {
            "harness": {
                "executable": "codex",
                "native_claw_tools": False,
                "model_tool_proxy": {"enabled": True},
                "timeout_s_per_instance": 60,
                "sandbox_mode": "danger-full-access",
                "live_sandbox_mode": "danger-full-access",
                "extra_args": ["--json"],
            }
        }
    )

    result = harness.run(task, _model(), dry_run=True)

    assert result.status == "dry_run"
    manifest = json.loads((task.output_dir / "harness_manifest.json").read_text(encoding="utf-8"))
    assert manifest["codex_tool_mode"] == "harness_cli_with_claw_model_proxy"
    assert manifest["native_harness_tools_disabled"] is False
    assert manifest["cmd"] is not None and "codex exec" in manifest["cmd"]
    assert manifest["model_visible_tools"] == ["Bash", "web_fetch"]
    assert manifest["model_tool_proxy"]["enabled"] is True
    assert manifest["model_tool_proxy"]["transport_tool"] == "exec_command"
    assert "model_tool_proxy=harness_cli_with_claw_model_proxy" in (task.output_dir / "DRY_RUN.txt").read_text(encoding="utf-8")
