from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harness_eval.harnesses.nanobot import NanobotHarness
from harness_eval.harnesses.openclaw import OpenClawHarness
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


def _task(tmp_path: Path, harness_name: str) -> BenchmarkTask:
    agent = tmp_path / harness_name / "agent"
    agent.mkdir(parents=True)
    return BenchmarkTask(
        benchmark="openclaw",
        task_id=f"T_proxy_mode_{harness_name}",
        row={},
        prompt="Fetch the evidence with the official tool and answer.",
        workspace=tmp_path / f"{harness_name}_workspace",
        output_dir=tmp_path / harness_name / "out",
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


@pytest.mark.parametrize(
    "harness_cls,transport_tool,argument_key",
    [(NanobotHarness, "exec", "command"), (OpenClawHarness, "exec", "command")],
)
def test_external_harness_dry_run_uses_cli_with_model_tool_proxy(tmp_path, harness_cls, transport_tool, argument_key):
    harness = harness_cls(
        {
            "harness": {
                "native_claw_tools": False,
                "timeout_s_per_instance": 60,
                "command_template": f"{sys.executable} -c 'print(\"cli would run\")'",
                "model_tool_proxy": {"enabled": True, "transport_tool_name": transport_tool, "transport_argument_key": argument_key},
            }
        }
    )
    task = _task(tmp_path, harness.name)

    result = harness.run(task, _model(), dry_run=True)

    assert result.status == "dry_run"
    manifest = json.loads((task.output_dir / "harness_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool_mode"] == "harness_cli_with_claw_model_proxy"
    assert manifest["native_harness_tools_disabled"] is False
    assert manifest["cmd"] is not None and "cli would run" in manifest["cmd"]
    assert manifest["model_visible_tools"] == ["Bash", "web_fetch"]
    assert manifest["model_tool_proxy"]["enabled"] is True
    assert manifest["model_tool_proxy"]["transport_tool"] == transport_tool


def test_openclaw_proxy_default_uses_harness_accepted_bare_model_arg(tmp_path):
    harness = OpenClawHarness({"harness": {"native_claw_tools": False, "timeout_s_per_instance": 60, "model_tool_proxy": {"enabled": True}}})
    task = _task(tmp_path, harness.name)

    result = harness.run(task, _model(), dry_run=True)

    manifest = json.loads((task.output_dir / "harness_manifest.json").read_text(encoding="utf-8"))
    assert result.status == "dry_run"
    assert "--profile clawh-" in manifest["cmd"]
    assert "--model claw_proxy/deepseek-test" in manifest["cmd"]
    assert "--model openai/deepseek-test" not in manifest["cmd"]
    profile_configs = list((task.output_dir / "openclaw_home").glob(".openclaw-*/openclaw.json"))
    assert len(profile_configs) == 1
    profile_config = json.loads(profile_configs[0].read_text(encoding="utf-8"))
    profile_provider = profile_config["models"]["providers"]["claw_proxy"]
    assert profile_provider["baseUrl"] == "http://127.0.0.1:9/v1"
    assert profile_provider["apiKey"] == "dummy"
    assert "name" not in profile_provider
    assert profile_provider["models"][0]["id"] == "deepseek-test"
    catalogs = list((task.output_dir / "openclaw_home").glob(".openclaw-*/agents/openclaw/agent/models.json"))
    assert len(catalogs) == 1
    catalog = json.loads(catalogs[0].read_text(encoding="utf-8"))
    provider = catalog["providers"]["claw_proxy"]
    assert provider["baseUrl"] == "http://127.0.0.1:9/v1"
    assert provider["apiKey"] == "dummy"
    assert provider["models"][0]["id"] == "deepseek-test"
    assert manifest["model_tool_proxy"]["transport_tool"] == "exec"


def test_nanobot_proxy_renders_per_task_provider_config(tmp_path):
    template = tmp_path / "nanobot_template.json"
    template.write_text(json.dumps({"providers": {"deepseek": {"api_base": "http://127.0.0.1:8012/v1"}}, "agents": {"defaults": {"provider": "deepseek"}}}), encoding="utf-8")
    harness = NanobotHarness({"harness": {"config_template": str(template), "model_tool_proxy": {"enabled": True}}})
    task = _task(tmp_path, harness.name)

    config_path = harness._render_nanobot_config(task, _model())
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config_path == task.output_dir / "nanobot_model_proxy_config.json"
    assert config["providers"]["deepseek"]["api_base"] == "http://127.0.0.1:9/v1"
    assert config["agents"]["defaults"]["model"] == "deepseek-test"
    assert config["tools"]["restrict_to_workspace"] is True
