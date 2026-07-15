import json
import sys
from pathlib import Path

from harness_eval.benchmarks import make_benchmark
from harness_eval.benchmarks.openclaw import OpenClawBenchmark
from harness_eval.harnesses.base import task_timeout_s
from harness_eval.harnesses.codex import CodexHarness
from harness_eval.harnesses.external_cli import ExternalCliHarness
from harness_eval.harnesses import make_harness
from harness_eval.io import load_yaml
from harness_eval.models import resolve_model
from harness_eval.run import _assign_row_indexes, load_run_config
from harness_eval.types import BenchmarkTask, HarnessResult, ModelProfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_env_default_expansion_and_model_resolution(monkeypatch):
    monkeypatch.delenv("SERVERLLM_BASE_URL", raising=False)
    profile = resolve_model("local_serverllm_qwen3_30b_moe", PROJECT_ROOT / "configs/models/models.yaml")

    assert profile.model == "qwen3-30b-a3b"
    assert profile.base_url == "http://127.0.0.1:8000/v1"
    assert profile.api_key_env == "VLLM_API_KEY"


def test_factory_layers_are_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    cfg = load_run_config(
        "configs/project.yaml",
        "configs/benchmarks/openclaw_sample.yaml",
        "configs/harnesses/nanobot.yaml",
    )
    benchmark = make_benchmark("claw-eval", cfg, tmp_path / "run")
    harness = make_harness("nanobot", cfg)

    assert benchmark.name == "openclaw"
    assert harness.name == "nanobot"
    assert cfg["project"]["root"] == str(PROJECT_ROOT.resolve())


def test_openclaw_sample_can_prepare_a_task_without_network(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    cfg = load_run_config(
        "configs/project.yaml",
        "configs/benchmarks/openclaw_sample.yaml",
        "configs/harnesses/codex.yaml",
    )
    benchmark = make_benchmark("openclaw", cfg, tmp_path / "run")
    rows = benchmark.load_rows(max_instances=1)
    task = benchmark.prepare_task(rows[0])

    assert task.task_id == "sample_echo_zh"
    assert task.workspace.exists()
    assert "hello harness" in task.prompt


def test_external_cli_timeout_returns_result_instead_of_raising(tmp_path):
    workspace = tmp_path / "workspace"
    output_dir = tmp_path / "output"
    workspace.mkdir()
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="timeout_case",
        row={},
        prompt="timeout prompt",
        workspace=workspace,
        output_dir=output_dir,
    )
    model = ModelProfile(
        name="test",
        provider="test",
        model="test-model",
        base_url="http://127.0.0.1:9/v1",
        api_key_env="TEST_API_KEY",
    )
    harness = ExternalCliHarness(
        {
            "harness": {
                "command_template": f"{sys.executable} -c \"import time; time.sleep(5)\"",
                "timeout_s_per_instance": 0.1,
            }
        }
    )

    result = harness.run(task, model)

    assert result.status == "timeout"
    assert result.returncode == -124
    assert result.error == "External CLI run timed out."
    assert (output_dir / "harness_manifest.json").exists()


def test_live_prompts_are_not_prefixed_with_bridge_contracts(tmp_path):
    workspace = tmp_path / "workspace"
    output_dir = tmp_path / "output"
    workspace.mkdir()
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="prompt_contract",
        row={},
        prompt="Original MiniHarness-visible prompt.",
        workspace=workspace,
        output_dir=output_dir,
        metadata={
            "live_tool_bridge": True,
            "live_tool_bridge_requested": True,
            "claw_tool_bridge_url": "http://127.0.0.1:1",
            "allowed_tools": ["Bash"],
            "helper_files": ["claw_tool", "claw_bash"],
        },
    )

    assert CodexHarness({"harness": {}})._prompt_for_codex(task) == task.prompt
    assert ExternalCliHarness({"harness": {}})._prompt_for_cli(task) == task.prompt


def test_task_timeout_prefers_official_task_environment():
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="budget_case",
        row={},
        prompt="budget prompt",
        workspace=PROJECT_ROOT,
        output_dir=PROJECT_ROOT,
        metadata={"timeout_seconds": 1800},
    )

    assert task_timeout_s(task, {"timeout_s_per_instance": 300}) == 1800.0


def test_task_timeout_falls_back_to_harness_config():
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="budget_case",
        row={},
        prompt="budget prompt",
        workspace=PROJECT_ROOT,
        output_dir=PROJECT_ROOT,
    )

    assert task_timeout_s(task, {"timeout_s_per_instance": 300}) == 300.0


def test_run_assigns_selected_order_indexes_without_mutating_rows():
    rows = [{"task_id": "a"}, {"task_id": "b", "_harness_eval_row_index": 99}]

    indexed = _assign_row_indexes(rows)

    assert indexed == [
        {"task_id": "a", "_harness_eval_row_index": 0},
        {"task_id": "b", "_harness_eval_row_index": 1},
    ]
    assert rows == [{"task_id": "a"}, {"task_id": "b", "_harness_eval_row_index": 99}]



def test_openclaw_finalize_collects_env_snapshot_after_agent_loop(tmp_path):
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "T_snapshot"
    task_dir.mkdir(parents=True)
    (task_dir / "input.txt").write_text("visible input\n", encoding="utf-8")
    (task_dir / "grader.sh").write_text("printf graded > result.txt\n", encoding="utf-8")
    (task_dir / "answer.txt").write_text("secret answer\n", encoding="utf-8")
    (task_dir / "task.yaml").write_text(
        """
task_id: T_snapshot
task_name: Snapshot Fixture
prompt:
  text: Use the input file.
  language: en
sandbox_files:
  - input.txt
sandbox_grader_files:
  - grader.sh
env_snapshot_commands:
  - bash grader.sh
env_snapshot_files:
  - result.txt
local_grader_files:
  - answer.txt
environment:
  timeout_seconds: 60
  max_turns: 3
  env_snapshot_timeout: 5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    prompt_template = tmp_path / "prompt.md"
    prompt_template.write_text("{{TASK_ID}}\n{{QUERY}}\n{{CLAW_EVAL_TOOLS}}\n", encoding="utf-8")
    cfg = {
        "project": {"root": str(tmp_path)},
        "benchmark": {
            "tasks_dir": str(tasks_dir),
            "fixture_root": str(tmp_path / "fixtures"),
            "prompt_template": str(prompt_template),
        },
    }
    benchmark = OpenClawBenchmark(cfg, tmp_path / "run")
    task = benchmark.prepare_task({"task_id": "T_snapshot", "query": "read input", "language": "en"})

    assert (task.workspace / "input.txt").exists()
    assert not (task.workspace / "grader.sh").exists()
    assert not (task.workspace / "answer.txt").exists()

    result = HarnessResult(task_id="T_snapshot", harness="codex", model="m", status="ok")
    result = benchmark.finalize_task_result(result, task)

    snapshot_path = Path(task.metadata["env_snapshot_path"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert (task.workspace / "grader.sh").exists()
    assert snapshot["cmd:bash grader.sh"]["exit_code"] == 0
    assert snapshot["file:result.txt"]["content"] == "graded"
    assert snapshot["local_file:answer.txt"]["content"] == "secret answer\n"
    assert result.metrics["env_snapshot"]["entries"] >= 4


def test_openclaw_evaluate_auto_converts_external_traces_before_grading(tmp_path):
    checker = tmp_path / "check_eval_input.py"
    checker.write_text(
        """
import json
import pathlib
import json
import sys
pred = pathlib.Path(sys.argv[1])
raw = pathlib.Path(sys.argv[2])
score_ready_dir = pathlib.Path(sys.argv[3])
assert pred.name == 'score_ready_predictions.jsonl'
assert raw.name == 'harness_predictions.jsonl'
assert pred.exists()
rows = [json.loads(line) for line in pred.read_text(encoding='utf-8').splitlines() if line.strip()]
assert rows and rows[0]['converted_trace_schema'] == 'claw_eval_minimal_v1'
assert pathlib.Path(rows[0]['trace_path']).exists()
assert score_ready_dir.exists()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    benchmark = OpenClawBenchmark(
        {
            "project": {"root": str(PROJECT_ROOT)},
            "benchmark": {
                "eval_command": f"{sys.executable} {checker} {{predictions}} {{raw_predictions}} {{score_ready_dir}}",
            },
        },
        run_dir,
    )
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "harness_predictions.jsonl").write_text(
        json.dumps(
            {
                "task_id": "T_eval",
                "harness": "nanobot",
                "model": "m",
                "query": "q",
                "final_message": "answer",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    ev = benchmark.evaluate(timeout_s=10)

    assert ev.status == "ok"
    assert ev.manifest["predictions_path"].endswith("score_ready_predictions.jsonl")
    assert ev.manifest["conversion"]["num_rows"] == 1



def test_service_ready_timeout_minimum_and_no_reuse(tmp_path):
    benchmark = OpenClawBenchmark(
        {
            "project": {"root": str(tmp_path)},
            "benchmark": {
                "service_ready_timeout_default": 30,
                "service_ready_timeout_min": 30,
                "service_ready_timeout_max": 120,
                "reuse_healthy_services": False,
                "kill_existing_services_on_port": True,
            },
        },
        tmp_path / "run",
    )
    service = {"name": "svc", "port": 9999, "ready_timeout": 5}

    assert benchmark._service_ready_timeout(service) == 30
    assert benchmark._reuse_healthy_services() is False


def test_live_bridge_is_safe_default_unless_disabled(tmp_path):
    benchmark = OpenClawBenchmark({"project": {"root": str(tmp_path)}, "benchmark": {}}, tmp_path / "run")
    legacy = OpenClawBenchmark({"project": {"root": str(tmp_path)}, "benchmark": {"live_tool_bridge": False}}, tmp_path / "run2")

    assert benchmark._live_bridge_enabled() is True
    assert legacy._live_bridge_enabled() is False


def test_openclaw_live_command_starts_from_driver_workspace_without_positional_arg(tmp_path):
    from harness_eval.harnesses.openclaw import OpenClawHarness

    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="Tcli",
        row={},
        prompt="prompt",
        workspace=tmp_path / "shadow_workspace",
        output_dir=tmp_path / "out",
        metadata={"live_tool_bridge_requested": True, "agent_workspace": str(tmp_path / "driver_workspace")},
    )
    task.output_dir.mkdir(parents=True, exist_ok=True)
    model = ModelProfile(name="m", provider="p", model="deepseek-v4-pro", base_url="http://127.0.0.1/v1", api_key_env="K")
    harness = OpenClawHarness({"harness": {}})
    command = harness._format_command(task, model, tmp_path / "stdout", tmp_path / "stderr", tmp_path / "trace.jsonl")

    assert command.startswith("cd ")
    assert str(tmp_path / "driver_workspace") in command.split(" && ", 1)[0]
    assert "openclaw agent --local" in command
    assert f"agent {tmp_path / 'driver_workspace'}" not in command
    assert "--cwd" not in command


def test_external_cli_marks_provider_quota_even_when_cli_exits_zero(tmp_path):
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="Tquota",
        row={},
        prompt="prompt",
        workspace=tmp_path / "workspace",
        output_dir=tmp_path / "out",
        metadata={"agent_workspace": str(tmp_path / "workspace")},
    )
    task.workspace.mkdir(parents=True)
    task.output_dir.mkdir(parents=True)
    model = ModelProfile(name="m", provider="p", model="m", base_url="http://127.0.0.1/v1", api_key_env="K")
    harness = ExternalCliHarness(
        {
            "harness": {
                "command_template": "printf 'The AI provider rejected the request because the API key is out of quota or the account is in arrears.'",
                "timeout_s_per_instance": 30,
            }
        }
    )

    result = harness.run(task, model)

    assert result.status == "provider_quota_error"
    assert "out of quota" in (result.error or "")


def test_codex_live_mode_avoids_nested_read_only_bwrap_sandbox(tmp_path):
    from harness_eval.harnesses.codex import CodexHarness

    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="Tcodex",
        row={},
        prompt="prompt",
        workspace=tmp_path / "workspace",
        output_dir=tmp_path / "out",
        metadata={"live_tool_bridge": True, "agent_workspace": str(tmp_path / "driver_workspace")},
    )
    model = ModelProfile(name="m", provider="p", model="m", base_url="http://127.0.0.1/v1", api_key_env="K")
    harness = CodexHarness({"harness": {"executable": "codex", "sandbox_mode": "danger-full-access", "extra_args": ["--json"], "allow_native_tool_bypass": False}})
    command = harness._command(task, model, tmp_path / "final.txt")

    assert "--sandbox" in command
    assert "danger-full-access" in command
    assert "read-only" not in command
