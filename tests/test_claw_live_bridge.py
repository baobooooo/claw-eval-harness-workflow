import json
from pathlib import Path

from harness_eval.analysis.trace_conversion import convert_prediction_rows
from harness_eval.analysis.tool_policy import enforce_tool_policy
from harness_eval.benchmarks.openclaw import OpenClawBenchmark
from harness_eval.harnesses.external_cli import ExternalCliHarness
from harness_eval.types import HarnessResult, ModelProfile


def _model() -> ModelProfile:
    return ModelProfile(
        name="test",
        provider="test",
        model="test-model",
        base_url="http://127.0.0.1:9/v1",
        api_key_env="TEST_API_KEY",
    )


def _make_live_task(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "T_live"
    task_dir.mkdir(parents=True)
    (task_dir / "input.txt").write_text("visible only through bridge\n", encoding="utf-8")
    (task_dir / "grader.sh").write_text("cat /workspace/out.txt > /workspace/result.txt\n", encoding="utf-8")
    (task_dir / "answer.txt").write_text("visible only through bridge\n", encoding="utf-8")
    (task_dir / "task.yaml").write_text(
        """
task_id: T_live
task_name: Live Bridge Fixture
prompt:
  text: Read /workspace/input.txt and write /workspace/out.txt.
  language: en
sandbox_files:
  - input.txt
sandbox_grader_files:
  - grader.sh
env_snapshot_commands:
  - bash /workspace/grader.sh
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
    benchmark = OpenClawBenchmark(
        {
            "project": {"root": str(tmp_path)},
            "benchmark": {
                "tasks_dir": str(tasks_dir),
                "fixture_root": str(tmp_path / "fixtures"),
                "prompt_template": str(prompt_template),
                "live_tool_bridge": True,
                "allow_host_sandbox_fallback": True,
            },
        },
        tmp_path / "run",
    )
    task = benchmark.prepare_task({"task_id": "T_live", "query": "use bridge", "language": "en"})
    return benchmark, task


def test_live_bridge_routes_tools_to_sandbox_and_writes_trace(tmp_path):
    benchmark, task = _make_live_task(tmp_path)
    model = _model()
    harness = ExternalCliHarness(
        {
            "harness": {
                "command_template": "./claw_tool Bash @payload.json && printf 'done through bridge' > {output_dir}/final_message.txt",
                "native_claw_tools": False,
                "timeout_s_per_instance": 30,
            }
        }
    )
    # The external harness should not use the shadow task workspace as cwd.
    assert Path(task.metadata["agent_workspace"]) != task.workspace
    assert not Path(task.metadata["agent_workspace"]).exists()

    with benchmark.task_run_context(task, model):
        agent_workspace = Path(task.metadata["agent_workspace"])
        assert agent_workspace.exists()
        assert (agent_workspace / "claw_tool").exists()
        assert not (agent_workspace / "input.txt").exists()
        (agent_workspace / "payload.json").write_text(
            json.dumps({"command": "cat /workspace/input.txt > /workspace/out.txt"}),
            encoding="utf-8",
        )
        result = harness.run(task, model)
        result = benchmark.finalize_task_result(result, task)
        result = enforce_tool_policy(result, task)

    assert result.status == "ok"
    assert result.trace_path and result.trace_path.endswith("claw_live_trace.jsonl")
    events = [json.loads(line) for line in Path(result.trace_path).read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events].count("tool_dispatch") == 1
    dispatch = next(event for event in events if event["type"] == "tool_dispatch")
    assert dispatch["tool_name"] == "Bash"
    assert "/exec" in dispatch["endpoint_url"]
    assert events[-1]["type"] == "trace_end"

    snapshot = json.loads(Path(task.metadata["env_snapshot_path"]).read_text(encoding="utf-8"))
    assert snapshot["file:result.txt"]["content"] == "visible only through bridge\n"
    assert snapshot["local_file:answer.txt"]["content"] == "visible only through bridge\n"
    assert result.metrics["tool_policy"]["enforced"] is True
    assert result.metrics["tool_policy"]["live_tool_bridge"] is True


def test_trace_conversion_rewrites_live_claw_trace_to_miniharness_shape(tmp_path):
    trace = tmp_path / "live.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps({"type": "trace_start", "trace_id": "x", "task_id": "T", "model": "m"}),
                json.dumps({"type": "message", "trace_id": "x", "message": {"role": "user", "content": [{"type": "text", "text": "fetch it"}]}}),
                json.dumps({"type": "tool_dispatch", "trace_id": "x", "tool_use_id": "1", "tool_name": "Bash", "endpoint_url": "http://127.0.0.1/exec", "request_body": {"command": "cat /workspace/input.txt"}, "response_status": 200, "response_body": {"output": "ok"}}),
                json.dumps({"type": "audit_snapshot", "trace_id": "x", "service_name": "sandbox", "audit_data": {"ok": True}}),
                json.dumps({"type": "trace_end", "trace_id": "x"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "T",
            "harness": "codex",
            "model": "m",
            "trace_path": str(trace),
            "trace_schema": "claw_eval_live_v1",
            "final_message": "done",
        }
    ]

    converted = convert_prediction_rows(rows, tmp_path / "score_ready")

    assert converted[0]["trace_path"] != str(trace)
    assert converted[0]["original_trace_path"] == str(trace)
    assert converted[0]["converted_trace_schema"] == "claw_eval_miniharness_from_live_bridge_v1"
    assert converted[0]["converted_tool_dispatch_count"] == 1
    converted_events = [json.loads(line) for line in Path(converted[0]["trace_path"]).read_text(encoding="utf-8").splitlines()]
    assistant_tool_uses = [
        event for event in converted_events
        if event.get("type") == "message"
        and event.get("message", {}).get("role") == "assistant"
        and event.get("message", {}).get("content", [{}])[0].get("type") == "tool_use"
    ]
    user_tool_results = [
        event for event in converted_events
        if event.get("type") == "message"
        and event.get("message", {}).get("role") == "user"
        and event.get("message", {}).get("content", [{}])[0].get("type") == "tool_result"
    ]
    assert assistant_tool_uses[0]["message"]["content"][0]["name"] == "Bash"
    assert user_tool_results[0]["message"]["content"][0]["tool_use_id"] == "1"
    assert [event["type"] for event in converted_events].count("tool_dispatch") == 1
    assert [event["type"] for event in converted_events].count("audit_snapshot") == 1
