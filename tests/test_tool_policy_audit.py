import json

from harness_eval.analysis.tool_policy import audit_prediction, enforce_tool_policy
from harness_eval.types import BenchmarkTask, HarnessResult


def test_tool_policy_audit_allows_official_helper_and_service_call(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {
        "allowed_tools": ["web_search"],
        "helper_files": ["claw_tool", "claw_web_search"],
        "exposed_tool_endpoints": [
            {"tool_name": "web_search", "url": "http://localhost:9114/web/search", "method": "POST"}
        ],
        "policy_sha256": "abc",
    }
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    (inst / "service_audit.json").write_text(
        json.dumps({"web_real": {"body": {"calls": [{"endpoint": "/web/search", "request_body": {"query": "x"}}]}}}),
        encoding="utf-8",
    )
    trace = inst / "codex_events.ndjson"
    trace.write_text(
        json.dumps(
            {
                "item": {
                    "type": "command_execution",
                    "command": "/bin/bash -lc './claw_web_search x'",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_prediction(
        {
            "task_id": "T1",
            "harness": "codex",
            "workspace": str(workspace),
            "trace_path": str(trace),
            "service_audit_path": str(inst / "service_audit.json"),
        }
    )

    assert audit["compliant"] is True
    assert audit["observed_tool_counts"] == {"web_search": 1}
    assert audit["violations"] == []


def test_tool_policy_audit_flags_direct_network_and_unallowed_endpoint(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {
        "allowed_tools": ["web_search"],
        "helper_files": ["claw_web_search"],
        "exposed_tool_endpoints": [
            {"tool_name": "web_search", "url": "http://localhost:9114/web/search", "method": "POST"}
        ],
    }
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    (inst / "service_audit.json").write_text(
        json.dumps({"web_real": {"body": {"calls": [{"endpoint": "/admin", "request_body": {}}]}}}),
        encoding="utf-8",
    )
    trace = inst / "codex_events.ndjson"
    trace.write_text(
        json.dumps(
            {
                "item": {
                    "type": "command_execution",
                    "command": "/bin/bash -lc 'curl https://example.com'",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_prediction(
        {
            "task_id": "T1",
            "harness": "codex",
            "workspace": str(workspace),
            "trace_path": str(trace),
            "service_audit_path": str(inst / "service_audit.json"),
        }
    )

    assert audit["compliant"] is False
    assert {item["type"] for item in audit["violations"]} == {
        "direct_network_access",
        "unauthorized_service_tool",
    }


def test_tool_policy_audit_reports_external_text_markers_as_warnings_and_helper_edits_as_violations(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {
        "allowed_tools": ["web_search"],
        "helper_files": ["claw_tool", "claw_web_search"],
        "exposed_tool_endpoints": [
            {"tool_name": "web_search", "url": "http://localhost:9114/web/search", "method": "POST"}
        ],
    }
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    (inst / "nanobot_trace.jsonl").write_text(
        "thinking about curl and urllib.request but not showing an executed shell command\n",
        encoding="utf-8",
    )
    (inst / "patch.diff").write_text(
        "diff --git a/claw_tool b/claw_tool\n--- a/claw_tool\n+++ b/claw_tool\n",
        encoding="utf-8",
    )

    audit = audit_prediction(
        {
            "task_id": "T1",
            "harness": "nanobot",
            "workspace": str(workspace),
            "trace_path": str(inst / "nanobot_trace.jsonl"),
        }
    )

    assert audit["compliant"] is False
    assert {item["type"] for item in audit["violations"]} == {"helper_file_modified"}
    assert audit["text_suspicion_count"] >= 1
    assert {item["type"] for item in audit["warnings"]} == {"text_direct_network_mention"}


def test_enforce_tool_policy_marks_non_bash_task_with_direct_network_as_violation(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {
        "allowed_tools": ["web_search"],
        "helper_files": ["claw_web_search"],
        "exposed_tool_endpoints": [
            {"tool_name": "web_search", "url": "http://localhost:9114/web/search", "method": "POST"}
        ],
    }
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    trace = inst / "codex_events.ndjson"
    trace.write_text(
        json.dumps({"item": {"type": "command_execution", "command": "/bin/bash -lc 'curl https://example.com'"}})
        + "\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="T1",
        row={},
        prompt="",
        workspace=workspace,
        output_dir=inst,
        metadata={"allowed_tools": ["web_search"], "tool_policy_path": str(workspace / "claw_eval_tool_policy.json")},
    )
    result = HarnessResult(
        task_id="T1",
        harness="codex",
        model="m",
        status="ok",
        trace_path=str(trace),
    )

    gated = enforce_tool_policy(result, task)

    assert gated.status == "tool_policy_violation"
    assert gated.error == "tool policy violation"
    assert gated.metrics["tool_policy"]["violation_count"] == 1
    assert gated.metrics["tool_policy"]["violations"][0]["type"] == "direct_network_access"
    assert (inst / "tool_policy_audit.json").exists()


def test_enforce_tool_policy_does_not_gate_direct_shell_when_bash_is_allowed(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {
        "allowed_tools": ["Bash"],
        "helper_files": [],
        "exposed_tool_endpoints": [],
    }
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    trace = inst / "codex_events.ndjson"
    trace.write_text(
        json.dumps({"item": {"type": "command_execution", "command": "/bin/bash -lc 'node -e 1'"}})
        + "\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="T2",
        row={},
        prompt="",
        workspace=workspace,
        output_dir=inst,
        metadata={"allowed_tools": ["Bash"], "tool_policy_path": str(workspace / "claw_eval_tool_policy.json")},
    )
    result = HarnessResult(task_id="T2", harness="codex", model="m", status="ok", trace_path=str(trace))

    gated = enforce_tool_policy(result, task)

    assert gated.status == "ok"
    assert gated.metrics["tool_policy"]["enforced"] is False
    assert gated.metrics["tool_policy"]["skip_reason"] == "bash_allowed"


def test_live_bridge_allows_network_commands_through_allowed_bash(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {
        "allowed_tools": ["Bash"],
        "helper_files": ["claw_tool", "claw_bash"],
        "exposed_tool_endpoints": [],
        "sandbox_tools": ["Bash"],
        "requires_bridge_tool_calls": True,
    }
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    trace = inst / "claw_live_trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "type": "tool_dispatch",
                "tool_name": "Bash",
                "request_body": {"command": "curl -fsSL https://www.sec.gov/"},
                "endpoint_url": "http://sandbox/exec",
                "response_status": 200,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="Tb",
        row={},
        prompt="",
        workspace=workspace,
        output_dir=inst,
        metadata={
            "allowed_tools": ["Bash"],
            "tool_policy_path": str(workspace / "claw_eval_tool_policy.json"),
            "live_tool_bridge": True,
            "requires_bridge_tool_calls": True,
        },
    )
    result = HarnessResult(task_id="Tb", harness="nanobot", model="m", status="ok", trace_path=str(trace))

    gated = enforce_tool_policy(result, task)

    assert gated.status == "ok"
    assert gated.metrics["tool_policy"]["compliant"] is True
    assert gated.metrics["tool_policy"]["violation_count"] == 0
    assert gated.metrics["tool_policy"]["observed_tool_counts"] == {"Bash": 1}
    assert gated.metrics["tool_policy"]["bash_network_allowed"] is True
    assert gated.metrics["tool_policy"]["allowed_bash_network_count"] == 1


def test_tool_policy_audit_allows_openclaw_metadata_read_warning(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {
        "allowed_tools": ["Bash", "Read"],
        "helper_files": ["claw_tool", "claw_read", "claw_eval_tool_policy.json"],
        "exposed_tool_endpoints": [],
    }
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    stderr = inst / "openclaw_stderr.log"
    stderr.write_text(
        "[tools] read failed: ENOENT: no such file or directory, access '/tmp/w/claw_eval_tool_policy.json' raw_params={\"path\":\"/tmp/w/claw_eval_tool_policy.json\"}\n",
        encoding="utf-8",
    )

    audit = audit_prediction(
        {
            "task_id": "T1",
            "harness": "openclaw",
            "workspace": str(workspace),
            "stderr_path": str(stderr),
        }
    )

    assert audit["compliant"] is True
    assert audit["violation_count"] == 0
    assert audit["warning_count"] == 1
    assert audit["warnings"][0]["reason"] == "driver_metadata_read_only"


def test_live_bridge_zero_tool_calls_is_hard_failure(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {
        "allowed_tools": ["web_search", "Bash"],
        "helper_files": ["claw_tool", "claw_web_search"],
        "exposed_tool_endpoints": [
            {"tool_name": "web_search", "url": "http://localhost:9114/web/search", "method": "POST"}
        ],
        "sandbox_tools": ["Bash"],
        "requires_bridge_tool_calls": True,
    }
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    trace = inst / "claw_live_trace.jsonl"
    trace.write_text(json.dumps({"type": "trace_start", "task_id": "T0"}) + "\n", encoding="utf-8")
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="T0",
        row={},
        prompt="",
        workspace=workspace,
        output_dir=inst,
        metadata={
            "allowed_tools": ["web_search", "Bash"],
            "tool_policy_path": str(workspace / "claw_eval_tool_policy.json"),
            "live_tool_bridge": True,
            "requires_bridge_tool_calls": True,
        },
    )
    result = HarnessResult(task_id="T0", harness="codex", model="m", status="ok", trace_path=str(trace))

    gated = enforce_tool_policy(result, task)

    assert gated.status == "no_bridge_tool_calls"
    assert gated.error == "no Claw-Eval live-bridge tool calls were observed"
    assert gated.metrics["tool_policy"]["tool_usage_failure_count"] == 1


def test_provider_quota_failure_is_not_reported_as_no_bridge_tool_calls(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {
        "allowed_tools": ["Bash"],
        "helper_files": ["claw_tool", "claw_bash"],
        "sandbox_tools": ["Bash"],
        "requires_bridge_tool_calls": True,
    }
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    trace = inst / "claw_live_trace.jsonl"
    trace.write_text(json.dumps({"type": "trace_start", "task_id": "Tquota"}) + "\n", encoding="utf-8")

    audit = audit_prediction(
        {
            "task_id": "Tquota",
            "harness": "nanobot",
            "status": "provider_quota_error",
            "workspace": str(workspace),
            "trace_path": str(trace),
            "live_tool_bridge": True,
            "trace_schema": "claw_eval_live_v1",
            "requires_bridge_tool_calls": True,
        }
    )

    assert audit["compliant"] is True
    assert audit["tool_usage_failure_count"] == 0
    assert audit["violations"] == []


def test_live_bridge_timeout_with_real_tool_calls_is_not_overwritten_by_text_mentions(tmp_path):
    inst = tmp_path / "inst"
    workspace = inst / "workspace"
    workspace.mkdir(parents=True)
    policy = {"allowed_tools": ["Bash"], "helper_files": ["claw_tool", "claw_bash"], "sandbox_tools": ["Bash"], "requires_bridge_tool_calls": True}
    (workspace / "claw_eval_tool_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    trace = inst / "claw_live_trace.jsonl"
    trace.write_text(
        json.dumps({"type": "tool_dispatch", "tool_name": "Bash", "request_body": {"command": "echo ok"}, "endpoint_url": "http://sandbox/exec"}) + "\n"
        + "thinking about curl but did not execute it\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        benchmark="openclaw",
        task_id="T3",
        row={},
        prompt="",
        workspace=workspace,
        output_dir=inst,
        metadata={"allowed_tools": ["Bash"], "tool_policy_path": str(workspace / "claw_eval_tool_policy.json"), "live_tool_bridge": True, "requires_bridge_tool_calls": True},
    )
    result = HarnessResult(task_id="T3", harness="nanobot", model="m", status="timeout", trace_path=str(trace), returncode=-124)

    gated = enforce_tool_policy(result, task)

    assert gated.status == "timeout"
    assert gated.metrics["tool_policy"]["violation_count"] == 0
    assert gated.metrics["tool_policy"]["observed_tool_counts"] == {"Bash": 1}
