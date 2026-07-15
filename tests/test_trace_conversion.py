import json
from pathlib import Path

from harness_eval.analysis.trace_conversion import convert_prediction_rows


def test_convert_prediction_rows_writes_claw_eval_traces_for_multiple_instances(tmp_path):
    codex_trace = tmp_path / "codex_events.ndjson"
    codex_trace.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "/bin/bash -lc './claw_web_search test'",
                    "aggregated_output": "search output",
                    "exit_code": 0,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "service_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "web_real": {
                    "body": {
                        "calls": [
                            {
                                "endpoint": "/web/search",
                                "request_body": {"query": "test"},
                                "response_body": {"results": []},
                                "timestamp": "2026-07-03T00:00:00+00:00",
                            }
                        ]
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "claw_eval_tool_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "exposed_tool_endpoints": [
                    {"tool_name": "web_search", "url": "http://localhost:9114/web/search", "method": "POST"}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "T001",
            "harness": "codex",
            "model": "deepseek-v4-pro",
            "query": "first question",
            "trace_path": str(codex_trace),
            "service_audit_path": str(audit_path),
            "tool_policy_path": str(policy_path),
            "final_message": "first answer",
        },
        {
            "task_id": "T002",
            "harness": "nanobot",
            "model": "deepseek-v4-pro",
            "query": "second question",
            "final_message": json.dumps({"payloads": [{"text": "second answer"}]}, ensure_ascii=False),
        },
    ]

    converted = convert_prediction_rows(rows, tmp_path / "converted", model="deepseek-v4-pro")

    assert len(converted) == 2
    assert converted[0]["original_trace_path"] == str(codex_trace)
    assert converted[0]["converted_tool_dispatch_count"] == 2
    assert converted[0]["converted_audit_snapshot_count"] == 1
    assert converted[0]["converted_successful_web_dispatch_count"] == 1
    assert converted[1]["converted_assistant_text_bytes"] == len("second answer".encode("utf-8"))
    for row in converted:
        trace_path = tmp_path / "converted" / "converted_traces" / (row["trace_file"])
        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        assert events[0]["type"] == "trace_start"
        assert events[0]["task_id"] == row["task_id"]
        assert events[-1]["type"] == "trace_end"
        if row["task_id"] == "T001":
            audit_events = [event for event in events if event["type"] == "audit_snapshot"]
            assert len(audit_events) == 1
            assert audit_events[0]["service_name"] == "web_real"
            assert audit_events[0]["audit_data"]["calls"][0]["endpoint"] == "/web/search"


def test_convert_prediction_rows_does_not_rewrite_existing_miniharness_trace_without_schema(tmp_path):
    trace = tmp_path / "mini.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps({"type": "trace_start", "trace_id": "m", "task_id": "Tmini", "model": "deepseek-v4-pro"}),
                json.dumps({"type": "message", "trace_id": "m", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "web_fetch", "input": {"url": "https://example.com"}}]}}),
                json.dumps({"type": "message", "trace_id": "m", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": [{"type": "text", "text": "ok"}]}]}}),
                json.dumps({"type": "tool_dispatch", "trace_id": "m", "tool_use_id": "tu1", "tool_name": "web_fetch", "request_body": {"url": "https://example.com"}, "response_status": 200, "response_body": {"ok": True}}),
                json.dumps({"type": "trace_end", "trace_id": "m"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    converted = convert_prediction_rows([
        {"task_id": "Tmini", "harness": "miniharness", "model": "deepseek-v4-pro", "trace_path": str(trace), "final_message": "done"}
    ], tmp_path / "converted")

    assert converted[0]["trace_path"] == str(trace)
    assert converted[0]["converted_trace_schema"] == "claw_eval_miniharness_passthrough_v1"
    events = [json.loads(line) for line in Path(converted[0]["trace_path"]).read_text(encoding="utf-8").splitlines()]
    # Existing MiniHarness-shaped transcript is preserved for judge instead of regenerated.
    assert sum(1 for event in events if event.get("type") == "message") == 2
    assert converted[0]["converted_tool_dispatch_count"] == 1


def test_live_conversion_preserves_miniharness_event_order_and_stable_generated_id(tmp_path):
    live = tmp_path / "claw_live_trace.jsonl"
    live.write_text(
        "\n".join(
            [
                json.dumps({"type": "trace_start", "trace_id": "live", "task_id": "Torder", "model": "m"}),
                json.dumps({"type": "message", "trace_id": "live", "message": {"role": "user", "content": [{"type": "input_text", "text": "do it"}]}}),
                # No tool_use_id on purpose: converter must generate one once and reuse it.
                json.dumps({"type": "tool_dispatch", "trace_id": "live", "tool_name": "web_search", "request_body": {"query": "x"}, "response_status": 200, "response_body": {"results": []}}),
                json.dumps({"type": "trace_end", "trace_id": "live"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    converted = convert_prediction_rows([
        {"task_id": "Torder", "harness": "nanobot", "model": "m", "trace_path": str(live), "trace_schema": "claw_eval_live_v1", "final_message": "done"}
    ], tmp_path / "converted")

    events = [json.loads(line) for line in Path(converted[0]["trace_path"]).read_text(encoding="utf-8").splitlines()]
    types = [event["type"] for event in events]
    tool_use_idx = next(i for i, event in enumerate(events) if event.get("type") == "message" and event.get("message", {}).get("content", [{}])[0].get("type") == "tool_use")
    dispatch_idx = next(i for i, event in enumerate(events) if event.get("type") == "tool_dispatch")
    result_idx = next(i for i, event in enumerate(events) if event.get("type") == "message" and event.get("message", {}).get("content", [{}])[0].get("type") == "tool_result")
    assert tool_use_idx < dispatch_idx < result_idx
    tool_use_id = events[tool_use_idx]["message"]["content"][0]["id"]
    assert events[dispatch_idx]["tool_use_id"] == tool_use_id
    assert events[result_idx]["message"]["content"][0]["tool_use_id"] == tool_use_id
    assert converted[0]["converted_message_shape"] == "assistant_tool_use_plus_tool_dispatch_plus_user_tool_result"


def test_converter_prefers_canonical_live_trace_next_to_native_trace(tmp_path):
    native = tmp_path / "opencode_trace.jsonl"
    native.write_text(json.dumps({"native": "not judge evidence"}) + "\n", encoding="utf-8")
    live = tmp_path / "claw_live_trace.jsonl"
    live.write_text(
        "\n".join(
            [
                json.dumps({"type": "trace_start", "trace_id": "live", "task_id": "Tlive", "model": "m"}),
                json.dumps({"type": "message", "trace_id": "live", "message": {"role": "user", "content": [{"type": "text", "text": "task"}]}}),
                json.dumps({"type": "tool_dispatch", "trace_id": "live", "tool_use_id": "tu-live", "tool_name": "Bash", "request_body": {"command": "echo ok"}, "response_status": 200, "response_body": {"output": "ok"}}),
                json.dumps({"type": "trace_end", "trace_id": "live"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    converted = convert_prediction_rows([
        {"task_id": "Tlive", "harness": "opencode", "model": "m", "trace_path": str(native), "trace_schema": "claw_eval_live_v1", "requires_bridge_tool_calls": True}
    ], tmp_path / "converted")

    assert converted[0]["original_trace_path"] == str(live)
    assert converted[0]["source_prediction_trace_path"] == str(native)
    assert converted[0]["converted_tool_dispatch_count"] == 1
    assert "conversion_warnings" not in converted[0]


def test_live_required_without_canonical_trace_warns_and_does_not_infer_from_native_or_audit(tmp_path):
    native = tmp_path / "codex_events.ndjson"
    native.write_text(
        json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "python3 ./claw_tool web_search @p.json", "exit_code": 0, "aggregated_output": "ok"}}) + "\n",
        encoding="utf-8",
    )
    audit = tmp_path / "service_audit.json"
    audit.write_text(json.dumps({"web_real": {"body": {"calls": [{"endpoint": "/web/search", "request_body": {"query": "x"}, "response_body": {"results": []}}]}}}), encoding="utf-8")

    converted = convert_prediction_rows([
        {"task_id": "Tmissing", "harness": "codex", "model": "m", "trace_path": str(native), "service_audit_path": str(audit), "trace_schema": "claw_eval_live_v1", "requires_bridge_tool_calls": True}
    ], tmp_path / "converted")

    events = [json.loads(line) for line in Path(converted[0]["trace_path"]).read_text(encoding="utf-8").splitlines()]
    assert sum(1 for event in events if event.get("type") == "tool_dispatch") == 0
    assert sum(1 for event in events if event.get("type") == "audit_snapshot") == 1
    assert converted[0]["converted_tool_dispatch_count"] == 0
    assert "requires_bridge_tool_calls_but_no_canonical_live_trace" in converted[0]["conversion_warnings"]
