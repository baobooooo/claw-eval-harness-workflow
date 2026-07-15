import json
from pathlib import Path

from harness_eval.analysis.tool_policy import model_proxy_history_markers


def test_model_proxy_history_markers_ignore_system_prompt_but_catch_structured_hidden_transport(tmp_path: Path):
    log = tmp_path / "proxy.jsonl"
    body = {
        "messages": [
            {"role": "system", "content": "OpenClaw mentions exec in its own system prompt."},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "Bash", "arguments": "{\"command\":\"ls\"}"}}]},
            {"role": "tool", "name": "Bash", "tool_call_id": "call_1", "content": "ok"},
        ],
        "tools": [{"type": "function", "function": {"name": "Bash"}}],
    }
    log.write_text(json.dumps({"event": "upstream_request", "body": body}) + "\n", encoding="utf-8")
    row = {"metrics": {"model_tool_proxy": {"log_path": str(log)}}}
    assert model_proxy_history_markers(row, {"Bash"}) == []

    body["messages"][1]["tool_calls"][0]["function"]["name"] = "exec"
    body["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{\"command\":\"python3 ./claw_tool Bash @.claw_tool_payloads/x.json\"}"
    body["messages"][2]["name"] = "exec"
    body["messages"][2]["content"] = "Command still running (session abc)"
    log.write_text(json.dumps({"event": "upstream_request", "body": body}) + "\n", encoding="utf-8")
    markers = model_proxy_history_markers(row, {"Bash"})
    reasons = {m["reason"] for m in markers}
    assert "assistant_tool_call_not_in_yaml_surface" in reasons
    assert "assistant_tool_call_arguments_expose_hidden_transport" in reasons
    assert "tool_result_name_not_in_yaml_surface" in reasons
    assert "tool_result_content_exposes_hidden_transport" in reasons
