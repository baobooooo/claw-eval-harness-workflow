from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assistant_text(row: dict[str, Any]) -> str:
    raw = str(row.get("final_message") or "")
    try:
        payload = json.loads(raw)
    except Exception:
        return raw
    if isinstance(payload, dict) and isinstance(payload.get("payloads"), list):
        texts: list[str] = []
        for item in payload["payloads"]:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
        if texts:
            return "\n\n".join(texts)
    return raw


def json_event(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, sort_keys=True)


def text_message(role: str, text: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }


def _path_if_exists(raw: Any) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.exists() else None


def _read_json(path: Path | None) -> Any:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None



def _read_trace_events(path: Path | None, *, limit: int = 10000) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:limit]:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _is_miniharness_message_event(event: dict[str, Any]) -> bool:
    if event.get("type") != "message":
        return False
    msg = event.get("message") if isinstance(event.get("message"), dict) else {}
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"}:
            return True
    return False


def _is_live_claw_trace(row: dict[str, Any], trace_path: Path | None) -> bool:
    if trace_path is None or not trace_path.exists():
        return False
    schema = str(row.get("trace_schema") or "")
    if schema.startswith("claw_eval_miniharness") or schema.startswith("claw_eval_minimal"):
        return False
    events = _read_trace_events(trace_path, limit=100)
    if any(_is_miniharness_message_event(ev) for ev in events):
        return False
    types = {str(ev.get("type")) for ev in events}
    if schema.startswith("claw_eval_live"):
        # A row may say live while trace_path still points to a native harness log.
        # Treat it as the canonical live trace only when it has the live trace
        # envelope.  It may legitimately have zero tool_dispatch events, in which
        # case conversion will emit a warning rather than inferring native calls.
        return "trace_start" in types
    # Only infer live-bridge conversion for raw external traces that have direct
    # ToolDispatch events but lack the MiniHarness assistant tool_use/user
    # tool_result message shape.  This avoids rewriting already-official
    # MiniHarness traces when a caller forgot to set trace_schema.
    if "trace_start" not in types or "tool_dispatch" not in types:
        return False
    return True



def _is_existing_miniharness_trace(row: dict[str, Any], trace_path: Path | None) -> bool:
    if trace_path is None or not trace_path.exists():
        return False
    schema = str(row.get("trace_schema") or "")
    if schema.startswith("claw_eval_miniharness") or schema.startswith("claw_eval_official_miniharness"):
        return True
    events = _read_trace_events(trace_path, limit=100)
    return any(_is_miniharness_message_event(ev) for ev in events)


def _passthrough_miniharness_trace(row: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    events = _read_trace_events(trace_path)
    tool_events = [ev for ev in events if ev.get("type") == "tool_dispatch"]
    audit_events = [ev for ev in events if ev.get("type") == "audit_snapshot"]
    text = assistant_text(row)
    if not text:
        assistant_messages = [ev for ev in events if ev.get("type") == "message" and _message_role(ev) == "assistant"]
        if assistant_messages:
            text = _message_text(assistant_messages[-1])
    converted_row = dict(row)
    converted_row["original_trace_path"] = str(trace_path)
    if row.get("trace_path") and str(row.get("trace_path")) != str(trace_path):
        converted_row["source_prediction_trace_path"] = row.get("trace_path")
    converted_row["trace_path"] = str(trace_path)
    converted_row["converted_trace_path"] = str(trace_path)
    converted_row["trace_file"] = trace_path.name
    converted_row["converted_trace_schema"] = "claw_eval_miniharness_passthrough_v1"
    converted_row["converted_assistant_text_bytes"] = len(text.encode("utf-8"))
    converted_row["converted_tool_dispatch_count"] = len(tool_events)
    converted_row["converted_audit_snapshot_count"] = len(audit_events)
    converted_row["converted_successful_web_dispatch_count"] = sum(
        1
        for ev in tool_events
        if ev.get("tool_name") in {"web_search", "web_fetch"} and int(ev.get("response_status", 500)) < 400
    )
    converted_row["converted_message_shape"] = "already_miniharness"
    return converted_row


def _message_role(event: dict[str, Any]) -> str:
    msg = event.get("message") if isinstance(event.get("message"), dict) else {}
    return str(msg.get("role") or "")


def _message_text(event: dict[str, Any]) -> str:
    msg = event.get("message") if isinstance(event.get("message"), dict) else {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # MiniHarness uses text blocks.  Some external traces use
                # Responses/Chat-compatible input_text or output_text blocks.
                if block.get("type") in {"text", "input_text", "output_text"} or "text" in block:
                    parts.append(str(block.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _clone_with_trace_id(event: dict[str, Any], trace_id: str) -> dict[str, Any]:
    out = json.loads(json.dumps(event, ensure_ascii=False))
    out["trace_id"] = trace_id
    return out




def _dispatch_with_stable_tool_use_id(dispatch: dict[str, Any], trace_id: str, *, index: int) -> dict[str, Any]:
    """Clone a ToolDispatch and ensure linked synthetic events share one id.

    MiniHarness traces link the assistant tool_use block, the ToolDispatch
    event, and the user tool_result block through the same tool_use_id.  Some
    external/live bridge dispatches already carry this field; when they do not,
    create it once here and reuse it for all converted events.
    """
    out = _clone_with_trace_id(dispatch, trace_id)
    if not out.get("tool_use_id"):
        name = safe_slug(out.get("tool_name"), "tool")
        out["tool_use_id"] = f"converted-{name}-{index + 1:04d}-{uuid.uuid4().hex[:8]}"
    return out


def _coerce_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except Exception:
            return {"_raw": raw}
        return dict(parsed) if isinstance(parsed, dict) else {"_value": parsed}
    return {}


def _dispatch_input(dispatch: dict[str, Any]) -> dict[str, Any]:
    for key in ("request_body", "tool_input", "input", "arguments"):
        if key in dispatch:
            value = _coerce_mapping(dispatch.get(key))
            if value or dispatch.get(key) in ({}, "{}", None):
                return value
    return {}


def _row_live_bridge_enabled(row: dict[str, Any]) -> bool:
    live_obj = row.get("live_tool_bridge")
    if isinstance(live_obj, dict):
        if live_obj.get("enabled") or live_obj.get("requested"):
            return True
    return bool(row.get("live_tool_bridge")) or str(row.get("trace_schema") or "") == "claw_eval_live_v1"


def _requires_canonical_bridge_dispatch(row: dict[str, Any]) -> bool:
    return bool(row.get("requires_bridge_tool_calls")) or _row_live_bridge_enabled(row)


def _candidate_live_trace_paths(row: dict[str, Any], input_trace_path: Path | None) -> list[Path]:
    """Return existing candidate canonical live traces, preserving priority.

    Agent adapters sometimes keep the harness-native transcript as trace_path
    and put the canonical Claw live trace beside it as claw_live_trace.jsonl.
    Conversion must use the canonical live trace when available; native Codex /
    NanoBot / OpenCode logs are not scored tool-use evidence.
    """
    raw_candidates: list[Any] = [
        row.get("original_trace_path"),
        row.get("claw_live_trace_path"),
        row.get("live_trace_path"),
        row.get("claw_eval_live_trace_path"),
    ]
    for container_key in ("metrics", "live_tool_bridge"):
        container = row.get(container_key)
        if isinstance(container, dict):
            for key in ("claw_live_trace_path", "live_trace_path", "trace_path", "path"):
                raw_candidates.append(container.get(key))
            nested = container.get("live_tool_bridge")
            if isinstance(nested, dict):
                for key in ("claw_live_trace_path", "live_trace_path", "trace_path", "path"):
                    raw_candidates.append(nested.get(key))

    # The prediction trace_path may itself be the canonical live trace.
    if input_trace_path is not None:
        raw_candidates.append(input_trace_path)
        raw_candidates.append(input_trace_path.parent / "claw_live_trace.jsonl")

    for base_key in ("output_dir", "workspace", "agent_workspace"):
        raw = row.get(base_key)
        if raw:
            base = Path(str(raw))
            raw_candidates.append(base / "claw_live_trace.jsonl")
            raw_candidates.append(base.parent / "claw_live_trace.jsonl")

    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        path = _path_if_exists(raw)
        if path is None:
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _find_live_trace_path(row: dict[str, Any], input_trace_path: Path | None) -> Path | None:
    for candidate in _candidate_live_trace_paths(row, input_trace_path):
        if _is_live_claw_trace(row, candidate):
            return candidate
    return None

def _tool_result_text(body: Any) -> str:
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    return json.dumps(body, ensure_ascii=False, sort_keys=True)


def _assistant_tool_use_message(trace_id: str, dispatch: dict[str, Any]) -> dict[str, Any]:
    tool_use_id = str(dispatch.get("tool_use_id") or f"tool-{uuid.uuid4().hex[:12]}")
    return {
        "type": "message",
        "trace_id": trace_id,
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": str(dispatch.get("tool_name") or ""),
                    "input": _dispatch_input(dispatch),
                }
            ],
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "timestamp": dispatch.get("timestamp") or now(),
    }


def _user_tool_result_message(trace_id: str, dispatch: dict[str, Any]) -> dict[str, Any]:
    tool_use_id = str(dispatch.get("tool_use_id") or f"tool-{uuid.uuid4().hex[:12]}")
    status = int(dispatch.get("response_status", 500) or 500)
    return {
        "type": "message",
        "trace_id": trace_id,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": _tool_result_text(dispatch.get("response_body"))}],
                    "is_error": bool(status >= 400),
                }
            ],
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "timestamp": dispatch.get("timestamp") or now(),
    }


def _converted_trace_end(original_end: dict[str, Any] | None, trace_id: str, *, tool_count: int, final_message_count: int) -> dict[str, Any]:
    if isinstance(original_end, dict):
        out = _clone_with_trace_id(original_end, trace_id)
    else:
        out = {"type": "trace_end", "trace_id": trace_id, "timestamp": now()}
    out.setdefault("model_input_tokens", 0)
    out.setdefault("model_output_tokens", 0)
    out.setdefault("input_tokens", out.get("model_input_tokens", 0))
    out.setdefault("output_tokens", out.get("model_output_tokens", 0))
    out.setdefault("total_tokens", int(out.get("input_tokens") or 0) + int(out.get("output_tokens") or 0))
    out.setdefault("model_time_s", 0.0)
    out.setdefault("tool_time_s", 0.0)
    out.setdefault("other_time_s", 0.0)
    out.setdefault("wall_time_s", 0.0)
    out.setdefault("scores", {"completion": 0.0, "robustness": 0.0, "communication": 0.0, "safety": 1.0, "efficiency_turns": 0, "efficiency_tokens": 0, "efficiency_wall_time_s": 0.0})
    out.setdefault("task_score", 0.0)
    out.setdefault("passed", False)
    out.setdefault("failure_modes", [])
    out.setdefault("user_agent_rounds", 0)
    out.setdefault("user_agent_max_rounds", 0)
    out.setdefault("user_agent_done", False)
    # MiniHarness total_turns is model turns.  Each synthetic tool_use message
    # corresponds to one model turn; add the final answer turn when present.
    out["total_turns"] = int(tool_count + final_message_count)
    return out


def _convert_live_trace_to_miniharness(row: dict[str, Any], trace_path: Path, out_dir: Path, *, model: str | None = None, record_index: int = 0) -> dict[str, Any]:
    """Rewrite an external live-bridge trace into MiniHarness-style JSONL.

    The live bridge records tool_dispatch events directly because the external
    harness, not MiniHarness, owns the model loop.  Official MiniHarness traces
    contain the model-visible conversation: assistant tool_use blocks followed
    by user tool_result blocks.  The judge should see that MiniHarness-shaped
    transcript, not the hidden exec/bridge transport.  This conversion is
    post-hoc only; it never changes the prompt or tools seen by the model.
    """
    events = _read_trace_events(trace_path)
    trace_dir = out_dir / "converted_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    task_id = str(row.get("task_id") or row.get("instance_id") or f"task_{record_index}")
    harness = str(row.get("harness") or "harness")
    model_name = str(model or row.get("model") or "")
    trace_id = f"converted-live-{safe_slug(harness, 'harness')}-{safe_slug(task_id, 'task')}-{uuid.uuid4().hex[:12]}"
    trace_file = "_".join([f"{record_index:04d}", safe_slug(harness, "harness"), safe_slug(task_id, "task")]) + ".jsonl"
    converted_path = trace_dir / trace_file

    original_start = next((ev for ev in events if ev.get("type") == "trace_start"), None)
    if isinstance(original_start, dict):
        start_event = _clone_with_trace_id(original_start, trace_id)
        start_event["task_id"] = task_id
        start_event["model"] = model_name or start_event.get("model") or ""
        start_event["persona"] = "external_harness_converted_to_miniharness"
    else:
        start_event = {"type": "trace_start", "trace_id": trace_id, "task_id": task_id, "model": model_name, "persona": "external_harness_converted_to_miniharness", "timestamp": now()}

    first_user = next((ev for ev in events if ev.get("type") == "message" and _message_role(ev) == "user"), None)
    user_text = _message_text(first_user) if isinstance(first_user, dict) else str(row.get("query") or row.get("instruction") or "")
    output_events: list[dict[str, Any]] = [
        start_event,
        {
            "type": "message",
            "trace_id": trace_id,
            "message": text_message("user", user_text),
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "timestamp": (first_user or {}).get("timestamp") if isinstance(first_user, dict) else now(),
        },
    ]

    dispatches = [ev for ev in events if ev.get("type") == "tool_dispatch"]
    normalized_dispatches = [
        _dispatch_with_stable_tool_use_id(dispatch, trace_id, index=i)
        for i, dispatch in enumerate(dispatches)
    ]
    for dispatch in normalized_dispatches:
        # Match native MiniHarness ordering: assistant emits tool_use, the
        # dispatcher records ToolDispatch, then the user receives tool_result.
        output_events.append(_assistant_tool_use_message(trace_id, dispatch))
        output_events.append(dispatch)
        output_events.append(_user_tool_result_message(trace_id, dispatch))

    final_text = assistant_text(row)
    if not final_text:
        assistant_messages = [ev for ev in events if ev.get("type") == "message" and _message_role(ev) == "assistant"]
        if assistant_messages:
            final_text = _message_text(assistant_messages[-1])
    final_message_count = 1 if final_text else 0
    if final_text:
        output_events.append({"type": "message", "trace_id": trace_id, "message": text_message("assistant", final_text), "usage": {"input_tokens": 0, "output_tokens": 0}, "timestamp": now()})

    media_events = [ev for ev in events if ev.get("type") == "media_load"]
    audit_events = [ev for ev in events if ev.get("type") == "audit_snapshot"]
    output_events.extend(_clone_with_trace_id(ev, trace_id) for ev in media_events)
    if audit_events:
        output_events.extend(_clone_with_trace_id(ev, trace_id) for ev in audit_events)
    else:
        # Some post-run packages keep service_audit.json but were written before
        # audit_snapshot events were appended to the live trace.  Rehydrate only
        # audit snapshots, never tool calls, from that file.
        audit_events = service_audit_snapshot_events(row, trace_id)
        output_events.extend(audit_events)
    original_end = next((ev for ev in events if ev.get("type") == "trace_end"), None)
    output_events.append(_converted_trace_end(original_end, trace_id, tool_count=len(dispatches), final_message_count=final_message_count))
    converted_path.write_text("\n".join(json_event(event) for event in output_events) + "\n", encoding="utf-8")

    converted_row = dict(row)
    converted_row["original_trace_path"] = str(trace_path)
    if row.get("trace_path") and str(row.get("trace_path")) != str(trace_path):
        converted_row["source_prediction_trace_path"] = row.get("trace_path")
    converted_row["trace_path"] = str(converted_path)
    converted_row["converted_trace_path"] = str(converted_path)
    converted_row["trace_file"] = trace_file
    converted_row["converted_trace_schema"] = "claw_eval_miniharness_from_live_bridge_v1"
    converted_row["converted_assistant_text_bytes"] = len(final_text.encode("utf-8"))
    converted_row["converted_tool_dispatch_count"] = len(dispatches)
    converted_row["converted_audit_snapshot_count"] = len(audit_events)
    converted_row["converted_successful_web_dispatch_count"] = sum(
        1
        for ev in dispatches
        if ev.get("tool_name") in {"web_search", "web_fetch"} and int(ev.get("response_status", 500)) < 400
    )
    converted_row["converted_message_shape"] = "assistant_tool_use_plus_tool_dispatch_plus_user_tool_result"
    warnings: list[str] = []
    if not dispatches and _requires_canonical_bridge_dispatch(row):
        warnings.append("requires_bridge_tool_calls_but_no_tool_dispatch_events")
    if warnings:
        converted_row["conversion_warnings"] = warnings
    return converted_row

def _service_audit_path_for(row: dict[str, Any]) -> Path | None:
    audit_path = _path_if_exists(row.get("service_audit_path"))
    if audit_path is None and row.get("workspace"):
        candidate = Path(str(row["workspace"])).parent / "service_audit.json"
        audit_path = candidate if candidate.exists() else None
    return audit_path


def codex_dispatches(row: dict[str, Any], trace_id: str) -> list[dict[str, Any]]:
    trace_path = _path_if_exists(row.get("trace_path"))
    if trace_path is None:
        return []
    dispatches: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        item = obj.get("item")
        if obj.get("type") != "item.completed" or not isinstance(item, dict):
            continue
        if item.get("type") != "command_execution":
            continue
        exit_code = item.get("exit_code")
        dispatches.append(
            {
                "type": "tool_dispatch",
                "trace_id": trace_id,
                "tool_use_id": f"codex-exec-{len(dispatches) + 1}",
                "tool_name": "exec",
                "endpoint_url": "harness://codex/exec",
                "request_body": {"command": item.get("command")},
                "response_status": 200 if exit_code == 0 else 500,
                "response_body": {
                    "exit_code": exit_code,
                    "output": str(item.get("aggregated_output") or "")[:4000],
                },
                "latency_ms": 0.0,
                "timestamp": now(),
            }
        )
    return dispatches


def openclaw_dispatches(row: dict[str, Any], trace_id: str) -> list[dict[str, Any]]:
    stderr_path = _path_if_exists(row.get("stderr_path"))
    if stderr_path is None:
        return []
    dispatches: list[dict[str, Any]] = []
    pattern = re.compile(r"\[tools\]\s+([A-Za-z0-9_-]+)\s+failed:\s+(.*?)\s+raw_params=(\{.*\})$")
    for line in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        tool_name, error, raw_params = match.groups()
        try:
            request_body = json.loads(raw_params)
        except Exception:
            request_body = {"raw_params": raw_params}
        dispatches.append(
            {
                "type": "tool_dispatch",
                "trace_id": trace_id,
                "tool_use_id": f"openclaw-{tool_name}-{len(dispatches) + 1}",
                "tool_name": tool_name,
                "endpoint_url": f"harness://openclaw/{tool_name}",
                "request_body": request_body,
                "response_status": 500,
                "response_body": {"error": error},
                "latency_ms": 0.0,
                "timestamp": now(),
            }
        )
    return dispatches


def _endpoint_path(value: Any) -> str:
    raw = str(value or "")
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        return parsed.path or raw
    return raw


def _endpoint_to_tool_map(row: dict[str, Any]) -> dict[str, str]:
    mapping = {
        "/web/search": "web_search",
        "/web/fetch": "web_fetch",
    }
    policy_path = _path_if_exists(row.get("tool_policy_path"))
    if policy_path is None:
        return mapping
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return mapping
    endpoints = policy.get("exposed_tool_endpoints") or policy.get("tool_endpoints") or []
    if not isinstance(endpoints, list):
        return mapping
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        tool_name = endpoint.get("tool_name")
        if not tool_name:
            continue
        mapping[_endpoint_path(endpoint.get("url"))] = str(tool_name)
    return mapping


def service_audit_dispatches(row: dict[str, Any], trace_id: str) -> list[dict[str, Any]]:
    audit = _read_json(_service_audit_path_for(row))
    if not isinstance(audit, dict):
        return []

    dispatches: list[dict[str, Any]] = []
    endpoint_to_tool = _endpoint_to_tool_map(row)
    for service_name, service_data in audit.items():
        body = service_data.get("body") if isinstance(service_data, dict) else None
        calls = body.get("calls") if isinstance(body, dict) else None
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            endpoint = str(call.get("endpoint") or "")
            endpoint_path = _endpoint_path(endpoint)
            tool_name = endpoint_to_tool.get(endpoint) or endpoint_to_tool.get(endpoint_path)
            if not tool_name:
                continue
            response_body = call.get("response_body")
            response_status = call.get("response_status", 200)
            if isinstance(response_body, dict) and isinstance(response_body.get("status_code"), int):
                response_status = int(response_body["status_code"])
            dispatches.append(
                {
                    "type": "tool_dispatch",
                    "trace_id": trace_id,
                    "tool_use_id": f"service-{service_name}-{tool_name}-{len(dispatches) + 1}",
                    "tool_name": tool_name,
                    "endpoint_url": endpoint,
                    "request_body": call.get("request_body") or {},
                    "response_status": int(response_status),
                    "response_body": response_body,
                    "latency_ms": 0.0,
                    "timestamp": call.get("timestamp") or now(),
                }
            )
    return dispatches


def service_audit_snapshot_events(row: dict[str, Any], trace_id: str) -> list[dict[str, Any]]:
    """Expose service /audit payloads as Claw-Eval AuditSnapshot events.

    The patched evaluator uses claw_eval.trace.reader.load_trace(), which only
    fills grader audit_data from audit_snapshot events.  ToolDispatch events are
    useful trajectory evidence, but they are not enough for service-state graders.
    """
    audit = _read_json(_service_audit_path_for(row))
    if not isinstance(audit, dict):
        return []

    events: list[dict[str, Any]] = []
    for service_name, service_data in audit.items():
        audit_url = ""
        audit_data: Any = service_data
        if isinstance(service_data, dict):
            audit_url = str(service_data.get("url") or "")
            if "body" in service_data:
                audit_data = service_data.get("body")
        if not isinstance(audit_data, dict):
            audit_data = {"body": audit_data}
        events.append(
            {
                "type": "audit_snapshot",
                "trace_id": trace_id,
                "service_name": str(service_name),
                "audit_url": audit_url,
                "audit_data": audit_data,
                "timestamp": now(),
            }
        )
    return events


def dispatches_for(row: dict[str, Any], trace_id: str) -> list[dict[str, Any]]:
    harness = str(row.get("harness") or "")
    dispatches: list[dict[str, Any]] = []
    if harness == "codex":
        dispatches.extend(codex_dispatches(row, trace_id))
    elif harness in {"openclaw", "opencode", "open_code"}:
        dispatches.extend(openclaw_dispatches(row, trace_id))
    dispatches.extend(service_audit_dispatches(row, trace_id))
    return dispatches


def safe_slug(value: Any, default: str) -> str:
    raw = str(value or default).strip() or default
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return slug.strip("._-") or default


def convert_prediction_row(
    row: dict[str, Any],
    out_dir: Path,
    *,
    model: str | None = None,
    record_index: int = 0,
) -> dict[str, Any]:
    input_trace_path = _path_if_exists(row.get("trace_path"))
    live_trace_path = _find_live_trace_path(row, input_trace_path)
    if live_trace_path is not None:
        return _convert_live_trace_to_miniharness(row, live_trace_path, out_dir, model=model, record_index=record_index)
    if _is_existing_miniharness_trace(row, input_trace_path):
        assert input_trace_path is not None
        return _passthrough_miniharness_trace(row, input_trace_path)

    trace_dir = out_dir / "converted_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    task_id = str(row.get("task_id") or row.get("instance_id") or f"task_{record_index}")
    harness = str(row.get("harness") or "harness")
    model_name = str(model or row.get("model") or "")
    text = assistant_text(row)
    trace_id = f"external-{safe_slug(harness, 'harness')}-{safe_slug(task_id, 'task')}-{uuid.uuid4().hex[:12]}"
    trace_file = "_".join(
        [
            f"{record_index:04d}",
            safe_slug(harness, "harness"),
            safe_slug(task_id, "task"),
        ]
    ) + ".jsonl"
    trace_path = trace_dir / trace_file
    events: list[dict[str, Any]] = [
        {
            "type": "trace_start",
            "trace_id": trace_id,
            "task_id": task_id,
            "model": model_name,
            "persona": "external_harness",
            "timestamp": now(),
        },
        {
            "type": "message",
            "trace_id": trace_id,
            "message": text_message("user", str(row.get("query") or row.get("instruction") or "")),
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "timestamp": now(),
        },
    ]
    # Only legacy non-live rows may infer dispatches from harness-native logs or
    # service audit.  Formal live-bridge rows must have canonical ToolDispatch
    # events; do not manufacture tool calls from Codex/NanoBot/OpenCode native
    # transcripts when those events are missing.
    tool_events = [] if _requires_canonical_bridge_dispatch(row) else dispatches_for(row, trace_id)
    normalized_tool_events = [
        _dispatch_with_stable_tool_use_id(dispatch, trace_id, index=i)
        for i, dispatch in enumerate(tool_events)
    ]
    for dispatch in normalized_tool_events:
        events.append(_assistant_tool_use_message(trace_id, dispatch))
        events.append(dispatch)
        events.append(_user_tool_result_message(trace_id, dispatch))
    audit_events = service_audit_snapshot_events(row, trace_id)
    events.extend(audit_events)
    final_message_count = 1 if text else 0
    if text:
        events.append(
            {
                "type": "message",
                "trace_id": trace_id,
                "message": text_message("assistant", text),
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "timestamp": now(),
            }
        )
    events.append(
        {
            "type": "trace_end",
            "trace_id": trace_id,
            "total_turns": int(len(normalized_tool_events) + final_message_count),
            "model_input_tokens": 0,
            "model_output_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "model_time_s": 0.0,
            "tool_time_s": 0.0,
            "other_time_s": 0.0,
            "wall_time_s": 0.0,
            "task_score": 0.0,
            "passed": False,
            "failure_modes": [],
            "user_agent_rounds": 0,
            "user_agent_max_rounds": 0,
            "user_agent_done": False,
            "timestamp": now(),
        }
    )
    trace_path.write_text("\n".join(json_event(event) for event in events) + "\n", encoding="utf-8")

    converted_row = dict(row)
    converted_row["original_trace_path"] = row.get("trace_path")
    converted_row["trace_path"] = str(trace_path)
    converted_row["converted_trace_path"] = str(trace_path)
    converted_row["trace_file"] = trace_file
    converted_row["converted_trace_schema"] = "claw_eval_minimal_v1"
    converted_row["converted_assistant_text_bytes"] = len(text.encode("utf-8"))
    converted_row["converted_tool_dispatch_count"] = len(tool_events)
    converted_row["converted_audit_snapshot_count"] = len(audit_events)
    converted_row["converted_successful_web_dispatch_count"] = sum(
        1
        for dispatch in tool_events
        if dispatch.get("tool_name") in {"web_search", "web_fetch"} and int(dispatch.get("response_status", 500)) < 400
    )
    converted_row["converted_message_shape"] = "minimal_assistant_text_with_inferred_tool_dispatches"
    warnings: list[str] = []
    if _requires_canonical_bridge_dispatch(row):
        warnings.append("requires_bridge_tool_calls_but_no_canonical_live_trace")
    if not tool_events and _requires_canonical_bridge_dispatch(row):
        warnings.append("requires_bridge_tool_calls_but_no_tool_dispatch_events")
    if warnings:
        converted_row["conversion_warnings"] = warnings
    return converted_row


def convert_prediction_rows(rows: list[dict[str, Any]], out_dir: Path, *, model: str | None = None) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [convert_prediction_row(row, out_dir, model=model, record_index=i) for i, row in enumerate(rows)]


def write_score_ready_outputs(rows: list[dict[str, Any]], out_dir: Path, *, model: str | None = None) -> dict[str, Any]:
    converted = convert_prediction_rows(rows, out_dir, model=model)
    pred_path = out_dir / "score_ready_predictions.jsonl"
    pred_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in converted) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "model": model,
        "predictions": str(pred_path),
        "trace_dir": str(out_dir / "converted_traces"),
        "num_rows": len(converted),
        "rows": [
            {
                "task_id": row.get("task_id"),
                "harness": row.get("harness"),
                "trace_path": row.get("trace_path"),
                "original_trace_path": row.get("original_trace_path"),
                "source_prediction_trace_path": row.get("source_prediction_trace_path"),
                "converted_trace_schema": row.get("converted_trace_schema"),
                "converted_message_shape": row.get("converted_message_shape"),
                "assistant_text_bytes": row.get("converted_assistant_text_bytes"),
                "tool_dispatch_count": row.get("converted_tool_dispatch_count"),
                "audit_snapshot_count": row.get("converted_audit_snapshot_count"),
                "successful_web_dispatch_count": row.get("converted_successful_web_dispatch_count"),
                "conversion_warnings": row.get("conversion_warnings", []),
            }
            for row in converted
        ],
        "warning_count": sum(len(row.get("conversion_warnings", [])) for row in converted),
        "rows_with_warnings": [
            {
                "task_id": row.get("task_id"),
                "harness": row.get("harness"),
                "conversion_warnings": row.get("conversion_warnings", []),
            }
            for row in converted
            if row.get("conversion_warnings")
        ],
    }
    (out_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
