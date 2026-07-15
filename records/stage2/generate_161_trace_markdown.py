#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_PREFIX = "general161_plus_smoke_20260709_175615"
RETRY_PREFIX = "formal_timeout_new3_retry_20260708_154131"
TASK_IDS = Path("records/stage2/general161_plus_smoke_20260709_175615_general161_ids.txt")
REUSED_IDS = Path("records/stage2/general161_plus_smoke_20260709_175615_reused_smoke_ids.txt")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def fmt_sec(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    if seconds < 1:
        return f"{sign}{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{sign}{seconds:.3f}s"
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{sign}{minutes}m{rest:05.2f}s"


def fmt_runtime(ms: float | int | None) -> str:
    if ms is None:
        return "not captured"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.3f}s"


def fmt_token_usage(usage: dict[str, Any] | None) -> str:
    if not usage:
        return "tok=n/a"
    input_tokens = usage.get("input_tokens", usage.get("input"))
    output_tokens = usage.get("output_tokens", usage.get("output"))
    cached = usage.get("cached_input_tokens", usage.get("cacheRead", usage.get("cache_read_tokens")))
    total = usage.get("total_tokens", usage.get("total"))
    parts = []
    if input_tokens is not None:
        parts.append(f"in {input_tokens}")
    if output_tokens is not None:
        parts.append(f"out {output_tokens}")
    if cached not in (None, 0):
        parts.append(f"cache {cached}")
    if total is not None:
        parts.append(f"total {total}")
    return "tok=" + "/".join(parts) if parts else "tok=n/a"


def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```\n"


def text_block(value: str) -> str:
    return "```text\n" + value.replace("```", "``\\`") + "\n```\n"


def sha_key(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def safe_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)


def load_id_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def locate_instance(repo: Path, harness: str, task_id: str, reused: set[str]) -> tuple[Path | None, str | None]:
    run_names: list[str]
    if task_id in reused:
        run_names = [f"{RETRY_PREFIX}_{harness}_deepseek_v4_pro"]
    elif harness == "openclaw":
        run_names = [
            f"{RUN_PREFIX}_openclaw_deepseek_v4_pro",
            f"{RUN_PREFIX}_openclaw_resume6core_deepseek_v4_pro",
        ]
    else:
        run_names = [f"{RUN_PREFIX}_{harness}_deepseek_v4_pro"]
    for run_name in run_names:
        inst = repo / "runs" / run_name / "instances" / task_id
        if inst.exists():
            return inst, run_name
    return None, None


def load_proxy(instance: Path, harness: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[dict[str, Any]]]]:
    proxy = instance / f"{harness}_model_tool_proxy.jsonl"
    events = list(read_jsonl(proxy)) if proxy.exists() else []
    requests = [row for row in events if row.get("event") == "upstream_request"]
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in events:
        if row.get("event") == "upstream_request":
            if current:
                groups.append(current)
                current = []
        elif row.get("event") == "rewrite_tool_call":
            current.append(row)
    if current:
        groups.append(current)
    return events, requests, groups


def trace_summary(instance: Path) -> tuple[datetime | None, datetime | None, list[dict[str, Any]], datetime | None, datetime | None, str | None]:
    trace_first = None
    user_ts = None
    dispatches: list[dict[str, Any]] = []
    assistant_ts = None
    closed_ts = None
    final_text = None
    trace = instance / "claw_live_trace.jsonl"
    if not trace.exists():
        return trace_first, user_ts, dispatches, assistant_ts, closed_ts, final_text
    for row in read_jsonl(trace):
        event = row.get("type") or row.get("event")
        ts = parse_ts(row.get("timestamp")) if row.get("timestamp") else None
        if event == "trace_start" and ts:
            trace_first = ts
        elif event == "message":
            message = row.get("message") or {}
            role = message.get("role")
            if role == "user" and user_ts is None:
                user_ts = ts
            elif role == "assistant":
                assistant_ts = ts
                final_text = flatten_content(message.get("content"))
        elif event == "agent_final_message":
            assistant_ts = ts
            final_text = row.get("text") or row.get("message") or final_text
        elif event == "tool_dispatch":
            latency = row.get("latency_ms")
            finish = ts.timestamp() if ts else None
            begin = finish - latency / 1000.0 if finish is not None and isinstance(latency, (int, float)) else None
            dispatches.append(
                {
                    "name": row.get("tool_name") or "unknown",
                    "latency_ms": latency,
                    "begin": begin,
                    "finish": finish,
                    "timestamp": ts,
                    "response_status": row.get("response_status"),
                    "response_body": row.get("response_body"),
                    "request_body": row.get("request_body"),
                    "endpoint_url": row.get("endpoint_url"),
                }
            )
        elif event == "trace_end":
            closed_ts = ts
    return trace_first, user_ts, dispatches, assistant_ts, closed_ts, final_text


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def intervals_overlap(dispatches: list[dict[str, Any]]) -> bool:
    intervals = [(d["begin"], d["finish"]) for d in dispatches if d.get("begin") is not None and d.get("finish") is not None]
    intervals.sort()
    for (_, first_finish), (second_begin, _) in zip(intervals, intervals[1:]):
        if second_begin < first_finish:
            return True
    return False


def codex_token_usage(instance: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    session_root = instance / "codex_home" / "sessions"
    usages: list[dict[str, Any]] = []
    total = None
    if session_root.exists():
        for path in sorted(session_root.rglob("rollout-*.jsonl")):
            for row in read_jsonl(path):
                if row.get("type") == "event_msg" and (row.get("payload") or {}).get("type") == "token_count":
                    info = row["payload"].get("info") or {}
                    if info.get("last_token_usage"):
                        usages.append(info["last_token_usage"])
                    if info.get("total_token_usage"):
                        total = info["total_token_usage"]
    if total is None and (instance / "codex_events.ndjson").exists():
        for row in read_jsonl(instance / "codex_events.ndjson"):
            if row.get("type") == "turn.completed" and row.get("usage"):
                usage = row["usage"]
                total = {
                    "input_tokens": usage.get("input_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                    "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
                }
    return usages, total, "codex token_count events" if usages else "codex turn.completed aggregate" if total else "not_recorded"


def openclaw_token_usage(instance: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    for path in sorted((instance / "openclaw_home").rglob("*.trajectory.jsonl")):
        for row in read_jsonl(path):
            if row.get("type") in {"model.completed", "trace.artifacts"}:
                usage = (row.get("data") or {}).get("usage")
                if usage:
                    return [], usage, "openclaw model.completed aggregate"
    return [], None, "not_recorded"


def token_usage(instance: Path, harness: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    if harness == "codex":
        return codex_token_usage(instance)
    if harness == "openclaw":
        return openclaw_token_usage(instance)
    return [], None, "not_recorded"


def extract_system_and_tools(requests: list[dict[str, Any]]) -> tuple[str | None, str | None, list[dict[str, Any]], dict[str, Any] | None]:
    if not requests:
        return None, None, [], None
    body = requests[0].get("body") or {}
    tools = body.get("tools") or []
    model_profile = {
        key: body.get(key)
        for key in ["model", "temperature", "max_tokens", "tool_choice", "parallel_tool_calls", "thinking", "reasoning"]
        if key in body
    }
    if "instructions" in body:
        system = body.get("instructions")
        developer = None
        for item in body.get("input") or []:
            if item.get("role") == "developer":
                developer = flatten_codex_content(item.get("content"))
                break
        return system, developer, tools, model_profile
    messages = body.get("messages") or []
    system_messages = [m for m in messages if m.get("role") == "system"]
    system = "\n\n".join(flatten_content(m.get("content")) for m in system_messages) if system_messages else None
    return system, None, tools, model_profile


def flatten_codex_content(content: Any) -> str:
    if isinstance(content, list):
        return "\n\n".join(str(item.get("text", item)) for item in content)
    return flatten_content(content)


def normalize_transcript_from_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for request in requests:
        body = request.get("body") or {}
        if "input" in body:
            for item in body.get("input") or []:
                entry = normalize_codex_item(item)
                if entry is None:
                    continue
                key = entry_key(entry)
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
        else:
            for msg in body.get("messages") or []:
                if msg.get("role") == "system":
                    continue
                entry = normalize_chat_message(msg)
                if entry is None:
                    continue
                key = entry_key(entry)
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
    return entries


def normalize_codex_item(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = item.get("type")
    role = item.get("role")
    if role == "developer":
        return None
    if item_type == "message" and role:
        return {"role": role, "payload": item}
    if item_type == "reasoning":
        return {"role": "assistant", "payload": item}
    if item_type == "function_call":
        return {"role": "assistant", "payload": item}
    if item_type == "function_call_output":
        return {"role": "tool", "payload": item}
    return None


def normalize_chat_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    role = msg.get("role")
    if role == "tool":
        return {"role": "tool", "payload": msg}
    if role in {"user", "assistant"}:
        return {"role": role, "payload": msg}
    return None


def entry_key(entry: dict[str, Any]) -> str:
    payload = entry.get("payload") or {}
    if payload.get("call_id"):
        return f"{entry['role']}:{payload.get('type')}:{payload.get('call_id')}:{sha_key(payload.get('output', payload))}"
    if payload.get("tool_call_id"):
        return f"tool:{payload.get('tool_call_id')}:{sha_key(payload.get('content', payload))}"
    tool_calls = payload.get("tool_calls") or []
    if tool_calls:
        ids = ",".join(str(call.get("id")) for call in tool_calls)
        return f"assistant_tool_calls:{ids}:{sha_key(payload.get('content'))}"
    return f"{entry['role']}:{sha_key(payload)}"


def captured_call_ids(entries: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    calls: set[str] = set()
    outputs: set[str] = set()
    for entry in entries:
        payload = entry.get("payload") or {}
        if payload.get("type") == "function_call" and payload.get("call_id"):
            calls.add(payload["call_id"])
        for call in payload.get("tool_calls") or []:
            if call.get("id"):
                calls.add(call["id"])
        if payload.get("type") == "function_call_output" and payload.get("call_id"):
            outputs.add(payload["call_id"])
        if payload.get("tool_call_id"):
            outputs.add(payload["tool_call_id"])
    return calls, outputs


def payload_for_call(instance: Path, call_id: str) -> Any:
    for path in instance.rglob(f"{call_id}.json"):
        if ".claw_tool_payloads" in str(path):
            try:
                return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                return None
    return None


def pending_text_by_call(instance: Path, harness: str) -> dict[str, str]:
    texts: dict[str, str] = {}
    if harness == "nanobot":
        for path in (instance / "agent_driver_workspace" / "sessions").glob("*.jsonl"):
            for row in read_jsonl(path):
                if row.get("role") == "assistant":
                    text = row.get("content") or ""
                    for call in row.get("tool_calls") or []:
                        if call.get("id") and text:
                            texts.setdefault(call["id"], text)
    elif harness == "openclaw":
        for path in (instance / "openclaw_home").rglob("agents/openclaw/sessions/*.jsonl"):
            if path.name.endswith(".trajectory.jsonl"):
                continue
            for row in read_jsonl(path):
                message = row.get("message") or {}
                if message.get("role") != "assistant":
                    continue
                text_parts = []
                call_ids = []
                for part in message.get("content") or []:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text") or "")
                    elif part.get("type") == "toolCall" and part.get("id"):
                        call_ids.append(part["id"])
                text = "\n".join(t for t in text_parts if t)
                for call_id in call_ids:
                    if text:
                        texts.setdefault(call_id, text)
    return texts


def append_pending_calls(
    entries: list[dict[str, Any]],
    groups: list[list[dict[str, Any]]],
    dispatches: list[dict[str, Any]],
    instance: Path,
    harness: str,
) -> None:
    calls_seen, outputs_seen = captured_call_ids(entries)
    text_by_call = pending_text_by_call(instance, harness)
    flat_rewrites = [rewrite for group in groups for rewrite in group]
    for index, rewrite in enumerate(flat_rewrites):
        call_id = rewrite.get("call_id")
        if not call_id:
            continue
        tool_name = rewrite.get("from_tool") or rewrite.get("tool_name") or "unknown"
        args = payload_for_call(instance, call_id)
        if call_id not in calls_seen:
            payload: dict[str, Any] = {
                "role": "assistant",
                "content": text_by_call.get(call_id),
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args, ensure_ascii=False) if args is not None else None,
                        },
                    }
                ],
                "note": "Recovered from proxy rewrite_tool_call; this model output was not present in a later upstream request history.",
            }
            entries.append({"role": "assistant", "payload": payload})
            calls_seen.add(call_id)
        if call_id not in outputs_seen and index < len(dispatches):
            dispatch = dispatches[index]
            if dispatch.get("name") == tool_name:
                payload = {
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": call_id,
                    "content": dispatch.get("response_body"),
                    "metadata": {
                        "endpoint_url": dispatch.get("endpoint_url"),
                        "response_status": dispatch.get("response_status"),
                        "latency_ms": dispatch.get("latency_ms"),
                    },
                    "note": "Recovered from claw_live_trace tool_dispatch; this tool result was not present in a later upstream request history.",
                }
                entries.append({"role": "tool", "payload": payload})
                outputs_seen.add(call_id)


def append_final_message(entries: list[dict[str, Any]], final_text: str | None) -> None:
    if not final_text:
        return
    if any(entry.get("role") == "assistant" and flatten_content((entry.get("payload") or {}).get("content")) == final_text for entry in entries):
        return
    entries.append({"role": "assistant", "payload": {"role": "assistant", "content": final_text, "note": "Captured final_message from claw_live_trace."}})


def build_tree(
    task_id: str,
    harness: str,
    groups: list[list[dict[str, Any]]],
    dispatches: list[dict[str, Any]],
    trace_first: datetime | None,
    user_ts: datetime | None,
    assistant_ts: datetime | None,
    closed_ts: datetime | None,
    per_call_usage: list[dict[str, Any]],
    total_usage: dict[str, Any] | None,
) -> str:
    first = trace_first or user_ts
    last_candidates = [ts for ts in (closed_ts, assistant_ts) if ts]
    last_dispatch_ts = max((d["timestamp"] for d in dispatches if d.get("timestamp")), default=None)
    if last_dispatch_ts:
        last_candidates.append(last_dispatch_ts)
    last = max(last_candidates) if last_candidates else None
    total = fmt_sec(last.timestamp() - first.timestamp()) if first and last else "unknown"
    initial_wait = user_ts.timestamp() - first.timestamp() if user_ts and first else None
    lines = [
        "workflow_tree:",
        f"- run: {task_id} / {harness} (total={total}, timezone=UTC, {fmt_token_usage(total_usage)})",
        f"  - user: task prompt (initial_wait={fmt_sec(initial_wait)}, tok=n/a)",
    ]
    cursor = user_ts.timestamp() if user_ts else (first.timestamp() if first else None)
    dispatch_index = 0
    tool_number = 1
    for group_index, group in enumerate(groups, 1):
        matched = []
        for rewrite in group:
            dispatch = dispatches[dispatch_index] if dispatch_index < len(dispatches) else None
            matched.append((rewrite, dispatch))
            if dispatch:
                dispatch_index += 1
        captured = [dispatch for _, dispatch in matched if dispatch]
        if not captured:
            wait = None
            execution = "not_dispatched"
        else:
            first_begin = min((dispatch["begin"] for dispatch in captured if dispatch["begin"] is not None), default=None)
            wait = first_begin - cursor if first_begin is not None and cursor is not None else None
            execution = "single" if len(group) == 1 else ("parallel" if intervals_overlap(captured) else "fanout_serial")
            finishes = [dispatch["finish"] for dispatch in captured if dispatch.get("finish") is not None]
            if finishes:
                cursor = max(finishes)
        usage = per_call_usage[group_index - 1] if group_index - 1 < len(per_call_usage) else None
        plural = "call" if len(group) == 1 else "calls"
        lines.append(
            f"    - assistant_turn#{group_index:02d}: {len(group)} tool {plural} "
            f"(wait={fmt_sec(wait)}, execution={execution}, {fmt_token_usage(usage)})"
        )
        for rewrite, dispatch in matched:
            name = rewrite.get("from_tool") or rewrite.get("tool_name") or "unknown"
            lines.append(f"      - tool#{tool_number:02d}: {name} (runtime={fmt_runtime(dispatch.get('latency_ms') if dispatch else None)}, tok=n/a)")
            tool_number += 1
    final_usage = per_call_usage[len(groups)] if len(per_call_usage) > len(groups) else None
    if assistant_ts:
        wait = assistant_ts.timestamp() - cursor if cursor is not None else None
        if wait is not None and wait < 0:
            lines.append(f"    - assistant_final_log: captured while previous tool was still running (wait={fmt_sec(wait)}, {fmt_token_usage(final_usage)})")
        else:
            lines.append(f"    - assistant_final: captured message (wait={fmt_sec(wait)}, {fmt_token_usage(final_usage)})")
            cursor = assistant_ts.timestamp()
    else:
        lines.append(f"    - assistant_final: missing in captured trace ({fmt_token_usage(final_usage)})")
    if closed_ts:
        wait = closed_ts.timestamp() - cursor if cursor is not None else None
        suffix = " before final tool completed" if wait is not None and wait < 0 else ""
        lines.append(f"  - trace: closed{suffix} (wait={fmt_sec(wait)}, {fmt_token_usage(total_usage)})")
    return "\n".join(lines) + "\n\n"


def render_trace(repo: Path, output_root: Path, task_id: str, harness: str, reused: set[str]) -> dict[str, Any]:
    instance, run_name = locate_instance(repo, harness, task_id, reused)
    if instance is None or run_name is None:
        return {"task_id": task_id, "harness": harness, "status": "missing_instance"}
    events, requests, groups = load_proxy(instance, harness)
    trace_first, user_ts, dispatches, assistant_ts, closed_ts, final_text = trace_summary(instance)
    per_call_usage, total_usage, token_source = token_usage(instance, harness)
    system_prompt, developer_prompt, tools, model_profile = extract_system_and_tools(requests)
    entries = normalize_transcript_from_requests(requests)
    append_pending_calls(entries, groups, dispatches, instance, harness)
    append_final_message(entries, final_text)

    task_dir = output_root / safe_name(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = task_dir / f"{safe_name(task_id)}__{harness}.md"
    header = [
        f"# {task_id} / {harness} Full Trajectory",
        "",
        "token_report:",
        f"- token_source: {token_source}",
        f"- total: {fmt_token_usage(total_usage).replace('tok=', '')}",
        f"- per_model_call_records: {len(per_call_usage)}",
        f"- upstream_requests: {len(requests)}",
        f"- tool_call_groups: {len(groups)}",
        f"- tool_dispatches: {len(dispatches)}",
        f"- run_dir: {run_name}",
        "",
    ]
    body: list[str] = []
    body.append(build_tree(task_id, harness, groups, dispatches, trace_first, user_ts, assistant_ts, closed_ts, per_call_usage, total_usage))
    body.append("model_profile:\n")
    body.append(json_block(model_profile or {}))
    body.append("\nsystem:\n")
    body.append(text_block(system_prompt or ""))
    if developer_prompt:
        body.append("\ndeveloper:\n")
        body.append(text_block(developer_prompt))
    body.append("\ntools:\n")
    body.append(json_block(tools))
    body.append("\ntranscript:\n")
    for entry_index, entry in enumerate(entries, 1):
        role = entry.get("role") or "unknown"
        body.append(f"\n{role} #{entry_index:03d}:\n")
        body.append(json_block(entry.get("payload")))
    out_path.write_text("\n".join(header) + "".join(body), encoding="utf-8")
    return {
        "task_id": task_id,
        "harness": harness,
        "status": "ok",
        "path": str(out_path.relative_to(output_root)),
        "bytes": out_path.stat().st_size,
        "run_dir": run_name,
        "token_source": token_source,
        "per_model_call_records": len(per_call_usage),
        "upstream_requests": len(requests),
        "tool_call_groups": len(groups),
        "tool_dispatches": len(dispatches),
        "total_usage": total_usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/data1/zjq/harness/codex-swebench-qwen3")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    repo = Path(args.repo)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    task_ids = load_id_file(repo / TASK_IDS)
    reused = set(load_id_file(repo / REUSED_IDS))
    if args.limit:
        task_ids = task_ids[: args.limit]
    manifest = []
    for task_id in task_ids:
        for harness in ["codex", "nanobot", "openclaw"]:
            manifest.append(render_trace(repo, output_root, task_id, harness, reused))
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Claw-Eval General161 Full Trajectory Manifest",
        "",
        f"- tasks: {len(task_ids)}",
        f"- expected_trace_files: {len(task_ids) * 3}",
        f"- ok_trace_files: {sum(1 for item in manifest if item['status'] == 'ok')}",
        f"- missing: {sum(1 for item in manifest if item['status'] != 'ok')}",
        "",
        "| task_id | harness | status | token_source | path |",
        "|---|---|---|---|---|",
    ]
    for item in manifest:
        lines.append(
            f"| {item['task_id']} | {item['harness']} | {item['status']} | "
            f"{item.get('token_source', '')} | {item.get('path', '')} |"
        )
    (output_root / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if all(item["status"] == "ok" for item in manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
