from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import append_jsonl, write_json


TOOL_HINTS = (
    "exec_command",
    "shell_command",
    "apply_patch",
    "read_file",
    "write_file",
    "list_files",
    "search",
    "tool_call",
    "function_call",
)


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _event_type(obj: dict[str, Any]) -> str:
    for key in ("type", "item_type", "event", "name"):
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return "unknown"


def _extract_text(obj: Any) -> str:
    texts: list[str] = []
    for d in _walk(obj):
        for key in ("text", "content", "message", "delta"):
            v = d.get(key)
            if isinstance(v, str):
                texts.append(v)
    # Keep bounded; raw events are preserved elsewhere.
    return "\n".join(texts)[:20000]


def _extract_tool(obj: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for d in _walk(obj):
        typ = str(d.get("type") or d.get("item_type") or d.get("name") or "")
        if any(h in typ for h in TOOL_HINTS):
            candidates.append(d)
        if any(k in d for k in ("cmd", "command", "tool_name", "arguments")):
            candidates.append(d)
    if not candidates:
        return None
    d = candidates[0]
    cmd = d.get("cmd") or d.get("command") or d.get("args") or d.get("arguments")
    if isinstance(cmd, list):
        cmd_s = " ".join(str(x) for x in cmd)
    elif isinstance(cmd, dict):
        cmd_s = json.dumps(cmd, ensure_ascii=False, sort_keys=True)
    elif cmd is None:
        cmd_s = ""
    else:
        cmd_s = str(cmd)
    return {
        "tool_type": str(d.get("type") or d.get("item_type") or d.get("tool_name") or d.get("name") or "unknown"),
        "command_or_args": cmd_s[:20000],
        "status": d.get("status") or d.get("state"),
        "exit_code": d.get("exit_code") or d.get("returncode"),
    }


def parse_events(input_path: str | Path, output_dir: str | Path, keep_raw: bool = True) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / "module_timeline.jsonl"
    tools_path = output_dir / "tool_events.jsonl"
    messages_path = output_dir / "agent_messages.jsonl"
    stats = {
        "input": str(input_path),
        "event_count": 0,
        "parse_errors": 0,
        "tool_event_count": 0,
        "agent_message_count": 0,
        "event_types": {},
    }
    for p in (timeline_path, tools_path, messages_path):
        if p.exists():
            p.unlink()

    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            raw = line.rstrip("\n")
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                stats["parse_errors"] += 1
                append_jsonl(timeline_path, {"seq": idx, "module": "codex.raw_text", "raw": raw[:20000]})
                continue

            if not isinstance(ev, dict):
                ev = {"value": ev}
            typ = _event_type(ev)
            stats["event_count"] += 1
            stats["event_types"][typ] = stats["event_types"].get(typ, 0) + 1

            module = "codex.event"
            if "agent_message" in typ or "assistant" in typ or "message" in typ:
                module = "codex.agent_message"
            elif "tool" in typ or "exec" in typ or "patch" in typ or _extract_tool(ev):
                module = "codex.tool"
            elif "turn" in typ or "thread" in typ:
                module = "codex.session"

            rec: dict[str, Any] = {
                "seq": idx,
                "module": module,
                "event_type": typ,
                "timestamp": ev.get("timestamp") or ev.get("time") or ev.get("created_at"),
            }
            text = _extract_text(ev)
            if text:
                rec["text_preview"] = text[:2000]
            tool = _extract_tool(ev)
            if tool:
                rec.update({f"tool_{k}": v for k, v in tool.items() if v is not None})
                append_jsonl(tools_path, {"seq": idx, **tool, "event_type": typ, "raw_event": ev if keep_raw else None})
                stats["tool_event_count"] += 1
            if module == "codex.agent_message":
                append_jsonl(messages_path, {"seq": idx, "event_type": typ, "text": text, "raw_event": ev if keep_raw else None})
                stats["agent_message_count"] += 1
            if keep_raw:
                rec["raw_event"] = ev
            append_jsonl(timeline_path, rec)

    write_json(output_dir / "codex_event_stats.json", stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Codex exec JSONL event stream into module/tool timelines.")
    ap.add_argument("input", help="codex_events.ndjson")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-raw", action="store_true", help="Do not copy raw Codex events into parsed files.")
    args = ap.parse_args()
    stats = parse_events(args.input, args.out_dir, keep_raw=not args.no_raw)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
