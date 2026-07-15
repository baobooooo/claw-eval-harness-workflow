"""Minimal /v1/responses -> /v1/chat/completions bridge for Codex CLI.

The bridge preserves the subset of Responses API semantics that Codex needs for
local shell-tool loops: streaming SSE, function-call tools, function-call output
history, and assistant messages. It is intentionally small and should stay a
compatibility layer over the vLLM chat/completions endpoint, not a general
Responses API implementation.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import time
import uuid
from typing import Any

try:
    import requests  # type: ignore
    from fastapi import FastAPI, Request  # type: ignore
    from fastapi.responses import JSONResponse, StreamingResponse  # type: ignore
except Exception:  # pragma: no cover
    requests = None
    FastAPI = None
    Request = None
    JSONResponse = None
    StreamingResponse = None


DEFAULT_RESPONSE_MAX_TOKENS = 2048


def _flatten_input(inp: Any) -> str:
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        chunks: list[str] = []
        for item in inp:
            if isinstance(item, dict):
                if isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                elif isinstance(item.get("content"), list):
                    for c in item["content"]:
                        if isinstance(c, dict) and isinstance(c.get("text"), str):
                            chunks.append(c["text"])
                        elif isinstance(c, str):
                            chunks.append(c)
                elif isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(inp)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                for key in ("text", "input_text", "output_text"):
                    if isinstance(item.get(key), str):
                        chunks.append(item[key])
                        break
        return "\n".join(chunks)
    if content is None:
        return ""
    return str(content)


def _reasoning_text_from_item(item: dict[str, Any]) -> str:
    for key in ("text", "reasoning_content"):
        if isinstance(item.get(key), str):
            return item[key]
    text = _content_to_text(item.get("content"))
    if text:
        return text
    return _content_to_text(item.get("summary"))


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("id"):
            ids.append(str(call["id"]))
    return ids


def _missing_tool_message(tool_call_id: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": "[missing tool result inserted by bridge: the client did not provide a tool message before the next non-tool message]",
    }


def _orphan_tool_message_as_user(message: dict[str, Any]) -> dict[str, Any]:
    tool_call_id = str(message.get("tool_call_id") or "")
    content = _content_to_text(message.get("content"))
    prefix = f"[orphan tool result converted by bridge for tool_call_id={tool_call_id}]"
    return {"role": "user", "content": f"{prefix}\n{content}".rstrip()}


def _sanitize_chat_messages_for_tool_order(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    sanitized: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        raw = messages[i]
        if not isinstance(raw, dict):
            i += 1
            continue
        message = dict(raw)
        expected_ids = _tool_call_ids(message) if message.get("role") == "assistant" else []
        if not expected_ids:
            if message.get("role") == "tool":
                sanitized.append(_orphan_tool_message_as_user(message))
            else:
                sanitized.append(message)
            i += 1
            continue

        message.setdefault("reasoning_content", "")
        sanitized.append(message)
        seen: set[str] = set()
        i += 1
        while i < len(messages):
            next_raw = messages[i]
            if not isinstance(next_raw, dict) or next_raw.get("role") != "tool":
                break
            next_message = dict(next_raw)
            tool_call_id = str(next_message.get("tool_call_id") or "")
            if tool_call_id in expected_ids and tool_call_id not in seen:
                sanitized.append(next_message)
                seen.add(tool_call_id)
            else:
                sanitized.append(_orphan_tool_message_as_user(next_message))
            i += 1
        for tool_call_id in expected_ids:
            if tool_call_id not in seen:
                sanitized.append(_missing_tool_message(tool_call_id))
    return sanitized


def _coalesce_consecutive_assistant_tool_calls(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    coalesced: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        raw = messages[i]
        if not isinstance(raw, dict):
            i += 1
            continue
        message = dict(raw)
        if message.get("role") != "assistant" or not _tool_call_ids(message):
            coalesced.append(message)
            i += 1
            continue

        combined = dict(message)
        combined["tool_calls"] = list(message.get("tool_calls") or [])
        reasoning_parts = []
        if isinstance(message.get("reasoning_content"), str) and message["reasoning_content"]:
            reasoning_parts.append(message["reasoning_content"])
        i += 1
        while i < len(messages):
            next_raw = messages[i]
            if not isinstance(next_raw, dict):
                i += 1
                continue
            next_message = dict(next_raw)
            if next_message.get("role") != "assistant" or not _tool_call_ids(next_message):
                break
            combined["tool_calls"].extend(list(next_message.get("tool_calls") or []))
            if isinstance(next_message.get("reasoning_content"), str) and next_message["reasoning_content"]:
                reasoning_parts.append(next_message["reasoning_content"])
            i += 1
        if reasoning_parts:
            combined["reasoning_content"] = "\n".join(reasoning_parts)
        coalesced.append(combined)
    return coalesced


def _sanitize_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    messages = _coalesce_consecutive_assistant_tool_calls(payload.get("messages"))
    sanitized["messages"] = _sanitize_chat_messages_for_tool_order(messages)
    return sanitized


def _stringify_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        text = _content_to_text(output)
        if text:
            return text
    try:
        return json.dumps(output, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(output)


def _responses_input_to_chat_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    input_items = payload.get("input", "")
    if isinstance(input_items, str):
        messages.append({"role": "user", "content": input_items})
        return messages
    if not isinstance(input_items, list):
        messages.append({"role": "user", "content": _flatten_input(input_items)})
        return messages

    pending_reasoning: str | None = None
    for item in input_items:
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue
        item_type = item.get("type")
        role = item.get("role")
        if item_type == "reasoning":
            pending_reasoning = _reasoning_text_from_item(item)
            continue
        if item_type == "message" or role in {"developer", "system", "user", "assistant"}:
            content = _content_to_text(item.get("content")) or _content_to_text(item.get("text"))
            if not content:
                continue
            if role == "assistant":
                message = {"role": "assistant", "content": content}
                if pending_reasoning is not None:
                    message["reasoning_content"] = pending_reasoning
                    pending_reasoning = None
                messages.append(message)
            elif role in {"developer", "system"}:
                messages.append({"role": "system", "content": content})
            else:
                messages.append({"role": "user", "content": content})
            continue
        if item_type == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}")
            arguments = item.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": pending_reasoning or "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": str(item.get("name") or ""), "arguments": arguments},
                        }
                    ],
                }
            )
            pending_reasoning = None
            continue
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                    "content": _stringify_tool_output(item.get("output", "")),
                }
            )
    return messages


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        function: dict[str, Any] = {"name": name}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        if isinstance(tool.get("parameters"), dict):
            function["parameters"] = tool["parameters"]
        out.append({"type": "function", "function": function})
    return out


def _chat_payload_from_responses(
    payload: dict[str, Any],
    default_max_tokens: int | None = DEFAULT_RESPONSE_MAX_TOKENS,
) -> dict[str, Any]:
    chat_payload: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": _responses_input_to_chat_messages(payload),
        "temperature": payload.get("temperature", 0),
    }
    if "max_output_tokens" in payload:
        chat_payload["max_tokens"] = payload.get("max_output_tokens")
    elif "max_tokens" in payload:
        chat_payload["max_tokens"] = payload.get("max_tokens")
    elif default_max_tokens is not None and default_max_tokens > 0:
        chat_payload["max_tokens"] = default_max_tokens
    tools = _responses_tools_to_chat_tools(payload.get("tools"))
    if tools:
        chat_payload["tools"] = tools
        chat_payload["tool_choice"] = "auto"
    return _sanitize_chat_payload({k: v for k, v in chat_payload.items() if v is not None})


def _repair_python_body_string_newlines(body: str) -> str:
    out: list[str] = []
    quote: str | None = None
    triple = False
    escaped = False
    i = 0
    while i < len(body):
        ch = body[i]
        if quote is None:
            if ch in {"'", '"'}:
                quote = ch
                triple = body.startswith(ch * 3, i)
                if triple:
                    out.append(ch * 3)
                    i += 3
                    continue
            out.append(ch)
            i += 1
            continue
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            i += 1
            continue
        if triple and body.startswith(quote * 3, i):
            out.append(quote * 3)
            quote = None
            triple = False
            i += 3
            continue
        if not triple and ch == quote:
            out.append(ch)
            quote = None
            i += 1
            continue
        if not triple and ch == "\n":
            out.append("\\n")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_python_heredoc_newlines(cmd: str) -> str:
    match = re.search(r"<<'([^']+)'\n", cmd)
    if not match:
        return cmd
    tag = match.group(1)
    body_start = match.end()
    body_end = cmd.rfind("\n" + tag, body_start)
    if body_end < 0:
        return cmd
    body = cmd[body_start:body_end]
    try:
        ast.parse(body)
        return cmd
    except SyntaxError:
        pass
    repaired = _repair_python_body_string_newlines(body)
    try:
        ast.parse(repaired)
    except SyntaxError:
        return cmd
    return cmd[:body_start] + repaired + cmd[body_end:]


def _normalize_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
        arguments = decoded
    if isinstance(arguments, dict) and isinstance(arguments.get("cmd"), str):
        arguments = dict(arguments)
        arguments["cmd"] = _repair_python_heredoc_newlines(arguments["cmd"])
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)


def _xml_tool_calls_from_text(content: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL):
        raw = match.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(raw.replace("\\'", "'"))
            except json.JSONDecodeError:
                continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        arguments = _normalize_tool_arguments(parsed.get("arguments", {}))
        if not isinstance(name, str) or not name:
            continue
        calls.append(
            {
                "type": "function_call",
                "call_id": f"call_{uuid.uuid4().hex}",
                "name": name,
                "arguments": arguments,
            }
        )
    return calls


def _message_to_response_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    reasoning_content = _content_to_text(message.get("reasoning_content"))
    if reasoning_content:
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_bridge_{uuid.uuid4().hex}",
                "summary": [],
                "text": reasoning_content,
                "content": [{"type": "reasoning_text", "text": reasoning_content}],
            }
        )
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            continue
        arguments = fn.get("arguments", "{}")
        arguments = _normalize_tool_arguments(arguments)
        items.append(
            {
                "type": "function_call",
                "call_id": str(call.get("id") or f"call_{uuid.uuid4().hex}"),
                "name": str(fn.get("name") or ""),
                "arguments": arguments,
            }
        )
    content = _content_to_text(message.get("content"))
    xml_calls = _xml_tool_calls_from_text(content)
    if xml_calls:
        items.extend(xml_calls)
        return items
    if content:
        items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]})
    return items


def _responses_usage(chat_usage: Any) -> dict[str, Any]:
    if not isinstance(chat_usage, dict):
        return {
            "input_tokens": 0,
            "input_tokens_details": None,
            "output_tokens": 0,
            "output_tokens_details": None,
            "total_tokens": 0,
        }
    input_tokens = int(chat_usage.get("input_tokens", chat_usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(chat_usage.get("output_tokens", chat_usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(chat_usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": chat_usage.get("input_tokens_details"),
        "output_tokens": output_tokens,
        "output_tokens_details": chat_usage.get("output_tokens_details"),
        "total_tokens": total_tokens,
    }


def _response_body(payload: dict[str, Any], chat_body: dict[str, Any]) -> dict[str, Any]:
    rid = f"resp_bridge_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    message = (chat_body.get("choices") or [{}])[0].get("message") or {}
    output = _message_to_response_items(message)
    output_text = "\n".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if isinstance(part, dict)
    )
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "model": payload.get("model"),
        "status": "completed",
        "output": output,
        "output_text": output_text,
        "usage": _responses_usage(chat_body.get("usage")),
    }


def _sse_response_text(resp: dict[str, Any]) -> str:
    def event(obj: dict[str, Any]) -> str:
        return f"event: {obj['type']}\ndata: {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))}\n\n"

    rid = str(resp["id"])
    chunks = [event({"type": "response.created", "response": {"id": rid}})]
    for item in resp.get("output") or []:
        chunks.append(event({"type": "response.output_item.done", "item": item}))
    chunks.append(event({"type": "response.completed", "response": {"id": rid, "usage": resp.get("usage", {})}}))
    return "".join(chunks)


def _json_response_from_upstream(response: Any) -> Any:
    if not getattr(response, "text", ""):
        return JSONResponse(status_code=response.status_code, content={})
    try:
        return JSONResponse(status_code=response.status_code, content=response.json())
    except Exception:
        status = response.status_code if response.status_code >= 300 else 502
        return JSONResponse(
            status_code=status,
            content={
                "error": str(getattr(response, "text", ""))[:4000],
                "upstream_status_code": response.status_code,
            },
        )


def _iter_upstream_bytes(response: Any):
    try:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk
    finally:
        response.close()


def create_app(
    target_base_url: str,
    api_key: str = "dummy",
    default_max_tokens: int | None = DEFAULT_RESPONSE_MAX_TOKENS,
):
    if FastAPI is None or requests is None:
        raise RuntimeError("Install optional bridge deps: pip install fastapi uvicorn requests")
    app = FastAPI(title="Responses-to-Chat Bridge")
    target = target_base_url.rstrip("/")

    @app.get("/v1/models")
    async def models():
        r = requests.get(f"{target}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        return _json_response_from_upstream(r)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        payload = _sanitize_chat_payload(payload)
        if payload.get("stream"):
            r = requests.post(
                f"{target}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=600,
                stream=True,
            )
            if r.status_code >= 300:
                return _json_response_from_upstream(r)
            return StreamingResponse(
                _iter_upstream_bytes(r),
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "text/event-stream"),
            )
        r = requests.post(f"{target}/chat/completions", json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=600)
        return _json_response_from_upstream(r)

    @app.post("/v1/responses")
    async def responses(request: Request):
        payload = await request.json()
        chat_payload = _chat_payload_from_responses(payload, default_max_tokens=default_max_tokens)
        r = requests.post(f"{target}/chat/completions", json=chat_payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=1800)
        if r.status_code >= 300:
            return JSONResponse(status_code=r.status_code, content={"error": r.text[:4000]})
        resp = _response_body(payload, r.json())
        if payload.get("stream"):
            return StreamingResponse(iter([_sse_response_text(resp).encode("utf-8")]), media_type="text/event-stream")
        return JSONResponse(content=resp)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="/v1/responses compatibility bridge for chat-only local backends.")
    ap.add_argument("--target-base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--default-max-tokens", type=int, default=DEFAULT_RESPONSE_MAX_TOKENS)
    args = ap.parse_args()
    try:
        import uvicorn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Install optional bridge deps: pip install fastapi uvicorn requests") from e
    uvicorn.run(
        create_app(args.target_base_url, args.api_key, default_max_tokens=args.default_max_tokens),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
