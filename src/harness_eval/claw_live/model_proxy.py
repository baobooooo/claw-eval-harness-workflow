from __future__ import annotations

import json
import os
import shlex
import threading
import time
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from harness_eval.types import BenchmarkTask, ModelProfile


_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
}


class ModelToolProxyError(RuntimeError):
    pass


def model_tool_proxy_config(hcfg: dict[str, Any]) -> dict[str, Any]:
    raw = hcfg.get("model_tool_proxy") or hcfg.get("claw_tool_model_proxy") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def model_tool_proxy_enabled(task: BenchmarkTask, hcfg: dict[str, Any]) -> bool:
    cfg = model_tool_proxy_config(hcfg)
    if "enabled" in cfg:
        return bool(cfg.get("enabled"))
    # The proxy is intentionally opt-in.  A live bridge only means that the
    # Claw-Eval sandbox/dispatcher exists; it should not silently replace the
    # selected harness' own agent loop.
    return False


def model_tool_proxy_dry_manifest(task: BenchmarkTask, model: ModelProfile, hcfg: dict[str, Any], *, harness_name: str) -> dict[str, Any]:
    cfg = model_tool_proxy_config(hcfg)
    return {
        "enabled": bool(model_tool_proxy_enabled(task, hcfg)),
        "mode": "harness_cli_with_claw_model_proxy",
        "harness": harness_name,
        "upstream_base_url": model.base_url,
        "base_url": "<allocated-at-runtime>",
        "transport_tool": _transport_tool_name(cfg, harness_name),
        "transport_argument_key": _transport_argument_key(cfg, harness_name),
        "model_visible_tools": _model_visible_tool_names(task),
        "log_path": str(task.output_dir / f"{harness_name}_model_tool_proxy.jsonl"),
    }


def _model_visible_tool_names(task: BenchmarkTask) -> list[str]:
    return [
        str(spec.get("name"))
        for spec in task.metadata.get("allowed_tool_specs") or []
        if isinstance(spec, dict) and spec.get("name")
    ]


def _normalise_tool_spec(raw: Any) -> dict[str, Any] | None:
    """Copy one Claw-Eval tool spec into OpenAI-compatible shape.

    Keep the user/model-visible surface byte-for-byte equivalent to the
    official MiniHarness conversion as far as this adapter can control it:
    name, description, schema, and order come from task.metadata without
    adding fallback prose or re-sorting.  Execution transport details are
    handled only after the model returns a tool call.
    """
    if not isinstance(raw, dict) or not raw.get("name"):
        return None
    schema = raw.get("input_schema") if "input_schema" in raw else raw.get("parameters")
    if not isinstance(schema, dict):
        schema = {}
    return {
        "name": str(raw.get("name")),
        "description": str(raw.get("description") or ""),
        "parameters": schema,
    }


def _chat_tool_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description") or "",
            "parameters": spec.get("parameters") if isinstance(spec.get("parameters"), dict) else {},
        },
    }


def _responses_tool_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": spec["name"],
        "description": spec.get("description") or "",
        "parameters": spec.get("parameters") if isinstance(spec.get("parameters"), dict) else {},
    }


def _tool_names_from_request(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for raw in payload.get("tools") or []:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not name and isinstance(raw.get("function"), dict):
            name = raw["function"].get("name")
        if name:
            names.add(str(name))
    return names


def _transport_tool_name(cfg: dict[str, Any], harness_name: str) -> str:
    raw = cfg.get("transport_tool_name") or cfg.get("tool_name")
    if raw:
        return str(raw)
    if harness_name == "codex":
        # Current Codex Responses API exposes its shell transport as
        # `exec_command` (not `shell`).  Returning `shell` makes Codex
        # accept the model response but then fail at the internal router with
        # "unsupported call: shell" before ./claw_tool can run.
        return "exec_command"
    if harness_name == "openclaw":
        # OpenClaw's native shell transport is logged/routed as `exec`.
        # The model never sees this name; it is only the hidden post-model
        # carrier for YAML Claw-Eval tool calls.
        return "exec"
    if harness_name == "nanobot":
        return "exec"
    return "exec"


def _transport_argument_key(cfg: dict[str, Any], harness_name: str) -> str:
    raw = cfg.get("transport_argument_key") or cfg.get("argument_key")
    if raw:
        return str(raw)
    if harness_name == "codex":
        return "cmd"
    return "command"


def _coerce_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except Exception:
            return {"_raw": raw}
        return dict(parsed) if isinstance(parsed, dict) else {"_value": parsed}
    return {}


def _json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _safe_call_id(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    text = "".join(out).strip("._")
    return text[:96] or uuid.uuid4().hex


def _call_id_keys(value: str | None) -> list[str]:
    """Return stable lookup aliases for harness-mutated tool call ids.

    Some Chat Completions harnesses sanitize ids before replaying history.
    OpenClaw, for example, can turn ``call_00_abc`` into ``call00abc``.
    The proxy must still restore that history to the original YAML tool call
    rather than leaking the hidden ``exec`` transport back to the model.
    """
    raw = str(value or "")
    if not raw:
        return []
    variants = [raw]
    compact = "".join(ch for ch in raw if ch.isalnum())
    if compact and compact not in variants:
        variants.append(compact)
    lowered = raw.lower()
    if lowered and lowered not in variants:
        variants.append(lowered)
    compact_lower = compact.lower()
    if compact_lower and compact_lower not in variants:
        variants.append(compact_lower)
    return variants


def _first_json_object_from_text(text: str) -> Any | None:
    stripped = text.lstrip()
    if not stripped:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _idx = decoder.raw_decode(stripped)
    except Exception:
        return None
    return value


def _canonical_tool_result_content(content: Any) -> Any:
    """Hide the native transport envelope from later model turns.

    Harness transports often return wrapper objects such as
    ``{"body": ..., "endpoint_url": ..., "tool_name": ...}`` or append
    native shell decorations like ``Exit code: 0``.  The upstream model should
    see the Claw-Eval tool result, not the helper command or transport metadata.
    """
    if isinstance(content, list):
        return [_canonical_tool_result_content(item) for item in content]
    if not isinstance(content, str):
        return content
    parsed = _first_json_object_from_text(content)
    if isinstance(parsed, dict):
        if "body" in parsed:
            body = parsed.get("body")
            return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True)
        hidden_keys = {"endpoint_url", "latency_ms", "tool_name", "tool_use_id", "status", "is_error"}
        if hidden_keys.intersection(parsed):
            cleaned = {k: v for k, v in parsed.items() if k not in hidden_keys}
            return json.dumps(cleaned, ensure_ascii=False, indent=2, sort_keys=True)
    return content


def _contains_hidden_transport_text(value: Any) -> bool:
    try:
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    except Exception:
        text = str(value)
    needles = ["claw_tool", ".claw_tool_payloads", "exec_command", "Command still running (session"]
    return any(needle in text for needle in needles)


def _redacted_headers(headers: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in {"authorization", "cookie", "set-cookie"}:
            out[key] = "<redacted>"
        else:
            out[key] = value
    return out


def _write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def _response_items_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    output = body.get("output")
    return output if isinstance(output, list) else []


def _make_response_sse(body: dict[str, Any]) -> bytes:
    rid = str(body.get("id") or f"resp_{uuid.uuid4().hex}")
    created = {"type": "response.created", "response": {k: v for k, v in body.items() if k != "output"}}
    events = [created]
    for idx, item in enumerate(_response_items_from_body(body)):
        if isinstance(item, dict):
            events.append({"type": "response.output_item.done", "output_index": idx, "item": item})
    events.append({"type": "response.completed", "response": body})
    chunks = []
    for event in events:
        chunks.append(f"event: {event['type']}\n")
        chunks.append("data: " + json.dumps(event, ensure_ascii=False) + "\n\n")
    return "".join(chunks).encode("utf-8")


def _make_chat_sse(body: dict[str, Any]) -> bytes:
    """Render a buffered Chat Completions response as OpenAI-style SSE.

    The OpenAI Python SDK expects streaming responses as incremental deltas:
    a role chunk, zero or more content/tool-call chunks, then a final chunk
    carrying only ``finish_reason``.  Sending the full assistant message and
    finish reason in the same first chunk is accepted by simple line readers,
    but can leave SDK stream consumers waiting or surfacing ``data: [DONE]`` as
    an error.
    """

    def make_chunk(index: int, delta: dict[str, Any], finish_reason: Any = None) -> dict[str, Any]:
        return {
            "id": body.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": body.get("created") or int(time.time()),
            "model": body.get("model"),
            "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
        }

    chunks: list[str] = []
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        choice_index = int(choice.get("index", 0) or 0)
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        role = message.get("role") or "assistant"
        events: list[dict[str, Any]] = [make_chunk(choice_index, {"role": role})]

        content = message.get("content")
        if content not in (None, ""):
            events.append(make_chunk(choice_index, {"content": content}))

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            deltas = []
            for idx, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                delta_call = dict(call)
                delta_call.setdefault("index", idx)
                deltas.append(delta_call)
            if deltas:
                events.append(make_chunk(choice_index, {"tool_calls": deltas}))

        func_call = message.get("function_call") if isinstance(message.get("function_call"), dict) else None
        if func_call:
            events.append(make_chunk(choice_index, {"function_call": func_call}))

        events.append(make_chunk(choice_index, {}, choice.get("finish_reason") or "stop"))
        for event in events:
            chunks.append("data: " + json.dumps(event, ensure_ascii=False) + "\n\n")

    usage = body.get("usage")
    if isinstance(usage, dict):
        usage_chunk = {
            "id": body.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": body.get("created") or int(time.time()),
            "model": body.get("model"),
            "choices": [],
            "usage": usage,
        }
        chunks.append("data: " + json.dumps(usage_chunk, ensure_ascii=False) + "\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


class ClawToolModelProxy:
    """OpenAI-compatible proxy that makes YAML Claw-Eval tools model-visible.

    Codex/OpenClaw/NanoBot still own the harness loop.  They send their normal
    OpenAI-compatible request, usually with a native shell/exec tool.  This
    proxy replaces those model-visible tools with the Claw-Eval YAML + sandbox
    tools from task.metadata.  If the model calls one of those Claw tools, the
    proxy rewrites the response back to one hidden transport tool call that the
    harness can execute, e.g. a harness-native exec transport such as `exec_command({cmd: "python3 ./claw_tool web_fetch @..."})`.
    The transport tool result is then replayed to the upstream model as the
    original Claw tool result in later turns.
    """

    def __init__(
        self,
        *,
        task: BenchmarkTask,
        model: ModelProfile,
        hcfg: dict[str, Any],
        harness_name: str,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.task = task
        self.model = model
        self.hcfg = hcfg
        self.cfg = model_tool_proxy_config(hcfg)
        self.harness_name = harness_name
        self.host = host
        self.port = int(self.cfg.get("port", port) or port)
        self.upstream_base_url = str(model.base_url).rstrip("/")
        if not self.upstream_base_url:
            raise ModelToolProxyError("model.base_url is required for Claw model tool proxy")
        self.allowed_specs = [s for s in (_normalise_tool_spec(raw) for raw in task.metadata.get("allowed_tool_specs") or []) if s]
        self.allowed_names = {str(spec["name"]) for spec in self.allowed_specs}
        self.transport_tool_name = _transport_tool_name(self.cfg, harness_name)
        self.transport_argument_key = _transport_argument_key(self.cfg, harness_name)
        self.command_template = str(
            self.cfg.get("command_template")
            or "python3 ./claw_tool {tool_name} @{payload_path}"
        )
        self.extra_transport_arguments = dict(self.cfg.get("extra_transport_arguments") or {})
        self.keep_native_tools = bool(self.cfg.get("keep_native_tools", False))
        self.strict_tool_surface = bool(self.cfg.get("strict_tool_surface", True))
        self.log_path = Path(str(self.cfg.get("log_path") or task.output_dir / f"{harness_name}_model_tool_proxy.jsonl"))
        self.payload_dir = Path(str(self.cfg.get("payload_dir") or Path(str(task.metadata.get("agent_workspace") or task.workspace)) / ".claw_tool_payloads"))
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str | None = None
        self._lock = threading.Lock()
        self._call_map: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "ClawToolModelProxy":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def start(self) -> str:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
                return

            def do_GET(self) -> None:  # noqa: N802
                proxy._handle(self)

            def do_POST(self) -> None:  # noqa: N802
                proxy._handle(self)

            def do_OPTIONS(self) -> None:  # noqa: N802
                proxy._send_json(self, 200, {"ok": True})

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        host, port = self._httpd.server_address
        self.base_url = f"http://{host}:{port}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, name=f"{self.harness_name}-claw-model-tool-proxy", daemon=True)
        self._thread.start()
        _write_jsonl(
            self.log_path,
            {
                "event": "start",
                "harness": self.harness_name,
                "base_url": self.base_url,
                "upstream_base_url": self.upstream_base_url,
                "allowed_tools": [str(spec["name"]) for spec in self.allowed_specs],
                "transport_tool": self.transport_tool_name,
                "transport_argument_key": self.transport_argument_key,
            },
        )
        return self.base_url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self.base_url:
            _write_jsonl(self.log_path, {"event": "stop", "harness": self.harness_name, "base_url": self.base_url})

    def proxied_model(self) -> ModelProfile:
        if not self.base_url:
            raise ModelToolProxyError("proxy has not been started")
        return replace(
            self.model,
            base_url=self.base_url,
            notes=((self.model.notes + "\n") if self.model.notes else "")
            + f"Claw-Eval model tool proxy for {self.harness_name}; upstream={self.upstream_base_url}",
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": "harness_cli_with_claw_model_proxy",
            "harness": self.harness_name,
            "base_url": self.base_url,
            "upstream_base_url": self.upstream_base_url,
            "transport_tool": self.transport_tool_name,
            "transport_argument_key": self.transport_argument_key,
            "command_template": self.command_template,
            # Preserve the exact model-visible order used in the API request.
            "model_visible_tools": [str(spec["name"]) for spec in self.allowed_specs],
            "log_path": str(self.log_path),
            "payload_dir": str(self.payload_dir),
            "keep_native_tools": self.keep_native_tools,
            "strict_tool_surface": self.strict_tool_surface,
        }

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        started = time.time()
        raw_body = self._read_body(handler)
        path = urlsplit(handler.path).path
        query = urlsplit(handler.path).query
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else None
        except Exception:
            payload = None
        request_id = uuid.uuid4().hex
        _write_jsonl(
            self.log_path,
            {
                "event": "request",
                "id": request_id,
                "method": handler.command,
                "path": handler.path,
                "headers": _redacted_headers(dict(handler.headers)),
                "body": payload if isinstance(payload, dict) else None,
            },
        )
        try:
            if handler.command == "GET" and path.rstrip("/") in {"", "/v1", "/health", "/v1/health"}:
                self._send_json(handler, 200, {"ok": True, "harness": self.harness_name, "upstream_base_url": self.upstream_base_url})
                return
            if isinstance(payload, dict) and path.endswith("/chat/completions"):
                status, headers, body = self._proxy_json_request(handler, path, query, payload, kind="chat")
                self._send_response(handler, status, headers, body)
                return
            if isinstance(payload, dict) and path.endswith("/responses"):
                status, headers, body = self._proxy_json_request(handler, path, query, payload, kind="responses")
                self._send_response(handler, status, headers, body)
                return
            status, headers, body = self._forward_raw(handler, path, query, raw_body)
            self._send_response(handler, status, headers, body)
        except Exception as exc:
            _write_jsonl(
                self.log_path,
                {
                    "event": "proxy_error",
                    "id": request_id,
                    "duration_s": round(time.time() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            self._send_json(handler, 502, {"error": f"Claw model tool proxy error: {type(exc).__name__}: {exc}"})

    @staticmethod
    def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
        length = int(handler.headers.get("Content-Length") or "0")
        return handler.rfile.read(length) if length else b""

    def _forward_url(self, path: str, query: str = "") -> str:
        if path.startswith("/v1/"):
            suffix = path[len("/v1") :]
        elif path == "/v1":
            suffix = ""
        else:
            suffix = path
        url = self.upstream_base_url + suffix
        if query:
            url = f"{url}?{query}"
        return url

    def _forward_headers(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        headers = {
            k: v
            for k, v in dict(handler.headers).items()
            if k.lower() not in _HOP_BY_HOP_HEADERS
        }
        # The proxy is local; keep the harness' Authorization if present so
        # experiments do not need a separate API key for the proxy itself.
        return headers

    def _forward_raw(self, handler: BaseHTTPRequestHandler, path: str, query: str, raw_body: bytes) -> tuple[int, dict[str, str], bytes]:
        with httpx.Client(timeout=None, trust_env=False) as client:
            resp = client.request(
                handler.command,
                self._forward_url(path, query),
                headers=self._forward_headers(handler),
                content=raw_body,
            )
        return resp.status_code, self._response_headers(resp.headers), resp.content

    def _proxy_json_request(self, handler: BaseHTTPRequestHandler, path: str, query: str, payload: dict[str, Any], *, kind: str) -> tuple[int, dict[str, str], bytes]:
        original_stream = bool(payload.get("stream"))
        upstream_payload = self._request_for_upstream(payload, kind=kind)
        if original_stream and bool(self.cfg.get("force_non_streaming", True)):
            upstream_payload["stream"] = False
            # OpenAI-compatible APIs reject stream_options when stream=false.
            # NanoBot always asks for include_usage during streaming, so remove
            # the option when the proxy de-streams the upstream request.
            upstream_payload.pop("stream_options", None)
        self._log_upstream_request(upstream_payload, kind=kind, path=path)
        with httpx.Client(timeout=None, trust_env=False) as client:
            resp = client.post(
                self._forward_url(path, query),
                headers={**self._forward_headers(handler), "Content-Type": "application/json"},
                json=upstream_payload,
            )
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            return resp.status_code, self._response_headers(resp.headers), resp.content
        try:
            body = resp.json()
        except Exception:
            return resp.status_code, self._response_headers(resp.headers), resp.content
        if isinstance(body, dict) and resp.status_code < 400:
            body = self._response_for_harness(body, kind=kind)
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if original_stream:
            if kind == "responses":
                return resp.status_code, {"Content-Type": "text/event-stream; charset=utf-8"}, _make_response_sse(body if isinstance(body, dict) else {})
            return resp.status_code, {"Content-Type": "text/event-stream; charset=utf-8"}, _make_chat_sse(body if isinstance(body, dict) else {})
        return resp.status_code, headers, raw

    @staticmethod
    def _response_headers(headers: httpx.Headers) -> dict[str, str]:
        return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}

    @staticmethod
    def _send_json(handler: BaseHTTPRequestHandler, status: int, body: Any) -> None:
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)

    @staticmethod
    def _send_response(handler: BaseHTTPRequestHandler, status: int, headers: dict[str, str], body: bytes) -> None:
        handler.send_response(status)
        for key, value in headers.items():
            if key.lower() in _HOP_BY_HOP_HEADERS:
                continue
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _request_for_upstream(self, payload: dict[str, Any], *, kind: str) -> dict[str, Any]:
        out = json.loads(json.dumps(payload, ensure_ascii=False))
        if kind == "chat":
            out["messages"] = self._chat_messages_for_upstream(out.get("messages"))
            claw_tools = [_chat_tool_from_spec(spec) for spec in self.allowed_specs]
        else:
            out["input"] = self._responses_input_for_upstream(out.get("input"))
            claw_tools = [_responses_tool_from_spec(spec) for spec in self.allowed_specs]
        native_tools = out.get("tools") if isinstance(out.get("tools"), list) else []
        if self.keep_native_tools:
            out["tools"] = claw_tools + native_tools
        else:
            out["tools"] = claw_tools
        # MiniHarness does not need an explicit tool_choice for normal OpenAI
        # tool calling.  Do not add one when the harness omitted it; only repair
        # an explicit "none" that would make the fixed tool surface unusable.
        if claw_tools and kind == "chat" and out.get("tool_choice") == "none":
            out["tool_choice"] = "auto"
        self._log_model_visible_surface(out, kind=kind)
        return out

    def _log_model_visible_surface(self, payload: dict[str, Any], *, kind: str) -> None:
        tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
        observed = []
        for raw in tools:
            if not isinstance(raw, dict):
                continue
            if kind == "chat" and isinstance(raw.get("function"), dict):
                fn = raw["function"]
                observed.append({
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters"),
                })
            elif kind == "responses":
                observed.append({
                    "name": raw.get("name"),
                    "description": raw.get("description"),
                    "parameters": raw.get("parameters"),
                })
        expected = [
            {"name": spec["name"], "description": spec.get("description") or "", "parameters": spec.get("parameters") if isinstance(spec.get("parameters"), dict) else {}}
            for spec in self.allowed_specs
        ]
        _write_jsonl(
            self.log_path,
            {
                "event": "model_visible_tool_surface",
                "kind": kind,
                "tool_names": [str(item.get("name")) for item in observed if item.get("name")],
                "matches_expected": observed == expected,
                "expected": expected,
                "observed": observed,
            },
        )

    def _log_upstream_request(self, payload: dict[str, Any], *, kind: str, path: str) -> None:
        tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
        tool_names: list[str] = []
        for raw in tools:
            if not isinstance(raw, dict):
                continue
            name = raw.get("name")
            if not name and isinstance(raw.get("function"), dict):
                name = raw["function"].get("name")
            if name:
                tool_names.append(str(name))
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else None
        input_items = payload.get("input") if isinstance(payload.get("input"), list) else None
        _write_jsonl(
            self.log_path,
            {
                "event": "upstream_request",
                "kind": kind,
                "path": path,
                "model": payload.get("model"),
                "stream": payload.get("stream"),
                "tool_names": tool_names,
                "message_count": len(messages) if messages is not None else None,
                "input_item_count": len(input_items) if input_items is not None else None,
                "body": payload,
            },
        )

    def _chat_messages_for_upstream(self, messages: Any) -> Any:
        if not isinstance(messages, list):
            return messages
        return [self._chat_message_for_upstream(item) if isinstance(item, dict) else item for item in messages]

    def _chat_message_for_upstream(self, message: dict[str, Any]) -> dict[str, Any]:
        out = dict(message)
        if out.get("role") == "tool":
            return self._chat_tool_result_for_upstream(out)
        tool_calls = out.get("tool_calls")
        if isinstance(tool_calls, list):
            rewritten = []
            for call in tool_calls:
                rewritten.append(self._chat_tool_call_for_upstream(call) if isinstance(call, dict) else call)
            out["tool_calls"] = rewritten
        return out

    def _remember_call_map(self, call_id: str, value: dict[str, Any]) -> None:
        for key in _call_id_keys(call_id):
            self._call_map[key] = value

    def _lookup_call_map(self, call_id: str | None) -> dict[str, Any] | None:
        for key in _call_id_keys(call_id):
            original = self._call_map.get(key)
            if original:
                return original
        return None

    def _chat_tool_call_for_upstream(self, call: dict[str, Any]) -> dict[str, Any]:
        out = json.loads(json.dumps(call, ensure_ascii=False))
        call_id = str(out.get("id") or "")
        func = out.get("function") if isinstance(out.get("function"), dict) else None
        if not call_id or func is None:
            return out
        with self._lock:
            original = self._lookup_call_map(call_id)
        if not original:
            return out
        func["name"] = original["tool_name"]
        func["arguments"] = original["arguments"]
        if _contains_hidden_transport_text(call):
            _write_jsonl(self.log_path, {"event": "restore_tool_call_history", "kind": "chat", "call_id": call_id, "restored_tool": original["tool_name"]})
        return out

    def _chat_tool_result_for_upstream(self, message: dict[str, Any]) -> dict[str, Any]:
        out = json.loads(json.dumps(message, ensure_ascii=False))
        call_id = str(out.get("tool_call_id") or "")
        with self._lock:
            original = self._lookup_call_map(call_id)
        if not original:
            return out
        if "name" in out:
            out["name"] = original["tool_name"]
        out["content"] = _canonical_tool_result_content(out.get("content"))
        if _contains_hidden_transport_text(message):
            _write_jsonl(self.log_path, {"event": "restore_tool_result_history", "kind": "chat", "call_id": call_id, "restored_tool": original["tool_name"]})
        return out

    def _responses_input_for_upstream(self, items: Any) -> Any:
        if not isinstance(items, list):
            return items
        out = []
        for item in items:
            if isinstance(item, dict):
                out.append(self._responses_item_for_upstream(item))
            else:
                out.append(item)
        return out

    def _responses_item_for_upstream(self, item: dict[str, Any]) -> dict[str, Any]:
        out = dict(item)
        if out.get("type") != "function_call":
            return out
        call_id = str(out.get("call_id") or out.get("id") or "")
        with self._lock:
            original = self._lookup_call_map(call_id)
        if not original:
            return out
        out["name"] = original["tool_name"]
        out["arguments"] = original["arguments"]
        return out

    def _response_for_harness(self, body: dict[str, Any], *, kind: str) -> dict[str, Any]:
        out = json.loads(json.dumps(body, ensure_ascii=False))
        if kind == "chat":
            for choice in out.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message") if isinstance(choice.get("message"), dict) else None
                if message is None:
                    continue
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    blocked = [call for call in tool_calls if isinstance(call, dict) and self._chat_tool_call_name(call) not in self.allowed_names]
                    if blocked and self.strict_tool_surface:
                        self._replace_chat_message_with_blocked_tool_notice(message, blocked)
                        choice["finish_reason"] = "stop"
                    else:
                        message["tool_calls"] = [self._chat_tool_call_for_harness(call) if isinstance(call, dict) else call for call in tool_calls]
                func_call = message.get("function_call") if isinstance(message.get("function_call"), dict) else None
                if func_call and str(func_call.get("name")) in self.allowed_names:
                    # Legacy Chat Completions function_call has no reliable id;
                    # leave it untouched rather than making history unreplayable.
                    pass
        else:
            items = out.get("output")
            if isinstance(items, list):
                out["output"] = [self._responses_item_for_harness(item) if isinstance(item, dict) else item for item in items]
        return out

    @staticmethod
    def _chat_tool_call_name(call: dict[str, Any]) -> str:
        func = call.get("function") if isinstance(call.get("function"), dict) else None
        return str(func.get("name") or "") if func else ""

    def _replace_chat_message_with_blocked_tool_notice(self, message: dict[str, Any], blocked: list[dict[str, Any]]) -> None:
        blocked_names = [self._chat_tool_call_name(call) for call in blocked if self._chat_tool_call_name(call)]
        _write_jsonl(
            self.log_path,
            {
                "event": "blocked_non_claw_tool_call",
                "kind": "chat",
                "blocked_tool_names": blocked_names,
                "allowed_tools": [str(spec["name"]) for spec in self.allowed_specs],
            },
        )
        message.pop("tool_calls", None)
        message["content"] = (
            "The requested tool call was blocked by the Claw-Eval fair-tool proxy because it is not in "
            "the task YAML/API tool surface. Use only these Claw-Eval tools: "
            + ", ".join(str(spec["name"]) for spec in self.allowed_specs)
        )

    def _chat_tool_call_for_harness(self, call: dict[str, Any]) -> dict[str, Any]:
        out = json.loads(json.dumps(call, ensure_ascii=False))
        func = out.get("function") if isinstance(out.get("function"), dict) else None
        if func is None:
            return out
        tool_name = str(func.get("name") or "")
        if tool_name not in self.allowed_names:
            return out
        call_id = str(out.get("id") or f"call_{uuid.uuid4().hex}")
        out["id"] = call_id
        arguments = _json_string(func.get("arguments", "{}"))
        transport_args = self._transport_arguments(call_id, tool_name, arguments)
        with self._lock:
            self._remember_call_map(call_id, {"tool_name": tool_name, "arguments": arguments, "transport_arguments": transport_args})
        func["name"] = self.transport_tool_name
        func["arguments"] = json.dumps(transport_args, ensure_ascii=False, sort_keys=True)
        _write_jsonl(self.log_path, {"event": "rewrite_tool_call", "kind": "chat", "call_id": call_id, "from_tool": tool_name, "to_tool": self.transport_tool_name, "transport_arguments": transport_args})
        return out

    def _responses_item_for_harness(self, item: dict[str, Any]) -> dict[str, Any]:
        out = dict(item)
        if out.get("type") != "function_call":
            return out
        tool_name = str(out.get("name") or "")
        if tool_name not in self.allowed_names:
            return out
        call_id = str(out.get("call_id") or out.get("id") or f"call_{uuid.uuid4().hex}")
        out["call_id"] = call_id
        arguments = _json_string(out.get("arguments", "{}"))
        transport_args = self._transport_arguments(call_id, tool_name, arguments)
        with self._lock:
            self._remember_call_map(call_id, {"tool_name": tool_name, "arguments": arguments, "transport_arguments": transport_args})
        out["name"] = self.transport_tool_name
        out["arguments"] = json.dumps(transport_args, ensure_ascii=False, sort_keys=True)
        _write_jsonl(self.log_path, {"event": "rewrite_tool_call", "kind": "responses", "call_id": call_id, "from_tool": tool_name, "to_tool": self.transport_tool_name, "transport_arguments": transport_args})
        return out

    def _transport_arguments(self, call_id: str, tool_name: str, arguments: str) -> dict[str, Any]:
        payload = _coerce_json_object(arguments)
        safe_id = _safe_call_id(call_id)
        payload_path = self.payload_dir / f"{safe_id}.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        agent_workspace = Path(str(self.task.metadata.get("agent_workspace") or self.task.workspace))
        try:
            rel_payload_path = os.path.relpath(payload_path, agent_workspace)
        except Exception:
            rel_payload_path = str(payload_path)
        claw_tool_path = agent_workspace / "claw_tool"
        command = self.command_template.format(
            tool_name=shlex.quote(tool_name),
            raw_tool_name=tool_name,
            payload_path=shlex.quote(rel_payload_path),
            raw_payload_path=rel_payload_path,
            absolute_payload_path=shlex.quote(str(payload_path)),
            raw_absolute_payload_path=str(payload_path),
            agent_workspace=shlex.quote(str(agent_workspace)),
            raw_agent_workspace=str(agent_workspace),
            claw_tool_path=shlex.quote(str(claw_tool_path)),
            raw_claw_tool_path=str(claw_tool_path),
            bridge_url=shlex.quote(str(self.task.metadata.get("claw_tool_bridge_url") or "")),
            raw_bridge_url=str(self.task.metadata.get("claw_tool_bridge_url") or ""),
            call_id=shlex.quote(call_id),
            raw_call_id=call_id,
        )
        transport_args = dict(self.extra_transport_arguments)
        transport_args[self.transport_argument_key] = command
        return transport_args
