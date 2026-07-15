from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from harness_eval.claw_live.trace import LiveTraceWriter


SANDBOX_PATH_MAP: dict[str, str] = {
    "Bash": "/exec",
    "Read": "/read",
    "Write": "/write",
    "Edit": "/edit",
    "Glob": "/glob",
    "Grep": "/grep",
    "BrowserScreenshot": "/screenshot",
    "ReadMedia": "/read_media",
    "Download": "/download",
}

FALLBACK_SANDBOX_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "Bash",
        "description": "Executes a given bash command and returns its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "timeout": {"type": "integer"},
                "timeout_seconds": {"type": "number"},
                "run_in_background": {"type": "boolean"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Reads a file from the Claw-Eval sandbox.",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["file_path"]},
    },
    {
        "name": "Write",
        "description": "Writes a file in the Claw-Eval sandbox.",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]},
    },
    {
        "name": "Edit",
        "description": "Edits a file in the Claw-Eval sandbox by exact string replacement.",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["file_path", "old_string", "new_string"]},
    },
    {"name": "Glob", "description": "Lists files in the Claw-Eval sandbox by glob pattern.", "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "Grep", "description": "Searches files in the Claw-Eval sandbox.", "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "output_mode": {"type": "string"}, "case_insensitive": {"type": "boolean"}}, "required": ["pattern"]}},
    {"name": "BrowserScreenshot", "description": "Captures a browser screenshot in the sandbox.", "input_schema": {"type": "object", "properties": {"url": {"type": "string"}, "wait_seconds": {"type": "number"}, "frame_count": {"type": "integer"}}, "required": ["url"]}},
    {"name": "ReadMedia", "description": "Reads video/image/PDF media from the sandbox.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "media_type": {"type": "string"}, "max_frames": {"type": "integer"}}, "required": ["path"]}},
    {"name": "Download", "description": "Downloads a file from the sandbox as base64.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}}, "required": ["path"]}},
]
FALLBACK_SANDBOX_TOOL_NAMES = frozenset(spec["name"] for spec in FALLBACK_SANDBOX_TOOL_SPECS)


@dataclass(slots=True)
class DispatchResult:
    status: int
    body: Any
    is_error: bool
    tool_use_id: str
    tool_name: str
    endpoint_url: str
    latency_ms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_use_id": self.tool_use_id,
            "tool_name": self.tool_name,
            "is_error": self.is_error,
            "status": self.status,
            "endpoint_url": self.endpoint_url,
            "latency_ms": self.latency_ms,
            "body": self.body,
        }


def _tool_specs_from_official() -> tuple[list[dict[str, Any]], set[str]]:
    try:
        from claw_eval.runner.sandbox_tools import SANDBOX_TOOLS, SANDBOX_TOOL_NAMES  # type: ignore
    except Exception:
        return list(FALLBACK_SANDBOX_TOOL_SPECS), set(FALLBACK_SANDBOX_TOOL_NAMES)
    specs = []
    for tool in SANDBOX_TOOLS:
        if hasattr(tool, "model_dump"):
            specs.append(tool.model_dump(mode="json"))
        else:
            specs.append({"name": tool.name, "description": tool.description, "input_schema": tool.input_schema})
    return specs, set(SANDBOX_TOOL_NAMES)


def _normalise_task_tool_specs(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in tool_specs:
        if not isinstance(spec, dict) or not spec.get("name"):
            continue
        out.append(
            {
                "name": str(spec.get("name")),
                "description": str(spec.get("description") or ""),
                "input_schema": spec.get("input_schema") if isinstance(spec.get("input_schema"), dict) else {},
            }
        )
    return out


def _sandbox_path(value: Any, *, default: str | None = None) -> str | None:
    if value in (None, ""):
        return default
    text = str(value)
    if text.startswith("/"):
        return text
    return "/workspace/" + text.lstrip("./")


def _translate_sandbox_payload(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    translated = dict(payload)
    if tool_name == "Bash":
        # Codex frequently emits {"cmd": "..."}; Claw-Eval's sandbox server
        # expects {"command": "..."}.  Accept both to avoid wrapper-level
        # JSON/argument churn while preserving the official tool boundary.
        if "cmd" in translated and "command" not in translated:
            translated["command"] = translated.pop("cmd")
        if "timeout" in translated and "timeout_seconds" not in translated:
            try:
                translated["timeout_seconds"] = max(1, int(translated.pop("timeout")) // 1000)
            except Exception:
                translated.pop("timeout", None)
        translated.pop("description", None)
        translated.pop("run_in_background", None)
    elif tool_name in {"Read", "Write", "Edit", "ReadMedia"}:
        if "file_path" in translated and "path" not in translated:
            translated["path"] = translated.pop("file_path")
        if "path" in translated:
            translated["path"] = _sandbox_path(translated.get("path"))
    elif tool_name == "Download":
        if "file_path" in translated and "path" not in translated:
            translated["path"] = translated.pop("file_path")
        if "destination" in translated and "path" not in translated:
            translated["path"] = translated.pop("destination")
        if "path" in translated:
            translated["path"] = _sandbox_path(translated.get("path"))
    elif tool_name in {"Glob", "Grep"}:
        if "path" in translated:
            translated["path"] = _sandbox_path(translated.get("path"), default="/workspace")
        elif "directory" in translated:
            translated["path"] = _sandbox_path(translated.pop("directory"), default="/workspace")
        else:
            translated["path"] = "/workspace"
    return translated


def _text_from_tool_result(result: Any) -> str:
    if hasattr(result, "content"):
        parts = []
        for block in getattr(result, "content", []) or []:
            if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                parts.append(str(block.text))
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        if parts:
            return "\n".join(parts)
    return str(result)


class ClawLiveDispatcher:
    """Dispatch tool calls through Claw-Eval's dispatchers and write trace live."""

    def __init__(
        self,
        *,
        trace_writer: LiveTraceWriter,
        endpoints: list[dict[str, Any]],
        task_tool_specs: list[dict[str, Any]],
        sandbox_url: str | None,
        strict_sandbox: bool = True,
    ) -> None:
        self.trace_writer = trace_writer
        self.sandbox_url = sandbox_url
        self.strict_sandbox = strict_sandbox
        self.endpoints = [dict(ep) for ep in endpoints]
        self.endpoint_map: dict[str, dict[str, Any]] = {
            str(ep.get("tool_name")): {
                "tool_name": str(ep.get("tool_name")),
                "url": str(ep.get("url")),
                "method": str(ep.get("method") or "POST"),
            }
            for ep in self.endpoints
            if ep.get("tool_name") and ep.get("url")
        }
        sandbox_specs, sandbox_names = _tool_specs_from_official()
        self.sandbox_tool_specs = sandbox_specs
        self.sandbox_tool_names = sandbox_names
        self.task_tool_specs = _normalise_task_tool_specs(task_tool_specs)
        self.tool_specs = self._dedupe_specs(self.task_tool_specs + self.sandbox_tool_specs)
        self.allowed_tool_names = {str(spec["name"]) for spec in self.tool_specs}
        self._client = httpx.Client(trust_env=False, timeout=120.0)
        self._official_dispatcher = None
        self._official_tool_use_cls = None
        self._init_official_dispatcher()

    @staticmethod
    def _dedupe_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for spec in specs:
            name = str(spec.get("name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(spec)
        # Preserve MiniHarness order: YAML task tools first, then official
        # sandbox tools in their declared order.  Do not sort alphabetically;
        # the OpenAI tools array order is part of the model-visible surface.
        return out

    def _init_official_dispatcher(self) -> None:
        try:
            from claw_eval.models.content import ToolUseBlock  # type: ignore
            from claw_eval.models.tool import ToolEndpoint  # type: ignore
            from claw_eval.runner.dispatcher import ToolDispatcher  # type: ignore
            from claw_eval.runner.sandbox_dispatcher import SandboxToolDispatcher  # type: ignore
        except Exception:
            return
        try:
            endpoint_map = {
                name: ToolEndpoint(tool_name=name, url=str(ep["url"]), method=str(ep.get("method") or "POST"))
                for name, ep in self.endpoint_map.items()
            }
            http_dispatcher = ToolDispatcher(endpoint_map)
            self._official_dispatcher = SandboxToolDispatcher(http_dispatcher, sandbox_url=self.sandbox_url)
            self._official_tool_use_cls = ToolUseBlock
        except Exception:
            self._official_dispatcher = None
            self._official_tool_use_cls = None

    def close(self) -> None:
        self._client.close()
        if self._official_dispatcher is not None and hasattr(self._official_dispatcher, "close"):
            self._official_dispatcher.close()

    def _trace_error_result(self, *, tool_name: str, payload: dict[str, Any], tool_use_id: str, status: int, body: dict[str, Any], endpoint_url: str = "") -> DispatchResult:
        event = {
            "type": "tool_dispatch",
            "trace_id": self.trace_writer.trace_id,
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "endpoint_url": endpoint_url,
            "request_body": payload,
            "response_status": status,
            "response_body": body,
            "latency_ms": 0.0,
        }
        self.trace_writer.tool_dispatch(event)
        return DispatchResult(status, body, True, tool_use_id, tool_name, endpoint_url, 0.0)

    def dispatch(self, tool_name: str, payload: dict[str, Any] | None = None, *, tool_use_id: str | None = None) -> DispatchResult:
        payload = dict(payload or {})
        tool_name = str(tool_name)
        if tool_name in self.sandbox_tool_names:
            payload = _translate_sandbox_payload(tool_name, payload)
        tool_use_id = tool_use_id or f"external-{tool_name}-{uuid.uuid4().hex[:12]}"
        if tool_name not in self.allowed_tool_names:
            body = {"error": f"tool '{tool_name}' is not allowed by this task"}
            return self._trace_error_result(tool_name=tool_name, payload=payload, tool_use_id=tool_use_id, status=403, body=body)

        if self._official_dispatcher is not None and self._official_tool_use_cls is not None:
            return self._dispatch_official(tool_name, payload, tool_use_id)
        return self._dispatch_fallback(tool_name, payload, tool_use_id)

    def _dispatch_official(self, tool_name: str, payload: dict[str, Any], tool_use_id: str) -> DispatchResult:
        tool_use = self._official_tool_use_cls(id=tool_use_id, name=tool_name, input=payload)
        try:
            result_tuple = self._official_dispatcher.dispatch(tool_use, self.trace_writer.trace_id)
            if len(result_tuple) == 3:
                result, event, extra_images = result_tuple
            else:  # pragma: no cover - older API fallback
                result, event = result_tuple
                extra_images = None
            self.trace_writer.tool_dispatch(event)
            status = int(getattr(event, "response_status", 200))
            body = getattr(event, "response_body", None)
            if body is None:
                body = {"content": _text_from_tool_result(result)}
            if extra_images:
                body = {"content": body, "extra_images": [img.model_dump(mode="json") if hasattr(img, "model_dump") else img for img in extra_images]}
            return DispatchResult(
                status=status,
                body=body,
                is_error=bool(getattr(result, "is_error", status >= 400)),
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                endpoint_url=str(getattr(event, "endpoint_url", "")),
                latency_ms=float(getattr(event, "latency_ms", 0.0) or 0.0),
            )
        except Exception as exc:
            body = {"error": str(exc)}
            event = {
                "type": "tool_dispatch",
                "trace_id": self.trace_writer.trace_id,
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "endpoint_url": "official://dispatcher/error",
                "request_body": payload,
                "response_status": 500,
                "response_body": body,
                "latency_ms": 0.0,
            }
            self.trace_writer.tool_dispatch(event)
            return DispatchResult(500, body, True, tool_use_id, tool_name, event["endpoint_url"], 0.0)

    def _dispatch_fallback(self, tool_name: str, payload: dict[str, Any], tool_use_id: str) -> DispatchResult:
        if tool_name in self.sandbox_tool_names:
            if not self.sandbox_url and self.strict_sandbox:
                body = {"error": "sandbox_url is required for sandbox tool dispatch"}
                self.trace_writer.tool_dispatch(
                    {
                        "type": "tool_dispatch",
                        "trace_id": self.trace_writer.trace_id,
                        "tool_use_id": tool_use_id,
                        "tool_name": tool_name,
                        "endpoint_url": "",
                        "request_body": payload,
                        "response_status": 500,
                        "response_body": body,
                        "latency_ms": 0.0,
                    }
                )
                return DispatchResult(500, body, True, tool_use_id, tool_name, "", 0.0)
            path = SANDBOX_PATH_MAP.get(tool_name)
            if not path:
                body = {"error": f"unknown sandbox tool: {tool_name}"}
                return self._trace_error_result(tool_name=tool_name, payload=payload, tool_use_id=tool_use_id, status=404, body=body)
            endpoint_url = f"{self.sandbox_url}{path}"
            req_body = payload
            method = "POST"
        else:
            endpoint = self.endpoint_map.get(tool_name)
            if endpoint is None:
                body = {"error": f"no endpoint registered for tool: {tool_name}"}
                return self._trace_error_result(tool_name=tool_name, payload=payload, tool_use_id=tool_use_id, status=404, body=body)
            endpoint_url = str(endpoint["url"])
            method = str(endpoint.get("method") or "POST")
            req_body = payload

        t0 = time.monotonic()
        try:
            resp = self._client.request(method, endpoint_url, json=req_body)
            latency_ms = (time.monotonic() - t0) * 1000.0
            try:
                body: Any = resp.json()
            except Exception:
                body = {"text": resp.text}
            status = int(resp.status_code)
            is_error = status >= 400
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            status = 500
            body = {"error": str(exc)}
            is_error = True

        self.trace_writer.tool_dispatch(
            {
                "type": "tool_dispatch",
                "trace_id": self.trace_writer.trace_id,
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "endpoint_url": endpoint_url,
                "request_body": payload,
                "response_status": status,
                "response_body": body,
                "latency_ms": latency_ms,
            }
        )
        return DispatchResult(status, body, is_error, tool_use_id, tool_name, endpoint_url, latency_ms)
