from __future__ import annotations

import json
import os
import shlex
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from harness_eval.claw_live.dispatcher import ClawLiveDispatcher
from harness_eval.claw_live.trace import LiveTraceWriter


def _send_json(handler: BaseHTTPRequestHandler, status: int, body: Any) -> None:
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    data = handler.rfile.read(length) if length else b"{}"
    if not data:
        return {}
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


class LiveToolBridgeServer:
    """HTTP bridge that external harnesses call for every Claw-Eval tool use."""

    def __init__(
        self,
        *,
        dispatcher: ClawLiveDispatcher,
        trace_writer: LiveTraceWriter,
        task_id: str,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.dispatcher = dispatcher
        self.trace_writer = trace_writer
        self.task_id = task_id
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str | None = None
        self.final_messages: list[str] = []

    def start(self) -> str:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    _send_json(self, 200, {"ok": True, "task_id": bridge.task_id, "trace_id": bridge.trace_writer.trace_id})
                    return
                if self.path == "/tools":
                    _send_json(
                        self,
                        200,
                        {
                            "task_id": bridge.task_id,
                            "trace_id": bridge.trace_writer.trace_id,
                            "allowed_tools": sorted(bridge.dispatcher.allowed_tool_names),
                            "tool_specs": bridge.dispatcher.tool_specs,
                        },
                    )
                    return
                _send_json(self, 404, {"error": f"unknown endpoint: {self.path}"})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    payload = _read_json(self)
                    if self.path == "/final":
                        text = str(payload.get("text") or payload.get("final_message") or "")
                        if text:
                            bridge.final_messages.append(text)
                            bridge.trace_writer.message("assistant", text)
                        _send_json(self, 200, {"ok": True})
                        return
                    if self.path == "/tool":
                        tool_name = str(payload.get("tool_name") or "")
                        tool_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}
                    elif self.path.startswith("/tool/"):
                        tool_name = unquote(self.path[len("/tool/") :])
                        tool_input = payload
                    else:
                        _send_json(self, 404, {"error": f"unknown endpoint: {self.path}"})
                        return
                    result = bridge.dispatcher.dispatch(tool_name, tool_input)
                    _send_json(self, result.status if result.status >= 400 else 200, result.to_json())
                except Exception as exc:
                    _send_json(self, 500, {"error": str(exc)})

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        actual_host, actual_port = self._httpd.server_address
        self.base_url = f"http://{actual_host}:{actual_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="claw-live-tool-bridge", daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @staticmethod
    def _write_executable(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    def write_client_files(self, workspace: Path) -> list[str]:
        if not self.base_url:
            raise RuntimeError("bridge is not started")
        workspace.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        bridge_url = self.base_url
        self._write_executable(
            workspace / "claw_tool",
            f'''#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request

BRIDGE_URL = os.environ.get("CLAW_TOOL_BRIDGE_URL", {bridge_url!r}).rstrip("/")

def usage() -> None:
    print("Usage: ./claw_tool <ToolName> JSON_PAYLOAD", file=sys.stderr)
    print("       ./claw_tool <ToolName> @payload.json", file=sys.stderr)
    print("       ./claw_tool <ToolName> -   # read JSON from stdin", file=sys.stderr)


def load_payload(raw):
    if raw is None or raw == "":
        return {{}}
    if raw == "-":
        raw = sys.stdin.read()
    elif raw.startswith("@"):
        with open(raw[1:], "r", encoding="utf-8") as f:
            raw = f.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("tool payload must be a JSON object")
    return payload

if len(sys.argv) < 2 or sys.argv[1] in {{"-h", "--help", "help"}}:
    usage()
    raise SystemExit(2)

tool_name = sys.argv[1]
try:
    payload = load_payload(sys.argv[2] if len(sys.argv) >= 3 else None)
except Exception as exc:
    print(f"Invalid JSON payload: {{exc}}", file=sys.stderr)
    raise SystemExit(2)

url = f"{{BRIDGE_URL}}/tool/{{tool_name}}"
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={{"Content-Type": "application/json"}}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=int(os.environ.get("CLAW_TOOL_TIMEOUT", "180"))) as resp:
        body = resp.read().decode("utf-8", "replace")
        status = resp.status
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    status = exc.code
try:
    parsed = json.loads(body)
except Exception:
    print(body)
    raise SystemExit(1 if status >= 400 else 0)
print(json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True))
raise SystemExit(1 if status >= 400 or parsed.get("is_error") else 0)
''',
        )
        files.append("claw_tool")

        # Convenience wrappers.  They all go through claw_tool, never directly
        # to a task service or local filesystem.
        wrapper_specs = {
            "claw_bash": "Bash",
            "claw_read": "Read",
            "claw_write": "Write",
            "claw_edit": "Edit",
            "claw_glob": "Glob",
            "claw_grep": "Grep",
            "claw_download": "Download",
            "claw_web_search": "web_search",
            "claw_web_fetch": "web_fetch",
        }
        for filename, tool_name in wrapper_specs.items():
            self._write_executable(
                workspace / filename,
                f'''#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -eq 0 ]]; then
  echo "Usage: ./{filename} '<json-payload>' or tool-specific plain args" >&2
  exit 2
fi
case {shlex.quote(tool_name)} in
  Bash)
    python3 ./claw_tool Bash "$(python3 -c 'import json,sys; print(json.dumps({{"command":" ".join(sys.argv[1:])}}))' "$@")"
    ;;
  Read)
    python3 ./claw_tool Read "$(python3 -c 'import json,sys; print(json.dumps({{"file_path":sys.argv[1]}}))' "$@")"
    ;;
  Write)
    python3 ./claw_tool Write "$(python3 -c 'import json,sys; print(json.dumps({{"file_path":sys.argv[1],"content":" ".join(sys.argv[2:])}}))' "$@")"
    ;;
  Edit)
    python3 ./claw_tool Edit "$(python3 -c 'import json,sys; print(json.dumps({{"file_path":sys.argv[1],"old_string":sys.argv[2],"new_string":" ".join(sys.argv[3:])}}))' "$@")"
    ;;
  Glob)
    python3 ./claw_tool Glob "$(python3 -c 'import json,sys; payload={{"pattern":sys.argv[1]}}; payload.update({{"path":sys.argv[2]}} if len(sys.argv)>2 else {{}}); print(json.dumps(payload))' "$@")"
    ;;
  Grep)
    python3 ./claw_tool Grep "$(python3 -c 'import json,sys; payload={{"pattern":sys.argv[1]}}; payload.update({{"path":sys.argv[2]}} if len(sys.argv)>2 else {{}}); payload.update({{"glob":sys.argv[3]}} if len(sys.argv)>3 else {{}}); print(json.dumps(payload))' "$@")"
    ;;
  Download)
    python3 ./claw_tool Download "$(python3 -c 'import json,sys; print(json.dumps({{"path":sys.argv[1]}}))' "$@")"
    ;;
  web_search)
    python3 ./claw_tool web_search "$(python3 -c 'import json,sys; print(json.dumps({{"query":" ".join(sys.argv[1:])}}))' "$@")"
    ;;
  web_fetch)
    python3 ./claw_tool web_fetch "$(python3 -c 'import json,sys; print(json.dumps({{"url":sys.argv[1]}}))' "$@")"
    ;;
  *)
    python3 ./claw_tool {shlex.quote(tool_name)} "$@"
    ;;
esac
''',
            )
            files.append(filename)

        tools_path = workspace / "claw_live_tools.json"
        tools_path.write_text(
            json.dumps(
                {
                    "bridge_url": bridge_url,
                    "task_id": self.task_id,
                    "trace_id": self.trace_writer.trace_id,
                    "allowed_tools": [str(spec["name"]) for spec in self.dispatcher.tool_specs],
                    "tool_specs": self.dispatcher.tool_specs,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        files.append("claw_live_tools.json")

        readme = workspace / "README_CLAW_LIVE_BRIDGE.md"
        readme.write_text(
            "# Claw-Eval live bridge\n\n"
            "This directory is only the external harness driver workspace. The scored task workspace lives inside the Claw-Eval sandbox container.\n\n"
            "All task actions must go through `./claw_tool <ToolName> '<json_payload>'`. Direct shell/file/browser/web tools in this driver workspace do not affect the scored sandbox state.\n\n"
            f"Bridge URL: `{bridge_url}`\n\n"
            "Allowed tools are listed in `claw_live_tools.json`; the signed policy copy is `claw_eval_tool_policy.json`.\n",
            encoding="utf-8",
        )
        files.append("README_CLAW_LIVE_BRIDGE.md")

        codex_note = workspace / "CODEX_CLAW_TOOL_USAGE.md"
        codex_note.write_text(
            "# Codex live-bridge usage\n\n"
            "When Codex invokes its shell/exec tool, use the schema key `cmd`, not `command`.\n"
            "The command itself should be short and should call one of the bridge helpers from this directory.\n\n"
            "Examples:\n"
            "- `./claw_read /workspace/input.txt`\n"
            "- `./claw_bash ls -la /workspace`\n"
            "- `./claw_web_search \"query text\"`\n"
            "- `./claw_tool Write '{\"file_path\":\"/workspace/out.txt\",\"content\":\"answer\"}'`\n",
            encoding="utf-8",
        )
        files.append("CODEX_CLAW_TOOL_USAGE.md")
        return files
