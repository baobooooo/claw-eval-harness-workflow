from __future__ import annotations

import base64
import glob as glob_mod
import json
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: Any) -> None:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    payload = json.loads(raw.decode("utf-8") or "{}")
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


class HostSandboxServer:
    """Small local fallback that mimics Claw-Eval's sandbox HTTP server.

    This exists for unit tests and offline smoke runs.  Real benchmark runs can
    require the official Docker SandboxRunner by setting
    benchmark.require_official_claw_sandbox=true.
    """

    def __init__(self, root: str | Path, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.sandbox_url: str | None = None

    def start(self) -> str:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover - noisy
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    _json_response(self, 200, {"ok": True, "mode": "host_fallback"})
                    return
                _json_response(self, 404, {"error": f"unknown endpoint: {self.path}"})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    payload = _read_json(self)
                    status, body = server.handle(self.path, payload)
                    _json_response(self, status, body)
                except Exception as exc:
                    _json_response(self, 500, {"error": str(exc)})

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        actual_host, actual_port = self._httpd.server_address
        self.sandbox_url = f"http://{actual_host}:{actual_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="host-sandbox", daemon=True)
        self._thread.start()
        return self.sandbox_url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _resolve(self, raw: str | None, *, must_be_under_root: bool = True) -> Path:
        if not raw:
            raise ValueError("missing path")
        text = str(raw)
        if text.startswith("/workspace/"):
            rel = text[len("/workspace/") :]
            path = self.root / rel
        elif text == "/workspace":
            path = self.root
        else:
            p = Path(text)
            path = p if p.is_absolute() else self.root / p
        path = path.resolve()
        if must_be_under_root:
            try:
                path.relative_to(self.root)
            except ValueError:
                raise ValueError(f"path escapes sandbox root: {raw}")
        return path

    def handle(self, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
        if path == "/exec":
            return 200, self.exec(payload)
        if path == "/read":
            return self.read(payload)
        if path == "/write":
            return self.write(payload)
        if path == "/write_b64":
            return self.write_b64(payload)
        if path == "/edit":
            return self.edit(payload)
        if path == "/glob":
            return 200, self.glob(payload)
        if path == "/grep":
            return 200, self.grep(payload)
        if path == "/download":
            return self.download(payload)
        if path in {"/screenshot", "/read_media"}:
            return 501, {"error": f"{path} requires the official Docker sandbox with media/browser support"}
        return 404, {"error": f"unknown endpoint: {path}"}

    def exec(self, payload: dict[str, Any]) -> dict[str, Any]:
        cmd = str(payload.get("command") or "")
        # The official Docker sandbox exposes /workspace.  In host fallback
        # mode, rewrite that path into the isolated fallback root so smoke tests
        # exercise the same command strings without touching the host root.
        cmd = cmd.replace("/workspace", str(self.root))
        timeout = float(payload.get("timeout_seconds") or 30)
        env = dict(os.environ)
        env.setdefault("HOME", str(self.root))
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.root,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": -1,
                "stdout": exc.stdout or "",
                "stderr": f"Command timed out after {timeout}s",
            }

    def read(self, payload: dict[str, Any]) -> tuple[int, Any]:
        path = self._resolve(str(payload.get("path") or payload.get("file_path") or ""))
        if not path.exists() or not path.is_file():
            return 404, {"error": f"File not found: {path}"}
        text = path.read_text(encoding="utf-8", errors="replace")
        offset = payload.get("offset")
        limit = payload.get("limit")
        if offset is not None or limit is not None:
            lines = text.splitlines(keepends=True)
            start = max(0, int(offset or 1) - 1)
            end = start + int(limit) if limit else len(lines)
            selected = lines[start:end]
            text = "".join(selected)
        return 200, {"content": text}

    def write(self, payload: dict[str, Any]) -> tuple[int, Any]:
        path = self._resolve(str(payload.get("path") or payload.get("file_path") or ""))
        content = str(payload.get("content") or "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return 200, {"written": str(path), "bytes": len(content.encode("utf-8"))}

    def write_b64(self, payload: dict[str, Any]) -> tuple[int, Any]:
        path = self._resolve(str(payload.get("path") or payload.get("file_path") or ""))
        raw = base64.b64decode(str(payload.get("content_b64") or payload.get("content") or ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return 200, {"written": str(path), "bytes": len(raw)}

    def edit(self, payload: dict[str, Any]) -> tuple[int, Any]:
        path = self._resolve(str(payload.get("path") or payload.get("file_path") or ""))
        if not path.exists():
            return 404, {"error": f"File not found: {path}"}
        old = str(payload.get("old_string") or "")
        new = str(payload.get("new_string") or "")
        replace_all = bool(payload.get("replace_all", False))
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return 400, {"error": "old_string not found"}
        if count > 1 and not replace_all:
            return 400, {"error": f"old_string found {count} times; use replace_all=true"}
        path.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1), encoding="utf-8")
        return 200, {"edited": str(path), "replacements": count if replace_all else 1}

    def glob(self, payload: dict[str, Any]) -> dict[str, Any]:
        pattern = str(payload.get("pattern") or "*")
        base = self._resolve(str(payload.get("path") or "/workspace"))
        matches = sorted(glob_mod.glob(str(base / pattern), recursive=True))
        return {"files": [str(Path(m).resolve()) for m in matches[:200]]}

    def grep(self, payload: dict[str, Any]) -> dict[str, Any]:
        pattern = str(payload.get("pattern") or "")
        base = self._resolve(str(payload.get("path") or "/workspace"))
        regex = re.compile(pattern, re.IGNORECASE if payload.get("case_insensitive") else 0)
        rows: list[str] = []
        for p in base.rglob("*") if base.is_dir() else [base]:
            if not p.is_file():
                continue
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                    if regex.search(line):
                        rows.append(f"{p}:{i}:{line}")
                        if len(rows) >= int(payload.get("head_limit") or 200):
                            return {"output": "\n".join(rows), "exit_code": 0}
            except Exception:
                continue
        return {"output": "\n".join(rows), "exit_code": 0 if rows else 1}

    def download(self, payload: dict[str, Any]) -> tuple[int, Any]:
        path = self._resolve(str(payload.get("path") or ""))
        if not path.exists() or not path.is_file():
            return 404, {"error": f"File not found: {path}"}
        max_bytes = int(payload.get("max_bytes") or 50_000_000)
        data = path.read_bytes()[:max_bytes]
        return 200, {"path": str(path), "content_b64": base64.b64encode(data).decode("ascii"), "size_bytes": len(data)}


def copy_files_into_host_sandbox(root: Path, task_dir: Path | None, files: list[str]) -> int:
    injected = 0
    if task_dir is None:
        return 0
    for rel in files:
        src = task_dir / rel
        if not src.exists():
            continue
        dst = root / rel
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() in {".sh", ".bash", ".py"}:
                try:
                    text = src.read_text(encoding="utf-8")
                    dst.write_text(text.replace("/workspace/", str(root.resolve()) + "/"), encoding="utf-8")
                except UnicodeDecodeError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
        injected += 1
    return injected
