from __future__ import annotations

import base64
import glob
import mimetypes
import os
import shlex
import subprocess
import sys
import hashlib
import json
import shutil
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

from harness_eval.benchmarks.base import BenchmarkAdapter
from harness_eval.io import append_jsonl, get_path, now_iso, quote_cmd, read_jsonl, run_cmd, safe_id, write_json
from harness_eval.types import BenchmarkTask, EvalResult, HarnessResult, ModelProfile
from harness_eval.claw_live.dispatcher import _tool_specs_from_official
from harness_eval.claw_live.runtime import ClawLiveRuntime


class OpenClawBenchmark(BenchmarkAdapter):
    """Claw-Eval/OpenClaw benchmark adapter in external-harness mode.

    Claw-Eval's stock path evaluates model capability through its own mini
    harness.  This adapter treats each Claw-Eval item as a prepared workspace
    task and delegates action to the selected harness (Codex, nanobot, or
    OpenClaw).  It writes a harness_predictions.jsonl file with trace paths so a
    patched Claw-Eval grader can score final state and trajectory.
    """

    name = "openclaw"

    def _project_root(self) -> Path:
        return Path(str(get_path(self.cfg, "project.root", "."))).resolve()

    def _live_bridge_enabled(self) -> bool:
        """Default Claw-Eval external-harness runs to strict live bridge.

        The stage-2 smoke failure showed that a generated benchmark config can
        silently omit ``live_tool_bridge: true`` even when the repository
        templates contain it.  For this adapter, live bridge is the safe default:
        set ``benchmark.live_tool_bridge: false`` only for explicit legacy helper
        tests or local diagnostics.
        """
        bcfg = self.cfg.get("benchmark", {}) if isinstance(self.cfg.get("benchmark"), dict) else {}
        for key in ("live_tool_bridge", "live_bridge_enabled", "strict_claw_eval_runtime"):
            if key in bcfg:
                return bool(bcfg.get(key))
        return bool(bcfg.get("default_live_tool_bridge", True))

    @staticmethod
    def _sandbox_tool_specs_for_live() -> list[dict[str, Any]]:
        specs, _names = _tool_specs_from_official()
        return [dict(spec) for spec in specs]

    def _external_claw_eval_root(self) -> Path:
        bcfg = self.cfg.get("benchmark", {})
        configured = bcfg.get("claw_eval_root")
        if configured:
            p = Path(str(configured))
            return p if p.is_absolute() else self._project_root() / p
        return self._project_root() / "external" / "claw-eval"

    def _tasks_dir(self) -> Path:
        bcfg = self.cfg.get("benchmark", {})
        configured = bcfg.get("tasks_dir")
        if configured:
            p = Path(str(configured))
            return p if p.is_absolute() else self._project_root() / p
        return self._external_claw_eval_root() / "tasks"

    def _task_yaml_path(self, task_id: str) -> Path | None:
        path = self._tasks_dir() / task_id / "task.yaml"
        return path if path.exists() else None

    def _load_official_task_yaml(self, task_id: str) -> dict[str, Any]:
        path = self._task_yaml_path(task_id)
        if path is None:
            return {}
        try:
            import yaml
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read Claw-Eval task metadata") from e
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _safe_endpoint_metadata(
        task_yaml: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        tool_specs = []
        for tool in task_yaml.get("tools") or []:
            if not isinstance(tool, dict) or not tool.get("name"):
                continue
            tool_specs.append(
                {
                    "name": str(tool.get("name")),
                    "description": str(tool.get("description") or ""),
                    "input_schema": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {},
                }
            )
        allowed_tools = [str(tool["name"]) for tool in tool_specs]
        allowed_tool_set = set(allowed_tools)
        services = []
        for svc in task_yaml.get("services") or []:
            if isinstance(svc, dict):
                services.append({k: svc.get(k) for k in ["name", "command", "port", "health_check", "health_check_method", "ready_timeout", "reset_endpoint", "env"] if k in svc})
        endpoints = []
        for ep in task_yaml.get("tool_endpoints") or []:
            if isinstance(ep, dict) and str(ep.get("tool_name")) in allowed_tool_set:
                endpoints.append({k: ep.get(k) for k in ["tool_name", "url", "method"] if k in ep})
        return services, endpoints, allowed_tools, tool_specs

    def _service_port_isolation_enabled(self) -> bool:
        bcfg = self.cfg.get("benchmark", {}) if isinstance(self.cfg.get("benchmark"), dict) else {}
        if "isolate_service_ports" in bcfg:
            return bool(bcfg.get("isolate_service_ports"))
        if "service_port_isolation" in bcfg:
            return bool(bcfg.get("service_port_isolation"))
        try:
            return int(bcfg.get("effective_max_workers") or 1) > 1
        except (TypeError, ValueError):
            return False

    def _service_port_base(self) -> int:
        bcfg = self.cfg.get("benchmark", {}) if isinstance(self.cfg.get("benchmark"), dict) else {}
        return int(bcfg.get("service_port_base") or 19000)

    def _service_port_stride(self) -> int:
        bcfg = self.cfg.get("benchmark", {}) if isinstance(self.cfg.get("benchmark"), dict) else {}
        return int(bcfg.get("service_port_stride") or 50)

    @staticmethod
    def _rewrite_url_port(value: Any, port_map: dict[int, int]) -> Any:
        if not isinstance(value, str) or not port_map:
            return value
        try:
            parts = urlsplit(value)
            current_port = parts.port
        except ValueError:
            return value
        if current_port not in port_map:
            return value
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        userinfo = ""
        if parts.username:
            userinfo = parts.username
            if parts.password is not None:
                userinfo += f":{parts.password}"
            userinfo += "@"
        netloc = f"{userinfo}{host}:{port_map[current_port]}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

    def _apply_service_port_isolation(
        self,
        services: list[dict[str, Any]],
        endpoints: list[dict[str, Any]],
        row: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        services = [dict(service) for service in services]
        endpoints = [dict(endpoint) for endpoint in endpoints]
        if not self._service_port_isolation_enabled():
            return services, endpoints, {"enabled": False}

        raw_index = row.get("_harness_eval_row_index")
        if raw_index in (None, ""):
            return services, endpoints, {"enabled": False, "reason": "missing_row_index"}
        try:
            row_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid _harness_eval_row_index for service isolation: {raw_index!r}") from exc

        original_ports: list[int] = []
        for service in services:
            port = self._service_port(service)
            if port is not None and port not in original_ports:
                original_ports.append(port)
        if not original_ports:
            return services, endpoints, {"enabled": False, "row_index": row_index, "port_map": {}}

        stride = self._service_port_stride()
        if stride < len(original_ports):
            raise RuntimeError(
                f"service_port_stride={stride} is too small for {len(original_ports)} service ports in one task"
            )
        block_start = self._service_port_base() + row_index * stride
        port_map = {port: block_start + offset for offset, port in enumerate(original_ports)}
        for new_port in port_map.values():
            if new_port <= 0 or new_port > 65535:
                raise RuntimeError(f"Allocated isolated service port is outside TCP range: {new_port}")

        for service in services:
            port = self._service_port(service)
            if port in port_map:
                new_port = port_map[port]
                service["port"] = new_port
                env = dict(service.get("env") or {})
                env["PORT"] = str(new_port)
                service["env"] = env
            for key in ("health_check", "reset_endpoint"):
                if key in service:
                    service[key] = self._rewrite_url_port(service[key], port_map)
        for endpoint in endpoints:
            if "url" in endpoint:
                endpoint["url"] = self._rewrite_url_port(endpoint["url"], port_map)

        return services, endpoints, {
            "enabled": True,
            "row_index": row_index,
            "port_map": {str(old): new for old, new in port_map.items()},
        }

    @staticmethod
    def _safe_environment_metadata(task_yaml: dict[str, Any]) -> dict[str, int]:
        env = task_yaml.get("environment")
        if not isinstance(env, dict):
            return {}
        out: dict[str, int] = {}
        for key in ("timeout_seconds", "max_turns"):
            value = env.get(key)
            if value in (None, ""):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                out[key] = parsed
        return out

    @staticmethod
    def _safe_list(value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item not in (None, "")]
        return []

    @classmethod
    def _safe_file_metadata(cls, task_yaml: dict[str, Any]) -> dict[str, Any]:
        environment = task_yaml.get("environment") if isinstance(task_yaml.get("environment"), dict) else {}
        fixtures = cls._safe_list(environment.get("fixtures")) if isinstance(environment, dict) else []
        env_snapshot_timeout = 10
        if isinstance(environment, dict) and environment.get("env_snapshot_timeout") not in (None, ""):
            try:
                env_snapshot_timeout = max(1, int(environment.get("env_snapshot_timeout")))
            except (TypeError, ValueError):
                env_snapshot_timeout = 10
        sandbox_files = cls._safe_list(task_yaml.get("sandbox_files"))
        if not sandbox_files:
            sandbox_files = fixtures
        return {
            "sandbox_files": sandbox_files,
            "sandbox_grader_files": cls._safe_list(task_yaml.get("sandbox_grader_files")),
            "env_snapshot_files": cls._safe_list(task_yaml.get("env_snapshot_files")),
            "env_snapshot_commands": cls._safe_list(task_yaml.get("env_snapshot_commands")),
            "local_grader_files": cls._safe_list(task_yaml.get("local_grader_files")),
            "env_snapshot_timeout": env_snapshot_timeout,
        }

    @staticmethod
    def _normalize_declared_path(rel_path: str) -> Path | None:
        rel = Path(str(rel_path))
        if rel.is_absolute() or any(part in {"..", ""} for part in rel.parts):
            return None
        return rel

    def _resolve_task_file(self, rel_path: str, task_dir: Path | None) -> Path | None:
        rel = self._normalize_declared_path(rel_path)
        if rel is None:
            return None
        roots = []
        if task_dir is not None:
            roots.append(task_dir)
        roots.append(self._external_claw_eval_root())
        roots.append(self._project_root())
        for root in roots:
            candidate = root / rel
            if candidate.exists():
                return candidate
        return None

    def _copy_declared_files(self, workspace: Path, task_dir: Path | None, files: list[str]) -> dict[str, Any]:
        copied: list[str] = []
        missing: list[str] = []
        rejected: list[str] = []
        for raw in files:
            rel = self._normalize_declared_path(raw)
            if rel is None:
                rejected.append(str(raw))
                continue
            src = self._resolve_task_file(str(rel), task_dir)
            if src is None:
                missing.append(str(rel))
                continue
            dst = workspace / rel
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied.append(str(rel))
        return {"copied": copied, "missing": missing, "rejected": rejected}

    @staticmethod
    def _read_snapshot_file(path: Path) -> dict[str, Any]:
        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "content": base64.b64encode(data).decode("ascii"),
                "mime_type": mime or "application/octet-stream",
                "encoding": "base64",
                "size_bytes": len(data),
            }
        return {
            "content": text,
            "mime_type": mime or "text/plain",
            "encoding": "utf-8",
            "size_bytes": len(data),
        }

    def _collect_env_snapshot_host(self, task: BenchmarkTask) -> dict[str, Any]:
        timeout = int(task.metadata.get("env_snapshot_timeout") or 10)
        snapshot: dict[str, Any] = {}
        # Preserve Claw-Eval's temporal firewall: grader-only files become visible
        # only after the external agent has stopped. Patch capture has already
        # happened inside the harness adapter, so these files do not pollute patches.
        grader_copy = self._copy_declared_files(
            task.workspace,
            Path(str(task.metadata["task_dir"])) if task.metadata.get("task_dir") else None,
            list(task.metadata.get("sandbox_grader_files") or []),
        )
        if any(grader_copy.values()):
            snapshot["grader_file_injection"] = grader_copy

        for cmd in list(task.metadata.get("env_snapshot_commands") or []):
            try:
                res = run_cmd(["bash", "-lc", str(cmd)], cwd=task.workspace, timeout=timeout, check=False)
                snapshot[f"cmd:{cmd}"] = {
                    "exit_code": res.returncode,
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                }
            except subprocess.TimeoutExpired:
                snapshot[f"cmd:{cmd}"] = {
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": f"Timed out after {timeout}s",
                }
            except Exception as exc:
                snapshot[f"cmd:{cmd}"] = {"error": str(exc)}

        max_files = int(self.cfg.get("benchmark", {}).get("env_snapshot_max_files", 50))
        for raw_pattern in list(task.metadata.get("env_snapshot_files") or []):
            pattern = str(raw_pattern)
            rel_pattern = self._normalize_declared_path(pattern)
            if rel_pattern is None:
                snapshot[f"file:{pattern}"] = {"error": "rejected unsafe path"}
                continue
            matches: list[Path]
            if any(ch in pattern for ch in "*?"):
                matches = [Path(p) for p in sorted(glob.glob(str(task.workspace / rel_pattern), recursive=True)) if Path(p).is_file()]
                matches = matches[:max_files]
                if not matches:
                    snapshot[f"file:{pattern}"] = {"error": "no files matched"}
            else:
                matches = [task.workspace / rel_pattern]
            for path in matches:
                try:
                    rel = path.resolve().relative_to(task.workspace.resolve())
                except ValueError:
                    snapshot[f"file:{pattern}"] = {"error": "resolved outside workspace"}
                    continue
                key = f"file:{rel.as_posix()}"
                if not path.exists() or not path.is_file():
                    snapshot[key] = {"error": f"File not found: {rel.as_posix()}"}
                    continue
                try:
                    snapshot[key] = self._read_snapshot_file(path)
                except Exception as exc:
                    snapshot[key] = {"error": str(exc)}

        task_dir = Path(str(task.metadata["task_dir"])) if task.metadata.get("task_dir") else None
        for raw in list(task.metadata.get("local_grader_files") or []):
            src = self._resolve_task_file(str(raw), task_dir)
            key = f"local_file:{raw}"
            if src is None or not src.is_file():
                snapshot[key] = {"error": f"File not found: {raw}"}
                continue
            try:
                snapshot[key] = self._read_snapshot_file(src)
            except Exception as exc:
                snapshot[key] = {"error": str(exc)}
        return snapshot

    def finalize_task_result(self, result: HarnessResult, task: BenchmarkTask) -> HarnessResult:
        runtime = task.metadata.get("_claw_live_runtime_obj")
        if runtime is not None:
            return runtime.finalize_result(result)
        needs_snapshot = any(
            task.metadata.get(key)
            for key in ("sandbox_grader_files", "env_snapshot_files", "env_snapshot_commands", "local_grader_files")
        )
        if not needs_snapshot:
            return result
        snapshot = self._collect_env_snapshot_host(task)
        snapshot_path = task.output_dir / "env_snapshot.json"
        write_json(snapshot_path, snapshot)
        task.metadata["env_snapshot_path"] = str(snapshot_path)
        result.metrics = {
            **result.metrics,
            "env_snapshot": {
                "path": str(snapshot_path),
                "entries": len(snapshot),
            },
        }
        return result


    @staticmethod
    def _tool_policy_hash(policy: dict[str, Any]) -> str:
        payload = json.dumps(policy, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _helper_files_for(endpoints: list[dict[str, Any]], *, live_bridge: bool = False) -> list[str]:
        endpoint_names = {str(ep.get("tool_name")) for ep in endpoints}
        if live_bridge:
            helper_files = [
                "claw_tool",
                "claw_bash",
                "claw_read",
                "claw_write",
                "claw_edit",
                "claw_glob",
                "claw_grep",
                "claw_download",
                "claw_eval_tool_policy.json",
                "claw_live_tools.json",
                "README_CLAW_LIVE_BRIDGE.md",
                "CODEX_CLAW_TOOL_USAGE.md",
            ]
            if "web_search" in endpoint_names:
                helper_files.append("claw_web_search")
            if "web_fetch" in endpoint_names:
                helper_files.append("claw_web_fetch")
            return sorted(helper_files)
        helper_files = ["claw_tool"] if endpoints else []
        if "web_search" in endpoint_names:
            helper_files.append("claw_web_search")
        if "web_fetch" in endpoint_names:
            helper_files.append("claw_web_fetch")
        return sorted(helper_files)

    @staticmethod
    def _render_tool_instructions(
        endpoints: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        *,
        helper_files: list[str],
        policy_sha256: str,
        environment: dict[str, int] | None = None,
        live_bridge: bool = False,
    ) -> str:
        allowed_tools = [str(spec.get("name")) for spec in tool_specs if spec.get("name")]
        endpoint_names = {str(ep.get("tool_name")) for ep in endpoints}
        environment = environment or {}
        policy_file_note = (
            "- Policy/bridge metadata files in this driver workspace: `./claw_eval_tool_policy.json`, `./claw_live_tools.json`, `./README_CLAW_LIVE_BRIDGE.md`."
            if live_bridge
            else "- Policy file in this workspace: `./claw_eval_tool_policy.json`."
        )
        lines: list[str] = [
            "Official Claw-Eval tool policy:",
            policy_file_note,
            f"- Policy SHA-256: `{policy_sha256}`.",
            "- Allowed tool names: " + (", ".join(f"`{name}`" for name in allowed_tools) if allowed_tools else "none"),
        ]
        if environment:
            budget = ", ".join(f"{key}={value}" for key, value in sorted(environment.items()))
            lines.append(f"- Official task budget: {budget}.")
        helper_commands = [name for name in helper_files if name.startswith("claw_") and "." not in name]
        if helper_commands:
            lines.append("- Exposed helper commands: " + ", ".join(f"`./{name}`" for name in helper_commands) + ".")
        else:
            lines.append("- No helper commands are exposed for this task.")
        if live_bridge:
            lines.extend(
                [
                    "- You are running in a driver workspace. The scored task workspace is inside a Claw-Eval sandbox container and is reachable only through the live bridge.",
                    "- Every official action must go through `./claw_tool <ToolName> '<json_payload>'` or one of the listed `./claw_*` wrappers.",
                    "- Native shell is only a transport for launching bridge helper commands; do not use native read/edit/write/browser/web tools for task state.",
                    "- It is safe to inspect only the bridge metadata files listed above; all task files must be accessed with bridge tools.",
                    "- Prefer relative sandbox paths such as `fixtures/input.json` or `output.txt` when using bridge tools; the bridge maps them to `/workspace/...` inside the Claw sandbox. Absolute `/workspace/...` paths are also accepted.",
                    "- For long payloads or URLs that a harness safety guard refuses to place directly on a shell command line, write a small payload JSON file in the driver workspace and call `./claw_tool ToolName @payload.json`.",
                    "- Codex shell tool reminder: when the harness asks for shell arguments, the schema key is `cmd`, not `command`; keep commands short and call helpers from `./`.",
                ]
            )
        lines.append("- Only use tools listed in the official policy for task-specific external information or service actions.")
        if "Bash" in allowed_tools:
            lines.append(
                "- `Bash` is an official Claw-Eval sandbox tool when listed above; shell and network commands must run through the Claw-Eval Bash bridge/tool, not native host web/browser tools."
            )
        else:
            lines.append(
                "- Do not use direct public-network access such as raw curl/wget/browser/native web tools except through the listed Claw-Eval helpers."
            )
        if endpoints:
            lines.extend(
                [
                    "",
                    "Generic official tool helper:",
                    "- Use `./claw_tool <tool_name> '<json_payload>'` to call an allowed Claw-Eval tool through the live bridge.",
                    "- For long JSON payloads, prefer stdin (`-`) with a here-doc; legacy writable helper mode also supports `@payload.json`.",
                    "- The helper rejects tool names that are not in the official allowed list.",
                ]
            )
        if "web_search" in endpoint_names or "web_fetch" in endpoint_names:
            lines.extend(
                [
                    "",
                    "Benchmark web tools:",
                    "- Use `./claw_web_search \"query text\"` to call the task's Claw-Eval `web_search` service.",
                    "- Use `./claw_web_fetch \"https://example.com/page\"` or `./claw_tool web_fetch @payload.json` to call the task's Claw-Eval `web_fetch` service.",
                    "- Include any useful web evidence in your final answer.",
                ]
            )
            if "Bash" in allowed_tools:
                lines.append("- When `Bash` is allowed, Bash network commands are also official only when executed through the Claw-Eval Bash bridge/tool.")
            else:
                lines.extend(
                    [
                        "- These commands are the only allowed web access path for this benchmark run.",
                        "- Do not use direct public-network access such as raw curl/wget/browser/native web tools except through these commands.",
                    ]
                )
        if tool_specs:
            lines.extend(["", "Allowed tool schemas:"])
            for spec in tool_specs:
                lines.append(f"- `{spec['name']}`: {spec.get('description') or 'No description.'}")
                lines.append("```json")
                lines.append(json.dumps(spec.get("input_schema") or {}, ensure_ascii=False, sort_keys=True))
                lines.append("```")
        return "\n".join(lines).strip()

    @staticmethod
    def _endpoint_for(endpoints: list[dict[str, Any]], tool_name: str) -> str | None:
        for ep in endpoints:
            if ep.get("tool_name") == tool_name and ep.get("url"):
                return str(ep["url"])
        return None

    @staticmethod
    def _write_text_executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    def _write_tool_helpers(self, workspace: Path, endpoints: list[dict[str, Any]]) -> None:
        endpoint_map = {
            str(ep.get("tool_name")): {
                "url": str(ep.get("url")),
                "method": str(ep.get("method") or "POST"),
            }
            for ep in endpoints
            if ep.get("tool_name") and ep.get("url")
        }
        if endpoint_map:
            self._write_text_executable(
                workspace / "claw_tool",
                f"""#!/usr/bin/env python3
import json
import sys
import urllib.error
import urllib.request

ENDPOINTS = {json.dumps(endpoint_map, ensure_ascii=False, sort_keys=True)}

def usage() -> None:
    allowed = ", ".join(sorted(ENDPOINTS.keys()))
    print("Usage: ./claw_tool <tool_name> '{{\\\"key\\\":\\\"value\\\"}}'  # allowed: " + allowed, file=sys.stderr)

def load_payload(raw: str | None) -> dict:
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

if len(sys.argv) < 2 or sys.argv[1] in {{"-h", "--help"}}:
    usage()
    raise SystemExit(2)

tool_name = sys.argv[1]
endpoint = ENDPOINTS.get(tool_name)
if endpoint is None:
    usage()
    print(f"Error: tool '{{tool_name}}' is not allowed by this task policy", file=sys.stderr)
    raise SystemExit(2)

try:
    payload = load_payload(sys.argv[2] if len(sys.argv) >= 3 else None)
except Exception as exc:
    print(f"Error: invalid JSON payload: {{exc}}", file=sys.stderr)
    raise SystemExit(2)

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(endpoint["url"], data=data, headers={{"Content-Type": "application/json"}}, method=endpoint["method"])
try:
    with urllib.request.urlopen(req, timeout=45) as resp:
        body = resp.read().decode("utf-8", "replace")
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    print(body)
    raise SystemExit(1)
print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
""",
            )
        search_url = self._endpoint_for(endpoints, "web_search")
        fetch_url = self._endpoint_for(endpoints, "web_fetch")
        if search_url:
            self._write_text_executable(
                workspace / "claw_web_search",
                f"""#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINT = {search_url!r}

if len(sys.argv) < 2:
    print("Usage: ./claw_web_search <query>", file=sys.stderr)
    raise SystemExit(2)

payload = {{
    "query": " ".join(sys.argv[1:]),
    "max_results": int(os.environ.get("CLAW_WEB_MAX_RESULTS", "5")),
}}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(ENDPOINT, data=data, headers={{"Content-Type": "application/json"}}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", "replace")
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    print(body)
    raise SystemExit(1)
print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
""",
            )
        if fetch_url:
            self._write_text_executable(
                workspace / "claw_web_fetch",
                f"""#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINT = {fetch_url!r}

if len(sys.argv) != 2:
    print("Usage: ./claw_web_fetch <url>", file=sys.stderr)
    raise SystemExit(2)

payload = {{
    "url": sys.argv[1],
    "timeout_seconds": int(os.environ.get("CLAW_WEB_TIMEOUT_SECONDS", "30")),
}}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(ENDPOINT, data=data, headers={{"Content-Type": "application/json"}}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=45) as resp:
        body = resp.read().decode("utf-8", "replace")
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    print(body)
    raise SystemExit(1)
print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
""",
            )

    @staticmethod
    def _service_audit_url(service: dict[str, Any]) -> str | None:
        reset = service.get("reset_endpoint")
        if isinstance(reset, str) and reset.endswith("/reset"):
            return reset[: -len("/reset")] + "/audit"
        health = service.get("health_check")
        if isinstance(health, str) and health.endswith("/health"):
            return health[: -len("/health")] + "/audit"
        return None

    @staticmethod
    def _http_request(method: str, url: str, *, timeout: float = 5.0, json_body: dict[str, Any] | None = None) -> tuple[int, Any]:
        import httpx

        with httpx.Client(trust_env=False, timeout=timeout) as client:
            if method.upper() == "GET":
                resp = client.get(url)
            else:
                resp = client.request(method.upper(), url, json=json_body)
        try:
            body: Any = resp.json()
        except Exception:
            body = resp.text
        return resp.status_code, body

    def _service_health_timeout(self) -> float:
        return float(self.cfg.get("benchmark", {}).get("service_health_timeout", 3.0))

    def _service_reset_timeout(self) -> float:
        return float(self.cfg.get("benchmark", {}).get("service_reset_timeout", 10.0))

    def _service_ready_timeout(self, service: dict[str, Any]) -> float:
        bcfg = self.cfg.get("benchmark", {})
        default = float(bcfg.get("service_ready_timeout_default", 30.0))
        minimum = float(bcfg.get("service_ready_timeout_min", 30.0))
        maximum = float(bcfg.get("service_ready_timeout_max", 120.0))
        raw = service.get("ready_timeout")
        try:
            declared = float(raw) if raw not in (None, "") else default
        except (TypeError, ValueError):
            declared = default
        return min(max(declared, minimum), maximum)

    def _reuse_healthy_services(self) -> bool:
        return bool(self.cfg.get("benchmark", {}).get("reuse_healthy_services", False))

    def _kill_existing_services_on_port(self) -> bool:
        return bool(self.cfg.get("benchmark", {}).get("kill_existing_services_on_port", True))

    @staticmethod
    def _service_port(service: dict[str, Any]) -> int | None:
        raw = service.get("port")
        if raw in (None, ""):
            return None
        try:
            port = int(raw)
        except (TypeError, ValueError):
            return None
        return port if port > 0 else None

    @staticmethod
    def _listener_pids_for_port(port: int) -> list[int]:
        commands = [
            ["bash", "-lc", f"lsof -ti tcp:{port} -sTCP:LISTEN 2>/dev/null || true"],
            ["bash", "-lc", f"fuser -n tcp {port} 2>/dev/null || true"],
        ]
        pids: set[int] = set()
        for cmd in commands:
            try:
                res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=2)
            except Exception:
                continue
            for token in res.stdout.replace("\n", " ").split():
                try:
                    pids.add(int(token))
                except ValueError:
                    continue
            if pids:
                break
        return sorted(pids)

    def _terminate_existing_service(self, service: dict[str, Any]) -> dict[str, Any]:
        name = str(service.get("name") or "service")
        port = self._service_port(service)
        report: dict[str, Any] = {"service": name, "port": port, "terminated_pids": [], "healthy_before": self._is_healthy(service)}
        if not report["healthy_before"]:
            return report
        if self._reuse_healthy_services():
            report["reused"] = True
            return report
        if not self._kill_existing_services_on_port() or port is None:
            raise RuntimeError(
                f"Service {name} is already healthy before this task. Refusing to reuse it because "
                "reuse_healthy_services=false; set kill_existing_services_on_port=true or clear the stale process."
            )
        pids = [pid for pid in self._listener_pids_for_port(port) if pid != os.getpid()]
        if not pids:
            raise RuntimeError(
                f"Service {name} is already healthy on port {port}, but no listener PID could be identified. "
                "Refusing to skip startup because that would leave cleanup ownership ambiguous."
            )
        for pid in pids:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(self._is_pid_alive(pid) for pid in pids):
            time.sleep(0.2)
        for pid in pids:
            if self._is_pid_alive(pid):
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and self._is_healthy(service):
            time.sleep(0.2)
        report["terminated_pids"] = pids
        report["healthy_after"] = self._is_healthy(service)
        if report["healthy_after"]:
            raise RuntimeError(f"Service {name} remained healthy after terminating stale listener(s) on port {port}: {pids}")
        return report

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _is_healthy(self, service: dict[str, Any]) -> bool:
        url = service.get("health_check")
        if not url:
            return False
        method = str(service.get("health_check_method") or "GET")
        try:
            status, _ = self._http_request(method, str(url), timeout=self._service_health_timeout())
            return status < 500
        except Exception:
            return False

    def _spawn_service(self, service: dict[str, Any], task: BenchmarkTask | None = None) -> tuple[subprocess.Popen[bytes], Path | None, Path | None]:
        command = str(service.get("command") or "")
        if not command:
            raise RuntimeError(f"Service {service.get('name')} has no command")
        cmd = shlex.split(command)
        if cmd and cmd[0] in {"python", "python3"}:
            cmd[0] = sys.executable
        env = dict(os.environ)
        for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(key, None)
        svc_env = service.get("env") if isinstance(service.get("env"), dict) else {}
        env.update({str(k): str(v) for k, v in svc_env.items()})
        if service.get("port"):
            env["PORT"] = str(service["port"])
        stdout_fh = subprocess.DEVNULL
        stderr_fh = subprocess.PIPE
        stdout_path: Path | None = None
        stderr_path: Path | None = None
        if task is not None:
            log_dir = task.output_dir / "service_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_name = safe_id(str(service.get("name") or service.get("port") or "service"))
            stdout_path = log_dir / f"{safe_name}.stdout.log"
            stderr_path = log_dir / f"{safe_name}.stderr.log"
            stdout_fh = stdout_path.open("ab")
            stderr_fh = stderr_path.open("ab")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self._external_claw_eval_root(),
                stdout=stdout_fh,
                stderr=stderr_fh,
                env=env,
            )
        finally:
            for fh in (stdout_fh, stderr_fh):
                if hasattr(fh, "close"):
                    try:
                        fh.close()
                    except Exception:
                        pass
        return proc, stdout_path, stderr_path

    def _reset_services(self, services: list[dict[str, Any]]) -> None:
        for service in services:
            reset = service.get("reset_endpoint")
            if not reset:
                continue
            try:
                self._http_request("POST", str(reset), timeout=self._service_reset_timeout())
            except Exception as exc:
                print(f"[WARN] reset failed for service {service.get('name')}: {exc}")

    def _collect_service_audit(self, task: BenchmarkTask, services: list[dict[str, Any]]) -> None:
        audit: dict[str, Any] = {}
        for service in services:
            url = self._service_audit_url(service)
            if not url:
                continue
            name = str(service.get("name") or url)
            try:
                status, body = self._http_request("GET", url, timeout=10.0)
                audit[name] = {"status": status, "url": url, "body": body}
            except Exception as exc:
                audit[name] = {"status": "error", "url": url, "error": str(exc)}
        if audit:
            write_json(task.output_dir / "service_audit.json", audit)

    @contextmanager
    def task_run_context(self, task: BenchmarkTask, model: ModelProfile | None = None) -> Iterator[None]:
        services = list(task.metadata.get("services") or [])
        spawned: list[tuple[dict[str, Any], subprocess.Popen[bytes], Path | None, Path | None]] = []
        runtime: ClawLiveRuntime | None = None
        service_report: dict[str, Any] = {"services": [], "ready_timeout_policy": {
            "default": self.cfg.get("benchmark", {}).get("service_ready_timeout_default", 30.0),
            "min": self.cfg.get("benchmark", {}).get("service_ready_timeout_min", 30.0),
            "max": self.cfg.get("benchmark", {}).get("service_ready_timeout_max", 120.0),
            "reuse_healthy_services": self._reuse_healthy_services(),
            "kill_existing_services_on_port": self._kill_existing_services_on_port(),
        }}
        try:
            for service in services:
                entry: dict[str, Any] = {"name": service.get("name"), "port": service.get("port")}
                cleanup_report = self._terminate_existing_service(service)
                entry["prestart_cleanup"] = cleanup_report
                if cleanup_report.get("reused"):
                    service_report["services"].append(entry)
                    continue
                proc, stdout_path, stderr_path = self._spawn_service(service, task)
                timeout = self._service_ready_timeout(service)
                entry.update({"ready_timeout": timeout, "stdout_path": str(stdout_path) if stdout_path else None, "stderr_path": str(stderr_path) if stderr_path else None})
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path and stderr_path.exists() else ""
                        entry.update({"status": "exited_early", "returncode": proc.returncode})
                        service_report["services"].append(entry)
                        write_json(task.output_dir / "service_runtime.json", service_report)
                        raise RuntimeError(f"Service {service.get('name')} exited early: {stderr[-1000:]}")
                    if self._is_healthy(service):
                        spawned.append((service, proc, stdout_path, stderr_path))
                        entry["status"] = "ready"
                        break
                    time.sleep(0.5)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path and stderr_path.exists() else ""
                    entry.update({"status": "ready_timeout", "returncode": proc.returncode})
                    service_report["services"].append(entry)
                    write_json(task.output_dir / "service_runtime.json", service_report)
                    raise RuntimeError(f"Service {service.get('name')} did not become ready within {timeout:.1f}s. Last stderr: {stderr[-1000:]}")
                service_report["services"].append(entry)
            if services:
                self._reset_services(services)

            if task.metadata.get("live_tool_bridge_requested"):
                runtime = ClawLiveRuntime(self.cfg, task, model)
                runtime.start()
                task.metadata["_claw_live_runtime_obj"] = runtime
            yield
        finally:
            if services:
                self._collect_service_audit(task, services)
            if runtime is not None:
                # Audit must be collected before the live writer closes so the
                # grader receives AuditSnapshot events in the same trace.
                runtime.write_audit_snapshots()
                runtime.stop(status="ok")
                task.metadata.pop("_claw_live_runtime_obj", None)
            for service, proc, _stdout_path, _stderr_path in reversed(spawned):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            if services:
                post_health = []
                for service in services:
                    post_health.append({"name": service.get("name"), "port": service.get("port"), "healthy_after_cleanup": self._is_healthy(service)})
                service_report["post_cleanup_health"] = post_health
                write_json(task.output_dir / "service_runtime.json", service_report)

    def load_rows(self, max_instances: int | None = None, instance_ids: set[str] | None = None) -> list[dict[str, Any]]:
        bcfg = self.cfg.get("benchmark", {})
        if max_instances is None:
            cfg_max = bcfg.get("max_instances")
            max_instances = int(cfg_max) if cfg_max not in (None, "") else None
        local_jsonl = bcfg.get("jsonl") or bcfg.get("dataset_jsonl")
        rows: list[dict[str, Any]] = []
        if local_jsonl:
            p = Path(str(local_jsonl))
            if not p.is_absolute():
                p = Path(get_path(self.cfg, "project.root", ".")) / p
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
        else:
            dataset_name = str(bcfg.get("dataset_name", "claw-eval/Claw-Eval"))
            split = str(bcfg.get("split", "test"))
            try:
                from datasets import load_dataset  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("datasets is required to load Claw-Eval from Hugging Face") from e
            rows = [dict(row) for row in load_dataset(dataset_name, split=split)]
        selected: list[dict[str, Any]] = []
        for row in rows:
            tid = str(row.get("task_id") or row.get("id") or row.get("instance_id"))
            if instance_ids and tid not in instance_ids:
                continue
            row.setdefault("task_id", tid)
            selected.append(row)
            if max_instances is not None and len(selected) >= max_instances:
                break
        if not selected:
            raise RuntimeError("No OpenClaw/Claw-Eval rows selected")
        out = self.run_dir / "records" / "dataset_subset.jsonl"
        if out.exists():
            out.unlink()
        for row in selected:
            append_jsonl(out, row)
        return selected

    @staticmethod
    def _safe_extract_tar(tar: tarfile.TarFile, target: Path) -> None:
        target_resolved = target.resolve()
        for member in tar.getmembers():
            member_path = target / member.name
            try:
                member_path.resolve().relative_to(target_resolved)
            except ValueError:
                raise RuntimeError(f"Unsafe fixture tar member path: {member.name}")
        tar.extractall(target)


    def _copy_fixture(self, row: dict[str, Any], workspace: Path) -> None:
        bcfg = self.cfg.get("benchmark", {})
        fixture_root = Path(str(bcfg.get("fixture_root", "data/claw_eval_fixtures")))
        if not fixture_root.is_absolute():
            fixture_root = Path(get_path(self.cfg, "project.root", ".")) / fixture_root
        fixtures = row.get("fixture") or row.get("fixtures") or []
        if isinstance(fixtures, str):
            fixtures = [fixtures]
        # If a fixtures tarball exists, extract only once into fixture_root.
        tar_path = Path(str(bcfg.get("fixtures_tar", fixture_root.with_suffix(".tar.gz"))))
        if not fixture_root.exists() and tar_path.exists():
            fixture_root.mkdir(parents=True, exist_ok=True)
            mode = "r:gz" if str(tar_path).endswith(".gz") else "r"
            with tarfile.open(tar_path, mode) as tar:
                self._safe_extract_tar(tar, fixture_root)
        for rel in fixtures:
            src = fixture_root / str(rel)
            dst = workspace / str(rel)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    def _render_prompt(self, row: dict[str, Any]) -> str:
        if self._live_bridge_enabled():
            return str(row.get("_claw_eval_prompt_text") or row.get("query") or row.get("instruction") or "")
        template_path = Path(get_path(self.cfg, "benchmark.prompt_template", "configs/prompts/openclaw_harness_prompt.md"))
        if not template_path.is_absolute():
            template_path = Path(get_path(self.cfg, "project.root", ".")) / template_path
        template = template_path.read_text(encoding="utf-8")
        endpoints = row.get("_claw_eval_tool_endpoints") or []
        tool_specs = row.get("_claw_eval_allowed_tool_specs") or []
        helper_files = self._helper_files_for(list(endpoints), live_bridge=self._live_bridge_enabled())
        policy_sha256 = self._tool_policy_hash(
            {
                "task_id": row.get("task_id", ""),
                "allowed_tools": row.get("_claw_eval_allowed_tools") or [],
                "allowed_tool_specs": tool_specs,
                "exposed_tool_endpoints": endpoints,
                "services": row.get("_claw_eval_services") or [],
                "service_port_isolation": row.get("_claw_eval_service_port_isolation") or {"enabled": False},
                "helper_files": helper_files,
                "environment": row.get("_claw_eval_environment") or {},
                "sandbox_files": (row.get("_claw_eval_file_metadata") or {}).get("sandbox_files", []),
                "sandbox_grader_files": (row.get("_claw_eval_file_metadata") or {}).get("sandbox_grader_files", []),
                "env_snapshot_files": (row.get("_claw_eval_file_metadata") or {}).get("env_snapshot_files", []),
                "env_snapshot_commands": (row.get("_claw_eval_file_metadata") or {}).get("env_snapshot_commands", []),
                "local_grader_files": (row.get("_claw_eval_file_metadata") or {}).get("local_grader_files", []),
            }
        )
        values = {
            "TASK_ID": row.get("task_id", ""),
            "QUERY": row.get("query", row.get("instruction", "")),
            "LANGUAGE": row.get("language", ""),
            "CATEGORY": row.get("category", ""),
            "FIXTURE": json.dumps(row.get("fixture", row.get("fixtures", [])), ensure_ascii=False),
            "CLAW_EVAL_TOOLS": self._render_tool_instructions(
                list(endpoints),
                list(tool_specs),
                helper_files=helper_files,
                policy_sha256=policy_sha256,
                environment=dict(row.get("_claw_eval_environment") or {}),
                live_bridge=self._live_bridge_enabled(),
            ),
        }
        text = template
        for k, v in values.items():
            text = text.replace("{{" + k + "}}", str(v))
        return text

    def prepare_task(self, row: dict[str, Any]) -> BenchmarkTask:
        task_id = str(row.get("task_id") or row.get("id") or row.get("instance_id"))
        out_dir = self.run_dir / "instances" / safe_id(task_id)
        workspace = out_dir / "workspace"
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "openclaw_task.json", row)

        task_yaml_path = self._task_yaml_path(task_id)
        task_yaml = self._load_official_task_yaml(task_id)
        task_dir = task_yaml_path.parent if task_yaml_path is not None else None
        prompt_meta = task_yaml.get("prompt") if isinstance(task_yaml.get("prompt"), dict) else {}
        official_prompt_text = str(prompt_meta.get("text") or row.get("query") or row.get("instruction") or "")
        official_language = str(prompt_meta.get("language") or row.get("language") or "")
        services, endpoints, task_allowed_tools, task_tool_specs = self._safe_endpoint_metadata(task_yaml)
        services, endpoints, service_port_isolation = self._apply_service_port_isolation(services, endpoints, row)
        environment = self._safe_environment_metadata(task_yaml)
        if row.get("timeout_seconds") not in (None, ""):
            try:
                environment["timeout_seconds"] = int(float(row["timeout_seconds"]))
            except (TypeError, ValueError):
                pass
        file_meta = self._safe_file_metadata(task_yaml)
        live_bridge = self._live_bridge_enabled()
        sandbox_tool_specs = self._sandbox_tool_specs_for_live() if live_bridge else []
        tool_specs = task_tool_specs + [spec for spec in sandbox_tool_specs if spec.get("name") not in set(task_allowed_tools)]
        allowed_tools = []
        _seen_tool_names: set[str] = set()
        for spec in tool_specs:
            name = str(spec.get("name") or "")
            if name and name not in _seen_tool_names:
                _seen_tool_names.add(name)
                allowed_tools.append(name)
        helper_files = self._helper_files_for(endpoints, live_bridge=live_bridge)
        policy = {
            "task_id": task_id,
            "allowed_tools": allowed_tools,
            "task_tools": task_allowed_tools,
            "sandbox_tools": [str(spec.get("name")) for spec in sandbox_tool_specs if spec.get("name")],
            "allowed_tool_specs": tool_specs,
            "task_tool_specs": task_tool_specs,
            "exposed_tool_endpoints": endpoints,
            "services": services,
            "service_port_isolation": service_port_isolation,
            "helper_files": helper_files,
            "environment": environment,
            "sandbox_files": file_meta["sandbox_files"],
            "sandbox_grader_files": file_meta["sandbox_grader_files"],
            "env_snapshot_files": file_meta["env_snapshot_files"],
            "env_snapshot_commands": file_meta["env_snapshot_commands"],
            "local_grader_files": file_meta["local_grader_files"],
            "requires_bridge_tool_calls": bool(
                live_bridge
                and (
                    task_allowed_tools
                    or endpoints
                    or sandbox_tool_specs
                    or file_meta["sandbox_files"]
                    or file_meta["env_snapshot_files"]
                    or file_meta["env_snapshot_commands"]
                    or file_meta["sandbox_grader_files"]
                )
            ),
        }
        policy["policy_sha256"] = self._tool_policy_hash(policy)
        row = {
            **row,
            "query": official_prompt_text,
            "language": official_language or row.get("language"),
            "_claw_eval_prompt_text": official_prompt_text,
            "_claw_eval_services": services,
            "_claw_eval_tool_endpoints": endpoints,
            "_claw_eval_allowed_tools": allowed_tools,
            "_claw_eval_task_tools": task_allowed_tools,
            "_claw_eval_sandbox_tools": policy["sandbox_tools"],
            "_claw_eval_allowed_tool_specs": tool_specs,
            "_claw_eval_task_tool_specs": task_tool_specs,
            "_claw_eval_service_port_isolation": service_port_isolation,
            "_claw_eval_environment": environment,
            "_claw_eval_file_metadata": file_meta,
        }
        self._copy_fixture(row, workspace)
        sandbox_copy = self._copy_declared_files(workspace, task_dir, list(file_meta["sandbox_files"]))
        if not live_bridge:
            self._write_tool_helpers(workspace, endpoints)
        write_json(
            out_dir / "claw_eval_tool_endpoints.json",
            {
                "services": services,
                "tool_endpoints": endpoints,
                "allowed_tools": allowed_tools,
                "allowed_tool_specs": tool_specs,
                "task_tools": task_allowed_tools,
                "sandbox_tools": policy["sandbox_tools"],
                "task_tool_specs": task_tool_specs,
                "service_port_isolation": service_port_isolation,
                "live_tool_bridge_requested": live_bridge,
                "agent_workspace": str(out_dir / "agent_driver_workspace") if live_bridge else str(workspace),
                "helper_files": helper_files,
                "environment": environment,
                "file_metadata": file_meta,
                "sandbox_copy": sandbox_copy,
                "policy_sha256": policy["policy_sha256"],
                "requires_bridge_tool_calls": policy["requires_bridge_tool_calls"],
            },
        )
        write_json(out_dir / "claw_eval_tool_policy.json", policy)
        write_json(workspace / "claw_eval_tool_policy.json", policy)
        prompt = self._render_prompt(row)
        (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        return BenchmarkTask(
            benchmark=self.name,
            task_id=task_id,
            row=row,
            prompt=prompt,
            workspace=workspace,
            output_dir=out_dir,
            metadata={
                "category": row.get("category"),
                "language": row.get("language"),
                "fixture": row.get("fixture"),
                "services": services,
                "tool_endpoints": endpoints,
                "allowed_tools": allowed_tools,
                "allowed_tool_specs": tool_specs,
                "task_tools": task_allowed_tools,
                "sandbox_tools": policy["sandbox_tools"],
                "task_tool_specs": task_tool_specs,
                "service_port_isolation": service_port_isolation,
                "live_tool_bridge_requested": live_bridge,
                "agent_workspace": str(out_dir / "agent_driver_workspace") if live_bridge else str(workspace),
                "helper_files": helper_files,
                "environment": environment,
                "timeout_seconds": environment.get("timeout_seconds"),
                "max_turns": environment.get("max_turns"),
                "tool_policy_path": str(out_dir / "claw_eval_tool_policy.json"),
                "workspace_tool_policy_path": str(workspace / "claw_eval_tool_policy.json"),
                "tool_policy_sha256": policy["policy_sha256"],
                "requires_bridge_tool_calls": policy["requires_bridge_tool_calls"],
                "task_yaml_path": str(task_yaml_path) if task_yaml_path else None,
                "task_dir": str(task_dir) if task_dir else None,
                "sandbox_files": file_meta["sandbox_files"],
                "sandbox_grader_files": file_meta["sandbox_grader_files"],
                "env_snapshot_files": file_meta["env_snapshot_files"],
                "env_snapshot_commands": file_meta["env_snapshot_commands"],
                "local_grader_files": file_meta["local_grader_files"],
                "env_snapshot_timeout": file_meta["env_snapshot_timeout"],
                "sandbox_copy": sandbox_copy,
            },
        )

    @property
    def predictions_path(self) -> Path:
        return self.run_dir / "harness_predictions.jsonl"

    def record_prediction(self, result: HarnessResult, task: BenchmarkTask) -> None:
        append_jsonl(
            self.predictions_path,
            {
                "task_id": task.task_id,
                "query": task.row.get("query", task.row.get("instruction", "")),
                "category": task.row.get("category"),
                "language": task.row.get("language"),
                "harness": result.harness,
                "model": result.model,
                "status": result.status,
                "workspace": str(task.workspace),
                "trace_path": result.trace_path,
                "trace_schema": "claw_eval_live_v1" if task.metadata.get("live_tool_bridge") else "external_harness_raw",
                "agent_workspace": task.metadata.get("agent_workspace"),
                "claw_tool_bridge_url": task.metadata.get("claw_tool_bridge_url"),
                "claw_sandbox_url": task.metadata.get("claw_sandbox_url"),
                "claw_sandbox_mode": task.metadata.get("claw_sandbox_mode"),
                "live_tool_bridge": {
                    "requested": bool(task.metadata.get("live_tool_bridge_requested")),
                    "enabled": bool(task.metadata.get("live_tool_bridge")),
                    "bridge_url": task.metadata.get("claw_tool_bridge_url"),
                    "sandbox_url": task.metadata.get("claw_sandbox_url"),
                    "sandbox_mode": task.metadata.get("claw_sandbox_mode"),
                    "trace_schema": "claw_eval_live_v1" if task.metadata.get("live_tool_bridge") else "external_harness_raw",
                },
                "requires_bridge_tool_calls": bool(task.metadata.get("requires_bridge_tool_calls")),
                "returncode": result.returncode,
                "duration_s": result.duration_s,
                "error": result.error,
                "metrics": result.metrics,
                "stdout_path": result.stdout_path,
                "stderr_path": result.stderr_path,
                "service_audit_path": str(task.output_dir / "service_audit.json") if (task.output_dir / "service_audit.json").exists() else None,
                "env_snapshot_path": task.metadata.get("env_snapshot_path"),
                "tool_policy_path": str(task.output_dir / "claw_eval_tool_policy.json") if (task.output_dir / "claw_eval_tool_policy.json").exists() else None,
                "tool_policy_sha256": task.metadata.get("tool_policy_sha256"),
                "task_yaml": task.metadata.get("task_yaml_path"),
                "task_dir": task.metadata.get("task_dir"),
                "final_message": result.final_message,
                "patch": result.patch,
            },
        )

    def evaluate(self, timeout_s: int | None = None) -> EvalResult:
        eval_dir = self.run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        bcfg = self.cfg.get("benchmark", {})
        raw_predictions = self.predictions_path.resolve()
        manifest = {"started_at": now_iso(), "raw_predictions_path": str(raw_predictions)}
        if not raw_predictions.exists():
            manifest.update({"status": "missing_predictions"})
            write_json(eval_dir / "eval_manifest.json", manifest)
            return EvalResult(self.name, self.run_dir, manifest["status"], manifest, [])

        predictions = raw_predictions
        convert_before_eval = bool(bcfg.get("convert_traces_before_eval", True))
        if convert_before_eval:
            from harness_eval.analysis.trace_conversion import write_score_ready_outputs

            rows = [row for row in read_jsonl(raw_predictions) if isinstance(row, dict)]
            conversion_dir = eval_dir / "score_ready"
            conversion_manifest = write_score_ready_outputs(
                rows,
                conversion_dir,
                model=str(bcfg.get("eval_model") or bcfg.get("model") or "") or None,
            )
            predictions = Path(str(conversion_manifest["predictions"])).resolve()
            manifest["conversion"] = conversion_manifest

        manifest["predictions_path"] = str(predictions)
        cmd_template = bcfg.get("eval_command")
        if not cmd_template:
            manifest.update(
                {
                    "status": "skipped_external_grader_not_configured",
                    "note": "Set benchmark.eval_command to a patched claw-eval grader command. The command receives {predictions}, {raw_predictions}, {run_dir}, {eval_dir}, and {score_ready_dir}.",
                }
            )
            write_json(eval_dir / "eval_manifest.json", manifest)
            return EvalResult(self.name, self.run_dir, manifest["status"], manifest, [])

        score_ready_dir = eval_dir / "score_ready"
        cmd_str = str(cmd_template).format(
            predictions=predictions,
            raw_predictions=raw_predictions,
            run_dir=self.run_dir,
            eval_dir=eval_dir,
            score_ready_dir=score_ready_dir,
            project_root=self._project_root(),
            claw_eval_root=self._external_claw_eval_root(),
            tasks_dir=self._tasks_dir(),
        )
        cmd = ["bash", "-lc", cmd_str]
        manifest["cmd"] = cmd_str
        write_json(eval_dir / "eval_manifest.json", manifest)
        start = time.time()
        res = run_cmd(cmd, cwd=self.run_dir, timeout=timeout_s, check=False, stdout_path=eval_dir / "claw_eval_stdout.log", stderr_path=eval_dir / "claw_eval_stderr.log")
        manifest.update({"returncode": res.returncode, "finished_at": now_iso(), "duration_s": round(time.time() - start, 3)})
        report_files = [str(p) for p in eval_dir.rglob("*") if p.is_file() and p.name != "eval_manifest.json"]
        manifest["report_files"] = report_files
        write_json(eval_dir / "eval_manifest.json", manifest)
        return EvalResult(self.name, self.run_dir, "ok" if res.returncode == 0 else "nonzero", manifest, report_files)
