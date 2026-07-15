from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from harness_eval.claw_live.bridge import LiveToolBridgeServer
from harness_eval.claw_live.dispatcher import ClawLiveDispatcher, FALLBACK_SANDBOX_TOOL_SPECS
from harness_eval.claw_live.host_sandbox import HostSandboxServer, copy_files_into_host_sandbox
from harness_eval.claw_live.trace import LiveTraceWriter
from harness_eval.io import now_iso, safe_id, write_json
from harness_eval.types import BenchmarkTask, HarnessResult, ModelProfile


@dataclass(slots=True)
class ClawLiveRuntimeState:
    trace_path: Path
    trace_id: str
    sandbox_url: str
    bridge_url: str
    sandbox_mode: str
    driver_workspace: Path
    sandbox_host_workspace: Path | None = None
    sandbox_public_network_disabled: bool = False
    sandbox_network_name: str | None = None


def _as_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return []


def _load_official_task(task_yaml_path: str | None) -> Any | None:
    if not task_yaml_path:
        return None
    try:
        from claw_eval.models.task import TaskDefinition  # type: ignore
    except Exception:
        return None
    try:
        return TaskDefinition.from_yaml(task_yaml_path)
    except Exception:
        return None


class _EnvironmentShim:
    def __init__(self, fixtures: list[str]) -> None:
        self.fixtures = fixtures


class _TaskShim:
    def __init__(self, task: BenchmarkTask) -> None:
        self.task_id = task.task_id
        self.task_file = task.metadata.get("task_yaml_path")
        self.sandbox_files = _as_list(task.metadata.get("sandbox_files"))
        self.sandbox_grader_files = _as_list(task.metadata.get("sandbox_grader_files"))
        self.environment = _EnvironmentShim(_as_list(task.metadata.get("fixture") or task.row.get("fixture") or task.row.get("fixtures")))


def _sandbox_config_from_cfg(cfg: dict[str, Any]) -> Any:
    from claw_eval.config import SandboxConfig  # type: ignore

    bcfg = cfg.get("benchmark", {})
    scfg = bcfg.get("sandbox", {}) if isinstance(bcfg.get("sandbox"), dict) else {}
    return SandboxConfig(
        enabled=True,
        image=str(scfg.get("image") or bcfg.get("sandbox_image") or "claw-eval-agent:latest"),
        docker_host=scfg.get("docker_host") or bcfg.get("docker_host"),
        memory_limit=str(scfg.get("memory_limit") or "4g"),
        cpu_limit=float(scfg.get("cpu_limit") or 2.0),
        sandbox_port=int(scfg.get("sandbox_port") or 8080),
        container_timeout=int(scfg.get("container_timeout") or 900),
        max_concurrent=int(scfg.get("max_concurrent") or 10),
        enable_browser=bool(scfg.get("enable_browser", True)),
        enable_shell=bool(scfg.get("enable_shell", True)),
        enable_file=bool(scfg.get("enable_file", True)),
    )


def _sandbox_health_timeout_from_cfg(cfg: dict[str, Any]) -> int:
    bcfg = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark"), dict) else {}
    scfg = bcfg.get("sandbox", {}) if isinstance(bcfg.get("sandbox"), dict) else {}
    value = scfg.get("health_timeout") or scfg.get("startup_timeout") or bcfg.get("sandbox_health_timeout") or 60
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 60
    return max(parsed, 15)


def _sandbox_public_network_disabled_from_cfg(cfg: dict[str, Any]) -> bool:
    bcfg = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark"), dict) else {}
    scfg = bcfg.get("sandbox", {}) if isinstance(bcfg.get("sandbox"), dict) else {}
    for key in ("disable_public_network", "block_public_network", "no_public_network"):
        if key in scfg:
            return bool(scfg.get(key))
        if key in bcfg:
            return bool(bcfg.get(key))
    # Live bridge runs should be strict by default: external information must
    # flow through YAML-declared tools such as web_search/web_fetch, not through
    # curl/wget/urllib inside the sandbox Bash tool.
    if ClawLiveRuntime.enabled(cfg):
        return True
    return False


def _sandbox_clear_proxy_env_from_cfg(cfg: dict[str, Any]) -> bool:
    bcfg = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark"), dict) else {}
    scfg = bcfg.get("sandbox", {}) if isinstance(bcfg.get("sandbox"), dict) else {}
    for key in ("clear_proxy_env", "disable_proxy_env"):
        if key in scfg:
            return bool(scfg.get(key))
        if key in bcfg:
            return bool(bcfg.get(key))
    return _sandbox_public_network_disabled_from_cfg(cfg)


class ClawLiveRuntime:
    """Per-task live runtime for external-harness Claw-Eval execution.

    The external agent process only receives a driver workspace containing bridge
    clients.  The scored filesystem lives in the Claw-Eval sandbox.  Every
    official tool call flows through LiveToolBridgeServer -> ClawLiveDispatcher ->
    Claw-Eval SandboxToolDispatcher/ToolDispatcher (or a labelled host fallback
    for tests), and is appended immediately to the Claw-Eval JSONL trace.
    """

    def __init__(self, cfg: dict[str, Any], task: BenchmarkTask, model: ModelProfile | None = None) -> None:
        self.cfg = cfg
        self.task = task
        self.model = model
        self.bcfg = cfg.get("benchmark", {})
        self.driver_workspace = Path(str(task.metadata.get("agent_workspace") or task.output_dir / "agent_driver_workspace"))
        self.trace_path = task.output_dir / "claw_live_trace.jsonl"
        self.trace_writer = LiveTraceWriter(self.trace_path)
        self.dispatcher: ClawLiveDispatcher | None = None
        self.bridge: LiveToolBridgeServer | None = None
        self.sandbox_runner = None
        self.sandbox_handle = None
        self.host_sandbox: HostSandboxServer | None = None
        self.host_sandbox_root: Path | None = None
        self.official_task: Any | None = None
        self.state: ClawLiveRuntimeState | None = None
        self._audit_snapshots_written = False
        self._sandbox_public_network_disabled = _sandbox_public_network_disabled_from_cfg(cfg)
        self._sandbox_clear_proxy_env = _sandbox_clear_proxy_env_from_cfg(cfg)
        self._sandbox_network = None
        self._sandbox_network_name: str | None = None

    @staticmethod
    def enabled(cfg: dict[str, Any]) -> bool:
        bcfg = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark"), dict) else {}
        for key in ("live_tool_bridge", "live_bridge_enabled", "strict_claw_eval_runtime"):
            if key in bcfg:
                return bool(bcfg.get(key))
        # Safe default for the external-harness Claw-Eval adapter.  Legacy helper
        # mode must now be requested explicitly with benchmark.live_tool_bridge=false.
        return bool(bcfg.get("default_live_tool_bridge", True))

    def _require_official(self) -> bool:
        return bool(self.bcfg.get("require_official_claw_sandbox") or self.bcfg.get("strict_claw_eval_runtime"))

    def _allow_host_fallback(self) -> bool:
        if self._require_official():
            return False
        return bool(self.bcfg.get("allow_host_sandbox_fallback", True))

    def __enter__(self) -> "ClawLiveRuntime":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        status = "error" if exc_type else "ok"
        self.stop(status=status)

    def start(self) -> ClawLiveRuntimeState:
        self.task.output_dir.mkdir(parents=True, exist_ok=True)
        self.driver_workspace.mkdir(parents=True, exist_ok=True)
        self.official_task = _load_official_task(self.task.metadata.get("task_yaml_path")) or _TaskShim(self.task)
        self.trace_writer.start(task_id=self.task.task_id, model=self.model.model if self.model else str(self.task.metadata.get("model") or ""))
        self.trace_writer.message("user", self.task.prompt)

        sandbox_url, sandbox_mode = self._start_sandbox()
        endpoints = list(self.task.metadata.get("tool_endpoints") or [])
        task_tool_specs = list(self.task.metadata.get("task_tool_specs") or self.task.metadata.get("allowed_tool_specs") or [])
        self.dispatcher = ClawLiveDispatcher(
            trace_writer=self.trace_writer,
            endpoints=endpoints,
            task_tool_specs=task_tool_specs,
            sandbox_url=sandbox_url,
            strict_sandbox=True,
        )
        self.bridge = LiveToolBridgeServer(
            dispatcher=self.dispatcher,
            trace_writer=self.trace_writer,
            task_id=self.task.task_id,
            host=str(self.bcfg.get("bridge_host", "127.0.0.1")),
            port=int(self.bcfg.get("bridge_port", 0) or 0),
        )
        bridge_url = self.bridge.start()
        helper_files = self.bridge.write_client_files(self.driver_workspace)
        policy_src = Path(str(self.task.metadata.get("tool_policy_path") or self.task.output_dir / "claw_eval_tool_policy.json"))
        if policy_src.exists():
            shutil.copy2(policy_src, self.driver_workspace / "claw_eval_tool_policy.json")
            if "claw_eval_tool_policy.json" not in helper_files:
                helper_files.append("claw_eval_tool_policy.json")
        self.task.metadata.update(
            {
                "live_tool_bridge": True,
                "agent_workspace": str(self.driver_workspace),
                "claw_live_trace_path": str(self.trace_path),
                "claw_tool_bridge_url": bridge_url,
                "claw_sandbox_url": sandbox_url,
                "claw_sandbox_mode": sandbox_mode,
                "live_helper_files": helper_files,
                "allowed_tools": [str(spec["name"]) for spec in self.dispatcher.tool_specs],
                "allowed_tool_specs": self.dispatcher.tool_specs,
                "helper_files": helper_files,
                "claw_sandbox_public_network_disabled": self._sandbox_public_network_disabled,
                "claw_sandbox_network_name": self._sandbox_network_name,
            }
        )
        write_json(
            self.task.output_dir / "claw_live_runtime.json",
            {
                "started_at": now_iso(),
                "trace_path": str(self.trace_path),
                "trace_id": self.trace_writer.trace_id,
                "sandbox_url": sandbox_url,
                "sandbox_mode": sandbox_mode,
                "bridge_url": bridge_url,
                "driver_workspace": str(self.driver_workspace),
                "helper_files": helper_files,
                "official_task_loaded": self.official_task is not None and self.official_task.__class__.__name__ != "_TaskShim",
                "sandbox_public_network_disabled": self._sandbox_public_network_disabled,
                "sandbox_network_name": self._sandbox_network_name,
                "sandbox_proxy_env_cleared": self._sandbox_clear_proxy_env,
            },
        )
        self.state = ClawLiveRuntimeState(
            trace_path=self.trace_path,
            trace_id=self.trace_writer.trace_id,
            sandbox_url=sandbox_url,
            bridge_url=bridge_url,
            sandbox_mode=sandbox_mode,
            driver_workspace=self.driver_workspace,
            sandbox_host_workspace=self.host_sandbox_root,
            sandbox_public_network_disabled=self._sandbox_public_network_disabled,
            sandbox_network_name=self._sandbox_network_name,
        )
        return self.state

    def _create_no_nat_network(self, run_id: str) -> str:
        if self.sandbox_runner is None:
            raise RuntimeError("sandbox runner is not initialized")
        network_name = safe_id(f"claw-no-public-{run_id}")[:63]
        docker_client = self.sandbox_runner._docker
        try:
            old_network = docker_client.networks.get(network_name)
            old_network.remove()
        except Exception:
            pass
        self._sandbox_network = docker_client.networks.create(
            network_name,
            driver="bridge",
            internal=False,
            options={"com.docker.network.bridge.enable_ip_masquerade": "false"},
            labels={"app": "claw-eval", "role": "agent-no-public-network", "run_id": run_id},
        )
        self._sandbox_network_name = network_name
        return network_name

    def _start_locked_down_container(self, run_id: str) -> Any:
        if self.sandbox_runner is None:
            raise RuntimeError("sandbox runner is not initialized")
        try:
            from claw_eval.runner.sandbox_runner import ContainerHandle  # type: ignore
        except Exception as exc:  # pragma: no cover - import checked earlier
            raise RuntimeError("Claw-Eval ContainerHandle is required for official sandbox lockdown") from exc

        network_name = self._create_no_nat_network(run_id)
        config = self.sandbox_runner._config
        env = {} if self._sandbox_clear_proxy_env else self.sandbox_runner._proxy_env()
        container = None
        try:
            container = self.sandbox_runner._docker.containers.run(
                image=self.sandbox_runner._image,
                detach=True,
                name=f"claw-agent-{run_id}",
                mem_limit=config.memory_limit,
                nano_cpus=int(config.cpu_limit * 1e9),
                network=network_name,
                ports={f"{config.sandbox_port}/tcp": None},
                labels={"app": "claw-eval", "role": "agent", "run_id": run_id, "public_network": "disabled"},
                environment=env,
            )

            host_port = self.sandbox_runner._get_mapped_port(container)
            sandbox_url = f"http://localhost:{host_port}"
            self.sandbox_runner._wait_healthy(f"{sandbox_url}/health")
            print(f"[sandbox] Container claw-agent-{run_id} started at {sandbox_url} (public network disabled)")
            return ContainerHandle(
                container=container,
                host_port=host_port,
                run_id=run_id,
                sandbox_url=sandbox_url,
            )
        except Exception:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            try:
                if self._sandbox_network is not None:
                    self._sandbox_network.remove()
            except Exception:
                pass
            self._sandbox_network = None
            self._sandbox_network_name = None
            raise

    def _start_sandbox(self) -> tuple[str, str]:
        task_dir = Path(str(self.task.metadata["task_dir"])) if self.task.metadata.get("task_dir") else None
        try:
            from claw_eval.runner.sandbox_runner import SandboxRunner  # type: ignore
        except Exception as exc:
            if not self._allow_host_fallback():
                raise RuntimeError(
                    "Official Claw-Eval SandboxRunner is required but cannot be imported. "
                    "Install claw-eval with sandbox extras or set benchmark.allow_host_sandbox_fallback=true for local tests."
                ) from exc
            return self._start_host_sandbox(task_dir)

        try:
            sandbox_config = _sandbox_config_from_cfg(self.cfg)
            self.sandbox_runner = SandboxRunner(sandbox_config)
            health_timeout = _sandbox_health_timeout_from_cfg(self.cfg)
            original_wait_healthy = getattr(self.sandbox_runner, "_wait_healthy", None)
            if callable(original_wait_healthy):
                def wait_healthy(url: str, timeout: int = health_timeout) -> None:
                    original_wait_healthy(url, timeout=health_timeout)

                self.sandbox_runner._wait_healthy = wait_healthy  # type: ignore[method-assign]
            run_id = safe_id(f"{self.task.task_id}-{os.getpid()}-{int(time.time() * 1000)}")[:48]
            if self._sandbox_public_network_disabled:
                self.sandbox_handle = self._start_locked_down_container(run_id)
            else:
                self.sandbox_handle = self.sandbox_runner.start_container(run_id=run_id)
            SandboxRunner.inject_files(self.sandbox_handle, self.official_task, task_dir=str(task_dir) if task_dir else None)
            return str(self.sandbox_handle.sandbox_url), "official_docker"
        except Exception as exc:
            if not self._allow_host_fallback():
                raise
            (self.task.output_dir / "official_sandbox_start_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
            return self._start_host_sandbox(task_dir)

    def _start_host_sandbox(self, task_dir: Path | None) -> tuple[str, str]:
        root = self.task.output_dir / "host_sandbox_workspace"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        sandbox_files = _as_list(self.task.metadata.get("sandbox_files"))
        if not sandbox_files:
            sandbox_files = _as_list(self.task.metadata.get("fixture"))
        copy_files_into_host_sandbox(root, task_dir, sandbox_files)
        self.host_sandbox_root = root
        self.host_sandbox = HostSandboxServer(root)
        return self.host_sandbox.start(), "host_fallback"

    def _post_json(self, url: str, payload: dict[str, Any], *, timeout: float = 30.0) -> tuple[int, Any]:
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            resp = client.post(url, json=payload)
        try:
            body: Any = resp.json()
        except Exception:
            body = {"text": resp.text}
        return int(resp.status_code), body

    @staticmethod
    def _workspace_path(path: str) -> str:
        text = str(path)
        if text.startswith("/workspace/"):
            return text
        if text == "/workspace":
            return text
        return "/workspace/" + text.lstrip("/")

    def _collect_sandbox_file(self, path: str) -> dict[str, Any]:
        sandbox_url = str(self.task.metadata.get("claw_sandbox_url") or "")
        sandbox_path = self._workspace_path(path)
        status, body = self._post_json(f"{sandbox_url}/download", {"path": sandbox_path, "max_bytes": 50_000_000}, timeout=60.0)
        if status >= 400:
            # Text fallback via /read for older sandbox servers.
            status2, body2 = self._post_json(f"{sandbox_url}/read", {"path": sandbox_path}, timeout=60.0)
            if status2 >= 400:
                return {"error": body2 if isinstance(body2, str) else body2.get("error", body2)}
            content = body2.get("content") if isinstance(body2, dict) else str(body2)
            return {"content": content, "mime_type": "text/plain", "encoding": "utf-8", "size_bytes": len(str(content).encode("utf-8"))}
        if not isinstance(body, dict):
            return {"content": str(body), "mime_type": "text/plain", "encoding": "utf-8", "size_bytes": len(str(body).encode("utf-8"))}
        raw_b64 = body.get("content_b64") or body.get("content")
        if not raw_b64:
            return {"error": body.get("error", "empty download response")}
        data = base64.b64decode(str(raw_b64))
        mime, _ = mimetypes.guess_type(path)
        try:
            text = data.decode("utf-8")
            return {"content": text, "mime_type": mime or "text/plain", "encoding": "utf-8", "size_bytes": len(data)}
        except UnicodeDecodeError:
            return {"content": base64.b64encode(data).decode("ascii"), "mime_type": mime or "application/octet-stream", "encoding": "base64", "size_bytes": len(data)}

    def _inject_grader_files(self) -> int:
        task_dir = self.task.metadata.get("task_dir")
        files = _as_list(self.task.metadata.get("sandbox_grader_files"))
        if not files:
            return 0
        if self.sandbox_runner is not None and self.sandbox_handle is not None:
            try:
                from claw_eval.runner.sandbox_runner import SandboxRunner  # type: ignore

                return int(SandboxRunner.inject_grader_files(self.sandbox_handle, self.official_task, task_dir=str(task_dir) if task_dir else None))
            except Exception as exc:
                (self.task.output_dir / "grader_injection_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
                return 0
        if self.host_sandbox_root is not None:
            return copy_files_into_host_sandbox(self.host_sandbox_root, Path(str(task_dir)) if task_dir else None, files)
        return 0

    def collect_env_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        injected = self._inject_grader_files()
        if injected:
            snapshot["grader_file_injection"] = {"injected": injected, "mode": self.task.metadata.get("claw_sandbox_mode")}
        sandbox_url = str(self.task.metadata.get("claw_sandbox_url") or "")
        timeout = int(self.task.metadata.get("env_snapshot_timeout") or 10)
        for cmd in _as_list(self.task.metadata.get("env_snapshot_commands")):
            status, body = self._post_json(
                f"{sandbox_url}/exec",
                {"command": cmd, "timeout_seconds": timeout},
                timeout=timeout + 5,
            )
            if isinstance(body, dict):
                snapshot[f"cmd:{cmd}"] = {
                    "exit_code": body.get("exit_code", 0 if status < 400 else status),
                    "stdout": body.get("stdout", ""),
                    "stderr": body.get("stderr", body.get("error", "")),
                }
            else:
                snapshot[f"cmd:{cmd}"] = {"exit_code": status, "stdout": str(body), "stderr": ""}
        for pattern in _as_list(self.task.metadata.get("env_snapshot_files")):
            # Exact paths are common in Claw-Eval.  For globs, use sandbox Glob first.
            if any(ch in pattern for ch in "*?"):
                if self.dispatcher is not None:
                    res = self.dispatcher.dispatch("Glob", {"pattern": pattern, "path": "/workspace"})
                    files = []
                    if isinstance(res.body, dict):
                        files = [str(p) for p in res.body.get("files") or []]
                    if not files:
                        snapshot[f"file:{pattern}"] = {"error": "no files matched"}
                    for item in files[:50]:
                        rel = item.split("/workspace/", 1)[-1] if "/workspace/" in item else item
                        snapshot[f"file:{rel}"] = self._collect_sandbox_file(rel)
                continue
            snapshot[f"file:{pattern}"] = self._collect_sandbox_file(pattern)
        # Local grader files remain host-only evidence and are never injected pre-run.
        task_dir = Path(str(self.task.metadata["task_dir"])) if self.task.metadata.get("task_dir") else None
        for raw in _as_list(self.task.metadata.get("local_grader_files")):
            src = task_dir / raw if task_dir else None
            key = f"local_file:{raw}"
            if src is None or not src.exists() or not src.is_file():
                snapshot[key] = {"error": f"File not found: {raw}"}
                continue
            data = src.read_bytes()
            mime, _ = mimetypes.guess_type(str(src))
            try:
                snapshot[key] = {"content": data.decode("utf-8"), "mime_type": mime or "text/plain", "encoding": "utf-8", "size_bytes": len(data)}
            except UnicodeDecodeError:
                snapshot[key] = {"content": base64.b64encode(data).decode("ascii"), "mime_type": mime or "application/octet-stream", "encoding": "base64", "size_bytes": len(data)}
        return snapshot

    def finalize_result(self, result: HarnessResult) -> HarnessResult:
        final = result.final_message or ""
        if final:
            # Avoid duplicating final messages if the bridge /final endpoint was used.
            if not self.bridge or final not in self.bridge.final_messages:
                self.trace_writer.message("assistant", final)
        needs_snapshot = any(
            self.task.metadata.get(key)
            for key in ("sandbox_grader_files", "env_snapshot_files", "env_snapshot_commands", "local_grader_files")
        )
        if needs_snapshot:
            snapshot = self.collect_env_snapshot()
            snapshot_path = self.task.output_dir / "env_snapshot.json"
            write_json(snapshot_path, snapshot)
            self.task.metadata["env_snapshot_path"] = str(snapshot_path)
            result.metrics = {**result.metrics, "env_snapshot": {"path": str(snapshot_path), "entries": len(snapshot), "mode": self.task.metadata.get("claw_sandbox_mode")}}
        result.trace_path = str(self.trace_path)
        result.metrics = {
            **result.metrics,
            "live_tool_bridge": {
                "enabled": True,
                "trace_path": str(self.trace_path),
                "bridge_url": self.task.metadata.get("claw_tool_bridge_url"),
                "sandbox_url": self.task.metadata.get("claw_sandbox_url"),
                "sandbox_mode": self.task.metadata.get("claw_sandbox_mode"),
                "driver_workspace": str(self.driver_workspace),
                "sandbox_public_network_disabled": self.task.metadata.get("claw_sandbox_public_network_disabled"),
                "sandbox_network_name": self.task.metadata.get("claw_sandbox_network_name"),
            },
        }
        return result

    def write_audit_snapshots(self) -> None:
        if self._audit_snapshots_written:
            return
        self._audit_snapshots_written = True
        path = self.task.output_dir / "service_audit.json"
        if not path.exists():
            return
        try:
            audit = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return
        if not isinstance(audit, dict):
            return
        for service_name, service_data in audit.items():
            audit_url = ""
            audit_data: Any = service_data
            if isinstance(service_data, dict):
                audit_url = str(service_data.get("url") or "")
                audit_data = service_data.get("body", service_data)
            if not isinstance(audit_data, dict):
                audit_data = {"body": audit_data}
            self.trace_writer.audit_snapshot(service_name=str(service_name), audit_url=audit_url, audit_data=audit_data)

    def stop(self, *, status: str = "ok") -> None:
        try:
            self.write_audit_snapshots()
        finally:
            self.trace_writer.end(status=status)
            self.trace_writer.close()
            if self.bridge is not None:
                self.bridge.stop()
                self.bridge = None
            if self.dispatcher is not None:
                self.dispatcher.close()
                self.dispatcher = None
            if self.host_sandbox is not None:
                self.host_sandbox.stop()
                self.host_sandbox = None
            if self.sandbox_runner is not None and self.sandbox_handle is not None:
                try:
                    self.sandbox_runner.stop_container(self.sandbox_handle)
                finally:
                    if self._sandbox_network is not None:
                        try:
                            self._sandbox_network.remove()
                        except Exception:
                            pass
                        self._sandbox_network = None
                        self._sandbox_network_name = None
                    self.sandbox_runner = None
                    self.sandbox_handle = None
