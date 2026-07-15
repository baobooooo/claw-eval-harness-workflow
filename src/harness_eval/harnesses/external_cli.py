from __future__ import annotations

import subprocess
import time
from math import ceil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from harness_eval.claw_live.model_proxy import ClawToolModelProxy, model_tool_proxy_dry_manifest, model_tool_proxy_enabled
from harness_eval.harnesses.base import HarnessAdapter, agent_workspace, task_timeout_s, task_timeout_multiplier
from harness_eval.harnesses.git_utils import get_patch, status_short, validate_patch
from harness_eval.harnesses.native_claw import (
    _make_native_provider,
    model_visible_tool_names,
    native_claw_tools_enabled as _native_claw_tools_enabled,
    run_native_claw_tools,
)
from harness_eval.io import now_iso, quote_cmd, run_cmd, write_json
from harness_eval.types import BenchmarkTask, HarnessResult, ModelProfile


def _live_bridge_manifest(task: BenchmarkTask) -> dict[str, Any]:
    return {
        "requested": bool(task.metadata.get("live_tool_bridge_requested")),
        "enabled": bool(task.metadata.get("live_tool_bridge")),
        "bridge_url": task.metadata.get("claw_tool_bridge_url"),
        "sandbox_url": task.metadata.get("claw_sandbox_url"),
        "sandbox_mode": task.metadata.get("claw_sandbox_mode"),
        "trace_schema": "claw_eval_live_v1" if task.metadata.get("live_tool_bridge") else "external_harness_raw",
    }


def _looks_like_harness_start_failure(rc: int, stderr_text: str, stdout_text: str = "") -> bool:
    if rc == 0:
        return False
    text = (stderr_text + "\n" + stdout_text).lower()
    needles = [
        "does not recognize option",
        "unknown option",
        "unrecognized option",
        "unexpected argument",
        "no such option",
        "command not found",
        "no such file or directory",
        "usage:",
    ]
    return any(needle in text for needle in needles)


def _looks_like_provider_quota_failure(stderr_text: str, stdout_text: str = "") -> bool:
    text = (stderr_text + "\n" + stdout_text).lower()
    needles = [
        "out of quota",
        "account is in arrears",
        "insufficient balance",
        "billing status",
        "quota exceeded",
        "credit balance",
    ]
    return any(needle in text for needle in needles)


class ExternalCliHarness(HarnessAdapter):
    name = "external-cli"
    default_command_template = "echo 'Configure harness.command_template for {harness}' && exit 2"

    def _command_template(self) -> str:
        hcfg = self.cfg.get("harness", {})
        return str(hcfg.get("command_template") or self.default_command_template)

    def _env(self, model: ModelProfile, task: BenchmarkTask | None = None) -> dict[str, str]:
        env = model.env()
        if model.api_key_value:
            env.setdefault("OPENAI_API_KEY", model.api_key_value)
            env.setdefault("ANTHROPIC_AUTH_TOKEN", model.api_key_value)
            env.setdefault("DEEPSEEK_API_KEY", model.api_key_value)
        env.setdefault("OPENAI_BASE_URL", model.base_url)
        env.setdefault("OPENAI_API_BASE", model.base_url)
        env.setdefault("OPENAI_API_BASE_URL", model.base_url)
        env.setdefault("ANTHROPIC_BASE_URL", model.base_url.rstrip("/v1"))
        # Several third-party harnesses select a named provider (for example
        # deepseek) and ignore OPENAI_BASE_URL.  Set the common provider-specific
        # aliases too.  These variables only point the harness' model transport
        # at the local proxy; the proxy still forwards to the fixed model.
        env.setdefault("DEEPSEEK_BASE_URL", model.base_url)
        env.setdefault("DEEPSEEK_API_BASE", model.base_url)
        env.setdefault("DEEPSEEK_API_BASE_URL", model.base_url)
        env.setdefault("OPENCLAW_BASE_URL", model.base_url)
        env.setdefault("OPENCLAW_API_BASE", model.base_url)
        env.setdefault("OPENCLAW_API_BASE_URL", model.base_url)
        env.setdefault("NANOBOT_BASE_URL", model.base_url)
        env.setdefault("NANOBOT_API_BASE", model.base_url)
        env.setdefault("NANOBOT_API_BASE_URL", model.base_url)
        env.setdefault("HARNESS_MODEL", model.model)
        env.setdefault("HARNESS_BASE_URL", model.base_url)
        env.setdefault("HARNESS_API_KEY_ENV", model.api_key_env)
        if task is not None:
            if task.metadata.get("claw_tool_bridge_url"):
                env.setdefault("CLAW_TOOL_BRIDGE_URL", str(task.metadata["claw_tool_bridge_url"]))
                env.setdefault("CLAW_EVAL_STRICT_TOOLS", "1")
            if task.metadata.get("claw_sandbox_url"):
                env.setdefault("CLAW_EVAL_SANDBOX_URL", str(task.metadata["claw_sandbox_url"]))
            if task.metadata.get("claw_live_trace_path"):
                env.setdefault("CLAW_EVAL_LIVE_TRACE_PATH", str(task.metadata["claw_live_trace_path"]))
            if task.metadata.get("timeout_seconds") is not None:
                env.setdefault("CLAW_EVAL_TIMEOUT_SECONDS", str(task.metadata["timeout_seconds"]))
            if task.metadata.get("max_turns") is not None:
                env.setdefault("CLAW_EVAL_MAX_TURNS", str(task.metadata["max_turns"]))
            env.setdefault("HARNESS_AGENT_WORKSPACE", str(agent_workspace(task)))
            env.setdefault("CLAW_EVAL_AGENT_WORKSPACE", str(agent_workspace(task)))
            env.setdefault("HARNESS_TASK_WORKSPACE", str(task.workspace))
            env.setdefault("OPENCLAW_WORKSPACE", str(agent_workspace(task)))
            env.setdefault("OPENCLAW_CWD", str(agent_workspace(task)))
            env.setdefault("NANOBOT_WORKSPACE", str(agent_workspace(task)))
            env.setdefault("NANOBOT_CWD", str(agent_workspace(task)))
            if task.metadata.get("allowed_tools"):
                env.setdefault("CLAW_EVAL_ALLOWED_TOOLS", ",".join(str(x) for x in task.metadata["allowed_tools"]))
            if task.metadata.get("helper_files") or task.metadata.get("live_helper_files"):
                helpers = task.metadata.get("helper_files") or task.metadata.get("live_helper_files") or []
                env.setdefault("CLAW_EVAL_HELPER_FILES", ",".join(str(x) for x in helpers))
        return env

    def _prompt_for_cli(self, task: BenchmarkTask) -> str:
        return task.prompt

    def _extra_format_context(self, task: BenchmarkTask, model: ModelProfile, stdout_path: Path, stderr_path: Path, trace_path: Path, prompt_file: Path) -> dict[str, Any]:
        return {}

    def _format_command(self, task: BenchmarkTask, model: ModelProfile, stdout_path: Path, stderr_path: Path, trace_path: Path) -> str:
        prompt_file = task.output_dir / "prompt.md"
        prompt_file.write_text(self._prompt_for_cli(task), encoding="utf-8")
        timeout_s = task_timeout_s(task, self.cfg.get("harness", {}))
        context: dict[str, Any] = {
            "harness": self.name,
            "workspace": agent_workspace(task),
            "agent_workspace": agent_workspace(task),
            "task_workspace": task.workspace,
            "output_dir": task.output_dir,
            "prompt_file": prompt_file,
            "trace_file": trace_path,
            "stdout_file": stdout_path,
            "stderr_file": stderr_path,
            "model": model.model,
            "base_url": model.base_url,
            "api_key_env": model.api_key_env,
            "task_id": task.task_id,
            "timeout_s": int(ceil(timeout_s)),
            "timeout_s_float": timeout_s,
        }
        context.update(self._extra_format_context(task, model, stdout_path, stderr_path, trace_path, prompt_file))
        return self._command_template().format(**context)

    def run(self, task: BenchmarkTask, model: ModelProfile, dry_run: bool = False) -> HarnessResult:
        hcfg = self.cfg.get("harness", {})
        start = time.time()
        task.output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = task.output_dir / f"{self.name}_stdout.log"
        stderr_path = task.output_dir / f"{self.name}_stderr.log"
        trace_path = task.output_dir / f"{self.name}_trace.jsonl"
        final_path = task.output_dir / "final_message.txt"
        timeout_s = task_timeout_s(task, hcfg)
        native_mode = _native_claw_tools_enabled(task, hcfg)
        proxy_enabled = bool(not native_mode and model_tool_proxy_enabled(task, hcfg))
        model_visible_tools = model_visible_tool_names(task)
        proxy_manifest: dict[str, Any] = (
            model_tool_proxy_dry_manifest(task, model, hcfg, harness_name=self.name)
            if proxy_enabled
            else {"enabled": False}
        )
        proxy_cm = (
            ClawToolModelProxy(task=task, model=model, hcfg=hcfg, harness_name=self.name)
            if proxy_enabled and not dry_run
            else nullcontext(None)
        )

        command: str | None = None
        shell_command: str | None = None
        error = None
        native_status = ""
        native_metrics: dict[str, Any] = {}
        rc = 0
        agent_model = model

        with proxy_cm as proxy:
            if proxy is not None:
                proxy_manifest = proxy.manifest()
                agent_model = proxy.proxied_model()
            if not native_mode:
                try:
                    command = self._format_command(task, agent_model, stdout_path, stderr_path, trace_path)
                    shell_command = f"set -o pipefail; {command}"
                except Exception as exc:
                    error = f"Harness command configuration error: {exc}"
                    stderr_path.write_text(error + "\n", encoding="utf-8")
                    manifest = {
                        "started_at": now_iso(),
                        "finished_at": now_iso(),
                        "harness": self.name,
                        "model_profile": model.to_json(),
                        "effective_model_profile": agent_model.to_json() if agent_model.base_url != model.base_url else None,
                        "workspace": str(agent_workspace(task)),
                        "task_workspace": str(task.workspace),
                        "timeout_s": timeout_s,
                        "tool_mode": "harness_cli_with_claw_model_proxy" if proxy_enabled else "external_cli",
                        "model_tool_proxy": proxy_manifest,
                        "live_tool_bridge": _live_bridge_manifest(task),
                        "status": "harness_config_error",
                        "error": error,
                    }
                    write_json(task.output_dir / "harness_manifest.json", manifest)
                    return HarnessResult(task.task_id, self.name, model.model, "harness_config_error", stdout_path=str(stdout_path), stderr_path=str(stderr_path), trace_path=str(trace_path), duration_s=0, returncode=2, error=error)
            manifest = {
                "started_at": now_iso(),
                "harness": self.name,
                "model_profile": model.to_json(),
                "effective_model_profile": agent_model.to_json() if agent_model.base_url != model.base_url else None,
                "cmd": shell_command,
                "workspace": str(agent_workspace(task)),
                "task_workspace": str(task.workspace),
                "timeout_s": timeout_s,
                "timeout_policy": {
                    "official_timeout_seconds": task.metadata.get("timeout_seconds"),
                    "timeout_multiplier": task_timeout_multiplier(hcfg),
                    "effective_timeout_seconds": timeout_s,
                },
                "max_turns": task.metadata.get("max_turns"),
                "tool_mode": "claw_native_tools" if native_mode else ("harness_cli_with_claw_model_proxy" if proxy_enabled else "external_cli"),
                "native_harness_tools_disabled": bool(native_mode),
                "model_visible_tools": model_visible_tools if (native_mode or proxy_enabled) else None,
                "model_tool_proxy": proxy_manifest,
                "live_tool_bridge": _live_bridge_manifest(task),
            }
            write_json(task.output_dir / "harness_manifest.json", manifest)
            if dry_run:
                if native_mode:
                    dry_run_text = f"{self.name} claw-native direct model loop\nmodel_visible_tools={', '.join(model_visible_tools)}\n"
                elif proxy_enabled:
                    dry_run_text = (
                        str(shell_command) + "\n"
                        + "model_tool_proxy=harness_cli_with_claw_model_proxy\n"
                        + f"model_visible_tools={', '.join(model_visible_tools)}\n"
                    )
                else:
                    dry_run_text = str(shell_command) + "\n"
                (task.output_dir / "DRY_RUN.txt").write_text(dry_run_text, encoding="utf-8")
                return HarnessResult(task.task_id, self.name, model.model, "dry_run", stdout_path=str(stdout_path), stderr_path=str(stderr_path), trace_path=str(trace_path), duration_s=0, returncode=0)

            if native_mode:
                rc, _final, native_status, native_metrics = run_native_claw_tools(
                    task,
                    model,
                    final_path,
                    stdout_path,
                    stderr_path,
                    timeout_s=timeout_s,
                    hcfg=hcfg,
                    provider_factory=_make_native_provider,
                    harness_name=self.name,
                )
            else:
                try:
                    res = run_cmd(["bash", "-lc", str(shell_command)], cwd=agent_workspace(task), env=self._env(agent_model, task), timeout=timeout_s, check=False, stdout_path=stdout_path, stderr_path=stderr_path)
                    rc = res.returncode
                except subprocess.TimeoutExpired:
                    rc = -124
                    error = "External CLI run timed out."
                    stderr_path.write_text(error + "\n", encoding="utf-8")

        patch_workspace = agent_workspace(task)
        patch = get_patch(patch_workspace)
        validation = validate_patch(patch_workspace, self.cfg.get("agent", {})) if patch else {"enabled": True, "ok": True}
        (task.output_dir / "patch.diff").write_text(patch, encoding="utf-8")
        (task.output_dir / "repo_status_final.txt").write_text(status_short(patch_workspace), encoding="utf-8")
        write_json(task.output_dir / "patch_validation.json", validation)
        final = final_path.read_text(encoding="utf-8", errors="replace") if final_path.exists() else ""
        if not final and stdout_path.exists():
            final = stdout_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        if rc == -124:
            status = "timeout"
        elif _looks_like_provider_quota_failure(stderr_text, stdout_text):
            status = "provider_quota_error"
            error = error or (stderr_text or stdout_text)[-2000:] or "provider quota or billing failure"
        elif native_mode and rc != 0:
            status = native_status or "nonzero"
            error = error or (stderr_text or stdout_text)[-2000:] or None
        elif not native_mode and _looks_like_harness_start_failure(rc, stderr_text, stdout_text):
            status = "harness_start_failed"
            error = error or (stderr_text or stdout_text)[-2000:] or "external harness failed before starting agent loop"
        elif native_mode and rc == 0:
            status = "ok" if (final or task.benchmark != "swe") else "zero_patch"
        elif rc == 0 and (patch or task.benchmark != "swe"):
            status = "ok"
        elif rc == 0:
            status = "zero_patch"
        else:
            status = "nonzero"
        if patch and not validation.get("ok", True):
            status = "patch_validation_failed"
        duration = round(time.time() - start, 3)
        manifest.update({"finished_at": now_iso(), "returncode": rc, "duration_s": duration, "status": status, "patch_bytes": len(patch.encode("utf-8")), "error": error, **native_metrics})
        write_json(task.output_dir / "harness_manifest.json", manifest)
        result_trace_path = str(task.metadata.get("claw_live_trace_path") or (trace_path if trace_path.exists() else stdout_path))
        metrics: dict[str, Any] = {"patch_validation": validation}
        if native_mode:
            metrics["native_claw_tools"] = native_metrics
        if proxy_enabled:
            metrics["model_tool_proxy"] = proxy_manifest
        return HarnessResult(
            task_id=task.task_id,
            harness=self.name,
            model=model.model,
            status=status,
            patch=patch,
            final_message=final,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            trace_path=result_trace_path,
            duration_s=duration,
            returncode=rc,
            metrics=metrics,
            error=error,
        )
