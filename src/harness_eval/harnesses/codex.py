from __future__ import annotations

import subprocess
import time
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
from harness_eval.io import get_path, now_iso, quote_cmd, run_cmd, write_json
from harness_eval.types import BenchmarkTask, HarnessResult, ModelProfile


def _wire_api(profile: ModelProfile, cfg_value: str | None = None) -> str:
    if cfg_value:
        return cfg_value
    if "responses" in profile.protocol:
        return "responses"
    return "chat"


class CodexHarness(HarnessAdapter):
    name = "codex"

    def _live_sandbox_mode(self, hcfg: dict[str, Any]) -> str:
        # In live bridge mode the real scored state is protected by the
        # Claw-Eval Docker sandbox.  Codex's own bwrap sandbox can fail inside
        # many CI/VM/container environments (for example RTM_NEWADDR permission
        # errors), so the driver workspace should use the least nested mode that
        # still lets Codex execute ./claw_* helpers.
        return str(hcfg.get("live_sandbox_mode") or hcfg.get("sandbox_mode") or "danger-full-access")

    def _render_config(self, task: BenchmarkTask, model: ModelProfile) -> Path:
        hcfg = self.cfg.get("harness", self.cfg.get("codex", {}))
        codex_home = task.output_dir / "codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        provider_id = str(hcfg.get("model_provider_id", model.provider.replace("-", "_").replace(".", "_")))
        wire_api = _wire_api(model, hcfg.get("wire_api"))
        if task.metadata.get("live_tool_bridge") and not hcfg.get("allow_native_tool_bypass", False):
            sandbox_mode = self._live_sandbox_mode(hcfg)
        else:
            default_sandbox = "workspace-write"
            sandbox_mode = str(hcfg.get("sandbox_mode", default_sandbox))
        approval_policy = str(hcfg.get("approval_policy", "never"))
        context_window = int(model.context_window or hcfg.get("context_window", 32768))
        force_no_native_network = bool(task.metadata.get("live_tool_bridge") and not hcfg.get("allow_native_tool_bypass", False))
        network_access = str(False if force_no_native_network else bool(hcfg.get("sandbox_network_access", False))).lower()
        text = f'''model = "{model.model}"
model_provider = "{provider_id}"
model_context_window = {context_window}
approval_policy = "{approval_policy}"
sandbox_mode = "{sandbox_mode}"
web_search = "disabled"

[model_providers.{provider_id}]
name = "{provider_id}"
env_key = "{model.api_key_env}"
base_url = "{model.base_url}"
wire_api = "{wire_api}"

[sandbox_workspace_write]
network_access = {network_access}
'''
        (codex_home / "config.toml").write_text(text, encoding="utf-8")
        return codex_home

    def _prompt_for_codex(self, task: BenchmarkTask) -> str:
        return task.prompt

    def _command(self, task: BenchmarkTask, model: ModelProfile, final_path: Path) -> list[str]:
        hcfg = self.cfg.get("harness", self.cfg.get("codex", {}))
        exe = str(hcfg.get("executable", "codex"))
        cmd = [exe, "exec", "-m", model.model, "-C", str(agent_workspace(task)), "--output-last-message", str(final_path)]
        cmd.append("--skip-git-repo-check")
        extra = hcfg.get("extra_args", ["--json"])
        if isinstance(extra, str):
            extra = extra.split()
        cmd.extend(str(x) for x in extra)
        sandbox = self._live_sandbox_mode(hcfg) if task.metadata.get("live_tool_bridge") and not hcfg.get("allow_native_tool_bypass", False) else hcfg.get("sandbox_mode", "workspace-write")
        if sandbox and "--sandbox" not in cmd and "--full-auto" not in cmd:
            cmd.extend(["--sandbox", str(sandbox)])
        cmd.append(self._prompt_for_codex(task))
        return cmd

    def run(self, task: BenchmarkTask, model: ModelProfile, dry_run: bool = False) -> HarnessResult:
        hcfg = self.cfg.get("harness", self.cfg.get("codex", {}))
        start = time.time()
        task.output_dir.mkdir(parents=True, exist_ok=True)
        final_path = task.output_dir / "final_message.txt"
        stdout_path = task.output_dir / "codex_events.ndjson"
        stderr_path = task.output_dir / "codex_stderr.log"
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

        native_metrics: dict[str, Any] = {}
        native_status = ""
        status = ""
        rc = 0
        agent_model = model
        with proxy_cm as proxy:
            if proxy is not None:
                proxy_manifest = proxy.manifest()
                agent_model = proxy.proxied_model()

            codex_home = self._render_config(task, agent_model)
            env = {
                "CODEX_HOME": str(codex_home),
                "NO_COLOR": "1",
                **agent_model.env(),
            }
            # Several CLIs inspect OpenAI-style names regardless of provider id.
            if agent_model.api_key_value:
                env.setdefault("OPENAI_API_KEY", agent_model.api_key_value)
            env.setdefault("OPENAI_BASE_URL", agent_model.base_url)
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
            cmd = self._command(task, agent_model, final_path)
            manifest = {
                "started_at": now_iso(),
                "harness": self.name,
                "model_profile": model.to_json(),
                "effective_model_profile": agent_model.to_json() if agent_model.base_url != model.base_url else None,
                "cmd": quote_cmd([*cmd[:-1], "<PROMPT>"]) if not native_mode else None,
                "workspace": str(agent_workspace(task)),
                "task_workspace": str(task.workspace),
                "codex_home": str(codex_home),
                "timeout_s": timeout_s,
                "timeout_policy": {
                    "official_timeout_seconds": task.metadata.get("timeout_seconds"),
                    "timeout_multiplier": task_timeout_multiplier(hcfg),
                    "effective_timeout_seconds": timeout_s,
                },
                "max_turns": task.metadata.get("max_turns"),
                "codex_tool_mode": "claw_native_tools" if native_mode else ("harness_cli_with_claw_model_proxy" if proxy_enabled else "codex_cli"),
                "tool_mode": "claw_native_tools" if native_mode else ("harness_cli_with_claw_model_proxy" if proxy_enabled else "codex_cli"),
                "codex_native_tools_disabled": bool(native_mode),
                "native_harness_tools_disabled": bool(native_mode),
                "model_visible_tools": model_visible_tools if (native_mode or proxy_enabled) else None,
                "model_tool_proxy": proxy_manifest,
                "live_tool_bridge": {
                    "requested": bool(task.metadata.get("live_tool_bridge_requested")),
                    "enabled": bool(task.metadata.get("live_tool_bridge")),
                    "bridge_url": task.metadata.get("claw_tool_bridge_url"),
                    "sandbox_url": task.metadata.get("claw_sandbox_url"),
                    "sandbox_mode": task.metadata.get("claw_sandbox_mode"),
                    "trace_schema": "claw_eval_live_v1" if task.metadata.get("live_tool_bridge") else "external_harness_raw",
                    "driver_sandbox_mode": self._live_sandbox_mode(hcfg) if task.metadata.get("live_tool_bridge") else str(hcfg.get("sandbox_mode", "workspace-write")),
                },
            }
            write_json(task.output_dir / "harness_manifest.json", manifest)
            if dry_run:
                if native_mode:
                    dry_run_text = "codex claw-native direct model loop\n" f"model_visible_tools={', '.join(model_visible_tools)}\n"
                elif proxy_enabled:
                    dry_run_text = (
                        str(manifest["cmd"]) + "\n"
                        + "model_tool_proxy=harness_cli_with_claw_model_proxy\n"
                        + f"model_visible_tools={', '.join(model_visible_tools)}\n"
                    )
                else:
                    dry_run_text = str(manifest["cmd"]) + "\n"
                (task.output_dir / "DRY_RUN.txt").write_text(dry_run_text, encoding="utf-8")
                return HarnessResult(task.task_id, self.name, model.model, "dry_run", trace_path=str(stdout_path), stdout_path=str(stdout_path), stderr_path=str(stderr_path), duration_s=0, returncode=0)

            if native_mode:
                rc, _final, native_status, native_metrics = run_native_claw_tools(
                    task,
                    model,
                    final_path,
                    stdout_path,
                    stderr_path,
                    timeout_s=timeout_s,
                    hcfg=hcfg,
                    dry_run=dry_run,
                    provider_factory=_make_native_provider,
                    metrics_prefix="codex",
                    harness_name=self.name,
                )
            else:
                try:
                    res = run_cmd(
                        cmd,
                        cwd=agent_workspace(task),
                        env=env,
                        timeout=timeout_s,
                        check=False,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        input_text="",
                    )
                    rc = res.returncode
                    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
                    if rc != 0 and any(s in stderr_text.lower() for s in ["unexpected argument", "unknown option", "unrecognized option"]):
                        retry_cmd = [str(hcfg.get("executable", "codex")), "exec", "-m", agent_model.model, "-C", str(agent_workspace(task)), "--json", "--skip-git-repo-check", "--output-last-message", str(final_path), self._prompt_for_codex(task)]
                        manifest["cmd_retry_minimal"] = quote_cmd([*retry_cmd[:-1], "<PROMPT>"])
                        write_json(task.output_dir / "harness_manifest.json", manifest)
                        res = run_cmd(retry_cmd, cwd=agent_workspace(task), env=env, timeout=timeout_s, check=False, stdout_path=stdout_path, stderr_path=stderr_path, input_text="")
                        rc = res.returncode
                except subprocess.TimeoutExpired:
                    rc = -124
                    stderr_path.write_text("Codex run timed out.\n", encoding="utf-8")

        patch_workspace = agent_workspace(task)
        patch = get_patch(patch_workspace)
        validation = validate_patch(patch_workspace, self.cfg.get("agent", {})) if patch else {"enabled": True, "ok": True}
        (task.output_dir / "patch.diff").write_text(patch, encoding="utf-8")
        (task.output_dir / "repo_status_final.txt").write_text(status_short(patch_workspace), encoding="utf-8")
        write_json(task.output_dir / "patch_validation.json", validation)
        if stdout_path.exists():
            try:
                from swecodex_harness.parse_codex_events import parse_events  # type: ignore

                parse_events(stdout_path, task.output_dir / "observability", keep_raw=True)
            except Exception as e:  # pragma: no cover
                write_json(task.output_dir / "observability_parse_error.json", {"error": str(e)})
        final = final_path.read_text(encoding="utf-8", errors="replace") if final_path.exists() else ""
        if rc == -124:
            status = "timeout"
        elif not native_mode:
            status = "ok" if rc == 0 and (patch or task.benchmark != "swe") else ("zero_patch" if rc == 0 else "nonzero")
        elif rc == 0 and native_status == "dry_run":
            status = "dry_run"
        elif rc == 0:
            status = "ok" if (final or task.benchmark != "swe") else "zero_patch"
        else:
            status = native_status or "nonzero"
        if patch and not validation.get("ok", True):
            status = "patch_validation_failed"
        duration = round(time.time() - start, 3)
        manifest.update({
            "finished_at": now_iso(),
            "returncode": rc,
            "duration_s": duration,
            "status": status,
            "patch_bytes": len(patch.encode("utf-8")),
            **native_metrics,
        })
        write_json(task.output_dir / "harness_manifest.json", manifest)
        result_trace_path = str(task.metadata.get("claw_live_trace_path") or stdout_path)
        metrics: dict[str, Any] = {"patch_validation": validation}
        if native_mode:
            metrics["codex_native_tools"] = native_metrics
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
        )
