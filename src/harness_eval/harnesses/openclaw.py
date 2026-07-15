from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any

from harness_eval.claw_live.model_proxy import model_tool_proxy_enabled
from harness_eval.harnesses.base import agent_workspace
from harness_eval.harnesses.external_cli import ExternalCliHarness
from harness_eval.types import BenchmarkTask, ModelProfile


class OpenClawHarness(ExternalCliHarness):
    name = "openclaw"
    default_command_template = (
        "openclaw{openclaw_profile_arg} agent --local --model {openclaw_model} "
        "--message-file {prompt_file} --session-key agent:openclaw:{task_id} "
        "--timeout {timeout_s} --json | tee {output_dir}/final_message.txt {trace_file}"
    )


    def _openclaw_profile_name(self, task: BenchmarkTask) -> str:
        hcfg = self.cfg.get("harness", {})
        explicit = hcfg.get("profile") or hcfg.get("openclaw_profile")
        if explicit:
            return str(explicit).format(task_id=task.task_id)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", task.task_id).strip("-")[:48] or "task"
        digest = hashlib.sha1(str(task.output_dir.resolve()).encode("utf-8")).hexdigest()[:10]
        return f"clawh-{slug}-{digest}"

    def _openclaw_home(self, task: BenchmarkTask) -> Path:
        hcfg = self.cfg.get("harness", {})
        raw = hcfg.get("openclaw_home")
        if raw:
            return Path(str(raw).format(task_id=task.task_id, output_dir=task.output_dir))
        return task.output_dir / "openclaw_home"

    def _openclaw_profile_dir(self, task: BenchmarkTask) -> Path:
        profile = self._openclaw_profile_name(task)
        return self._openclaw_home(task) / f".openclaw-{profile}"

    def _openclaw_config_path(self, task: BenchmarkTask) -> Path:
        return self._openclaw_profile_dir(task) / "openclaw.json"

    def _openclaw_agent_dir(self, task: BenchmarkTask) -> Path:
        return self._openclaw_profile_dir(task) / "agents" / "openclaw" / "agent"

    def _openclaw_provider_config(self, task: BenchmarkTask, model: ModelProfile) -> tuple[str, dict[str, Any]]:
        hcfg = self.cfg.get("harness", {})
        provider_id = str(hcfg.get("openclaw_proxy_provider") or "claw_proxy")
        return provider_id, {
            "baseUrl": model.base_url,
            # The per-task proxy is local and does not require a real provider
            # secret. A literal key satisfies OpenClaw's auth check without
            # depending on global auth DBs.
            "apiKey": str(hcfg.get("openclaw_proxy_api_key") or "dummy"),
            "api": "openai-completions",
            "models": [
                {
                    "id": model.model,
                    "name": model.model,
                    "api": "openai-completions",
                    "reasoning": False,
                    "input": ["text"],
                    "contextWindow": model.context_window or 128000,
                    "maxTokens": model.max_output_tokens or 16384,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "compat": {"supportsUsageInStreaming": True, "maxTokensField": "max_tokens"},
                }
            ],
        }

    def _write_openclaw_profile_config(self, task: BenchmarkTask, provider_id: str, provider_config: dict[str, Any]) -> Path:
        path = self._openclaw_config_path(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                config: Any = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}
            if not isinstance(config, dict):
                config = {}
        else:
            config = {}
        models = config.get("models")
        if not isinstance(models, dict):
            models = {}
            config["models"] = models
        providers = models.get("providers")
        if not isinstance(providers, dict):
            providers = {}
            models["providers"] = providers
        providers[provider_id] = provider_config
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _render_openclaw_model_catalog(self, task: BenchmarkTask, model: ModelProfile) -> Path | None:
        hcfg = self.cfg.get("harness", {})
        if not model_tool_proxy_enabled(task, hcfg):
            return None
        provider_id, provider_config = self._openclaw_provider_config(task, model)
        self._write_openclaw_profile_config(task, provider_id, provider_config)
        catalog = {"providers": {provider_id: provider_config}}
        agent_dir = self._openclaw_agent_dir(task)
        agent_dir.mkdir(parents=True, exist_ok=True)
        path = agent_dir / "models.json"
        path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _openclaw_model_arg(self, task: BenchmarkTask, model: ModelProfile) -> str:
        hcfg = self.cfg.get("harness", {})
        explicit = hcfg.get("model_arg") or hcfg.get("openclaw_model")
        if explicit:
            return str(explicit).format(model=model.model, provider=model.provider, base_url=model.base_url)
        # Do not synthesize ``openai/{model}`` in proxy mode.  OpenClaw validates
        # the CLI model argument against its own registry before it reads the
        # OpenAI-compatible base URL env vars, and ``openai/deepseek-v4-pro`` is
        # rejected there.  Use the harness-accepted model id by default; callers
        # that need a provider-qualified alias can still set ``model_arg`` or
        # ``model_provider_prefix`` explicitly in the harness config.
        provider_prefix = hcfg.get("model_provider_prefix")
        if provider_prefix:
            return f"{provider_prefix}/{model.model}"
        if model_tool_proxy_enabled(task, hcfg):
            return f"{hcfg.get('openclaw_proxy_provider') or 'claw_proxy'}/{model.model}"
        return model.model

    def _extra_format_context(self, task: BenchmarkTask, model: ModelProfile, stdout_path: Path, stderr_path: Path, trace_path: Path, prompt_file: Path) -> dict[str, Any]:
        catalog_path = self._render_openclaw_model_catalog(task, model)
        profile = self._openclaw_profile_name(task) if catalog_path is not None else ""
        return {
            "openclaw_model": self._openclaw_model_arg(task, model),
            "openclaw_profile": profile,
            "openclaw_profile_arg": f" --profile {shlex.quote(profile)}" if profile else "",
            "openclaw_models_json": catalog_path or "",
        }

    def _env(self, model: ModelProfile, task: BenchmarkTask | None = None) -> dict[str, str]:
        env = super()._env(model, task)
        hcfg = self.cfg.get("harness", {})
        if task is not None and model_tool_proxy_enabled(task, hcfg):
            env["HOME"] = str(self._openclaw_home(task))
            env["OPENCLAW_CONFIG_PATH"] = str(self._openclaw_config_path(task))
            env.setdefault("OPENCLAW_MODEL_PROVIDER", str(hcfg.get("openclaw_proxy_provider") or "claw_proxy"))
            env.setdefault("OPENCLAW_PROVIDER", str(hcfg.get("openclaw_proxy_provider") or "claw_proxy"))
            # Some OpenClaw builds use OpenAI-style env names in addition to
            # models.json.  Keep these aliases aligned with the per-task proxy.
            env["OPENAI_BASE_URL"] = model.base_url
            env["OPENAI_API_BASE"] = model.base_url
            env["OPENAI_API_BASE_URL"] = model.base_url
            env["DEEPSEEK_BASE_URL"] = model.base_url
            env["DEEPSEEK_API_BASE"] = model.base_url
            env["DEEPSEEK_API_BASE_URL"] = model.base_url
        return env

    def _format_command(self, task: BenchmarkTask, model: ModelProfile, stdout_path: Path, stderr_path: Path, trace_path: Path) -> str:
        # Current OpenClaw CLI does not accept a workspace positional argument.
        # Run the process from the driver workspace instead; that is the only
        # workspace containing bridge helpers, while scored state remains in the
        # MiniHarness/Claw-Eval Docker sandbox behind the bridge.
        formatted = super()._format_command(task, model, stdout_path, stderr_path, trace_path)
        return f"cd {shlex.quote(str(agent_workspace(task)))} && {formatted}"
