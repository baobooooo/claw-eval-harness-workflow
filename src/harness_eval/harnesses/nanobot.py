from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness_eval.harnesses.external_cli import ExternalCliHarness
from harness_eval.types import BenchmarkTask, ModelProfile


def _read_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            obj = yaml.safe_load(text)
        except Exception:
            obj = None
    else:
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
    return obj if isinstance(obj, dict) else {}


def _resolve_template(raw: Any) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([
            Path.cwd() / path,
            Path(__file__).resolve().parents[3] / path,  # repository root when running from source checkout
        ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return path


class NanobotHarness(ExternalCliHarness):
    name = "nanobot"
    default_command_template = (
        "nanobot agent --workspace {workspace} --config {nanobot_config_file} "
        "--session harness:{task_id} --no-markdown --no-logs "
        "--message \"$(<{prompt_file})\" | tee {output_dir}/final_message.txt {trace_file}"
    )

    def _render_nanobot_config(self, task: BenchmarkTask, model: ModelProfile) -> Path:
        hcfg = self.cfg.get("harness", {})
        template_path = _resolve_template(
            hcfg.get("nanobot_config_template")
            or hcfg.get("config_template")
            or hcfg.get("config_path")
            or hcfg.get("nanobot_config")
        )
        cfg = _read_config(template_path)
        agents = cfg.setdefault("agents", {})
        defaults = agents.setdefault("defaults", {})
        if not isinstance(defaults, dict):
            defaults = {}
            agents["defaults"] = defaults
        provider_id = str(hcfg.get("provider_id") or defaults.get("provider") or model.provider or "deepseek")
        defaults["model"] = model.model
        defaults["provider"] = provider_id
        if model.max_output_tokens:
            defaults.setdefault("max_tokens", model.max_output_tokens)
        if model.context_window:
            defaults.setdefault("context_window_tokens", model.context_window)

        providers = cfg.setdefault("providers", {})
        provider_cfg = providers.setdefault(provider_id, {})
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}
            providers[provider_id] = provider_cfg
        # NanoBot reads provider.api_base from this file; in formal proxy mode it
        # must point at the per-task local model_tool_proxy, not the original
        # shared DeepSeek bridge.  The proxy forwards to the fixed model itself.
        provider_cfg["api_base"] = model.base_url
        provider_cfg.setdefault("base_url", model.base_url)
        provider_cfg.setdefault("api_key", "${" + model.api_key_env + "}")

        tools = cfg.setdefault("tools", {})
        if isinstance(tools, dict):
            tools.setdefault("restrict_to_workspace", True)
            web_cfg = tools.setdefault("web", {})
            if isinstance(web_cfg, dict):
                web_cfg.setdefault("enable", False)

        out = task.output_dir / "nanobot_model_proxy_config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out

    def _extra_format_context(self, task: BenchmarkTask, model: ModelProfile, stdout_path: Path, stderr_path: Path, trace_path: Path, prompt_file: Path) -> dict[str, Any]:
        config_file = self._render_nanobot_config(task, model)
        return {"nanobot_config_file": config_file}
