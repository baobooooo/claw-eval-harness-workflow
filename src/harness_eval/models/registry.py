from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harness_eval.io import load_yaml
from harness_eval.types import ModelProfile


def _coerce_profile(name: str, raw: dict[str, Any]) -> ModelProfile:
    api_key_env = str(raw.get("api_key_env") or raw.get("env_key") or "OPENAI_API_KEY")
    api_key_value = raw.get("api_key_value")
    if api_key_value is None and api_key_env:
        api_key_value = os.environ.get(api_key_env)
    return ModelProfile(
        name=name,
        provider=str(raw.get("provider", name)),
        model=str(raw.get("model") or raw.get("served_model_name") or name),
        base_url=str(raw.get("base_url") or raw.get("endpoint") or ""),
        api_key_env=api_key_env,
        api_key_value=api_key_value,
        protocol=str(raw.get("protocol", "openai_chat")),
        context_window=int(raw["context_window"]) if raw.get("context_window") is not None else None,
        max_output_tokens=int(raw["max_output_tokens"]) if raw.get("max_output_tokens") is not None else None,
        extra_headers=dict(raw.get("extra_headers") or {}),
        extra_body=dict(raw.get("extra_body") or {}),
        notes=str(raw.get("notes", "")),
    )


def load_model_profiles(path: str | Path = "configs/models/models.yaml") -> dict[str, ModelProfile]:
    data = load_yaml(path)
    raw_profiles = data.get("models", data)
    if not isinstance(raw_profiles, dict):
        raise ValueError(f"{path} must contain a mapping of model profiles")
    return {name: _coerce_profile(name, raw) for name, raw in raw_profiles.items() if isinstance(raw, dict)}


def resolve_model(name: str, path: str | Path = "configs/models/models.yaml") -> ModelProfile:
    profiles = load_model_profiles(path)
    if name not in profiles:
        known = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown model profile {name!r}. Known profiles: {known}")
    return profiles[name]
