from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base and return base."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return _expand(data)


def load_config(project_config: str | Path, stage_config: str | Path | None = None) -> dict[str, Any]:
    cfg = load_yaml(project_config)
    if stage_config:
        deep_update(cfg, load_yaml(stage_config))
    return cfg


def get_path(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def ensure_dirs(cfg: Mapping[str, Any]) -> None:
    for dotted in ["project.root", "project.data_root", "project.runs_root", "project.external_root"]:
        p = get_path(cfg, dotted)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)
