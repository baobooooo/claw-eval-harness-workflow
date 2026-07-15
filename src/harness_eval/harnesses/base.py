from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from harness_eval.types import BenchmarkTask, HarnessResult, ModelProfile


def _float_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def task_timeout_multiplier(hcfg: dict[str, Any]) -> float:
    """Multiplier applied to official Claw-Eval YAML time budgets.

    Keep this separate from ``timeout_s_per_instance``: the former preserves
    per-task YAML differences such as 600s vs 1800s, while the latter remains a
    fallback for tasks with no official budget.
    """
    raw = hcfg.get("timeout_multiplier")
    if raw is None:
        raw = hcfg.get("timeout_s_multiplier")
    parsed = _float_value(raw)
    return parsed if parsed is not None else 1.0


def task_timeout_s(task: BenchmarkTask, hcfg: dict[str, Any], default: float = 3600.0) -> float:
    official = _float_value(task.metadata.get("timeout_seconds"))
    if official is not None:
        return official * task_timeout_multiplier(hcfg)
    configured = _float_value(hcfg.get("timeout_s_per_instance"))
    return configured if configured is not None else float(default)


def agent_workspace(task: BenchmarkTask) -> Path:
    """Workspace visible to the external harness process.

    In strict Claw-Eval live mode this is a driver workspace containing only
    bridge clients.  The scored filesystem lives behind the Claw sandbox URL.
    """
    raw = task.metadata.get("agent_workspace")
    return Path(str(raw)) if raw else task.workspace


class HarnessAdapter(ABC):
    name: str

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    @abstractmethod
    def run(self, task: BenchmarkTask, model: ModelProfile, dry_run: bool = False) -> HarnessResult:
        raise NotImplementedError
