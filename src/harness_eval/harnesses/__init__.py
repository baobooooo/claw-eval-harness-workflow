from __future__ import annotations

from typing import Any

from harness_eval.harnesses.base import HarnessAdapter
from harness_eval.harnesses.codex import CodexHarness
from harness_eval.harnesses.nanobot import NanobotHarness
from harness_eval.harnesses.openclaw import OpenClawHarness


def make_harness(name: str, cfg: dict[str, Any]) -> HarnessAdapter:
    normalized = name.strip().lower().replace("_", "-")
    if normalized == "codex":
        return CodexHarness(cfg)
    if normalized == "nanobot":
        return NanobotHarness(cfg)
    if normalized == "openclaw":
        return OpenClawHarness(cfg)
    raise ValueError(f"Unknown harness adapter: {name}")


__all__ = ["HarnessAdapter", "CodexHarness", "NanobotHarness", "OpenClawHarness", "make_harness"]
