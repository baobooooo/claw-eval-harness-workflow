from __future__ import annotations

from pathlib import Path
from typing import Any

from harness_eval.benchmarks.base import BenchmarkAdapter
from harness_eval.benchmarks.openclaw import OpenClawBenchmark
from harness_eval.benchmarks.swe import SweBenchBenchmark


def make_benchmark(name: str, cfg: dict[str, Any], run_dir: Path) -> BenchmarkAdapter:
    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"swe", "swebench", "swe-bench"}:
        return SweBenchBenchmark(cfg, run_dir)
    if normalized in {"openclaw", "claw-eval", "claw", "claw_eval"}:
        return OpenClawBenchmark(cfg, run_dir)
    raise ValueError(f"Unknown benchmark adapter: {name}")


__all__ = ["BenchmarkAdapter", "SweBenchBenchmark", "OpenClawBenchmark", "make_benchmark"]
