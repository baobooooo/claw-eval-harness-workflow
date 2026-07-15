from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

from harness_eval.types import BenchmarkTask, EvalResult, HarnessResult


class BenchmarkAdapter(ABC):
    name: str

    def __init__(self, cfg: dict[str, Any], run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir

    @abstractmethod
    def load_rows(self, max_instances: int | None = None, instance_ids: set[str] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def prepare_task(self, row: dict[str, Any]) -> BenchmarkTask:
        raise NotImplementedError

    def task_run_context(self, task: BenchmarkTask, model: Any | None = None) -> ContextManager[None]:
        """Optional per-task runtime context.

        Benchmark adapters can override this to start task-declared mock
        services before an external harness runs, then collect service audit
        evidence afterwards. The default is a no-op for benchmarks that do not
        need supporting services.
        """
        return nullcontext()

    def finalize_task_result(self, result: HarnessResult, task: BenchmarkTask) -> HarnessResult:
        """Optional post-run hook before policy gating and prediction writing.

        Claw-Eval external-harness mode uses this hook to collect environment
        snapshots after the external agent loop has finished, preserving the
        temporal firewall around grader-only files while still feeding the
        original evaluator enough state to score the task.
        """
        return result

    @abstractmethod
    def record_prediction(self, result: HarnessResult, task: BenchmarkTask) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, timeout_s: int | None = None) -> EvalResult:
        raise NotImplementedError
