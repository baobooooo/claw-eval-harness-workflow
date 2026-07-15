from pathlib import Path

from harness_eval.harnesses.base import task_timeout_s
from harness_eval.types import BenchmarkTask


def _task(timeout_seconds=None):
    return BenchmarkTask(
        benchmark="openclaw",
        task_id="T_timeout",
        row={},
        prompt="",
        workspace=Path("/tmp/workspace"),
        output_dir=Path("/tmp/out"),
        metadata={"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {},
    )


def test_task_timeout_uses_official_yaml_budget_times_multiplier():
    assert task_timeout_s(_task(600), {"timeout_multiplier": 2.0, "timeout_s_per_instance": 3600}) == 1200
    assert task_timeout_s(_task("1800"), {"timeout_multiplier": "2", "timeout_s_per_instance": 600}) == 3600


def test_task_timeout_fallback_is_not_multiplied_without_official_budget():
    assert task_timeout_s(_task(), {"timeout_multiplier": 2.0, "timeout_s_per_instance": 900}) == 900
