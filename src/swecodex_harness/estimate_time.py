from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .common import read_jsonl


def estimate_from_run(run_dir: str | Path, target_instances: int, eval_seconds_per_instance: float | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifests = read_jsonl(run_dir / "records" / "instance_manifests.jsonl")
    durations = [float(m["duration_s"]) for m in manifests if isinstance(m, dict) and isinstance(m.get("duration_s"), (int, float))]
    if not durations:
        # Conservative placeholder until pilot data exists.
        gen_mean = 20 * 60.0
        source = "default_heuristic_no_pilot_data"
    else:
        gen_mean = statistics.mean(durations)
        source = "pilot_mean_duration_s"
    eval_s = eval_seconds_per_instance if eval_seconds_per_instance is not None else 5 * 60.0
    total_gen = gen_mean * target_instances
    total_eval = eval_s * target_instances
    return {
        "source": source,
        "pilot_instances": len(durations),
        "target_instances": target_instances,
        "generation_mean_s_per_instance": round(gen_mean, 2),
        "eval_s_per_instance_assumption": round(eval_s, 2),
        "estimated_generation_gpu_hours": round(total_gen / 3600, 2),
        "estimated_eval_wall_hours_before_parallelism": round(total_eval / 3600, 2),
        "estimated_total_before_parallelism_hours": round((total_gen + total_eval) / 3600, 2),
        "note": "SWE-bench Docker evaluation wall-time depends strongly on image/env cache and max_workers; use the first pilot run to replace assumptions.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate benchmark time from a pilot run.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-instances", type=int, default=300)
    ap.add_argument("--eval-seconds-per-instance", type=float, default=None)
    args = ap.parse_args()
    print(json.dumps(estimate_from_run(args.run_dir, args.target_instances, args.eval_seconds_per_instance), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
