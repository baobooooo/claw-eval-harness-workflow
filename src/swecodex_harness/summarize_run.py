from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_json


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifests = read_jsonl(run_dir / "records" / "instance_manifests.jsonl")
    durations = [float(m["duration_s"]) for m in manifests if isinstance(m, dict) and isinstance(m.get("duration_s"), (int, float))]
    patch_bytes = [int(m.get("patch_bytes", 0)) for m in manifests if isinstance(m, dict)]
    statuses: dict[str, int] = {}
    for m in manifests:
        if isinstance(m, dict):
            statuses[str(m.get("status", "unknown"))] = statuses.get(str(m.get("status", "unknown")), 0) + 1

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_instances": len(manifests),
        "statuses": statuses,
        "duration_s": {
            "total": round(sum(durations), 3),
            "mean": round(statistics.mean(durations), 3) if durations else None,
            "median": round(statistics.median(durations), 3) if durations else None,
            "min": round(min(durations), 3) if durations else None,
            "max": round(max(durations), 3) if durations else None,
        },
        "patch_bytes": {
            "total": sum(patch_bytes),
            "mean": round(statistics.mean(patch_bytes), 1) if patch_bytes else None,
            "zero_patch_count": sum(1 for x in patch_bytes if x == 0),
        },
    }
    eval_manifest = _load_json(run_dir / "eval" / "eval_manifest.json")
    if eval_manifest:
        summary["eval_manifest"] = eval_manifest
    metrics_path = run_dir / "metrics" / "metrics_selected.jsonl"
    if metrics_path.exists():
        lines = read_jsonl(metrics_path)
        summary["metrics_samples"] = len(lines)
        # Keep last sample keys only; raw metrics are preserved separately.
        if lines and isinstance(lines[-1], dict):
            summary["last_metrics_keys"] = sorted((lines[-1].get("metrics") or {}).keys())[:200]
    write_json(run_dir / "analysis" / "summary.json", summary)
    md = [
        f"# Run Summary: `{run_dir.name}`",
        "",
        f"- instances: {summary['num_instances']}",
        f"- statuses: `{json.dumps(statuses, ensure_ascii=False, sort_keys=True)}`",
        f"- total generation time: {summary['duration_s']['total']} s",
        f"- mean / median generation time: {summary['duration_s']['mean']} s / {summary['duration_s']['median']} s",
        f"- zero-patch count: {summary['patch_bytes']['zero_patch_count']}",
        "",
        "Artifacts to inspect:",
        "",
        "- `records/resolved_config.json`",
        "- `records/runtime_snapshot.json`",
        "- `predictions.jsonl`",
        "- `instances/<instance_id>/observability/module_timeline.jsonl`",
        "- `metrics/metrics_selected.jsonl` and `metrics/raw_prometheus/`",
        "- `kv_events.jsonl` when KV events are enabled",
    ]
    (run_dir / "analysis" / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a Codex+SWE-bench run.")
    ap.add_argument("run_dir")
    args = ap.parse_args()
    print(json.dumps(summarize(args.run_dir), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
