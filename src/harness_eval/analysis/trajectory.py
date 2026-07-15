from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harness_eval.io import read_jsonl, write_json


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        out[status] = out.get(status, 0) + 1
    return out


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    rows = [x for x in read_jsonl(run_dir / "records" / "harness_results.jsonl") if isinstance(x, dict)]
    tool_event_counts = []
    agent_message_counts = []
    for inst in (run_dir / "instances").glob("*") if (run_dir / "instances").exists() else []:
        stats_path = inst / "observability" / "codex_event_stats.json"
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                tool_event_counts.append(int(stats.get("tool_event_count", 0)))
                agent_message_counts.append(int(stats.get("agent_message_count", 0)))
            except Exception:
                pass
    summary = {
        "run_dir": str(run_dir),
        "num_instances": len(rows),
        "statuses": _status_counts(rows),
        "patch_bytes_total": sum(int(r.get("patch_bytes", 0) or 0) for r in rows),
        "patch_bytes_mean": (sum(int(r.get("patch_bytes", 0) or 0) for r in rows) / len(rows)) if rows else None,
        "duration_s_total": sum(float(r.get("duration_s", 0) or 0) for r in rows),
        "codex_tool_events_mean": (sum(tool_event_counts) / len(tool_event_counts)) if tool_event_counts else None,
        "codex_agent_messages_mean": (sum(agent_message_counts) / len(agent_message_counts)) if agent_message_counts else None,
    }
    write_json(run_dir / "analysis" / "trajectory_summary.json", summary)
    return summary


def compare_runs(run_dirs: list[str | Path], out: str | Path | None = None) -> dict[str, Any]:
    summaries = [summarize_run(p) for p in run_dirs]
    comparison = {"runs": summaries}
    if out:
        write_json(out, comparison)
    return comparison


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize or compare harness trajectories.")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(json.dumps(compare_runs(args.run_dirs, args.out), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
