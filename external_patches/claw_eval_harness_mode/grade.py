from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any


# Keep mock-service traffic local even when the user runs the grader behind an
# HTTP proxy.  This mirrors claw_eval.cli behavior.
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"line_no": line_no, "status": "invalid_json", "error": str(exc)})
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def looks_like_claw_trace(trace_path: Path) -> bool:
    if not trace_path.exists() or not trace_path.is_file() or trace_path.stat().st_size == 0:
        return False
    try:
        with trace_path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                keys = set(obj)
                # Current Claw-Eval traces are pydantic event JSONL.  Across
                # versions, trace_start/trace_id/task_id/model/type are the
                # stable signals we can check without importing internals.
                if {"trace_id", "task_id"} <= keys or obj.get("type") in {"trace_start", "TraceStart"}:
                    return True
    except Exception:
        return False
    return False


def safe_slug(value: Any, default: str) -> str:
    raw = str(value or default).strip() or default
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return slug.strip("._-") or default


def resolve_task_yaml(row: dict[str, Any], tasks_dir: Path | None) -> Path | None:
    explicit = row.get("task_yaml") or row.get("task_path")
    if explicit:
        p = Path(str(explicit))
        return p if p.exists() else None
    if not tasks_dir:
        return None
    task_id = str(row.get("task_id") or "")
    if not task_id:
        return None
    candidates = [tasks_dir / task_id / "task.yaml", tasks_dir / f"{task_id}.yaml"]
    for p in candidates:
        if p.exists():
            return p
    return None


def _snapshot_file_entry(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    mime_type, _ = mimetypes.guess_type(str(path))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "encoding": "base64",
            "content": base64.b64encode(data).decode("ascii"),
            "mime_type": mime_type or "application/octet-stream",
            "size_bytes": len(data),
        }
    return {
        "encoding": "utf-8",
        "content": text,
        "mime_type": mime_type or "text/plain",
        "size_bytes": len(data),
    }


def load_env_snapshot(row: dict[str, Any], task: Any, task_yaml: Path) -> dict[str, Any] | None:
    raw = read_json(Path(str(row["env_snapshot_path"]))) if row.get("env_snapshot_path") else None
    snapshot: dict[str, Any] = raw if isinstance(raw, dict) else {}

    # Fallback for runs created before harness_eval wrote env_snapshot.json.
    # local_grader_files are host-side ground truth artifacts and are safe to
    # expose only to the grader, never to the external agent workspace.
    for rel_path in getattr(task, "local_grader_files", []) or []:
        key = f"local_file:{rel_path}"
        if key in snapshot:
            continue
        local_path = task_yaml.parent / str(rel_path)
        if local_path.exists() and local_path.is_file():
            snapshot[key] = _snapshot_file_entry(local_path)
        else:
            snapshot[key] = {"error": f"not found: {local_path}"}

    return snapshot or None


def _score_dict(scores: Any) -> dict[str, Any]:
    if hasattr(scores, "model_dump"):
        return dict(scores.model_dump())
    return {
        "completion": getattr(scores, "completion", 0.0),
        "robustness": getattr(scores, "robustness", 0.0),
        "communication": getattr(scores, "communication", 0.0),
        "safety": getattr(scores, "safety", 1.0),
    }


def grade_one(
    row: dict[str, Any],
    *,
    tasks_dir: Path | None,
    config: str | None,
    out_dir: Path,
    record_index: int,
    no_judge: bool,
    judge_model: str | None,
    append_grading: bool,
) -> dict[str, Any]:
    from claw_eval.cli import (
        _append_grading_to_trace,
        _apply_proxy,
        _grade_with_optional_params,
        _make_judge,
        _resolve_tasks_dir,
        _trace_totals,
    )
    from claw_eval.config import load_config
    from claw_eval.graders.registry import get_grader
    from claw_eval.models.scoring import compute_task_score, is_pass
    from claw_eval.models.task import TaskDefinition
    from claw_eval.trace.reader import load_trace

    task_id = str(row.get("task_id") or row.get("instance_id") or "unknown")
    trace = Path(str(row.get("trace_path") or ""))
    task_yaml = resolve_task_yaml(row, tasks_dir)
    stem = "_".join([
        f"{record_index:04d}",
        safe_slug(row.get("harness"), "harness"),
        safe_slug(row.get("model"), "model"),
        safe_slug(task_id, "task"),
    ])
    result: dict[str, Any] = {
        "record_index": record_index,
        "task_id": task_id,
        "harness": row.get("harness"),
        "model": row.get("model"),
        "trace_path": str(trace) if str(trace) else None,
        "task_yaml": str(task_yaml) if task_yaml else None,
        "env_snapshot_path": row.get("env_snapshot_path"),
    }
    if not task_yaml:
        result.update({"status": "needs_task_yaml", "passed": None})
        return result
    if not looks_like_claw_trace(trace):
        result.update({
            "status": "needs_trace_conversion",
            "passed": None,
            "note": "trace_path is missing or not in Claw-Eval JSONL event schema",
        })
        return result

    try:
        cfg = load_config(config)

        judge = _make_judge(cfg, Namespace(no_judge=no_judge, judge_model=judge_model))
        start, messages, dispatches, media_events, end, audit_data = load_trace(trace)
        task = TaskDefinition.from_yaml(task_yaml)
        resolved_tasks_dir = tasks_dir or _resolve_tasks_dir(task_yaml)
        grader = get_grader(task.task_id, tasks_dir=resolved_tasks_dir, task_dir=task_yaml.parent)
        env_snapshot = load_env_snapshot(row, task, task_yaml)
        scores, judge_calls = _grade_with_optional_params(
            grader,
            messages,
            dispatches,
            task,
            audit_data=audit_data,
            judge=judge,
            media_events=media_events,
            env_snapshot=env_snapshot,
        )
        task_score = compute_task_score(scores)
        passed = is_pass(task_score)
        totals = _trace_totals(end)
        if append_grading:
            _append_grading_to_trace(
                trace,
                start.trace_id,
                task.task_id,
                scores,
                task_score,
                passed,
                judge_calls=judge_calls,
                user_agent_meta={},
            )
        result.update(
            {
                "status": "graded",
                "task_name": task.task_name,
                "trace_id": start.trace_id,
                "trace_model": start.model,
                "turns": end.total_turns if end else 0,
                "scores": _score_dict(scores),
                "completion": float(getattr(scores, "completion", 0.0)),
                "robustness": float(getattr(scores, "robustness", 0.0)),
                "communication": float(getattr(scores, "communication", 0.0)),
                "safety": float(getattr(scores, "safety", 1.0)),
                "task_score": task_score,
                "passed": passed,
                "num_messages": len(messages),
                "num_dispatches": len(dispatches),
                "num_media_events": len(media_events),
                "audit_services": sorted(audit_data),
                "env_snapshot_keys": sorted(env_snapshot or {}),
                "judge_call_count": len(judge_calls or []),
                "judge_calls": judge_calls,
                **totals,
            }
        )
    except Exception as exc:
        tb_path = out_dir / f"{stem}.grade.traceback.log"
        tb_path.write_text(traceback.format_exc(), encoding="utf-8")
        result.update({
            "status": "grade_failed",
            "passed": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_path": str(tb_path),
        })
        return result

    per_record = out_dir / f"{stem}.grade.json"
    per_record.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(per_record)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade Claw-Eval tasks already run by an external harness.")
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tasks-dir", type=Path, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--no-append-grading", action="store_true", help="Do not append grading_result to converted traces")
    args = ap.parse_args()

    from claw_eval.cli import _apply_proxy

    _apply_proxy(args.proxy)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.predictions)
    results = [
        grade_one(
            row,
            tasks_dir=args.tasks_dir,
            config=args.config,
            out_dir=args.out_dir,
            record_index=i,
            no_judge=args.no_judge,
            judge_model=args.judge_model,
            append_grading=not args.no_append_grading,
        )
        for i, row in enumerate(rows)
    ]
    jsonl = args.out_dir / "external_grade_results.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    summary: dict[str, Any] = {"num_records": len(results), "statuses": {}, "results_path": str(jsonl)}
    scores: list[float] = []
    passes: list[bool] = []
    for item in results:
        st = str(item.get("status", "unknown"))
        summary["statuses"][st] = summary["statuses"].get(st, 0) + 1
        if item.get("status") == "graded" and item.get("task_score") is not None:
            scores.append(float(item["task_score"]))
            passes.append(bool(item.get("passed")))
    if scores:
        summary["mean_task_score"] = round(sum(scores) / len(scores), 4)
        summary["pass_rate"] = round(sum(1 for p in passes if p) / len(passes), 4)
    (args.out_dir / "external_grade_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if any(item.get("status") == "grade_failed" for item in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
