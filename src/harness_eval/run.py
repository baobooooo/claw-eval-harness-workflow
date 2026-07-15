from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from harness_eval.benchmarks import make_benchmark
from harness_eval.harnesses import make_harness
from harness_eval.analysis.tool_policy import enforce_tool_policy
from harness_eval.io import append_jsonl, deep_update, get_path, load_yaml, safe_id, write_json
from harness_eval.models import resolve_model


def _parse_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    p = Path(value)
    if p.exists():
        return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")}
    return {x.strip() for x in value.split(",") if x.strip()}


def _assign_row_indexes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "_harness_eval_row_index": idx} for idx, row in enumerate(rows)]


def _resolve_max_workers(cli_value: int | None, cfg: dict[str, Any]) -> int:
    raw = cli_value
    if raw is None:
        raw = get_path(cfg, "run.max_workers") or get_path(cfg, "benchmark.max_workers") or 1
    try:
        workers = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("--max-workers must be an integer") from exc
    if workers < 1:
        raise ValueError("--max-workers must be >= 1")
    return workers


def load_run_config(project_config: str, benchmark_config: str | None, harness_config: str | None) -> dict[str, Any]:
    cfg = load_yaml(project_config)
    root = Path(str(get_path(cfg, "project.root", ".")))
    if str(root) == ".":
        cfg.setdefault("project", {})["root"] = str(Path.cwd())
    for path in [benchmark_config, harness_config]:
        if path:
            deep_update(cfg, load_yaml(path))
    # Normalize relative project paths against project.root.
    root = Path(str(get_path(cfg, "project.root", Path.cwd()))).resolve()
    cfg.setdefault("project", {})["root"] = str(root)
    for key in ["data_root", "runs_root", "external_root"]:
        value = get_path(cfg, f"project.{key}")
        if value:
            p = Path(str(value))
            if not p.is_absolute():
                cfg["project"][key] = str(root / p)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Run benchmark × harness × model evaluation.")
    ap.add_argument("--project-config", default="configs/project.yaml")
    ap.add_argument("--benchmark-config", default=None)
    ap.add_argument("--harness-config", default=None)
    ap.add_argument("--model-config", default="configs/models/models.yaml")
    ap.add_argument("--benchmark", required=True, choices=["swe", "openclaw", "claw-eval", "swebench"])
    ap.add_argument("--harness", required=True, choices=["codex", "nanobot", "openclaw"])
    ap.add_argument("--model", required=True, help="Model profile name in configs/models/models.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--instance-ids", default=None, help="Comma-separated IDs or a file with one ID per line")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-timeout-s", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=None, help="Run benchmark instances concurrently inside this harness run")
    args = ap.parse_args()

    cfg = load_run_config(args.project_config, args.benchmark_config, args.harness_config)
    try:
        max_workers = _resolve_max_workers(args.max_workers, cfg)
    except ValueError as exc:
        ap.error(str(exc))
    cfg.setdefault("run", {})["max_workers"] = max_workers
    bcfg = cfg.setdefault("benchmark", {})
    bcfg["effective_max_workers"] = max_workers
    if max_workers > 1 and "isolate_service_ports" not in bcfg and "service_port_isolation" not in bcfg:
        bcfg["isolate_service_ports"] = True
    model = resolve_model(args.model, args.model_config)
    run_id = args.run_id or f"{args.benchmark}_{args.harness}_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    runs_root = Path(str(get_path(cfg, "project.runs_root", "runs")))
    run_dir = runs_root / safe_id(run_id)
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "records" / "resolved_config.json", cfg)
    write_json(run_dir / "records" / "model_profile.json", model.to_json())

    benchmark = make_benchmark(args.benchmark, cfg, run_dir)
    selected_ids = _parse_ids(args.instance_ids)
    rows = _assign_row_indexes(benchmark.load_rows(args.max_instances, selected_ids))

    results = []
    result_jsonl = run_dir / "records" / "harness_results.jsonl"
    if result_jsonl.exists():
        result_jsonl.unlink()

    def run_one(row: dict[str, Any]) -> tuple[Any, Any]:
        task = benchmark.prepare_task(row)
        harness = make_harness(args.harness, cfg)
        run_context = nullcontext() if args.dry_run else benchmark.task_run_context(task, model)
        with run_context:
            result = harness.run(task, model, dry_run=args.dry_run)
            if not args.dry_run:
                result = benchmark.finalize_task_result(result, task)
        if not args.dry_run and (task.metadata.get("tool_policy_path") or "allowed_tools" in task.metadata):
            result = enforce_tool_policy(result, task)
        return result, task

    def record_one(result: Any, task: Any) -> None:
        benchmark.record_prediction(result, task)
        rec = result.to_json()
        rec.update(
            {
                "benchmark": task.benchmark,
                "workspace": str(task.workspace),
                "output_dir": str(task.output_dir),
                "row_index": task.row.get("_harness_eval_row_index"),
            }
        )
        append_jsonl(result_jsonl, rec)
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False, sort_keys=True), flush=True)

    if max_workers == 1 or len(rows) <= 1:
        for row in rows:
            result, task = run_one(row)
            record_one(result, task)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(run_one, row): row for row in rows}
            for future in as_completed(future_to_row):
                result, task = future.result()
                record_one(result, task)

    summary: dict[str, Any] = {
        "run_id": run_dir.name,
        "benchmark": args.benchmark,
        "harness": args.harness,
        "model_profile": args.model,
        "model": model.model,
        "num_instances": len(results),
        "statuses": {},
        "dry_run": args.dry_run,
        "max_workers": max_workers,
    }
    for rec in results:
        status = str(rec.get("status", "unknown"))
        summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
    if not args.no_eval and not args.dry_run:
        ev = benchmark.evaluate(timeout_s=args.eval_timeout_s)
        summary["eval"] = ev.manifest
    write_json(run_dir / "analysis" / "summary.json", summary)
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
