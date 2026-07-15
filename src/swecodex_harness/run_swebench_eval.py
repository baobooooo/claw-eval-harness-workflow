from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .common import now_iso, quote_cmd, run_cmd, write_json
from .config import get_path, load_config


def run_eval(cfg: dict[str, Any], predictions_path: str | Path, run_id: str | None = None, timeout_s: int | None = None) -> dict[str, Any]:
    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    predictions_path = predictions_path.resolve()
    if run_id is None:
        run_id = predictions_path.parent.name
    run_dir = predictions_path.parent
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(get_path(cfg, "swebench.dataset_name")),
        "--split",
        str(get_path(cfg, "swebench.split", "test")),
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(get_path(cfg, "swebench.max_workers_eval", 8)),
        "--run_id",
        str(run_id),
        "--cache_level",
        str(get_path(cfg, "swebench.cache_level", "env")),
    ]
    if get_path(cfg, "swebench.clean_eval", False):
        cmd.append("--clean")
    start = time.time()
    manifest: dict[str, Any] = {
        "started_at": now_iso(),
        "run_id": run_id,
        "predictions_path": str(predictions_path),
        "cmd": quote_cmd(cmd),
    }
    write_json(eval_dir / "eval_manifest.json", manifest)
    res = run_cmd(cmd, cwd=run_dir, timeout=timeout_s, check=False, stdout_path=eval_dir / "swebench_eval_stdout.log", stderr_path=eval_dir / "swebench_eval_stderr.log")
    manifest.update({"returncode": res.returncode, "finished_at": now_iso(), "duration_s": round(time.time() - start, 3)})
    # Copy known report files if SWE-bench wrote them under cwd or logs.
    for candidate in run_dir.rglob("*.json"):
        if candidate == eval_dir / "eval_manifest.json":
            continue
        if "report" in candidate.name.lower() or "results" in candidate.name.lower():
            manifest.setdefault("candidate_result_files", []).append(str(candidate))
    write_json(eval_dir / "eval_manifest.json", manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SWE-bench harness evaluation on a predictions.jsonl file.")
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--stage-config", default=None)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timeout-s", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.stage_config)
    manifest = run_eval(cfg, args.predictions, args.run_id, args.timeout_s)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
