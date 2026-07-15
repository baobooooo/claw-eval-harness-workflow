from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from harness_eval.benchmarks.base import BenchmarkAdapter
from harness_eval.io import append_jsonl, get_path, now_iso, quote_cmd, run_cmd, safe_id, write_json
from harness_eval.types import BenchmarkTask, EvalResult, HarnessResult


class SweBenchBenchmark(BenchmarkAdapter):
    """SWE-bench adapter that exposes repo checkout as a generic harness task."""

    name = "swe"

    def _load_dataset(self, dataset_name: str, split: str):
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("datasets is required for SWE-bench loading: pip install datasets") from e
        return load_dataset(dataset_name, split=split)

    def load_rows(self, max_instances: int | None = None, instance_ids: set[str] | None = None) -> list[dict[str, Any]]:
        bcfg = self.cfg.get("benchmark", {})
        if max_instances is None:
            cfg_max = bcfg.get("max_instances", get_path(self.cfg, "swebench.max_instances"))
            max_instances = int(cfg_max) if cfg_max not in (None, "") else None
        dataset_name = str(bcfg.get("dataset_name") or get_path(self.cfg, "swebench.dataset_name", "princeton-nlp/SWE-bench_Lite"))
        split = str(bcfg.get("split") or get_path(self.cfg, "swebench.split", "test"))
        rows: list[dict[str, Any]] = []
        for row in self._load_dataset(dataset_name, split):
            item = dict(row)
            iid = str(item.get("instance_id"))
            if instance_ids and iid not in instance_ids:
                continue
            rows.append(item)
            if max_instances is not None and len(rows) >= max_instances:
                break
        if not rows:
            raise RuntimeError("No SWE-bench rows selected")
        out = self.run_dir / "records" / "dataset_subset.jsonl"
        if out.exists():
            out.unlink()
        for row in rows:
            append_jsonl(out, row)
        return rows

    def _render_prompt(self, row: dict[str, Any]) -> str:
        template_path = Path(get_path(self.cfg, "benchmark.prompt_template", "configs/prompts/codex_swebench_prompt.md"))
        if not template_path.is_absolute():
            template_path = Path(get_path(self.cfg, "project.root", ".")) / template_path
        template = template_path.read_text(encoding="utf-8")
        values = {
            "INSTANCE_ID": row.get("instance_id", ""),
            "REPO": row.get("repo", ""),
            "BASE_COMMIT": row.get("base_commit", ""),
            "PROBLEM_STATEMENT": row.get("problem_statement", ""),
            "HINTS_TEXT": row.get("hints_text", ""),
            "TEST_PATCH": row.get("test_patch", ""),
            "FAIL_TO_PASS": row.get("FAIL_TO_PASS", row.get("fail_to_pass", "")),
            "PASS_TO_PASS": row.get("PASS_TO_PASS", row.get("pass_to_pass", "")),
            "RUN_TESTS_INSIDE_CODEX": get_path(self.cfg, "agent.run_tests_inside_codex", False),
        }
        text = template
        for k, v in values.items():
            text = text.replace("{{" + k + "}}", str(v))
        return text

    def prepare_task(self, row: dict[str, Any]) -> BenchmarkTask:
        # Reuse the existing, hardened repo checkout logic from the original project.
        from swecodex_harness.run_codex_on_swebench import _prepare_repo  # type: ignore

        iid = str(row["instance_id"])
        out_dir = self.run_dir / "instances" / safe_id(iid)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "swebench_instance.json", row)
        repo = _prepare_repo(row, self.cfg, out_dir)
        prompt = self._render_prompt(row)
        (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        return BenchmarkTask(
            benchmark=self.name,
            task_id=iid,
            row=row,
            prompt=prompt,
            workspace=Path(repo),
            output_dir=out_dir,
            repo=str(row.get("repo")),
            base_commit=str(row.get("base_commit")),
            metadata={"dataset_name": get_path(self.cfg, "swebench.dataset_name"), "split": get_path(self.cfg, "swebench.split")},
        )

    @property
    def predictions_path(self) -> Path:
        return self.run_dir / "predictions.jsonl"

    def record_prediction(self, result: HarnessResult, task: BenchmarkTask) -> None:
        append_jsonl(
            self.predictions_path,
            {
                "instance_id": task.task_id,
                "model_name_or_path": result.model,
                "model_patch": result.patch,
            },
        )

    def evaluate(self, timeout_s: int | None = None) -> EvalResult:
        eval_dir = self.run_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        predictions = self.predictions_path.resolve()
        if not predictions.exists():
            raise FileNotFoundError(predictions)
        run_id = self.run_dir.name
        swe_cfg = self.cfg.get("swebench", {})
        bench_cfg = self.cfg.get("benchmark", {})
        cmd = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            str(bench_cfg.get("dataset_name") or swe_cfg.get("dataset_name", "princeton-nlp/SWE-bench_Lite")),
            "--split",
            str(bench_cfg.get("split") or swe_cfg.get("split", "test")),
            "--predictions_path",
            str(predictions),
            "--max_workers",
            str(swe_cfg.get("max_workers_eval", 8)),
            "--run_id",
            run_id,
            "--cache_level",
            str(swe_cfg.get("cache_level", "env")),
        ]
        if swe_cfg.get("clean_eval", False):
            cmd.append("--clean")
        manifest = {"started_at": now_iso(), "cmd": quote_cmd(cmd), "predictions_path": str(predictions)}
        write_json(eval_dir / "eval_manifest.json", manifest)
        start = time.time()
        res = run_cmd(cmd, cwd=self.run_dir, timeout=timeout_s, check=False, stdout_path=eval_dir / "swebench_eval_stdout.log", stderr_path=eval_dir / "swebench_eval_stderr.log")
        manifest.update({"returncode": res.returncode, "finished_at": now_iso(), "duration_s": round(time.time() - start, 3)})
        for p in self.run_dir.rglob("*.json"):
            if p == eval_dir / "eval_manifest.json":
                continue
            if "report" in p.name.lower() or "result" in p.name.lower():
                manifest.setdefault("candidate_result_files", []).append(str(p))
        write_json(eval_dir / "eval_manifest.json", manifest)
        return EvalResult(self.name, self.run_dir, "ok" if res.returncode == 0 else "nonzero", manifest, manifest.get("candidate_result_files", []))
