from pathlib import Path
from types import SimpleNamespace

from swecodex_harness import run_swebench_eval


def test_run_eval_passes_absolute_predictions_path_when_cwd_is_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = Path("runs/stage1")
    run_dir.mkdir(parents=True)
    predictions = run_dir / "predictions.jsonl"
    predictions.write_text('{"instance_id":"demo","model_patch":"diff --git a/a b/a\\n"}\n', encoding="utf-8")
    captured = {}

    def fake_run_cmd(cmd, cwd=None, timeout=None, check=False, stdout_path=None, stderr_path=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        if stdout_path is not None:
            stdout_path.write_text("", encoding="utf-8")
        if stderr_path is not None:
            stderr_path.write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_swebench_eval, "run_cmd", fake_run_cmd)

    run_swebench_eval.run_eval(
        {
            "swebench": {
                "dataset_name": "princeton-nlp/SWE-bench_Lite",
                "split": "test",
                "max_workers_eval": 1,
                "cache_level": "env",
            }
        },
        predictions,
        run_id="eval_path_test",
        timeout_s=10,
    )

    pred_arg = captured["cmd"][captured["cmd"].index("--predictions_path") + 1]
    assert Path(pred_arg).is_absolute()
    assert Path(pred_arg).exists()
    assert captured["cwd"] == run_dir.resolve()
