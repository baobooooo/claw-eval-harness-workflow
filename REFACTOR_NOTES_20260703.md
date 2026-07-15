# Claw-Eval external-harness mode refactor notes — 2026-07-03

This refactor keeps the **Claw-Eval evaluator** as the source of truth while letting Codex, nanobot, or OpenClaw provide the external agent loop.

## Key fixes

1. **Separated external agent execution from evaluator preparation**
   - Added `BenchmarkAdapter.finalize_task_result()`.
   - `harness_eval.run` now calls the post-run hook after the external harness exits and before tool-policy gating / prediction writing.

2. **Preserved Claw-Eval's temporal firewall**
   - `sandbox_files` are copied into the workspace before the external harness runs.
   - `sandbox_grader_files` are injected only after the external harness has stopped.
   - `env_snapshot_commands`, `env_snapshot_files`, and `local_grader_files` are collected into `env_snapshot.json` for the evaluator.
   - Grader-only files are not present in the workspace during the external harness run.

3. **Converted external traces into score-ready Claw-Eval traces**
   - `OpenClawBenchmark.evaluate()` now converts `harness_predictions.jsonl` to `eval/score_ready/score_ready_predictions.jsonl` before invoking `benchmark.eval_command`.
   - Converted traces use a Claw-Eval-compatible JSONL schema with `trace_start`, `message`, `tool_dispatch`, `audit_snapshot`, and `trace_end` events.
   - Service `/audit` payloads now produce both trajectory `tool_dispatch` evidence and `audit_snapshot` events, which `claw_eval.trace.reader.load_trace()` requires to populate `audit_data` for graders.

4. **Made the patched grader actually grade converted traces**
   - Rewrote `claw_eval_harness_mode.grade` so it calls Claw-Eval internals directly instead of shelling out to `claw-eval grade`.
   - It now passes `audit_data`, `media_events`, and `env_snapshot` into graders through Claw-Eval's optional-parameter grading path.
   - It writes `external_grade_results.jsonl`, `external_grade_summary.json`, and per-record `*.grade.json` files.

5. **Removed brittle absolute paths from smoke config**
   - `benchmark.eval_command` can now use `{project_root}`, `{claw_eval_root}`, and `{tasks_dir}` placeholders.
   - The stage-2 smoke config no longer hard-codes `/data1/...` paths.

6. **Added missing sample dataset file**
   - Added `data/sample_claw_eval.jsonl`, fixing the local sample test and enabling no-network sample task preparation.

7. **Hardened fixture tar extraction**
   - Added a traversal guard before extracting fixture tarballs.

## Useful commands

Run the test suite:

```bash
pytest -q
```

Run the external-harness grader from this checkout:

```bash
PYTHONPATH=external/claw-eval/src:$PYTHONPATH \
python -m claw_eval_harness_mode.grade \
  --predictions <run_dir>/eval/score_ready/score_ready_predictions.jsonl \
  --tasks-dir external/claw-eval/tasks \
  --out-dir <run_dir>/eval \
  --no-judge
```

The intended `benchmark.eval_command` pattern is:

```yaml
benchmark:
  eval_command: >-
    PYTHONPATH={claw_eval_root}/src:$PYTHONPATH
    python -m claw_eval_harness_mode.grade
    --predictions {predictions}
    --tasks-dir {tasks_dir}
    --out-dir {eval_dir}
```

## Validation performed

- `pytest -q` → `37 passed`
- Direct patched-grader smoke test with a minimal synthetic Claw-Eval task → `graded`, `task_score=1.0`, `pass_rate=1.0`

## Remaining caveats

- I did not run a full Codex / nanobot / OpenClaw live benchmark in this environment.
- Full Claw-Eval scoring still depends on having the official `external/claw-eval/tasks` files and any required mock-service dependencies installed.
- If a third-party harness bypasses the generated `claw_tool` helpers and directly calls services or the public network, the tool-policy gate may mark the instance as a violation before scoring.
