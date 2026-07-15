# Claw-Eval external harness mode

Claw-Eval's stock `claw-eval batch` command runs its own agent loop.  That is fine for model capability, but it confounds harness comparisons.  This project instead runs the selected harness first and writes:

```text
runs/<run_id>/harness_predictions.jsonl
```

The optional sidecar module in `external_patches/claw_eval_harness_mode/` can be copied into a Claw-Eval checkout and used as the `benchmark.eval_command` target.  It consumes `harness_predictions.jsonl` and grades records that contain a Claw-Eval-compatible trace path.  For harness traces that are not yet Claw-Eval trace schema, it writes a structured `needs_trace_conversion` result instead of silently pretending the score is valid.

Install into a cloned Claw-Eval repo:

```bash
python scripts/patch_claw_eval_for_harness_mode.py --claw-eval-root /path/to/claw-eval
```

Then configure:

```yaml
benchmark:
  eval_command: >-
    python -m claw_eval_harness_mode.grade
    --predictions {predictions}
    --tasks-dir /path/to/claw-eval/tasks
    --out-dir {eval_dir}
```

This sidecar is intentionally conservative.  It does not fake Claw-Eval trace events for Codex/nanobot.  For Codex/nanobot traces, use the emitted workspace/trace/final_message for manual trajectory analysis, or add a trace converter once the target Claw-Eval trace schema is fixed in your environment.
