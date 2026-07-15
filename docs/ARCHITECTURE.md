# Architecture

## Why this refactor

The original project optimized one path:

```text
SWE-bench -> Codex CLI -> local Qwen3 MoE -> SWE-bench evaluator
```

That path is useful for one baseline, but it cannot answer whether score changes come from the model or from the harness.  The refactor introduces three independent axes.

## Axis 1: BenchmarkAdapter

A benchmark adapter owns dataset loading, workspace/task preparation, prediction recording, and evaluation.

```python
class BenchmarkAdapter:
    def load_rows(...): ...
    def prepare_task(row) -> BenchmarkTask: ...
    def record_prediction(result, task): ...
    def evaluate(...): ...
```

Implemented adapters:

- `SweBenchBenchmark`: reuses the old repo checkout and emits SWE-bench predictions.
- `OpenClawBenchmark`: loads Claw-Eval/OpenClaw rows, prepares a workspace from fixtures, delegates to an external harness, and emits `harness_predictions.jsonl`.

## Axis 2: HarnessAdapter

A harness adapter owns the agent loop.  It receives a prepared task and a model profile.

```python
class HarnessAdapter:
    def run(task: BenchmarkTask, model: ModelProfile, dry_run=False) -> HarnessResult: ...
```

Implemented adapters:

- `CodexHarness`: renders `CODEX_HOME/config.toml`, runs `codex exec`, parses Codex events, collects patch and validation.
- `NanobotHarness`: generic CLI wrapper configured by `configs/harnesses/nanobot.yaml`.
- `OpenClawHarness`: generic CLI/ACP wrapper configured by `configs/harnesses/openclaw.yaml`.

The generic CLI wrapper intentionally uses templates so the harness command can be adapted without touching Python code.

## Axis 3: ModelProfile

A model profile is endpoint metadata, not an LLM client in the runner:

```python
@dataclass
class ModelProfile:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    protocol: str
```

This keeps harness behavior separable from model behavior.  The same `mimo_v2_5_pro` profile can be passed to Codex, nanobot, and OpenClaw.

## Claw-Eval external-harness mode

The stock Claw-Eval command owns the model/tool loop.  That measures a combined `model + Claw-Eval mini harness` system.  In this project the `OpenClawBenchmark` adapter stops at task/workspace preparation and lets the selected external harness act.  Evaluation then consumes `harness_predictions.jsonl` and trace/workspace paths.

This makes the primary comparison:

```text
same task + same model + different harness -> different outcome/trajectory
```

## Output contract

Every run writes:

```text
runs/<run_id>/records/resolved_config.json
runs/<run_id>/records/model_profile.json
runs/<run_id>/records/dataset_subset.jsonl
runs/<run_id>/records/harness_results.jsonl
runs/<run_id>/instances/<task_id>/prompt.md
runs/<run_id>/instances/<task_id>/harness_manifest.json
runs/<run_id>/instances/<task_id>/patch.diff
runs/<run_id>/instances/<task_id>/patch_validation.json
runs/<run_id>/analysis/summary.json
```

Benchmark-specific outputs:

```text
SWE:      runs/<run_id>/predictions.jsonl
OpenClaw: runs/<run_id>/harness_predictions.jsonl
```
