# Validation performed in this refactor pass

```bash
PYTHONPATH=src python -m compileall -q src
PYTHONPATH=src pytest -q
```

Result at packaging time: `18 passed`.

Also manually checked a no-network OpenClaw sample dry-run:

```bash
PYTHONPATH=src python -m harness_eval.run \
  --benchmark openclaw \
  --benchmark-config configs/benchmarks/openclaw_sample.yaml \
  --harness openclaw \
  --harness-config configs/harnesses/openclaw.yaml \
  --model local_serverllm_qwen3_30b_moe \
  --max-instances 1 \
  --dry-run --no-eval
```
