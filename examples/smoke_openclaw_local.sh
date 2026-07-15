#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python -m harness_eval.run \
  --benchmark openclaw \
  --benchmark-config configs/benchmarks/openclaw_claw_eval.yaml \
  --harness codex \
  --harness-config configs/harnesses/codex.yaml \
  --model local_serverllm_qwen3_30b_moe \
  --max-instances 1 \
  --dry-run --no-eval
