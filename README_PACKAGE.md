# Claw-Eval Harness Smoke4 Code + Instances Package

This package contains the current harness integration code and the exact four non-multimodal smoke instances used in the latest comparison work.

Included:
- `src/`, `configs/`, `tests/`, selected `records/stage2` scripts/configs
- selected instance ids: `selected_instances/same4_fixed_ids.txt`
- local smoke dataset jsonl: `selected_instances/dataset/same4_instances.jsonl`
- four official task YAML files under `selected_instances/tasks/*/task.yaml`
- compact latest smoke evidence: per-instance manifests and tool-policy audits for Codex, NanoBot, and OpenClaw

Excluded:
- Python virtualenv and installed dependencies
- full `external/claw-eval` dataset/source tree
- full fixture data
- full run directories, trajectories, model logs, and large generated artifacts

Latest useful run ids represented in compact evidence:
- Codex: `same4_agent_judge_proxy_fix_retry_20260707_054133_codex_deepseek_v4_pro`
- NanoBot: `nano_openclaw_concurrent_20260707_082831_nanobot_deepseek_v4_pro`
- OpenClaw: `absbridge_openclaw4_20260707_095922_openclaw_deepseek_v4_pro`

Note: The latest OpenClaw code fixes provider auth/profile config and absolute `claw_tool` routing. The latest compact evidence shows OpenClaw bridge dispatches are now observed and tool policy is compliant, with remaining timeouts on some instances.
