# Claw-Eval Harness Integration Code Package

Created: 2026-07-03

This package is intended for code review and analysis. It includes the project code and configuration for integrating the Codex, NanoBot, and OpenClaw harnesses with Claw-Eval/OpenClaw-style tool policies.

## Included

- Project source code under `src/`
- Harness configs under `configs/`
- Stage scripts under `records/stage1/` and `records/stage2/`
- Tests, schemas, helper scripts, docs, and external patch files
- Claw-Eval source code only: `external/claw-eval/src`, `mock_services`, and related top-level config/code files
- Smoke selection metadata for:
  - `T161zh_automation_failure_recovery`
  - `T033zh_ops_review_dashboard`
  - `T044_service_outage_research`

## Excluded

- Runtime environments such as `.venv`
- Full Claw-Eval task dataset: `external/claw-eval/tasks`
- Claw-Eval downloaded data and fixtures: `external/claw-eval/data`, `data/claw_eval_fixtures`
- Run outputs and result traces: `runs/`, `records/stage2/smoke_score_ready/`, smoke logs, audits, and predictions
- Cache/build artifacts such as `__pycache__`, `.pytest_cache`, `.cache`, and egg-info metadata

The smoke run was intentionally stopped before completion. Per the packaging requirement, detailed smoke result artifacts are not included.
