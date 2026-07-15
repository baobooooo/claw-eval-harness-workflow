# Claw-Eval Harness Workflow

Public package for the Claw-Eval harness comparison work.

This repository contains the experiment code used to run Claw-Eval through three external harnesses:

- `codex`
- `nanobot`
- `openclaw`

The base model profile used by the experiment is `deepseek-v4-pro`.

## Included

- Source code under `src/`
- Harness and benchmark configuration under `configs/`
- Scripts, schemas, docs, tests, and helper tools
- External patch code under `external_patches/`
- Public-safe judge result JSON files under `judge_results/external_grade/`

## Excluded

The repository intentionally excludes benchmark data and runtime artifacts:

- Claw-Eval datasets and fixtures
- Agent workspaces
- Run directories
- Raw traces and logs
- Virtual environments and caches
- Model weights

Full redacted workflow traces are published as a GitHub release asset:

https://github.com/baobooooo/claw-eval-harness-workflow/releases/tag/workflow-traces-20260715

## Judge Results

`judge_results/external_grade/` contains `.grade.json` files of the same form as:

`0000_codex_deepseek-v4-pro_T009zh_contact_lookup.grade.json`

These are per-instance judge outputs only. Larger judge JSONL files, grader workspaces, task data, and fixtures are not included.

