# ClawEval fair harness comparison notes (2026-07-06)

## Goal

Compare MiniHarness, Codex, OpenClaw, and NanoBot on ClawEval while fixing:

- model: DeepSeek V4 Pro;
- task-action tools: the exact MiniHarness-visible ClawEval task YAML tools, plus the same official ClawEval sandbox tools when the MiniHarness run exposes sandbox tools.

The selected harness should still own planning, memory/context management, compression, long-context behavior, retry policy, and agent-loop policy.

## First model request invariant

For Codex/OpenClaw/NanoBot formal runs, the first request sent upstream to DeepSeek should satisfy this invariant:

```text
messages/input: harness-native, not normalized to MiniHarness
API tools:     MiniHarness/ClawEval tool surface, same names, descriptions, schemas, and order
routing:       hidden after the model response, not described in messages/input
```

Concretely, the proxy only rewrites the OpenAI-compatible `tools` array before the request reaches DeepSeek. It preserves the harness-generated `messages`/`input` on the first request. On later turns, it only rewrites previous hidden transport calls back into the original ClawEval tool calls so the model sees a coherent ClawEval trajectory.

The model-visible tool order is preserved as MiniHarness order: YAML task tools first, then official sandbox tools in the official order, with first occurrence winning on duplicates. The code no longer sorts tools alphabetically and no longer injects fallback tool descriptions.

## Main design

The formal path is:

```text
Codex/OpenClaw/NanoBot CLI
        │ normal OpenAI-compatible request
        ▼
local ClawToolModelProxy
        │ only API tools are replaced with ClawEval tools
        ▼
DeepSeek V4 Pro
        │ ClawEval tool call, e.g. web_fetch({...})
        ▼
local ClawToolModelProxy
        │ hidden rewrite to harness transport call,
        │ e.g. exec_command({cmd: "python3 ./claw_tool web_fetch @payload.json"}) for Codex
        │      or exec({command: "python3 ./claw_tool web_fetch @payload.json"}) for OpenClaw/NanoBot
        ▼
Codex/OpenClaw/NanoBot native executor
        │ calls ./claw_tool in driver workspace
        ▼
ClawEval live bridge / MiniHarness dispatcher / official sandbox
```

Thus the model sees the same task tools as MiniHarness, while the selected harness still performs its own loop and context handling.

## Important configuration flags

Formal experiments should use:

```yaml
harness:
  native_claw_tools: false
  model_tool_proxy:
    enabled: true
```

`native_claw_tools: true` remains only as a direct-model-loop escape hatch for debugging bridge/tool behavior. It is no longer auto-enabled just because `benchmark.live_tool_bridge` is true.

## Files touched

- `src/harness_eval/claw_live/model_proxy.py`: local OpenAI-compatible tool proxy; preserves message/input content, injects exact ordered ClawEval tool surface, logs `model_visible_tool_surface` for audit.
- `src/harness_eval/claw_live/dispatcher.py`: preserves MiniHarness tool order instead of sorting alphabetically.
- `src/harness_eval/claw_live/runtime.py` and `bridge.py`: preserve ordered tool names in runtime metadata.
- `src/harness_eval/benchmarks/openclaw.py`: preserves ordered allowed tool names in policy metadata.
- `src/harness_eval/harnesses/codex.py`: starts Codex CLI with proxied model endpoint when configured; hidden transport defaults to Codex `exec_command`, matching the current Codex Responses API native tool.
- `src/harness_eval/harnesses/external_cli.py`: same for OpenClaw/NanoBot adapters; exports provider/base-url aliases so CLIs hit the per-task proxy instead of the shared bridge directly.
- `src/harness_eval/harnesses/native_claw.py`: direct loop is now explicit-only; adds a MiniHarness prompt fallback for environments without installed `claw_eval` package.
- `configs/harnesses/{codex,nanobot,openclaw}.yaml`: enables formal proxy mode and leaves native harness loops active; OpenClaw is forced through OpenAI-compatible provider selection and NanoBot receives a rendered per-task proxy config.
- `tests/test_claw_model_tool_proxy.py`: verifies exact ordered tool injection, first-request message preservation, hidden transport rewriting, and history replay.

## Validation

The package test suite passes:

```text
63 passed
```
