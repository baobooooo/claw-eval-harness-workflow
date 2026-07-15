# Claw-Eval strict external harness mode

This project now implements scheme B as a live runtime instead of a post-hoc helper bridge.

The execution boundary is:

```text
Codex/Nanobot/OpenClaw agent loop
  -> driver workspace helpers (`./claw_tool`, `./claw_bash`, ...)
  -> LiveToolBridgeServer
  -> ClawLiveDispatcher
  -> Claw-Eval ToolDispatcher / SandboxToolDispatcher
  -> Claw-Eval Docker sandbox or task mock service
  -> LiveTraceWriter appends Claw-Eval JSONL events immediately
```

The external harness does **not** receive the scored task workspace as its cwd.  It receives a driver workspace that contains only bridge clients and policy files.  The real task state lives inside the Claw-Eval sandbox container under `/workspace`.  Direct native shell/file/browser tools in the driver workspace cannot modify the scored sandbox state.

For official runs, use:

```yaml
benchmark:
  live_tool_bridge: true
  require_official_claw_sandbox: true
  sandbox:
    image: claw-eval-agent:latest
    memory_limit: 4g
    cpu_limit: 2.0
    sandbox_port: 8080
```

`allow_host_sandbox_fallback: true` is only for unit tests and smoke tests on machines without Docker or the official `claw_eval` package.  It is labelled as `host_fallback` in `claw_live_runtime.json` and should not be used for final numbers.

## Temporal firewall

Pre-run:

- `sandbox_files` / fixtures are injected into the sandbox.
- `sandbox_grader_files`, `local_grader_files`, and env snapshot outputs are not visible to the external harness.

Post-run, while the sandbox is still alive:

- `sandbox_grader_files` are injected.
- `env_snapshot_commands` run inside the sandbox.
- `env_snapshot_files` are collected from the sandbox.
- `local_grader_files` are read from the host task directory only for the grader evidence file.

## Trace behavior

`claw_live_trace.jsonl` is live Claw-Eval schema, not a converted approximation.  `evaluate()` detects `trace_schema=claw_eval_live_v1` and passes it through instead of synthesizing a post-hoc trace.

Expected trace events:

```text
trace_start
message              # user prompt given to the external harness
tool_dispatch         # every bridge call, written immediately
message              # final response, if available
[audit_snapshot ...]  # mock service /audit payloads, after service collection
trace_end
```

## Tool policy

In live mode the policy gate is always enforced, even when `Bash` is an allowed tool.  `Bash` is allowed only as a Claw-Eval sandbox tool through the bridge; native harness shell commands happen in the driver workspace and do not count as official task actions.
