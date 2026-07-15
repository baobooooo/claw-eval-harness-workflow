# Formal run timeout and fairness fixes — 2026-07-08

This patch keeps the agent/harness comparison definition intact while fixing the two formal-run blockers found in the smoke4 review.

## 1. Timeout policy

Formal external-harness runs now use:

```text
effective_timeout_seconds = task.yaml environment.timeout_seconds × harness.timeout_multiplier
```

The default multiplier in `configs/harnesses/{codex,nanobot,openclaw}.yaml` is now `2.0`.

`timeout_s_per_instance` is now only a fallback for tasks that do not define an official YAML timeout. The smoke script no longer writes a uniform `timeout_seconds: 1200` into each dataset row, so tasks like `T067zh_synopsys_china_revenue` keep their larger YAML budget:

```text
T067: 1800 × 2 = 3600
T101:  600 × 2 = 1200
T139:  600 × 2 = 1200
T164:  600 × 2 = 1200
```

The smoke runner also computes an outer harness-run timeout from the selected task count, max workers, and max effective instance timeout, instead of hard-coding `7200` for the entire harness run.

## 2. Hidden transport history restoration

`src/harness_eval/claw_live/model_proxy.py` now restores Chat Completions history more strictly:

- handles harness-mutated tool call ids, e.g. `call_00_x` → `call00x`;
- rewrites assistant history calls from hidden transport tools back to the original YAML Claw-Eval tool name and arguments;
- rewrites role=`tool` result names like `exec` back to the original YAML tool name when present;
- strips hidden transport wrapper metadata from tool-result content, such as `endpoint_url`, `tool_name`, `tool_use_id`, and appended shell `Exit code` decoration;
- blocks non-YAML native tool calls in strict mode rather than letting `exec` / `process` leak into the agent path.

This does not change the first-round prompt/messages. It only affects post-model transport and later replayed history.

## 3. OpenClaw transport synchronization

OpenClaw hidden transport now passes:

```yaml
extra_transport_arguments:
  background: false
  yieldMs: 600000
  timeout: 600
```

This is intended to prevent OpenClaw from returning `Command still running (session ...)` to the model for the hidden `python3 claw_tool ...` transport command.

## 4. Fairness gate before judge

A new gate script was added:

```text
records/stage2/fairness_gate.py
```

The smoke runner calls it after conversion and tool-compliance audit, before judge. By default:

```text
REQUIRE_FAIRNESS_GATE=true
```

The gate fails if:

- `conversion_manifest.warning_count != 0`;
- tool policy audit has actual violations;
- model history still exposes hidden transport evidence;
- any `model_visible_tool_surface.matches_expected` event is false;
- a run is missing model proxy logs.

The runner writes:

```text
records/stage2/<LABEL>_fairness_gate.json
```

## 5. Tests added/updated

New/updated tests cover:

- official YAML timeout × multiplier;
- fallback timeout not being multiplied;
- Chat history restoration under mutated tool-call ids;
- sanitization of role=`tool` result envelopes;
- strict blocking of non-YAML native tool calls;
- model-proxy history audit for hidden transport leakage.
