from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from harness_eval.types import BenchmarkTask, HarnessResult


DIRECT_NETWORK_RE = re.compile(
    r"\b(curl|wget|lynx|links|w3m)\b|"
    r"\b(requests|httpx)\s*\.|"
    r"\burllib\.request\b|"
    r"\bpython3?\s+-c\b.*\b(http|urllib|requests|httpx)\b",
    re.IGNORECASE,
)
POLICY_INSTRUCTION_RE = re.compile(
    r"do not use direct public-network access|"
    r"only allowed web access path|"
    r"raw curl/wget/browser/native web tools",
    re.IGNORECASE,
)

OPENCLAW_NATIVE_TOOL_RE = re.compile(r"\[tools\]\s+([A-Za-z0-9_-]+)\s+(failed|succeeded):")
ALLOWED_DRIVER_METADATA_FILES = {
    "claw_eval_tool_policy.json",
    "claw_live_tools.json",
    "README_CLAW_LIVE_BRIDGE.md",
    "CODEX_CLAW_TOOL_USAGE.md",
    "AGENTS.md",
}
STARTUP_FAILURE_STATUSES = {"harness_start_failed", "harness_config_error"}
NON_TOOL_FAILURE_STATUSES = STARTUP_FAILURE_STATUSES | {"provider_quota_error", "model_call_failed"}


def read_text(path: Path | None, *, limit: int | None = None) -> str:
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _bash_network_allowed(policy: dict[str, Any], allowed_tools: set[str]) -> bool:
    sandbox_tools = {str(name) for name in policy.get("sandbox_tools") or []}
    return "Bash" in allowed_tools or "Bash" in sandbox_tools


def first_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                obj = json.loads(line)
            except Exception:
                return {}
            return obj if isinstance(obj, dict) else {}
    return {}


def load_prediction_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "harness_predictions.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _policy_path_for(row: dict[str, Any]) -> Path | None:
    raw = row.get("tool_policy_path")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
    workspace = row.get("workspace")
    if workspace:
        path = Path(str(workspace)) / "claw_eval_tool_policy.json"
        if path.exists():
            return path
        parent_path = Path(str(workspace)).parent / "claw_eval_tool_policy.json"
        if parent_path.exists():
            return parent_path
    return None


def _endpoint_path_to_tool(policy: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for endpoint in policy.get("exposed_tool_endpoints") or []:
        if not isinstance(endpoint, dict):
            continue
        tool_name = endpoint.get("tool_name")
        url = endpoint.get("url")
        if not tool_name or not url:
            continue
        path = urlparse(str(url)).path
        if path:
            out[path] = str(tool_name)
    return out


def service_audit_tools(service_audit_path: Path | None, policy: dict[str, Any]) -> list[dict[str, Any]]:
    audit = read_json(service_audit_path)
    if not isinstance(audit, dict):
        return []
    path_to_tool = _endpoint_path_to_tool(policy)
    observed: list[dict[str, Any]] = []
    for service_name, service_data in audit.items():
        body = service_data.get("body") if isinstance(service_data, dict) else None
        calls = body.get("calls") if isinstance(body, dict) else None
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            endpoint = str(call.get("endpoint") or "")
            tool_name = path_to_tool.get(endpoint, endpoint.rsplit("/", 1)[-1] if endpoint else "")
            observed.append(
                {
                    "source": "service_audit",
                    "service": service_name,
                    "endpoint": endpoint,
                    "tool_name": tool_name,
                    "request_body": call.get("request_body") or {},
                }
            )
    return observed


def trace_dispatch_tools(trace_path: Path | None) -> list[dict[str, Any]]:
    if trace_path is None or not trace_path.exists():
        return []
    observed: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "tool_dispatch":
            continue
        observed.append(
            {
                "source": "live_trace",
                "tool_name": str(obj.get("tool_name") or ""),
                "endpoint": str(obj.get("endpoint_url") or ""),
                "request_body": obj.get("request_body") or {},
                "response_status": obj.get("response_status"),
            }
        )
    return observed


def _dedupe_observed_tools(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep trace and audit evidence without double-counting the same service call.

    Live traces already include task-service dispatches.  Service audit is kept as
    secondary evidence, but when the same endpoint/body appears in both sources
    it should not inflate observed_tool_counts.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        tool = str(item.get("tool_name") or "")
        endpoint = str(item.get("endpoint") or "")
        try:
            body = json.dumps(item.get("request_body") or {}, ensure_ascii=False, sort_keys=True)
        except Exception:
            body = str(item.get("request_body") or {})
        key = (tool, endpoint, body)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def live_trace_actual_command_lines(trace_path: Path | None) -> list[str]:
    lines: list[str] = []
    for item in trace_dispatch_tools(trace_path):
        if item.get("tool_name") != "Bash":
            continue
        body = item.get("request_body") if isinstance(item.get("request_body"), dict) else {}
        command = body.get("command") or body.get("cmd") or body.get("script")
        if command:
            lines.append(str(command))
    return lines


HIDDEN_TRANSPORT_NEEDLES = ("claw_tool", ".claw_tool_payloads", "exec_command", "Command still running (session")


def _contains_hidden_transport(value: Any) -> bool:
    try:
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    except Exception:
        text = str(value)
    return any(needle in text for needle in HIDDEN_TRANSPORT_NEEDLES)


def _model_proxy_log_path(row: dict[str, Any]) -> Path | None:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    proxy = metrics.get("model_tool_proxy") if isinstance(metrics.get("model_tool_proxy"), dict) else {}
    raw = proxy.get("log_path")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
    workspace = Path(str(row["workspace"])) if row.get("workspace") else None
    if workspace is not None:
        harness = str(row.get("harness") or "")
        candidate = workspace.parent / f"{harness}_model_tool_proxy.jsonl"
        if candidate.exists():
            return candidate
    return None


def _chat_tool_name(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else None
    return str(fn.get("name") or "") if fn else ""


def model_proxy_history_markers(row: dict[str, Any], allowed_tools: set[str]) -> list[dict[str, str]]:
    """Hard-gate hidden transport leakage in upstream model history.

    System/developer prompts remain harness-owned, so this scans only structured
    tool-call history in the sanitized upstream requests: assistant tool_calls,
    role=tool messages, and Responses API function_call items.
    """
    path = _model_proxy_log_path(row)
    if path is None:
        return []
    markers: list[dict[str, str]] = []
    request_index = -1
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if not isinstance(ev, dict) or ev.get("event") != "upstream_request":
            continue
        request_index += 1
        body = ev.get("body") if isinstance(ev.get("body"), dict) else {}
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        for message_index, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list):
                for call_index, call in enumerate(msg["tool_calls"]):
                    if not isinstance(call, dict):
                        continue
                    name = _chat_tool_name(call)
                    if name and name not in allowed_tools:
                        markers.append({"source": "model_proxy_upstream", "reason": "assistant_tool_call_not_in_yaml_surface", "request_index": str(request_index), "message_index": str(message_index), "call_index": str(call_index), "tool_name": name})
                    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                    if _contains_hidden_transport(fn.get("arguments")):
                        markers.append({"source": "model_proxy_upstream", "reason": "assistant_tool_call_arguments_expose_hidden_transport", "request_index": str(request_index), "message_index": str(message_index), "call_index": str(call_index), "tool_name": name})
            if msg.get("role") == "tool":
                name = str(msg.get("name") or "")
                if name and name not in allowed_tools:
                    markers.append({"source": "model_proxy_upstream", "reason": "tool_result_name_not_in_yaml_surface", "request_index": str(request_index), "message_index": str(message_index), "tool_name": name})
                if _contains_hidden_transport(msg.get("content")):
                    markers.append({"source": "model_proxy_upstream", "reason": "tool_result_content_exposes_hidden_transport", "request_index": str(request_index), "message_index": str(message_index), "tool_name": name})
        input_items = body.get("input") if isinstance(body.get("input"), list) else []
        for item_index, item in enumerate(input_items):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                name = str(item.get("name") or "")
                if name and name not in allowed_tools:
                    markers.append({"source": "model_proxy_upstream", "reason": "responses_function_call_not_in_yaml_surface", "request_index": str(request_index), "item_index": str(item_index), "tool_name": name})
                if _contains_hidden_transport(item.get("arguments")):
                    markers.append({"source": "model_proxy_upstream", "reason": "responses_function_call_arguments_expose_hidden_transport", "request_index": str(request_index), "item_index": str(item_index), "tool_name": name})
    return markers


def codex_commands(path: Path | None) -> list[str]:
    commands: list[str] = []
    if path is None or not path.exists():
        return commands
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        item = obj.get("item") if isinstance(obj, dict) else None
        if isinstance(item, dict) and item.get("type") == "command_execution" and item.get("command"):
            commands.append(str(item["command"]))
    return commands


def _line_uses_allowed_helper(line: str, helper_files: set[str]) -> bool:
    return any(re.search(rf"(^|[/\s'\".]){re.escape(helper)}(\s|$|['\"])", line) for helper in helper_files)


def direct_network_markers(lines: list[str], helper_files: list[str], *, source: str = "actual_execution") -> list[dict[str, str]]:
    helpers = set(helper_files)
    markers: list[dict[str, str]] = []
    for line in lines:
        if POLICY_INSTRUCTION_RE.search(line):
            continue
        if _line_uses_allowed_helper(line, helpers):
            continue
        match = DIRECT_NETWORK_RE.search(line)
        if match:
            markers.append({"pattern": match.group(0), "line": line[:500], "source": source})
    return markers


def _is_allowed_driver_metadata_read(tool_name: str, line: str) -> bool:
    if tool_name.lower() not in {"read", "open", "cat"}:
        return False
    return any(name in line for name in ALLOWED_DRIVER_METADATA_FILES)


def openclaw_native_tool_markers(stderr_path: Path | None, allowed_tools: set[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    markers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for line in read_text(stderr_path).splitlines():
        match = OPENCLAW_NATIVE_TOOL_RE.search(line)
        if not match:
            continue
        tool_name = match.group(1)
        if _is_allowed_driver_metadata_read(tool_name, line):
            warnings.append({"type": "metadata_read_warning", "tool_name": tool_name, "line": line[:500], "reason": "driver_metadata_read_only"})
            continue
        if tool_name in allowed_tools:
            markers.append({"tool_name": tool_name, "line": line[:500], "reason": "native_tool_not_claw_eval_helper"})
        else:
            markers.append({"tool_name": tool_name, "line": line[:500], "reason": "tool_not_allowed"})
    return markers, warnings


def helper_modification_markers(workspace: Path | None, helper_files: list[str]) -> list[dict[str, str]]:
    if workspace is None:
        return []
    patch_path = workspace.parent / "patch.diff"
    text = read_text(patch_path)
    if not text:
        return []
    markers: list[dict[str, str]] = []
    for helper in helper_files:
        pattern = re.compile(rf"(^|\n)(diff --git a/{re.escape(helper)}\b|--- a/{re.escape(helper)}\b|\+\+\+ b/{re.escape(helper)}\b)")
        if pattern.search(text):
            markers.append({"helper": helper, "patch_path": str(patch_path)})
    return markers


def _requires_bridge_tool_calls(row: dict[str, Any], policy: dict[str, Any]) -> bool:
    if "requires_bridge_tool_calls" in row:
        return bool(row.get("requires_bridge_tool_calls"))
    if "requires_bridge_tool_calls" in policy:
        return bool(policy.get("requires_bridge_tool_calls"))
    live = bool(row.get("live_tool_bridge")) or row.get("trace_schema") == "claw_eval_live_v1"
    if not live:
        return False
    return bool(
        policy.get("task_tools")
        or policy.get("exposed_tool_endpoints")
        or policy.get("sandbox_tools")
        or policy.get("sandbox_files")
        or policy.get("env_snapshot_files")
        or policy.get("env_snapshot_commands")
        or policy.get("sandbox_grader_files")
    )


def _direct_network_markers_for_actual_execution(
    harness: str,
    trace_path: Path | None,
    stdout_path: Path | None,
    helper_files: list[str],
    *,
    bash_network_allowed: bool,
) -> tuple[list[dict[str, str]], int]:
    official_bash_lines = live_trace_actual_command_lines(trace_path)
    native_lines: list[str] = []
    if harness == "codex":
        native_lines.extend(codex_commands(trace_path))
        native_lines.extend(codex_commands(stdout_path))

    markers: list[dict[str, str]] = []
    allowed_bash_network_count = 0
    official_bash_markers = direct_network_markers(official_bash_lines, helper_files, source="live_bash")
    if bash_network_allowed:
        allowed_bash_network_count = len(official_bash_markers)
    else:
        markers.extend(official_bash_markers)
    markers.extend(direct_network_markers(native_lines, helper_files, source="native_execution"))
    return markers, allowed_bash_network_count


def _text_suspicion_lines(harness: str, trace_path: Path | None, stdout_path: Path | None, stderr_path: Path | None) -> list[str]:
    # Only free-form text is scanned here.  Hits are warnings, not hard gates.
    lines: list[str] = []
    if harness != "codex":
        for path in (trace_path, stdout_path, stderr_path):
            text = read_text(path, limit=250_000)
            for line in text.splitlines():
                # Skip structured live tool dispatch JSON; actual Bash commands are
                # handled by _direct_network_markers_for_actual_execution.
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("type") == "tool_dispatch":
                        continue
                except Exception:
                    pass
                lines.append(line)
    return lines


def _harness_failure_markers(row: dict[str, Any]) -> list[dict[str, str]]:
    status = str(row.get("status") or "")
    if status not in STARTUP_FAILURE_STATUSES:
        return []
    marker = {"type": "harness_start_failed", "status": status}
    error = row.get("error")
    if error:
        marker["error"] = str(error)[:500]
    return [marker]


def audit_prediction(row: dict[str, Any]) -> dict[str, Any]:
    policy_path = _policy_path_for(row)
    policy = read_json(policy_path)
    if not isinstance(policy, dict):
        policy = {}
    allowed_tools = set(str(x) for x in policy.get("allowed_tools") or [])
    helper_files = [str(x) for x in policy.get("helper_files") or []]
    workspace = Path(str(row["workspace"])) if row.get("workspace") else None
    service_audit_path = Path(str(row["service_audit_path"])) if row.get("service_audit_path") else None
    if service_audit_path is None and workspace is not None:
        candidate = workspace.parent / "service_audit.json"
        service_audit_path = candidate if candidate.exists() else None
    trace_path = Path(str(row["trace_path"])) if row.get("trace_path") else None
    stdout_path = Path(str(row["stdout_path"])) if row.get("stdout_path") else None
    stderr_path = Path(str(row["stderr_path"])) if row.get("stderr_path") else None
    harness = str(row.get("harness") or "")
    bash_network_allowed = _bash_network_allowed(policy, allowed_tools)

    observed_tools = _dedupe_observed_tools(trace_dispatch_tools(trace_path) + service_audit_tools(service_audit_path, policy))
    unauthorized_tools = [
        item for item in observed_tools if item.get("tool_name") and str(item.get("tool_name")) not in allowed_tools
    ]

    direct_markers, allowed_bash_network_count = _direct_network_markers_for_actual_execution(
        harness,
        trace_path,
        stdout_path,
        helper_files,
        bash_network_allowed=bash_network_allowed,
    )
    text_markers = direct_network_markers(_text_suspicion_lines(harness, trace_path, stdout_path, stderr_path), helper_files, source="free_text_mention")

    warnings: list[dict[str, Any]] = []
    for item in text_markers:
        warnings.append({"type": "text_direct_network_mention", **item})
    if harness == "openclaw":
        native_tool_markers, native_tool_warnings = openclaw_native_tool_markers(stderr_path, allowed_tools)
        warnings.extend(native_tool_warnings)
    else:
        native_tool_markers = []
    helper_markers = helper_modification_markers(workspace, helper_files)
    harness_failure_markers = _harness_failure_markers(row)
    model_history_markers = model_proxy_history_markers(row, allowed_tools)

    observed_counts: dict[str, int] = {}
    for item in observed_tools:
        tool_name = str(item.get("tool_name") or "")
        if tool_name:
            observed_counts[tool_name] = observed_counts.get(tool_name, 0) + 1

    requires_bridge = _requires_bridge_tool_calls(row, policy)
    live = bool(row.get("live_tool_bridge")) or row.get("trace_schema") == "claw_eval_live_v1"
    tool_usage_failures: list[dict[str, Any]] = []
    if live and requires_bridge and not observed_counts and str(row.get("status") or "") not in NON_TOOL_FAILURE_STATUSES:
        tool_usage_failures.append(
            {
                "type": "no_bridge_tool_calls",
                "reason": "live_tool_bridge run produced zero Claw-Eval tool dispatches",
            }
        )

    violations: list[dict[str, Any]] = []
    for item in unauthorized_tools:
        violations.append({"type": "unauthorized_service_tool", **item})
    for item in direct_markers:
        violations.append({"type": "direct_network_access", **item})
    for item in native_tool_markers:
        violations.append({"type": "native_tool_attempt", **item})
    for item in helper_markers:
        violations.append({"type": "helper_file_modified", **item})
    for item in model_history_markers:
        violations.append({"type": "model_history_hidden_transport", **item})
    for item in tool_usage_failures:
        violations.append(item)
    for item in harness_failure_markers:
        violations.append(item)

    real_violations = [item for item in violations if item.get("type") not in {"harness_start_failed"}]
    return {
        "task_id": row.get("task_id"),
        "harness": harness,
        "status": row.get("status"),
        "policy_path": str(policy_path) if policy_path else None,
        "policy_sha256": policy.get("policy_sha256"),
        "allowed_tools": sorted(allowed_tools),
        "helper_files": helper_files,
        "observed_tool_counts": observed_counts,
        "bash_network_allowed": bash_network_allowed,
        "allowed_bash_network_count": allowed_bash_network_count,
        "requires_bridge_tool_calls": requires_bridge,
        "service_audit_path": str(service_audit_path) if service_audit_path else None,
        "violation_count": len(violations),
        "actual_violation_count": len(real_violations),
        "text_suspicion_count": len(text_markers),
        "harness_failure_count": len(harness_failure_markers),
        "tool_usage_failure_count": len(tool_usage_failures),
        "model_history_hidden_transport_count": len(model_history_markers),
        "violations": violations,
        "warning_count": len(warnings),
        "warnings": warnings,
        "compliant": len(violations) == 0,
    }


def audit_run_dir(run_dir: Path) -> dict[str, Any]:
    rows = load_prediction_rows(run_dir)
    audits = [audit_prediction(row) for row in rows]
    return {
        "run_dir": str(run_dir),
        "prediction_count": len(rows),
        "compliant": all(item["compliant"] for item in audits),
        "violation_count": sum(int(item["violation_count"]) for item in audits),
        "actual_violation_count": sum(int(item.get("actual_violation_count", 0)) for item in audits),
        "text_suspicion_count": sum(int(item.get("text_suspicion_count", 0)) for item in audits),
        "harness_failure_count": sum(int(item.get("harness_failure_count", 0)) for item in audits),
        "tool_usage_failure_count": sum(int(item.get("tool_usage_failure_count", 0)) for item in audits),
        "model_history_hidden_transport_count": sum(int(item.get("model_history_hidden_transport_count", 0)) for item in audits),
        "warning_count": sum(int(item["warning_count"]) for item in audits),
        "instances": audits,
    }


def _row_from_result(result: HarnessResult, task: BenchmarkTask) -> dict[str, Any]:
    service_audit_path = task.output_dir / "service_audit.json"
    tool_policy_path = task.metadata.get("tool_policy_path") or task.output_dir / "claw_eval_tool_policy.json"
    return {
        "task_id": task.task_id,
        "harness": result.harness,
        "status": result.status,
        "error": result.error,
        "workspace": str(task.workspace),
        "trace_path": result.trace_path,
        "stdout_path": result.stdout_path,
        "stderr_path": result.stderr_path,
        "service_audit_path": str(service_audit_path) if service_audit_path.exists() else None,
        "tool_policy_path": str(tool_policy_path),
        "live_tool_bridge": bool(task.metadata.get("live_tool_bridge")),
        "trace_schema": "claw_eval_live_v1" if task.metadata.get("live_tool_bridge") else "external_harness_raw",
        "requires_bridge_tool_calls": bool(task.metadata.get("requires_bridge_tool_calls", False)),
    }


def _status_for_policy_violation(result: HarnessResult, audit: dict[str, Any]) -> str:
    types = {str(item.get("type")) for item in audit.get("violations") or []}
    if result.status in STARTUP_FAILURE_STATUSES or "harness_start_failed" in types:
        return "harness_start_failed" if result.status != "harness_config_error" else result.status
    if "no_bridge_tool_calls" in types:
        return "no_bridge_tool_calls"
    if result.status == "timeout":
        return "timeout"
    return "tool_policy_violation"


def enforce_tool_policy(result: HarnessResult, task: BenchmarkTask) -> HarnessResult:
    """Apply the post-run hard gate for Claw-Eval allowed tools.

    In strict live-bridge mode, the gate is based on real bridge dispatches and
    actual executed command lines.  Free-form reasoning text is only reported as
    suspicion, so a model mentioning ``curl`` is not treated the same as actually
    executing it.
    """
    allowed_tools = {str(name) for name in task.metadata.get("allowed_tools") or []}
    audit = audit_prediction(_row_from_result(result, task))
    live_bridge = bool(task.metadata.get("live_tool_bridge"))
    if "Bash" in allowed_tools and not live_bridge:
        audit["enforced"] = False
        audit["skip_reason"] = "bash_allowed"
    else:
        audit["enforced"] = True
        audit["live_tool_bridge"] = live_bridge
        if live_bridge and not result.trace_path:
            audit.setdefault("violations", []).append({"type": "missing_live_trace"})
            audit["violation_count"] = int(audit.get("violation_count", 0)) + 1
        if audit.get("violation_count", 0):
            result.status = _status_for_policy_violation(result, audit)
            if result.status == "no_bridge_tool_calls":
                result.error = "no Claw-Eval live-bridge tool calls were observed"
            elif result.status in STARTUP_FAILURE_STATUSES:
                result.error = result.error or "harness start failed"
            else:
                result.error = "tool policy violation"
    result.metrics = {**result.metrics, "tool_policy": audit}
    (task.output_dir / "tool_policy_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
