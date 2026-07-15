from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from harness_eval.types import BenchmarkTask, ModelProfile


ProviderFactory = Callable[[ModelProfile], Any]
ToolDispatcher = Callable[[str, str, dict[str, Any]], dict[str, Any]]


def _make_native_provider(model: ModelProfile) -> Any:
    from claw_eval.runner.providers.openai_compat import OpenAICompatProvider  # type: ignore

    return OpenAICompatProvider(
        model_id=model.model,
        api_key=model.api_key_value,
        base_url=model.base_url,
        extra_body=model.extra_body,
    )


def native_claw_tools_enabled(task: BenchmarkTask, hcfg: dict[str, Any]) -> bool:
    # This direct provider loop is useful as an escape hatch and for focused
    # bridge tests, but it must not be enabled merely because Claw-Eval live
    # tools are available.  Auto-enabling it bypasses Codex/OpenClaw/NanoBot
    # themselves and therefore removes the harness variable from the experiment.
    return bool(hcfg.get("native_claw_tools") or hcfg.get("direct_model_loop"))


def _tool_specs_from_raw(raw_specs: list[Any]) -> list[Any]:
    from claw_eval.models.tool import ToolSpec  # type: ignore

    specs: list[Any] = []
    for raw in raw_specs:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        specs.append(ToolSpec.model_validate(raw))
    return specs


def native_tool_specs(task: BenchmarkTask) -> list[Any]:
    return _tool_specs_from_raw(task.metadata.get("allowed_tool_specs") or [])


def native_task_tool_specs(task: BenchmarkTask) -> list[Any]:
    raw_task_specs = task.metadata.get("task_tool_specs")
    if isinstance(raw_task_specs, list):
        return _tool_specs_from_raw(raw_task_specs)
    sandbox_names = {str(name) for name in task.metadata.get("sandbox_tools") or []}
    raw_allowed = [
        spec
        for spec in task.metadata.get("allowed_tool_specs") or []
        if isinstance(spec, dict) and str(spec.get("name")) not in sandbox_names
    ]
    return _tool_specs_from_raw(raw_allowed)


def native_sandbox_tool_specs(task: BenchmarkTask) -> list[Any]:
    sandbox_names = {str(name) for name in task.metadata.get("sandbox_tools") or []}
    if not sandbox_names:
        return []
    raw_specs = [
        spec
        for spec in task.metadata.get("allowed_tool_specs") or []
        if isinstance(spec, dict) and str(spec.get("name")) in sandbox_names
    ]
    return _tool_specs_from_raw(raw_specs)


def model_visible_tool_names(task: BenchmarkTask) -> list[str]:
    return [
        str(spec.get("name"))
        for spec in task.metadata.get("allowed_tool_specs") or []
        if isinstance(spec, dict) and spec.get("name")
    ]


def dispatch_native_tool(bridge_url: str, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    url = f"{bridge_url.rstrip('/')}/tool/{tool_name}"
    with httpx.Client(trust_env=False, timeout=180.0) as client:
        resp = client.post(url, json=tool_input)
    try:
        body = resp.json()
    except Exception:
        body = {"body": resp.text, "status": resp.status_code, "is_error": resp.status_code >= 400}
    if not isinstance(body, dict):
        body = {"body": body, "status": resp.status_code, "is_error": resp.status_code >= 400}
    body.setdefault("status", resp.status_code)
    body.setdefault("is_error", resp.status_code >= 400)
    return body


def _native_tool_result_text(payload: dict[str, Any]) -> str:
    body = payload.get("body", payload)
    if isinstance(body, str):
        return body
    return json.dumps(body, ensure_ascii=False, sort_keys=True)


def _text_messages(system_prompt: str, user_prompt: str) -> list[Any]:
    from claw_eval.models.content import TextBlock  # type: ignore
    from claw_eval.models.message import Message  # type: ignore

    return [
        Message(role="system", content=[TextBlock(text=system_prompt)]),
        Message(role="user", content=[TextBlock(text=user_prompt)]),
    ]


class _FallbackTextMessage:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text
        self.content = text


def _fallback_text_messages(system_prompt: str, user_prompt: str) -> list[Any]:
    return [_FallbackTextMessage("system", system_prompt), _FallbackTextMessage("user", user_prompt)]


def _fallback_miniharness_system_prompt(task: BenchmarkTask) -> str:
    rows: list[str] = []
    for raw in task.metadata.get("allowed_tool_specs") or []:
        if isinstance(raw, dict) and raw.get("name"):
            description = str(raw.get("description") or "Available for this task.").strip()
            rows.append(f"- {raw['name']}: {description}")
    tools_text = "\n".join(rows) if rows else "(none)"
    environment = dict(task.metadata.get("environment") or {})
    if task.metadata.get("timeout_seconds") is not None:
        environment.setdefault("timeout_seconds", task.metadata["timeout_seconds"])
    if task.metadata.get("max_turns") is not None:
        environment.setdefault("max_turns", task.metadata["max_turns"])
    env_text = ""
    if environment:
        env_text = "\n\nEnvironment:\n" + "\n".join(f"- {k}={v}" for k, v in sorted(environment.items()))
    return f"You are MiniHarness, the Claw-Eval reference agent harness.\n\nAvailable tools:\n{tools_text}{env_text}"


def miniharness_initial_messages(task: BenchmarkTask) -> list[Any]:
    try:
        from claw_eval.config import PromptConfig  # type: ignore
        from claw_eval.models.task import TaskDefinition  # type: ignore
        from claw_eval.runner.system_prompt import build_system_prompt  # type: ignore
    except Exception:
        return _fallback_text_messages(_fallback_miniharness_system_prompt(task), task.prompt)

    environment = dict(task.metadata.get("environment") or {})
    if task.metadata.get("timeout_seconds") is not None:
        environment.setdefault("timeout_seconds", int(task.metadata["timeout_seconds"]))
    if task.metadata.get("max_turns") is not None:
        environment.setdefault("max_turns", int(task.metadata["max_turns"]))
    task_def = TaskDefinition.model_validate(
        {
            "task_id": task.task_id,
            "task_name": task.row.get("task_name") or task.task_id,
            "category": task.metadata.get("category") or task.row.get("category") or "",
            "prompt": {
                "text": task.prompt,
                "language": task.metadata.get("language") or task.row.get("language") or "en",
            },
            "tools": [
                spec.model_dump(mode="json") if hasattr(spec, "model_dump") else spec
                for spec in native_task_tool_specs(task)
            ],
            "environment": environment,
        }
    )
    if task.metadata.get("task_yaml_path"):
        task_def.task_file = str(task.metadata["task_yaml_path"])
    system_prompt = build_system_prompt(
        task_def,
        PromptConfig(),
        extra_tools=native_sandbox_tool_specs(task),
    )
    return _text_messages(system_prompt, task.prompt)


def _workspace_for_prompt(task: BenchmarkTask) -> str:
    return str(task.metadata.get("claw_workspace_path") or "/workspace")


def _agents_md_sections(task: BenchmarkTask) -> str:
    root = Path(str(task.workspace))
    if not root.exists():
        return ""
    try:
        path = root.resolve() / "AGENTS.md"
    except OSError:
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not content:
        return ""
    return f'## AGENTS.md instructions\n\n<AGENTS.md path="{path}">\n{content}\n</AGENTS.md>'


def codex_system_prompt(task: BenchmarkTask) -> str:
    prompt = """You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer.

## General

- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)

## Editing constraints

- Default to ASCII when editing or creating files. Only introduce non-ASCII or other Unicode characters when there is a clear justification and the file already uses them.
- Add succinct code comments that explain what is going on if code is not self-explanatory. You should not add comments like "Assigns the value to the variable", but a brief comment might be useful ahead of a complex code block that the user would otherwise have to spend time parsing out. Usage of these comments should be rare.
- Try to use apply_patch for single file edits, but it is fine to explore other options to make the edit if it does not work well. Do not use apply_patch for changes that are auto-generated or when scripting is more efficient.
- You may be in a dirty git worktree.
    * NEVER revert existing changes you did not make unless explicitly requested, since these changes were made by the user.
    * If asked to make a commit or code edits and there are unrelated changes to your work or changes that you didn't make in those files, don't revert those changes.
    * If the changes are in files you've touched recently, read carefully and work with the changes rather than reverting them.
    * If the changes are in unrelated files, ignore them and don't revert them.
- Do not amend a commit unless explicitly requested to do so.
- While you are working, you might notice unexpected changes that you didn't make. If this happens, STOP IMMEDIATELY and ask the user how they would like to proceed.
- **NEVER** use destructive commands like `git reset --hard` or `git checkout --` unless specifically requested or approved by the user.

## Plan tool

When using the planning tool:
- Skip using the planning tool for straightforward tasks.
- Do not make single-step plans.
- When you made a plan, update it after having performed one of the sub-tasks that you shared on the plan.

## Special user requests

- If the user makes a simple request which you can fulfill by running a terminal command, you should do so.
- If the user asks for a "review", default to a code review mindset: prioritise identifying bugs, risks, behavioural regressions, and missing tests.

## Presenting your work and final message

You are producing plain text that will later be styled by the CLI. Follow these rules exactly. Formatting should make results easy to scan, but not feel mechanical.

- Default: be very concise; friendly coding teammate tone.
- Ask only when needed; suggest ideas; mirror the user's style.
- For substantial work, summarize clearly; follow final-answer formatting.
- Skip heavy formatting for simple confirmations.
- Don't dump large files you've written; reference paths only.
- No "save/copy this file" - User is on the same machine.
- Offer logical next steps briefly; add verify steps if you couldn't do something.
"""
    agents = _agents_md_sections(task)
    if agents:
        prompt += f"\n{agents}\n"
    return prompt


def nanobot_system_prompt(task: BenchmarkTask) -> str:
    workspace = _workspace_for_prompt(task)
    return f"""## Runtime
channel=cli; system=POSIX; harness=nanobot

## Workspace
Your workspace is at: {workspace}
- Long-term memory: {workspace}/memory/MEMORY.md (automatically managed by Dream - do not edit directly)
- History log: {workspace}/memory/history.jsonl (append-only JSONL; prefer built-in `grep` for search).
- Custom skills: {workspace}/skills/{{skill-name}}/SKILL.md

## Platform Policy (POSIX)
- You are running on a POSIX system. Prefer UTF-8 and standard shell tools.
- Use file tools when they are simpler or more reliable than shell commands.

## Format Hint
Output is rendered in a terminal. Avoid markdown headings and tables. Use plain text with minimal formatting.

## Search & Discovery

- Prefer built-in `grep` over `exec` for workspace search.
- On broad searches, use `grep(output_mode="count")` to scope before requesting full content.

Reply directly with text for the current conversation. Do not use the 'message' tool for normal replies in the current chat.
When you need to call tools before answering, do not include the final user-visible answer in the same assistant message as the tool calls. Wait for the tool results, then answer once.

# Tool Usage Notes

Tool signatures are provided automatically via function calling. This section documents the general tool contract and non-obvious usage patterns.

## General Tool Contract

- Use the narrowest structured tool that directly matches the task.
- Use read-only discovery before writes when state is uncertain.
- Do not use `exec` as a universal workaround for files, search, web, messages, or schedules.
- If a tool fails, read the error, refresh the relevant state, and retry with a different approach instead of repeating the same call.
- After meaningful changes, verify with the smallest reliable check: re-read changed state, run targeted tests, or inspect command output.
- Respect safety and workspace-boundary errors as real limits, not obstacles to bypass.

## Discovery and Reading

- Use file discovery before reading when a path is uncertain.
- Use grep for content search inside the workspace; prefer it over shell grep for ordinary searches.
- Binary or oversized files may be skipped to keep results readable.

## File and Coding Workflows

- For code or config changes, the default loop is: locate, inspect, edit, then verify.
- Use patch-style edits as the default code editing approach.
- If an edit fails, re-read, narrow the context, and try a smaller patch rather than switching to shell `sed` or `echo`.

## Process Execution

- Use process execution for tests, builds, package commands, git commands, and other process execution.
- Prefer dedicated file/search tools over `cat`, shell `find`, shell `grep`, `sed`, or `echo` for ordinary workspace inspection and edits.
- Use non-interactive flags such as `-y` or `--yes` when available.
"""


def _openclaw_tool_snippets(task: BenchmarkTask) -> str:
    rows: list[str] = []
    for raw in task.metadata.get("allowed_tool_specs") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        description = str(raw.get("description") or "Available for this task.").strip()
        rows.append(f"- {raw['name']}: {description}")
    return "\n".join(rows) if rows else "(none)"


def openclaw_system_prompt(task: BenchmarkTask) -> str:
    workspace = _workspace_for_prompt(task)
    date = time.strftime("%Y-%m-%d")
    return f"""You are an expert coding assistant operating inside OpenClaw's embedded coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{_openclaw_tool_snippets(task)}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
- Be concise in your responses
- Show file paths clearly when working with files
- Prefer dedicated search and file tools over shell commands for ordinary workspace exploration

Current date: {date}
Current working directory: {workspace}
"""


def harness_initial_messages(task: BenchmarkTask, harness_name: str | None) -> list[Any]:
    normalized = (harness_name or "").lower()
    if normalized == "codex":
        return _text_messages(codex_system_prompt(task), task.prompt)
    if normalized == "nanobot":
        return _text_messages(nanobot_system_prompt(task), task.prompt)
    if normalized == "openclaw":
        return _text_messages(openclaw_system_prompt(task), task.prompt)
    return miniharness_initial_messages(task)


def run_native_claw_tools(
    task: BenchmarkTask,
    model: ModelProfile,
    final_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    *,
    timeout_s: float,
    hcfg: dict[str, Any] | None = None,
    dry_run: bool = False,
    provider_factory: ProviderFactory = _make_native_provider,
    tool_dispatcher: ToolDispatcher = dispatch_native_tool,
    metrics_prefix: str | None = None,
    harness_name: str | None = None,
) -> tuple[int, str, str, dict[str, Any]]:
    from claw_eval.models.content import TextBlock, ToolResultBlock  # type: ignore
    from claw_eval.models.message import Message  # type: ignore

    hcfg = hcfg or {}
    bridge_url = str(task.metadata.get("claw_tool_bridge_url") or "")
    if not bridge_url:
        raise RuntimeError("native Claw tool mode requires task.metadata['claw_tool_bridge_url']")
    tool_specs = native_tool_specs(task)
    tool_names = [str(tool.name) for tool in tool_specs]
    max_turns = int(task.metadata.get("max_turns") or hcfg.get("max_turns") or 50)
    provider = provider_factory(model)
    messages: list[Any] = harness_initial_messages(task, harness_name)
    total_input_tokens = 0
    total_output_tokens = 0
    transcript: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s

    base_metrics: dict[str, Any] = {
        "tool_mode": "claw_native_tools",
        "native_harness_tools_disabled": True,
        "model_visible_tools": tool_names,
        "model_visible_prompt_style": harness_name or "miniharness",
        "turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    if metrics_prefix:
        base_metrics[f"{metrics_prefix}_tool_mode"] = "claw_native_tools"
        base_metrics[f"{metrics_prefix}_native_tools_disabled"] = True

    if dry_run:
        final_path.write_text("", encoding="utf-8")
        stdout_path.write_text(
            json.dumps({"mode": "claw_native_tools", "model_visible_tools": tool_names}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return 0, "", "dry_run", base_metrics

    final = ""
    status = "ok"
    rc = 0
    try:
        for turn in range(1, max_turns + 1):
            if time.monotonic() > deadline:
                rc = -124
                status = "timeout"
                break
            response, usage = provider.chat(messages, tools=tool_specs)
            total_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            messages.append(response)
            response_json = response.model_dump(mode="json") if hasattr(response, "model_dump") else {"response": str(response)}
            transcript.append({"turn": turn, "assistant": response_json})
            tool_uses = [block for block in getattr(response, "content", []) or [] if getattr(block, "type", None) == "tool_use"]
            if not tool_uses:
                final = str(getattr(response, "text", "") or "")
                break
            result_blocks = []
            for tool_use in tool_uses:
                tool_name = str(tool_use.name)
                tool_input = dict(tool_use.input or {})
                dispatch_payload = tool_dispatcher(bridge_url, tool_name, tool_input)
                transcript.append(
                    {
                        "turn": turn,
                        "tool_name": tool_name,
                        "tool_use_id": tool_use.id,
                        "tool_input": tool_input,
                        "tool_result": dispatch_payload,
                    }
                )
                result_blocks.append(
                    ToolResultBlock(
                        tool_use_id=tool_use.id,
                        content=[TextBlock(text=_native_tool_result_text(dispatch_payload))],
                        is_error=bool(dispatch_payload.get("is_error")),
                    )
                )
            messages.append(Message(role="user", content=result_blocks))
        else:
            status = "max_turns_exhausted"
            rc = 1
    except Exception as exc:
        status = "nonzero"
        rc = 1
        stderr_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    else:
        stderr_path.write_text("", encoding="utf-8")

    final_path.write_text(final, encoding="utf-8")
    stdout_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in transcript) + ("\n" if transcript else ""),
        encoding="utf-8",
    )
    metrics = {
        **base_metrics,
        "turns": len([item for item in transcript if "assistant" in item]),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }
    return rc, final, status, metrics
