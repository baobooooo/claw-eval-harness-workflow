from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class ModelProfile:
    """Resolved model endpoint configuration used by a harness.

    The runner does not call the model directly for agentic harnesses.  It passes
    this profile to Codex/nanobot/OpenClaw so the harness can wire its native
    provider configuration to the same model endpoint.
    """

    name: str
    provider: str
    model: str
    base_url: str
    api_key_env: str
    api_key_value: str | None = None
    protocol: str = "openai_chat"
    context_window: int | None = None
    max_output_tokens: int | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def env(self) -> dict[str, str]:
        env = {}
        if self.api_key_value is not None:
            env[self.api_key_env] = self.api_key_value
        return env

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "protocol": self.protocol,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
            "notes": self.notes,
        }


@dataclass(slots=True)
class BenchmarkTask:
    benchmark: str
    task_id: str
    row: dict[str, Any]
    prompt: str
    workspace: Path
    output_dir: Path
    repo: str | None = None
    base_commit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HarnessResult:
    task_id: str
    harness: str
    model: str
    status: str
    patch: str = ""
    final_message: str = ""
    stdout_path: str | None = None
    stderr_path: str | None = None
    trace_path: str | None = None
    duration_s: float | None = None
    returncode: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "harness": self.harness,
            "model": self.model,
            "status": self.status,
            "patch_bytes": len(self.patch.encode("utf-8")),
            "final_message": self.final_message,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "trace_path": self.trace_path,
            "duration_s": self.duration_s,
            "returncode": self.returncode,
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass(slots=True)
class EvalResult:
    benchmark: str
    run_dir: Path
    status: str
    manifest: dict[str, Any] = field(default_factory=dict)
    report_files: list[str] = field(default_factory=list)
