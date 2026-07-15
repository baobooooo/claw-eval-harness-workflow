from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_message(role: str, text: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


class LiveTraceWriter:
    """Append-only Claw-Eval-compatible JSONL trace writer for external loops.

    The official Claw-Eval TraceWriter is tied to Pydantic event objects emitted
    by its native agent loop.  External harness mode still needs a live trace,
    so this writer emits the same JSON event schema while allowing bridge calls
    to be recorded immediately as they happen.
    """

    def __init__(self, path: str | Path, *, trace_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_id = trace_id or str(uuid.uuid4())
        self._fh = None
        self._lock = threading.Lock()
        self._started = False
        self._ended = False
        self._tool_time_s = 0.0
        self._tool_count = 0
        self._turn_count = 0
        self._wall_start = time.monotonic()

    def __enter__(self) -> "LiveTraceWriter":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        if self._fh is None or self._fh.closed:
            self._fh = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def write_event(self, event: dict[str, Any] | Any) -> None:
        """Write either a plain dict or a Claw-Eval Pydantic event object."""
        if hasattr(event, "model_dump"):
            payload = event.model_dump(mode="json")
        elif isinstance(event, dict):
            payload = event
        else:
            payload = dict(event)
        payload.setdefault("trace_id", self.trace_id)
        payload.setdefault("timestamp", utc_now())
        with self._lock:
            self.open()
            assert self._fh is not None
            self._fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            self._fh.flush()

    def start(self, *, task_id: str, model: str, persona: str = "external_harness_live_bridge") -> None:
        if self._started:
            return
        self._started = True
        self.write_event(
            {
                "type": "trace_start",
                "trace_id": self.trace_id,
                "task_id": task_id,
                "model": model,
                "persona": persona,
            }
        )

    def message(self, role: str, text: str, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._turn_count += 1 if role == "assistant" else 0
        self.write_event(
            {
                "type": "message",
                "trace_id": self.trace_id,
                "message": text_message(role, text),
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }
        )

    def tool_dispatch(self, dispatch: dict[str, Any] | Any) -> None:
        started = time.monotonic()
        if hasattr(dispatch, "model_dump"):
            payload = dispatch.model_dump(mode="json")
        else:
            payload = dict(dispatch)
        payload.setdefault("type", "tool_dispatch")
        payload.setdefault("trace_id", self.trace_id)
        payload.setdefault("latency_ms", 0.0)
        self._tool_count += 1
        self._tool_time_s += float(payload.get("latency_ms") or 0.0) / 1000.0
        # If a caller did not compute latency, count writer overhead minimally.
        if not payload.get("latency_ms"):
            self._tool_time_s += max(0.0, time.monotonic() - started)
        self.write_event(payload)

    def audit_snapshot(self, *, service_name: str, audit_url: str, audit_data: dict[str, Any]) -> None:
        self.write_event(
            {
                "type": "audit_snapshot",
                "trace_id": self.trace_id,
                "service_name": service_name,
                "audit_url": audit_url,
                "audit_data": audit_data,
            }
        )

    def end(self, *, status: str = "ok", failure_modes: list[str] | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        wall = max(0.0, time.monotonic() - self._wall_start)
        failures = list(failure_modes or [])
        if status not in {"ok", "dry_run"} and status not in failures:
            failures.append(status)
        self.write_event(
            {
                "type": "trace_end",
                "trace_id": self.trace_id,
                "total_turns": self._turn_count,
                "model_input_tokens": 0,
                "model_output_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model_time_s": 0.0,
                "tool_time_s": round(self._tool_time_s, 6),
                "other_time_s": 0.0,
                "wall_time_s": round(wall, 6),
                "scores": {
                    "completion": 0.0,
                    "robustness": 0.0,
                    "communication": 0.0,
                    "safety": 1.0,
                    "efficiency_turns": self._turn_count,
                    "efficiency_tokens": 0,
                    "efficiency_wall_time_s": round(wall, 6),
                },
                "task_score": 0.0,
                "passed": False,
                "failure_modes": failures,
                "user_agent_rounds": 0,
                "user_agent_max_rounds": 0,
                "user_agent_done": False,
            }
        )
