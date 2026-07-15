from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from harness_eval.claw_live.dispatcher import ClawLiveDispatcher
from harness_eval.claw_live.model_proxy import ClawToolModelProxy, _make_chat_sse, model_tool_proxy_enabled
from harness_eval.types import BenchmarkTask, ModelProfile


def _model(base_url: str) -> ModelProfile:
    return ModelProfile(
        name="deepseek_test",
        provider="deepseek",
        model="deepseek-test",
        base_url=base_url,
        api_key_env="TEST_API_KEY",
        api_key_value="test-key",
        protocol="openai_chat",
    )


def _task(tmp_path: Path) -> BenchmarkTask:
    agent = tmp_path / "agent"
    agent.mkdir(parents=True)
    (agent / "claw_tool").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    return BenchmarkTask(
        benchmark="openclaw",
        task_id="T_proxy",
        row={},
        prompt="Fetch the evidence.",
        workspace=tmp_path / "workspace",
        output_dir=tmp_path / "out",
        metadata={
            "live_tool_bridge": True,
            "agent_workspace": str(agent),
            "claw_tool_bridge_url": "http://127.0.0.1:9999",
            "allowed_tool_specs": [
                {
                    "name": "Bash",
                    "description": "Executes a bash command in the Claw-Eval sandbox container.",
                    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
                },
                {
                    "name": "web_fetch",
                    "description": "Fetch full webpage content for a URL.",
                    "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                },
            ],
        },
    )


class _Upstream:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                upstream.requests.append({"path": self.path, "payload": payload})
                body = upstream.responses.pop(0)
                raw = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        host, port = self._httpd.server_address
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{host}:{port}/v1"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def test_model_tool_proxy_injects_yaml_tools_and_replays_chat_history(tmp_path):
    upstream = _Upstream(
        [
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://example.com\"}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {"id": "chatcmpl-2", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]},
        ]
    )
    base_url = upstream.start()
    try:
        task = _task(tmp_path)
        hcfg = {"model_tool_proxy": {"enabled": True}}
        assert model_tool_proxy_enabled(task, hcfg) is True
        with ClawToolModelProxy(task=task, model=_model(base_url), hcfg=hcfg, harness_name="codex") as proxy:
            proxied_model = proxy.proxied_model()
            with httpx.Client(trust_env=False, timeout=10.0) as client:
                first = client.post(
                    f"{proxied_model.base_url}/chat/completions",
                    json={
                        "model": "deepseek-test",
                        "messages": [
                            {"role": "system", "content": "Harness-native system prompt."},
                            {"role": "user", "content": "fetch"},
                        ],
                        "tools": [{"type": "function", "function": {"name": "exec_command", "parameters": {"type": "object"}}}],
                    },
                ).json()
                tool_call = first["choices"][0]["message"]["tool_calls"][0]
                assert tool_call["function"]["name"] == "exec_command"
                transport_args = json.loads(tool_call["function"]["arguments"])
                assert transport_args["cmd"].startswith("python3 ./claw_tool web_fetch @")
                payload_ref = transport_args["cmd"].split("@", 1)[1]
                payload_path = Path(task.metadata["agent_workspace"]) / payload_ref
                assert json.loads(payload_path.read_text(encoding="utf-8")) == {"url": "https://example.com"}

                second = client.post(
                    f"{proxied_model.base_url}/chat/completions",
                    json={
                        "model": "deepseek-test",
                        "messages": [
                            {"role": "system", "content": "Harness-native system prompt."},
                            {"role": "user", "content": "fetch"},
                            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                            {"role": "tool", "tool_call_id": "call_1", "content": "fetched evidence"},
                        ],
                        "tools": [{"type": "function", "function": {"name": "exec_command", "parameters": {"type": "object"}}}],
                    },
                ).json()
                assert second["choices"][0]["message"]["content"] == "done"
    finally:
        upstream.stop()

    first_payload = upstream.requests[0]["payload"]
    assert first_payload["messages"] == [
        {"role": "system", "content": "Harness-native system prompt."},
        {"role": "user", "content": "fetch"},
    ]
    assert [tool["function"]["name"] for tool in first_payload["tools"]] == ["Bash", "web_fetch"]
    assert [tool["function"]["description"] for tool in first_payload["tools"]] == [
        "Executes a bash command in the Claw-Eval sandbox container.",
        "Fetch full webpage content for a URL.",
    ]
    assert "exec_command" not in {tool["function"]["name"] for tool in first_payload["tools"]}
    replayed_call = upstream.requests[1]["payload"]["messages"][2]["tool_calls"][0]
    assert replayed_call["function"]["name"] == "web_fetch"
    assert replayed_call["function"]["arguments"] == "{\"url\":\"https://example.com\"}"
    proxy_events = [json.loads(line) for line in (tmp_path / "out" / "codex_model_tool_proxy.jsonl").read_text(encoding="utf-8").splitlines()]
    upstream_events = [event for event in proxy_events if event.get("event") == "upstream_request"]
    assert len(upstream_events) == 2
    assert upstream_events[0]["body"]["messages"] == first_payload["messages"]
    assert [tool["function"]["name"] for tool in upstream_events[0]["body"]["tools"]] == ["Bash", "web_fetch"]
    assert upstream_events[1]["body"]["messages"][2]["tool_calls"][0]["function"]["name"] == "web_fetch"


def test_model_tool_proxy_command_template_can_use_absolute_claw_tool_and_payload(tmp_path):
    task = _task(tmp_path)
    proxy = ClawToolModelProxy(
        task=task,
        model=_model("http://127.0.0.1:9/v1"),
        hcfg={"model_tool_proxy": {"enabled": True, "command_template": "python3 {claw_tool_path} {tool_name} @{absolute_payload_path}"}},
        harness_name="openclaw",
    )

    transport_args = proxy._transport_arguments("call abs", "web_fetch", "{\"url\":\"https://example.com\"}")

    agent = Path(task.metadata["agent_workspace"])
    assert transport_args["command"].startswith(f"python3 {agent / 'claw_tool'} web_fetch @")
    payload_ref = transport_args["command"].split("@", 1)[1]
    payload_path = Path(payload_ref)
    assert payload_path.is_absolute()
    assert payload_path.parent == agent / ".claw_tool_payloads"
    assert json.loads(payload_path.read_text(encoding="utf-8")) == {"url": "https://example.com"}


def test_model_tool_proxy_drops_stream_options_when_de_streaming_upstream(tmp_path):
    upstream = _Upstream([
        {
            "id": "chatcmpl-stream-options",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        }
    ])
    base_url = upstream.start()
    try:
        task = _task(tmp_path)
        with ClawToolModelProxy(task=task, model=_model(base_url), hcfg={"model_tool_proxy": {"enabled": True}}, harness_name="nanobot") as proxy:
            with httpx.Client(trust_env=False, timeout=10.0) as client:
                response = client.post(
                    f"{proxy.proxied_model().base_url}/chat/completions",
                    json={
                        "model": "deepseek-test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                )
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
    finally:
        upstream.stop()

    upstream_payload = upstream.requests[0]["payload"]
    assert upstream_payload["stream"] is False
    assert "stream_options" not in upstream_payload


def test_chat_stream_sse_uses_incremental_openai_chunks_for_tool_calls():
    body = {
        "id": "chatcmpl-stream",
        "object": "chat.completion",
        "created": 123,
        "model": "deepseek-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{\"command\":\"python3 ./claw_tool web_fetch @payload.json\"}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    data_lines = [line.removeprefix("data: ") for line in _make_chat_sse(body).decode("utf-8").splitlines() if line.startswith("data: ")]

    assert data_lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in data_lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[0]["choices"][0]["finish_reason"] is None
    tool_delta = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["index"] == 0
    assert tool_delta["id"] == "call_1"
    assert tool_delta["function"]["name"] == "exec"
    assert chunks[1]["choices"][0]["finish_reason"] is None
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_model_tool_proxy_translates_responses_api_function_call(tmp_path):
    upstream = _Upstream(
        [
            {
                "id": "resp_1",
                "object": "response",
                "output": [
                    {"type": "function_call", "call_id": "call_rsp", "name": "web_fetch", "arguments": "{\"url\":\"https://example.org\"}"}
                ],
            }
        ]
    )
    base_url = upstream.start()
    try:
        task = _task(tmp_path)
        hcfg = {"model_tool_proxy": {"enabled": True}}
        with ClawToolModelProxy(task=task, model=_model(base_url), hcfg=hcfg, harness_name="codex") as proxy:
            with httpx.Client(trust_env=False, timeout=10.0) as client:
                body = client.post(
                    f"{proxy.proxied_model().base_url}/responses",
                    json={"model": "deepseek-test", "input": "fetch", "tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}]},
                ).json()
        item = body["output"][0]
        assert item["name"] == "exec_command"
        assert "./claw_tool web_fetch" in json.loads(item["arguments"])["cmd"]
    finally:
        upstream.stop()

    assert [tool["name"] for tool in upstream.requests[0]["payload"]["tools"]] == ["Bash", "web_fetch"]


class _TraceStub:
    trace_id = "trace-test"

    def tool_dispatch(self, event):  # pragma: no cover - not used by this order test
        pass


def test_dispatcher_preserves_miniharness_tool_order():
    dispatcher = ClawLiveDispatcher(
        trace_writer=_TraceStub(),
        endpoints=[],
        task_tool_specs=[
            {"name": "web_fetch", "description": "fetch", "input_schema": {"type": "object"}},
            {"name": "custom_tool", "description": "custom", "input_schema": {"type": "object"}},
        ],
        sandbox_url="http://127.0.0.1:1",
    )
    try:
        names = [spec["name"] for spec in dispatcher.tool_specs[:4]]
        # MiniHarness order is task YAML tools first, then sandbox tools.
        assert names[:2] == ["web_fetch", "custom_tool"]
        assert "Bash" in [spec["name"] for spec in dispatcher.tool_specs]
        assert names != sorted(names)
    finally:
        dispatcher.close()


def test_model_tool_proxy_restores_chat_history_with_mutated_ids_and_sanitizes_tool_result(tmp_path):
    upstream = _Upstream(
        [
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_00_AbC",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": "{\"command\":\"ls\"}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {"id": "chatcmpl-2", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
        ]
    )
    base_url = upstream.start()
    try:
        task = _task(tmp_path)
        with ClawToolModelProxy(task=task, model=_model(base_url), hcfg={"model_tool_proxy": {"enabled": True}}, harness_name="openclaw") as proxy:
            with httpx.Client(trust_env=False, timeout=10.0) as client:
                first = client.post(
                    f"{proxy.proxied_model().base_url}/chat/completions",
                    json={"model": "deepseek-test", "messages": [{"role": "user", "content": "list"}], "tools": []},
                ).json()
                transport_call = first["choices"][0]["message"]["tool_calls"][0]
                assert transport_call["function"]["name"] == "exec"
                # OpenClaw-style history can drop underscores from ids and can put
                # the hidden transport name on role=tool messages.  Upstream must
                # see the original YAML tool call/result instead.
                mutated_call = json.loads(json.dumps(transport_call))
                mutated_call["id"] = "call00AbC"
                second = client.post(
                    f"{proxy.proxied_model().base_url}/chat/completions",
                    json={
                        "model": "deepseek-test",
                        "messages": [
                            {"role": "user", "content": "list"},
                            {"role": "assistant", "content": None, "tool_calls": [mutated_call]},
                            {
                                "role": "tool",
                                "name": "exec",
                                "tool_call_id": "call00AbC",
                                "content": '{"body":{"exit_code":0,"stdout":"a\\n","stderr":""},"endpoint_url":"http://localhost:1/exec","tool_name":"Bash","tool_use_id":"hidden"}\n\nExit code: 0',
                            },
                        ],
                        "tools": [],
                    },
                ).json()
                assert second["choices"][0]["message"]["content"] == "ok"
    finally:
        upstream.stop()

    replay = upstream.requests[1]["payload"]["messages"]
    assert replay[1]["tool_calls"][0]["function"]["name"] == "Bash"
    assert replay[1]["tool_calls"][0]["function"]["arguments"] == "{\"command\":\"ls\"}"
    assert replay[2]["name"] == "Bash"
    assert "endpoint_url" not in replay[2]["content"]
    assert "claw_tool" not in json.dumps(replay, ensure_ascii=False)
    assert json.loads(replay[2]["content"])["stdout"] == "a\n"


def test_model_tool_proxy_blocks_non_yaml_native_tool_calls_in_strict_mode(tmp_path):
    upstream = _Upstream(
        [
            {
                "id": "chatcmpl-native",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "call_native", "type": "function", "function": {"name": "exec", "arguments": "{\"command\":\"ls\"}"}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        ]
    )
    base_url = upstream.start()
    try:
        task = _task(tmp_path)
        with ClawToolModelProxy(task=task, model=_model(base_url), hcfg={"model_tool_proxy": {"enabled": True}}, harness_name="openclaw") as proxy:
            with httpx.Client(trust_env=False, timeout=10.0) as client:
                body = client.post(
                    f"{proxy.proxied_model().base_url}/chat/completions",
                    json={"model": "deepseek-test", "messages": [{"role": "user", "content": "bad"}], "tools": []},
                ).json()
    finally:
        upstream.stop()

    msg = body["choices"][0]["message"]
    assert "tool_calls" not in msg
    assert "blocked" in msg["content"]
    assert body["choices"][0]["finish_reason"] == "stop"
