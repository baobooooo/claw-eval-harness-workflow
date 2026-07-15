import swecodex_harness.responses_to_chat_bridge as bridge
from swecodex_harness.responses_to_chat_bridge import (
    DEFAULT_RESPONSE_MAX_TOKENS,
    _chat_payload_from_responses,
    _message_to_response_items,
    _sanitize_chat_messages_for_tool_order,
    create_app,
)


def test_bridge_adds_default_max_tokens_for_unbounded_responses_request():
    payload = {"model": "qwen3-30b-a3b", "input": "inspect the checkout"}

    chat_payload = _chat_payload_from_responses(payload)

    assert chat_payload["max_tokens"] == DEFAULT_RESPONSE_MAX_TOKENS


def test_bridge_preserves_explicit_max_output_tokens():
    payload = {"model": "qwen3-30b-a3b", "input": "short", "max_output_tokens": 37}

    chat_payload = _chat_payload_from_responses(payload)

    assert chat_payload["max_tokens"] == 37


def test_bridge_converts_qwen_xml_tool_call_to_responses_function_call():
    message = {
        "content": """<tool_call>
{"name": "shell", "arguments": {"cmd": "find . -maxdepth 2 -type f | head"}}
</tool_call>"""
    }

    items = _message_to_response_items(message)

    assert len(items) == 1
    assert items[0]["type"] == "function_call"
    assert items[0]["name"] == "shell"
    assert "find ." in items[0]["arguments"]


def test_bridge_preserves_deepseek_reasoning_content_in_response_items():
    message = {
        "reasoning_content": "Need to inspect the workspace first.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
            }
        ],
    }

    items = _message_to_response_items(message)

    assert items[0]["type"] == "reasoning"
    assert items[0]["text"] == "Need to inspect the workspace first."
    assert items[1]["type"] == "function_call"


def test_bridge_replays_responses_reasoning_content_to_chat_tool_call():
    payload = {
        "model": "deepseek-v4-pro",
        "input": [
            {"type": "message", "role": "user", "content": "inspect"},
            {"type": "reasoning", "text": "Need to run a command."},
            {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            {"type": "message", "role": "user", "content": "continue"},
        ],
    }

    chat_payload = _chat_payload_from_responses(payload)

    assistant = chat_payload["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["reasoning_content"] == "Need to run a command."


def test_bridge_coalesces_consecutive_responses_function_calls_into_one_chat_message():
    payload = {
        "model": "deepseek-v4-pro",
        "input": [
            {"type": "message", "role": "user", "content": "inspect"},
            {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"one\"}"},
            {"type": "function_call", "call_id": "call_2", "name": "shell", "arguments": "{\"cmd\":\"two\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "one"},
            {"type": "function_call_output", "call_id": "call_2", "output": "two"},
            {"type": "message", "role": "user", "content": "continue"},
        ],
    }

    chat_payload = _chat_payload_from_responses(payload)

    assert [message["role"] for message in chat_payload["messages"]] == ["user", "assistant", "tool", "tool", "user"]
    assert len(chat_payload["messages"][1]["tool_calls"]) == 2
    assert [message["tool_call_id"] for message in chat_payload["messages"][2:4]] == ["call_1", "call_2"]


def test_bridge_chat_completions_streams_upstream_sse(monkeypatch):
    from fastapi.testclient import TestClient

    class Upstream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}
        text = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'

        def iter_content(self, chunk_size=None):
            yield self.text.encode("utf-8")

        def close(self):
            pass

    calls = {}

    def fake_post(url, json, headers, timeout, stream=False):
        calls["url"] = url
        calls["stream"] = stream
        calls["payload"] = json
        return Upstream()

    monkeypatch.setattr(bridge.requests, "post", fake_post)
    client = TestClient(create_app("http://upstream/v1", "test-key"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-pro", "messages": [], "stream": True},
    )

    assert response.status_code == 200
    assert calls["url"] == "http://upstream/v1/chat/completions"
    assert calls["stream"] is True
    assert response.text == Upstream.text


def test_bridge_sanitizes_chat_tool_calls_before_non_tool_message():
    messages = [
        {"role": "system", "content": "policy"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "continue"},
    ]

    sanitized = _sanitize_chat_messages_for_tool_order(messages)

    assert [m["role"] for m in sanitized] == ["system", "assistant", "tool", "user"]
    assert sanitized[2]["tool_call_id"] == "call_1"
    assert "missing" in sanitized[2]["content"].lower()
    assert sanitized[1]["reasoning_content"] == ""


def test_bridge_chat_completions_forwards_sanitized_messages(monkeypatch):
    from fastapi.testclient import TestClient

    class Upstream:
        status_code = 200
        text = '{"choices":[{"message":{"content":"ok"}}]}'
        headers = {"content-type": "application/json"}

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    calls = {}

    def fake_post(url, json, headers, timeout, stream=False):
        calls["payload"] = json
        return Upstream()

    monkeypatch.setattr(bridge.requests, "post", fake_post)
    client = TestClient(create_app("http://upstream/v1", "test-key"))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-v4-pro",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
        },
    )

    assert response.status_code == 200
    assert [m["role"] for m in calls["payload"]["messages"]] == ["assistant", "tool", "user"]


def test_bridge_chat_completions_wraps_non_json_upstream(monkeypatch):
    from fastapi.testclient import TestClient

    class Upstream:
        status_code = 502
        text = "<html>bad gateway</html>"
        headers = {"content-type": "text/html"}

        def json(self):
            raise ValueError("not json")

    def fake_post(url, json, headers, timeout, stream=False):
        return Upstream()

    monkeypatch.setattr(bridge.requests, "post", fake_post)
    client = TestClient(create_app("http://upstream/v1", "test-key"))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-v4-pro", "messages": []},
    )

    assert response.status_code == 502
    assert response.json()["error"] == "<html>bad gateway</html>"
