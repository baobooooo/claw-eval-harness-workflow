from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any


def _request(method: str, url: str, payload: dict[str, Any] | None = None, api_key: str = "dummy", timeout: float = 30.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec: local endpoint by default
            body = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body) if body else None
            except json.JSONDecodeError:
                parsed = body[:2000]
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "body": parsed}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": e.code, "body": body[:4000]}
    except Exception as e:
        return {"ok": False, "status": None, "body": str(e)}


def preflight(base_url: str, model: str, api_key: str = "dummy", require_responses: bool = True) -> dict[str, Any]:
    base = base_url.rstrip("/")
    out: dict[str, Any] = {"base_url": base, "model": model, "require_responses": require_responses}
    out["models"] = _request("GET", f"{base}/models", api_key=api_key)
    out["chat_completions"] = _request(
        "POST",
        f"{base}/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "Return the word ok."}], "max_tokens": 2, "temperature": 0},
        api_key=api_key,
    )
    out["responses"] = _request(
        "POST",
        f"{base}/responses",
        {"model": model, "input": "Return the word ok.", "max_output_tokens": 2, "temperature": 0},
        api_key=api_key,
    )
    out["codex_compatible"] = bool(out["responses"]["ok"] or not require_responses)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Check whether an OpenAI-compatible endpoint can serve Codex CLI.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--allow-no-responses", action="store_true")
    args = ap.parse_args()
    result = preflight(args.base_url, args.model, args.api_key, require_responses=not args.allow_no_responses)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["codex_compatible"] else 2)


if __name__ == "__main__":
    main()
