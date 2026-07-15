from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse


def _try_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return raw.decode("utf-8", errors="replace")[:20000]


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in {"authorization", "cookie", "set-cookie"}:
            redacted[k] = "<redacted>"
        else:
            redacted[k] = v
    return redacted


def _write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def create_app(target_base_url: str, log_path: str) -> FastAPI:
    app = FastAPI(title="OpenAI-compatible logging proxy")
    target = target_base_url.rstrip("/")
    out_path = Path(log_path)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request):
        started = time.time()
        raw_body = await request.body()
        req_headers = dict(request.headers)
        forward_headers = {
            k: v
            for k, v in req_headers.items()
            if k.lower() not in {"host", "content-length", "connection", "accept-encoding"}
        }
        url = f"{target}/{path}"
        query = request.url.query
        if query:
            url = f"{url}?{query}"

        request_record: dict[str, Any] = {
            "event": "request",
            "t": started,
            "method": request.method,
            "path": "/" + path,
            "target_url": url,
            "headers": _redact_headers(req_headers),
            "body": _try_json(raw_body),
        }
        _write_jsonl(out_path, request_record)

        async with httpx.AsyncClient(timeout=None) as client:
            req = client.build_request(request.method, url, headers=forward_headers, content=raw_body)
            upstream = await client.send(req, stream=True)
            chunks: list[bytes] = []

            async def stream_body():
                try:
                    async for chunk in upstream.aiter_bytes():
                        chunks.append(chunk)
                        yield chunk
                finally:
                    await upstream.aclose()
                    body = b"".join(chunks)
                    _write_jsonl(
                        out_path,
                        {
                            "event": "response",
                            "t": time.time(),
                            "duration_s": round(time.time() - started, 3),
                            "method": request.method,
                            "path": "/" + path,
                            "status_code": upstream.status_code,
                            "headers": _redact_headers(dict(upstream.headers)),
                            "body": _try_json(body),
                        },
                    )

            response_headers = {
                k: v
                for k, v in upstream.headers.items()
                if k.lower() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}
            }
            media_type = upstream.headers.get("content-type")
            return StreamingResponse(stream_body(), status_code=upstream.status_code, headers=response_headers, media_type=media_type)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Transparent OpenAI-compatible logging proxy.")
    ap.add_argument("--target-base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--log-path", required=True)
    args = ap.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.target_base_url, args.log_path), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
