from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .common import write_json


def _post(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec: local endpoint by default
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return {"ok": True, "status": resp.status, "body": json.loads(body)}
            except json.JSONDecodeError:
                return {"ok": True, "status": resp.status, "body": body[:10000]}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "body": e.read().decode("utf-8", errors="replace")[:10000]}
    except Exception as e:
        return {"ok": False, "status": None, "body": str(e)}


def probe(base_url: str) -> dict[str, Any]:
    base = base_url.rstrip("/")
    return {
        "base_url": base,
        "prefix_cache_inspect": _post(f"{base}/prefix_cache/inspect"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe Sparse-vLLM prefix-cache debugging endpoints.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = probe(args.base_url)
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
