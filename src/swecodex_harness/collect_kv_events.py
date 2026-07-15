from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

from .common import append_jsonl, now_iso


def collect(endpoint: str, out_path: str | Path, topic: bytes = b"", duration_s: float | None = None, stop_file: str | None = None, hwm: int = 100000) -> None:
    try:
        import zmq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyzmq is required for KV event collection: pip install pyzmq") from e
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.set_hwm(hwm)
    sock.setsockopt(zmq.SUBSCRIBE, topic)
    sock.connect(endpoint)
    start = time.time()
    seq = 0
    while True:
        if stop_file and Path(stop_file).exists():
            break
        if duration_s is not None and time.time() - start >= duration_s:
            break
        try:
            parts = sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.01)
            continue
        rec: dict[str, Any] = {
            "seq": seq,
            "timestamp": now_iso(),
            "endpoint": endpoint,
            "parts_base64": [base64.b64encode(p).decode("ascii") for p in parts],
            "parts_utf8_preview": [p[:4096].decode("utf-8", errors="replace") for p in parts],
        }
        append_jsonl(out_path, rec)
        seq += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect vLLM KV cache events from ZMQ publisher.")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="")
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--hwm", type=int, default=100000)
    args = ap.parse_args()
    collect(args.endpoint, args.out, topic=args.topic.encode(), duration_s=args.duration_s, stop_file=args.stop_file, hwm=args.hwm)


if __name__ == "__main__":
    main()
