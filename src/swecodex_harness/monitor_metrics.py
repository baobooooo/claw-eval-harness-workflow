from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .common import append_jsonl, now_iso, run_cmd


DEFAULT_PATTERNS = [
    r"vllm:.*kv.*",
    r"vllm:.*prefix.*",
    r"vllm:.*cache.*",
    r"vllm:.*time.*",
    r"vllm:.*tokens.*",
    r"vllm:.*requests.*",
]


def fetch_text(url: str, timeout: float = 5.0) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec: local metrics endpoint by default
        return resp.read().decode("utf-8", errors="replace")


def parse_prometheus(text: str, patterns: list[str]) -> dict[str, float]:
    compiled = [re.compile(p) for p in patterns]
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split()[0]
        bare_name = name.split("{")[0]
        if compiled and not any(p.search(bare_name) for p in compiled):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            out[name] = float(parts[1])
        except ValueError:
            continue
    return out


def sample_nvidia_smi() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        res = run_cmd(cmd, timeout=10, check=False)
        return {"returncode": res.returncode, "csv": res.stdout.strip()}
    except Exception as e:
        return {"error": str(e)}


def monitor(url: str, out_dir: str | Path, interval_s: float, duration_s: float | None, raw: bool, patterns: list[str], stop_file: str | None) -> None:
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw_prometheus"
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / "metrics_selected.jsonl"
    gpu_path = out_dir / "nvidia_smi.jsonl"
    start = time.time()
    seq = 0
    while True:
        if stop_file and Path(stop_file).exists():
            break
        if duration_s is not None and time.time() - start >= duration_s:
            break
        ts = now_iso()
        rec: dict[str, Any] = {"seq": seq, "timestamp": ts, "url": url}
        try:
            text = fetch_text(url)
            if raw:
                (raw_dir / f"metrics_{seq:08d}.prom").write_text(text, encoding="utf-8")
            rec["metrics"] = parse_prometheus(text, patterns)
            rec["ok"] = True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            rec.update({"ok": False, "error": str(e)})
        append_jsonl(selected_path, rec)
        append_jsonl(gpu_path, {"seq": seq, "timestamp": ts, **sample_nvidia_smi()})
        seq += 1
        time.sleep(interval_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape vLLM/Sparse-vLLM metrics and GPU telemetry into JSONL.")
    ap.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--interval-s", type=float, default=2.0)
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument("--pattern", action="append", default=[])
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()
    patterns = args.pattern or DEFAULT_PATTERNS
    monitor(args.url, args.out_dir, args.interval_s, args.duration_s, not args.no_raw, patterns, args.stop_file)


if __name__ == "__main__":
    main()
