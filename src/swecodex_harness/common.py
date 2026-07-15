from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_instance_id(instance_id: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "__" for c in instance_id)


def quote_cmd(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in args)


def run_cmd(
    args: list[str],
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    input_text: str | None = None,
) -> CommandResult:
    proc_env = os.environ.copy()
    if env:
        proc_env.update({str(k): str(v) for k, v in env.items()})

    stdout_target = subprocess.PIPE if stdout_path is None else open(stdout_path, "w", encoding="utf-8", errors="replace")
    stderr_target = subprocess.PIPE if stderr_path is None else open(stderr_path, "w", encoding="utf-8", errors="replace")
    try:
        p = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=proc_env,
            input=input_text,
            text=True,
            stdout=stdout_target,
            stderr=stderr_target,
            timeout=timeout,
        )
    finally:
        if stdout_path is not None and hasattr(stdout_target, "close"):
            stdout_target.close()
        if stderr_path is not None and hasattr(stderr_target, "close"):
            stderr_target.close()

    stdout = "" if stdout_path is not None else (p.stdout or "")
    stderr = "" if stderr_path is not None else (p.stderr or "")
    result = CommandResult(args=args, returncode=p.returncode, stdout=stdout, stderr=stderr)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {p.returncode}: {quote_cmd(args)}\n"
            f"stdout:\n{stdout[-4000:]}\n"
            f"stderr:\n{stderr[-4000:]}"
        )
    return result


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    out: list[Any] = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"_raw": line, "_parse_error": True})
    return out


def render_template(template: str, values: Mapping[str, Any]) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)
