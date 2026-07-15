from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


@dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def quote_cmd(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in args)


def safe_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "__" for c in value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"{p} must contain a YAML mapping")
    return expand_env(obj)


_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_str(text: str) -> str:
    """Expand $VAR, ${VAR}, and the common ${VAR:-default} form."""

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value is None or value == "":
            return default if default is not None else ""
        return value

    text = _ENV_DEFAULT_RE.sub(repl, text)
    return os.path.expandvars(os.path.expanduser(text))


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_str(value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def get_path(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(p)


def append_jsonl(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[Any] = []
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
    if stdout_path is not None:
        Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)
    stdout_target = subprocess.PIPE if stdout_path is None else open(stdout_path, "w", encoding="utf-8", errors="replace")
    stderr_target = subprocess.PIPE if stderr_path is None else open(stderr_path, "w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.run(
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
        if stdout_path is not None:
            stdout_target.close()
        if stderr_path is not None:
            stderr_target.close()
    res = CommandResult(
        args=args,
        returncode=proc.returncode,
        stdout="" if stdout_path is not None else (proc.stdout or ""),
        stderr="" if stderr_path is not None else (proc.stderr or ""),
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {quote_cmd(args)}\n"
            f"stdout:\n{res.stdout[-4000:]}\nstderr:\n{res.stderr[-4000:]}"
        )
    return res


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)
