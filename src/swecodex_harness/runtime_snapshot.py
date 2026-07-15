from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from .common import run_cmd, write_json
from .config import get_path


PACKAGES = ["vllm", "swebench", "datasets", "torch", "transformers", "pyzmq", "yaml"]


def _pkg_version(name: str) -> str | None:
    try:
        if name == "yaml":
            name = "PyYAML"
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_rev(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    if not (p / ".git").exists() and not (p / "HEAD").exists():
        return {"path": str(p), "exists": True, "git": False}
    out = {"path": str(p), "exists": True, "git": True}
    for key, cmd in {
        "rev": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "status": ["git", "status", "--short"],
    }.items():
        try:
            out[key] = run_cmd(cmd, cwd=p, timeout=30, check=False).stdout.strip()
        except Exception as e:
            out[key] = f"ERROR: {e}"
    return out


def runtime_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    external_root = Path(get_path(cfg, "project.external_root", "external"))
    snap: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "env": {
            k: os.environ.get(k)
            for k in ["CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "MAMBA_EXE", "CUDA_VISIBLE_DEVICES", "PATH", "CODEX_HOME"]
            if os.environ.get(k) is not None
        },
        "executables": {x: shutil.which(x) for x in ["codex", "git", "docker", "nvidia-smi", "python"]},
        "packages": {name: _pkg_version(name) for name in PACKAGES},
        "git": {
            "project": _git_rev(get_path(cfg, "project.root", ".")),
            "codex": _git_rev(external_root / "codex"),
            "swe_bench": _git_rev(external_root / "SWE-bench"),
            "sparse_vllm": _git_rev(external_root / "Sparse-vLLM"),
        },
    }
    try:
        snap["nvidia_smi"] = run_cmd(["nvidia-smi"], timeout=20, check=False).stdout
    except Exception as e:
        snap["nvidia_smi"] = f"ERROR: {e}"
    return snap


def write_runtime_snapshot(path: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    snap = runtime_snapshot(cfg)
    write_json(path, snap)
    return snap


def main() -> None:
    import argparse
    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--stage-config", default=None)
    ap.add_argument("--out", default="runtime_snapshot.json")
    args = ap.parse_args()
    cfg = load_config(args.config, args.stage_config)
    write_runtime_snapshot(args.out, cfg)
    print(args.out)


if __name__ == "__main__":
    main()
