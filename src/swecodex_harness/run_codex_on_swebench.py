from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .common import append_jsonl, now_iso, quote_cmd, render_template, run_cmd, safe_instance_id, sha256_text, write_json
from .config import ensure_dirs, get_path, load_config
from .parse_codex_events import parse_events
from .runtime_snapshot import write_runtime_snapshot


_COMMON_SEARCH_TERMS = {
    "about",
    "after",
    "before",
    "being",
    "between",
    "could",
    "error",
    "expected",
    "from",
    "have",
    "into",
    "issue",
    "should",
    "that",
    "their",
    "there",
    "this",
    "when",
    "where",
    "with",
    "would",
}
_TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".jinja",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_SKIP_PATH_PARTS = {".git", ".mypy_cache", ".pytest_cache", "__pycache__", "build", "dist", "node_modules"}


def _load_dataset_rows(dataset_name: str, split: str, max_instances: int | None, instance_ids: set[str] | None) -> list[dict[str, Any]]:
    offline_keys = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    old_offline_env = {k: os.environ.get(k) for k in offline_keys}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("datasets is required: pip install datasets") from e
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception:
        for k, v in old_offline_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ds = load_dataset(dataset_name, split=split)
    finally:
        for k, v in old_offline_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rows: list[dict[str, Any]] = []
    for row in ds:
        item = dict(row)
        iid = str(item.get("instance_id"))
        if instance_ids and iid not in instance_ids:
            continue
        rows.append(item)
        if max_instances is not None and len(rows) >= max_instances:
            break
    if not rows:
        raise RuntimeError("No SWE-bench instances selected. Check dataset/split/instance_ids.")
    return rows


def _parse_instance_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    p = Path(value)
    if p.exists():
        return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    return {x.strip() for x in value.split(",") if x.strip()}


def _repo_url(repo: str, cfg: dict[str, Any] | None = None) -> str:
    if repo.startswith("http://") or repo.startswith("https://") or repo.endswith(".git"):
        url = repo
    else:
        url = f"https://github.com/{repo}.git"
    prefix = str(get_path(cfg or {}, "agent.github_url_prefix", "") or "").strip()
    if prefix:
        return prefix.rstrip("/") + "/" + url
    return url


def _git_network_env(cfg: dict[str, Any]) -> dict[str, str] | None:
    env: dict[str, str] = {}
    low_speed_limit = get_path(cfg, "agent.git_http_low_speed_limit")
    low_speed_time = get_path(cfg, "agent.git_http_low_speed_time")
    if low_speed_limit is not None:
        env["GIT_HTTP_LOW_SPEED_LIMIT"] = str(low_speed_limit)
    if low_speed_time is not None:
        env["GIT_HTTP_LOW_SPEED_TIME"] = str(low_speed_time)
    return env or None


def _run_git(args: list[str], cfg: dict[str, Any], cwd: Path | None = None, timeout: float | None = None, check: bool = True, stdout_path: Path | None = None) -> Any:
    return run_cmd(args, cwd=cwd, env=_git_network_env(cfg), timeout=timeout, check=check, stdout_path=stdout_path)


def _move_failed_path(path: Path) -> Path:
    target = path.with_name(path.name + f".failed_{time.strftime('%Y%m%d_%H%M%S')}")
    suffix = 1
    while target.exists():
        target = path.with_name(path.name + f".failed_{time.strftime('%Y%m%d_%H%M%S')}_{suffix}")
        suffix += 1
    shutil.move(str(path), str(target))
    return target


def _mirror_is_usable(mirror: Path) -> bool:
    if not mirror.exists():
        return False
    bare = run_cmd(["git", "rev-parse", "--is-bare-repository"], cwd=mirror, timeout=60, check=False)
    if bare.returncode != 0 or bare.stdout.strip() != "true":
        return False
    refs = run_cmd(["git", "show-ref", "--head"], cwd=mirror, timeout=60, check=False)
    return refs.returncode == 0 and bool(refs.stdout.strip())


def _clone_mirror(repo: str, mirror: Path, cfg: dict[str, Any]) -> None:
    attempts = int(get_path(cfg, "agent.git_clone_retries", 2) or 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        if mirror.exists():
            _move_failed_path(mirror)
        try:
            _run_git(["git", "clone", "--mirror", _repo_url(repo, cfg), str(mirror)], cfg, timeout=1800)
            return
        except Exception as e:
            last_error = e
            if mirror.exists():
                _move_failed_path(mirror)
            if attempt < attempts:
                time.sleep(min(30, 5 * attempt))
    if last_error is not None:
        raise last_error


def _prepare_repo_mirror(row: dict[str, Any], cfg: dict[str, Any], inst_dir: Path) -> Path:
    data_root = Path(get_path(cfg, "project.data_root"))
    mirror_root = data_root / "repo_mirrors"
    mirror_root.mkdir(parents=True, exist_ok=True)
    repo = str(row["repo"])
    mirror = mirror_root / (repo.replace("/", "__") + ".git")
    worktree = inst_dir / "repo"
    base_commit = str(row["base_commit"])

    if not _mirror_is_usable(mirror):
        _clone_mirror(repo, mirror, cfg)
    elif get_path(cfg, "agent.repo_update_mirror", False):
        _run_git(["git", "remote", "update", "--prune"], cfg, cwd=mirror, timeout=900)

    if worktree.exists():
        shutil.rmtree(worktree)
    _run_git(["git", "clone", str(mirror), str(worktree)], cfg, timeout=900)
    _run_git(["git", "checkout", base_commit], cfg, cwd=worktree, timeout=300)
    _run_git(["git", "submodule", "update", "--init", "--recursive"], cfg, cwd=worktree, timeout=900, check=False)
    _run_git(["git", "status", "--short"], cfg, cwd=worktree, timeout=60, check=False, stdout_path=inst_dir / "repo_status_initial.txt")
    return worktree


def _prepare_repo_fetch_commit(row: dict[str, Any], cfg: dict[str, Any], inst_dir: Path) -> Path:
    repo = str(row["repo"])
    base_commit = str(row["base_commit"])
    worktree = inst_dir / "repo"
    cached = _prepare_repo_from_local_instance_cache(row, cfg, inst_dir)
    if cached is not None:
        return cached
    if worktree.exists():
        shutil.rmtree(worktree)
    worktree.mkdir(parents=True, exist_ok=True)
    _run_git(["git", "init"], cfg, cwd=worktree, timeout=120)
    _run_git(["git", "remote", "add", "origin", _repo_url(repo, cfg)], cfg, cwd=worktree, timeout=120)

    attempts = int(get_path(cfg, "agent.git_clone_retries", 2) or 1)
    last_error: Exception | None = None
    fetch_variants = [
        ["git", "fetch", "--depth", "1", "--filter=blob:none", "origin", base_commit],
        ["git", "fetch", "--depth", "1", "origin", base_commit],
    ]
    for attempt in range(1, attempts + 1):
        for fetch_cmd in fetch_variants:
            try:
                _run_git(fetch_cmd, cfg, cwd=worktree, timeout=1800)
                _run_git(["git", "checkout", "--detach", "FETCH_HEAD"], cfg, cwd=worktree, timeout=300)
                _run_git(["git", "submodule", "update", "--init", "--recursive"], cfg, cwd=worktree, timeout=900, check=False)
                _run_git(["git", "status", "--short"], cfg, cwd=worktree, timeout=60, check=False, stdout_path=inst_dir / "repo_status_initial.txt")
                return worktree
            except Exception as e:
                last_error = e
        if attempt < attempts:
            time.sleep(min(30, 5 * attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch {repo}@{base_commit}")


def _prepare_repo_from_local_instance_cache(row: dict[str, Any], cfg: dict[str, Any], inst_dir: Path) -> Path | None:
    runs_root = Path(get_path(cfg, "project.runs_root"))
    iid_dir = safe_instance_id(str(row["instance_id"]))
    base_commit = str(row["base_commit"])
    worktree = inst_dir / "repo"
    if not runs_root.exists():
        return None
    for candidate_git in sorted(runs_root.glob(f"*/instances/{iid_dir}/repo/.git"), reverse=True):
        candidate = candidate_git.parent
        if candidate.resolve() == worktree.resolve():
            continue
        has_commit = run_cmd(["git", "cat-file", "-e", f"{base_commit}^{{commit}}"], cwd=candidate, timeout=60, check=False)
        if has_commit.returncode != 0:
            continue
        if worktree.exists():
            shutil.rmtree(worktree)
        run_cmd(["git", "clone", "--no-hardlinks", str(candidate), str(worktree)], timeout=300)
        run_cmd(["git", "checkout", "--detach", base_commit], cwd=worktree, timeout=300)
        run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=worktree, timeout=900, check=False)
        run_cmd(["git", "status", "--short"], cwd=worktree, timeout=60, check=False, stdout_path=inst_dir / "repo_status_initial.txt")
        (inst_dir / "repo_source.txt").write_text(f"local_instance_cache: {candidate}\n", encoding="utf-8")
        return worktree
    return None


def _prepare_repo(row: dict[str, Any], cfg: dict[str, Any], inst_dir: Path) -> Path:
    strategy = str(get_path(cfg, "agent.clone_strategy", "mirror") or "mirror").strip().lower()
    if strategy == "fetch_commit":
        try:
            return _prepare_repo_fetch_commit(row, cfg, inst_dir)
        except Exception:
            if not get_path(cfg, "agent.fetch_commit_fallback_to_mirror", False):
                raise
            return _prepare_repo_mirror(row, cfg, inst_dir)
    if strategy != "mirror":
        raise ValueError(f"Unknown agent.clone_strategy: {strategy!r}")
    return _prepare_repo_mirror(row, cfg, inst_dir)


def _render_codex_config(cfg: dict[str, Any], codex_home: Path) -> None:
    template_path = Path("configs/codex/config.toml.template")
    if not template_path.exists():
        # Support running from installed package while cwd is project root expected by scripts.
        template_path = Path(get_path(cfg, "project.root")) / "configs/codex/config.toml.template"
    template = template_path.read_text(encoding="utf-8")
    values = {
        "SERVED_MODEL_NAME": get_path(cfg, "model.served_model_name"),
        "MODEL_PROVIDER_ID": get_path(cfg, "codex.model_provider_id"),
        "MODEL_PROVIDER_NAME": get_path(cfg, "codex.model_provider_id"),
        "MODEL_CONTEXT_WINDOW": get_path(cfg, "model.context_window", 65536),
        "APPROVAL_POLICY": get_path(cfg, "codex.approval_policy", "never"),
        "SANDBOX_MODE": get_path(cfg, "codex.sandbox_mode", "workspace-write"),
        "ENV_KEY": get_path(cfg, "codex.env_key", "VLLM_API_KEY"),
        "BASE_URL": get_path(cfg, "codex.base_url"),
        "WIRE_API": get_path(cfg, "codex.wire_api", "responses"),
    }
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text(render_template(template, values), encoding="utf-8")


def _render_prompt(row: dict[str, Any], cfg: dict[str, Any]) -> str:
    prompt_path = Path(get_path(cfg, "agent.prompt_template", "configs/prompts/codex_swebench_prompt.md"))
    if not prompt_path.exists():
        prompt_path = Path(get_path(cfg, "project.root")) / prompt_path
    template = prompt_path.read_text(encoding="utf-8")
    values = {
        "INSTANCE_ID": row.get("instance_id", ""),
        "REPO": row.get("repo", ""),
        "BASE_COMMIT": row.get("base_commit", ""),
        "PROBLEM_STATEMENT": row.get("problem_statement", ""),
        "HINTS_TEXT": row.get("hints_text", ""),
        "TEST_PATCH": row.get("test_patch", ""),
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", row.get("fail_to_pass", "")),
        "PASS_TO_PASS": row.get("PASS_TO_PASS", row.get("pass_to_pass", "")),
        "RUN_TESTS_INSIDE_CODEX": get_path(cfg, "agent.run_tests_inside_codex", False),
    }
    return render_template(template, values)


def _get_patch(repo_dir: Path) -> str:
    # `git add -N` makes newly-created files visible to git diff without staging content.
    run_cmd(["git", "add", "-N", "."], cwd=repo_dir, timeout=120, check=False)
    result = run_cmd(["git", "diff", "--binary"], cwd=repo_dir, timeout=120, check=False)
    run_cmd(["git", "reset", "--quiet"], cwd=repo_dir, timeout=120, check=False)
    return result.stdout


def _cmd_summary(result: Any) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout": (result.stdout or "")[-4000:],
        "stderr": (result.stderr or "")[-4000:],
    }


def _validate_worktree_patch(repo_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(get_path(cfg, "agent.validate_generated_patch", True))
    result: dict[str, Any] = {"enabled": enabled, "ok": True}
    if not enabled:
        return result

    diff_check = run_cmd(["git", "diff", "--check"], cwd=repo_dir, timeout=120, check=False)
    result["git_diff_check"] = _cmd_summary(diff_check)
    if diff_check.returncode != 0:
        result["ok"] = False

    names = run_cmd(["git", "diff", "--name-only", "--diff-filter=ACMRT"], cwd=repo_dir, timeout=120, check=False)
    changed_files = [line.strip() for line in names.stdout.splitlines() if line.strip()]
    result["changed_files"] = changed_files

    py_files = [path for path in changed_files if path.endswith(".py") and (repo_dir / path).exists()]
    result["python_files"] = py_files
    if py_files and bool(get_path(cfg, "agent.py_compile_changed_python", True)):
        py_compile = run_cmd([sys.executable, "-m", "py_compile", *py_files], cwd=repo_dir, timeout=120, check=False)
        result["python_compile"] = _cmd_summary(py_compile)
        if py_compile.returncode != 0:
            result["ok"] = False

    return result


def _tracked_files(repo_dir: Path) -> list[str]:
    res = run_cmd(["git", "ls-files"], cwd=repo_dir, timeout=120, check=False)
    if res.returncode != 0:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def _paths_from_patch(patch: str) -> set[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        raw_paths: list[str] = []
        if line.startswith("diff --git "):
            raw_paths.extend(line.split()[2:4])
        elif line.startswith(("--- ", "+++ ")):
            raw_paths.append(line.split(maxsplit=1)[1])
        for raw in raw_paths:
            if raw in {"/dev/null", "a/dev/null", "b/dev/null"}:
                continue
            if raw.startswith(("a/", "b/")):
                raw = raw[2:]
            if raw:
                paths.add(raw)
    return paths


def _search_terms(*texts: str, limit: int = 40) -> list[str]:
    counts: dict[str, int] = {}
    for text in texts:
        for match in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text or ""):
            term = match.lower()
            if term in _COMMON_SEARCH_TERMS:
                continue
            counts[term] = counts.get(term, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _is_text_candidate(path: str) -> bool:
    if any(part in _SKIP_PATH_PARTS for part in Path(path).parts):
        return False
    suffix = Path(path).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return True
    return suffix == "" and "/" not in path


def _file_score(path: str, text: str, terms: list[str], forced_paths: set[str]) -> int:
    lowered_path = path.lower()
    lowered_text = text.lower()
    score = 0
    if path in forced_paths:
        score += 100
    if "/test" in lowered_path or lowered_path.startswith("test"):
        score += 8
    for term in terms:
        if term in lowered_path:
            score += 12
        score += min(6, lowered_text.count(term))
    return score


def _compact_file_text(path: str, text: str, terms: list[str], max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    lowered = text.lower()
    windows: list[tuple[int, int]] = []
    for term in terms[:20]:
        idx = lowered.find(term)
        if idx >= 0:
            windows.append((max(0, idx - max_chars // 8), min(len(text), idx + max_chars // 8)))
    if not windows:
        return text[:max_chars] + "\n...[truncated]\n"
    windows.sort()
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1] + 200:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    chunks: list[str] = []
    budget = max_chars
    for start, end in merged:
        chunk = text[start:end]
        if len(chunk) > budget:
            chunk = chunk[:budget]
        chunks.append(f"...[{path}:{start}]...\n{chunk}")
        budget -= len(chunk)
        if budget <= 0:
            break
    return "\n".join(chunks)


def _build_repo_context(row: dict[str, Any], cfg: dict[str, Any], repo_dir: Path) -> str:
    max_total = int(get_path(cfg, "agent.direct_patch_fallback_context_chars", 120000) or 120000)
    max_per_file = int(get_path(cfg, "agent.direct_patch_fallback_max_file_chars", 24000) or 24000)
    problem = str(row.get("problem_statement", ""))
    hints = str(row.get("hints_text", ""))
    test_patch = str(row.get("test_patch", ""))
    forced_paths = _paths_from_patch(test_patch)
    terms = _search_terms(problem, hints, test_patch)
    scored: list[tuple[int, str, str]] = []
    for path in _tracked_files(repo_dir):
        if not _is_text_candidate(path):
            continue
        full_path = repo_dir / path
        try:
            if full_path.stat().st_size > 300_000 and path not in forced_paths:
                continue
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        score = _file_score(path, text[:200_000], terms, forced_paths)
        if score > 0:
            scored.append((score, path, text))
    scored.sort(key=lambda item: (-item[0], item[1]))

    chunks = [
        "Relevant repository files selected by lexical overlap with the issue and test patch.",
        "If a test file is included, use it only to infer expected behavior; do not edit tests unless necessary.",
        "",
    ]
    used = 0
    for score, path, text in scored[:24]:
        body = _compact_file_text(path, text, terms, max_per_file)
        block = f"### {path} (score={score})\n```text\n{body}\n```\n"
        if used + len(block) > max_total:
            break
        chunks.append(block)
        used += len(block)
    if used == 0:
        chunks.append("No relevant text files were selected.\n")
    return "\n".join(chunks)


def _responses_endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/responses"


def _extract_response_text(obj: Any) -> str:
    if isinstance(obj, dict) and isinstance(obj.get("output_text"), str):
        return obj["output_text"]
    parts: list[str] = []
    outputs = obj.get("output") if isinstance(obj, dict) else None
    if isinstance(outputs, list):
        for item in outputs:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
            elif isinstance(content, str):
                parts.append(content)
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
    return "\n".join(part for part in parts if part)


def _call_responses_api(cfg: dict[str, Any], prompt: str) -> tuple[dict[str, Any], str]:
    payload = {
        "model": str(get_path(cfg, "model.served_model_name")),
        "input": prompt,
        "stream": False,
        "temperature": float(get_path(cfg, "agent.direct_patch_fallback_temperature", 0.0) or 0.0),
        "max_output_tokens": int(get_path(cfg, "agent.direct_patch_fallback_max_output_tokens", 4096) or 4096),
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _responses_endpoint(str(get_path(cfg, "codex.base_url"))),
        data=data,
        headers={
            "Authorization": f"Bearer {get_path(cfg, 'codex.api_key_value', 'dummy')}",
            "Content-Type": "application/json",
            "Connection": "close",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(get_path(cfg, "agent.direct_patch_fallback_timeout_s", 900) or 900)) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Responses API HTTP {e.code}: {raw[-4000:]}") from e
    parsed = json.loads(raw)
    return parsed, _extract_response_text(parsed)


def _extract_unified_diff(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fence_matches = re.findall(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fence_matches or [text]
    for candidate in candidates:
        candidate = candidate.strip()
        idx = candidate.find("diff --git ")
        if idx >= 0:
            return candidate[idx:].rstrip() + "\n"
    for candidate in candidates:
        candidate = candidate.strip()
        idx = candidate.find("--- ")
        if idx >= 0 and "\n+++ " in candidate[idx:]:
            return candidate[idx:].rstrip() + "\n"
    return ""


def _split_diff_by_file(diff: str) -> list[str]:
    lines = diff.splitlines(keepends=True)
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("diff --git ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return ["".join(block).rstrip() + "\n" for block in blocks if "".join(block).strip()]


def _diff_block_paths(block: str) -> set[str]:
    paths: set[str] = set()
    first = block.splitlines()[0] if block.splitlines() else ""
    if first.startswith("diff --git "):
        for raw in first.split()[2:4]:
            if raw.startswith(("a/", "b/")):
                raw = raw[2:]
            if raw and raw != "/dev/null":
                paths.add(raw)
    for line in block.splitlines():
        if line.startswith(("--- ", "+++ ")):
            raw = line.split(maxsplit=1)[1]
            if raw.startswith(("a/", "b/")):
                raw = raw[2:]
            if raw and raw not in {"/dev/null", "dev/null"}:
                paths.add(raw)
    return paths


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


def _apply_candidate_diff(repo_dir: Path, diff: str, cfg: dict[str, Any]) -> tuple[bool, str]:
    apply_cmd = ["git", "apply", "--recount", "--whitespace=fix", "-"]
    allow_tests = bool(get_path(cfg, "agent.direct_patch_fallback_allow_test_edits", False))
    applied = 0
    logs: list[str] = []
    for index, block in enumerate(_split_diff_by_file(diff), start=1):
        paths = _diff_block_paths(block)
        label = ",".join(sorted(paths)) or f"block_{index}"
        if paths and not allow_tests and all(_is_test_path(path) for path in paths):
            logs.append(f"SKIP {label}: test-only patch block")
            continue
        check = run_cmd([*apply_cmd[:2], "--check", *apply_cmd[2:]], cwd=repo_dir, input_text=block, timeout=120, check=False)
        if check.returncode != 0:
            logs.append(f"FAIL {label}: {(check.stdout + chr(10) + check.stderr).strip()}")
            continue
        apply_res = run_cmd(apply_cmd, cwd=repo_dir, input_text=block, timeout=120, check=False)
        apply_log = (apply_res.stdout + "\n" + apply_res.stderr).strip()
        if apply_res.returncode == 0:
            applied += 1
            logs.append(f"APPLY {label}: ok")
        else:
            logs.append(f"FAIL {label}: {apply_log or f'git apply failed with code {apply_res.returncode}'}")
    return applied > 0, "\n".join(logs)


def _build_direct_patch_prompt(row: dict[str, Any], cfg: dict[str, Any], repo_dir: Path, prior_error: str = "") -> str:
    repo_context = _build_repo_context(row, cfg, repo_dir)
    retry_section = ""
    if prior_error:
        retry_section = (
            "\nPrevious candidate patch failed to apply. Produce a corrected unified diff.\n"
            f"git apply error:\n```\n{prior_error[-4000:]}\n```\n"
        )
    return f"""You are generating a SWE-bench benchmark prediction for a local repository checkout.
Return only a unified diff that can be applied with `git apply --whitespace=fix -`.
Do not include Markdown fences, explanations, or test edits unless the issue explicitly requires changing tests.
Keep the patch small and source-focused.
/no_think

Repository: {row.get("repo", "")}
Base commit: {row.get("base_commit", "")}
Instance ID: {row.get("instance_id", "")}

Issue statement:

{row.get("problem_statement", "")}

Hints:

{row.get("hints_text", "")}

Test patch for expected behavior only. Do not apply it:

{row.get("test_patch", "")}
{retry_section}
Repository context:

{repo_context}
"""


def _run_direct_patch_fallback(row: dict[str, Any], cfg: dict[str, Any], repo_dir: Path, inst_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"enabled": bool(get_path(cfg, "agent.direct_patch_fallback", False)), "used": False, "applied": False}
    if not info["enabled"]:
        return info
    info["used"] = True
    attempts = int(get_path(cfg, "agent.direct_patch_fallback_attempts", 2) or 1)
    prior_error = ""
    for attempt in range(1, attempts + 1):
        prompt = _build_direct_patch_prompt(row, cfg, repo_dir, prior_error)
        (inst_dir / f"direct_patch_prompt_attempt{attempt}.md").write_text(prompt, encoding="utf-8")
        try:
            response_json, response_text = _call_responses_api(cfg, prompt)
        except Exception as e:
            prior_error = str(e)
            (inst_dir / f"direct_patch_error_attempt{attempt}.txt").write_text(prior_error, encoding="utf-8")
            continue
        write_json(inst_dir / f"direct_patch_response_attempt{attempt}.json", response_json)
        (inst_dir / f"direct_patch_text_attempt{attempt}.txt").write_text(response_text, encoding="utf-8")
        diff = _extract_unified_diff(response_text)
        (inst_dir / f"direct_patch_candidate_attempt{attempt}.diff").write_text(diff, encoding="utf-8")
        if not diff:
            prior_error = "The model response did not contain a unified diff."
            (inst_dir / f"direct_patch_apply_attempt{attempt}.log").write_text(prior_error + "\n", encoding="utf-8")
            continue
        applied, apply_log = _apply_candidate_diff(repo_dir, diff, cfg)
        (inst_dir / f"direct_patch_apply_attempt{attempt}.log").write_text(apply_log + "\n", encoding="utf-8")
        applied_patch = _get_patch(repo_dir) if applied else ""
        if applied and applied_patch:
            info.update(
                {
                    "applied": True,
                    "attempts": attempt,
                    "patch_candidate_bytes": len(diff.encode("utf-8")),
                    "patch_candidate_sha256": sha256_text(diff),
                }
            )
            return info
        prior_error = apply_log or "No source patch blocks applied."
        if applied and not applied_patch:
            prior_error += "\nApplied candidate produced no git diff."
    info.update({"attempts": attempts, "error": prior_error[-4000:]})
    return info


def _codex_cmd(cfg: dict[str, Any], repo_dir: Path, prompt: str, final_path: Path) -> list[str]:
    exe = str(get_path(cfg, "codex.executable", "codex"))
    model = str(get_path(cfg, "model.served_model_name"))
    cmd = [exe, "exec", "-m", model, "-C", str(repo_dir), "--output-last-message", str(final_path)]
    extra = get_path(cfg, "codex.extra_args", []) or []
    if isinstance(extra, str):
        extra = extra.split()
    cmd.extend(str(x) for x in extra)
    # Keep this after extra args so the run manifest shows effective sandbox explicitly.
    sandbox = get_path(cfg, "codex.sandbox_mode")
    if sandbox and "--sandbox" not in cmd and "--full-auto" not in cmd:
        cmd.extend(["--sandbox", str(sandbox)])
    cmd.append(prompt)
    return cmd


def _minimal_codex_cmd(cfg: dict[str, Any], repo_dir: Path, prompt: str, final_path: Path) -> list[str]:
    exe = str(get_path(cfg, "codex.executable", "codex"))
    model = str(get_path(cfg, "model.served_model_name"))
    # Fallback for Codex CLI versions whose exec subcommand rejects some global flags.
    return [exe, "exec", "-m", model, "-C", str(repo_dir), "--json", "--output-last-message", str(final_path), prompt]


def _build_codex_retry_prompt(base_prompt: str, reason: str, attempt: int) -> str:
    return (
        base_prompt.rstrip()
        + f"""

Retry attempt {attempt}:
The previous Codex attempt did not leave a valid source patch in the git worktree.
Harness validation result:

{reason[-3000:]}

You must now make the source edit with a shell command before any final answer.
Do not describe the intended patch without editing a file.
Do not write "I will run" or "let me run"; run the command instead.
Use a Python heredoc for file edits; do not use `sed -i`.
If inserting source lines, use `splitlines(keepends=True)` plus an `out` list.
Do not put a newline inside a single-quoted or double-quoted Python string.
Do not use `python -c` for source edits.
Run `git diff --stat` after editing and only finish if it is non-empty.
"""
    )


def _reset_generated_worktree(repo_dir: Path) -> None:
    run_cmd(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, timeout=120, check=False)
    run_cmd(["git", "clean", "-fd"], cwd=repo_dir, timeout=120, check=False)


def _format_retry_reason(rc: int | None, patch: str, patch_validation: dict[str, Any], final_path: Path | None = None) -> str:
    if not patch:
        chunks = [f"codex_returncode={rc}; git diff --binary was empty."]
    elif patch_validation.get("ok", True):
        chunks = [f"codex_returncode={rc}; patch was valid."]
    else:
        chunks = [f"codex_returncode={rc}; generated patch failed local validation."]
    for key in ("git_diff_check", "python_compile"):
        value = patch_validation.get(key)
        if isinstance(value, dict):
            stdout = value.get("stdout") or ""
            stderr = value.get("stderr") or ""
            chunks.append(f"{key} rc={value.get('returncode')}\n{stdout}\n{stderr}".strip())
    if final_path is not None and final_path.exists():
        final_text = final_path.read_text(encoding="utf-8", errors="replace").strip()
        if final_text:
            chunks.append("last_final_message_excerpt:\n" + final_text[-2000:])
    return "\n\n".join(chunks)


def _path_for_attempt(path: Path, attempt: int) -> Path:
    if attempt == 1:
        return path
    return path.with_name(f"{path.stem}_attempt{attempt}{path.suffix}")


def _run_codex_command(
    cfg: dict[str, Any],
    repo_dir: Path,
    prompt: str,
    final_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
    attempt: int,
) -> int:
    cmd = _codex_cmd(cfg, repo_dir, prompt, final_path)
    if attempt == 1:
        manifest["codex_command"] = quote_cmd([*cmd[:-1], "<PROMPT>"])
    else:
        manifest.setdefault("codex_retry_commands", []).append(quote_cmd([*cmd[:-1], "<PROMPT>"]))
    try:
        res = run_cmd(
            cmd,
            cwd=repo_dir,
            env=env,
            timeout=float(get_path(cfg, "codex.timeout_s_per_instance", 3600)),
            check=False,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        rc = res.returncode
        stderr_txt = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        if rc != 0 and any(s in stderr_txt.lower() for s in ["unexpected argument", "unknown option", "unrecognized option"]):
            retry_cmd = _minimal_codex_cmd(cfg, repo_dir, prompt, final_path)
            if attempt == 1:
                manifest["codex_command_retry"] = quote_cmd([*retry_cmd[:-1], "<PROMPT>"])
            else:
                manifest.setdefault("codex_retry_commands", []).append(quote_cmd([*retry_cmd[:-1], "<PROMPT>"]))
            res = run_cmd(
                retry_cmd,
                cwd=repo_dir,
                env=env,
                timeout=float(get_path(cfg, "codex.timeout_s_per_instance", 3600)),
                check=False,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            rc = res.returncode
    except subprocess.TimeoutExpired:
        rc = -124
        stderr_path.write_text("Codex run timed out.\n", encoding="utf-8")
    return rc


def _run_codex_instance(row: dict[str, Any], cfg: dict[str, Any], run_dir: Path, predictions_path: Path, resume: bool, dry_run: bool) -> dict[str, Any]:
    iid = str(row["instance_id"])
    inst_dir = run_dir / "instances" / safe_instance_id(iid)
    inst_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = inst_dir / "manifest.json"
    if resume and manifest_path.exists() and (inst_dir / "patch.diff").exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {**existing, "resumed": True}

    write_json(inst_dir / "swebench_instance.json", row)
    start = time.time()
    manifest: dict[str, Any] = {
        "instance_id": iid,
        "repo": row.get("repo"),
        "base_commit": row.get("base_commit"),
        "started_at": now_iso(),
        "stage": get_path(cfg, "stage.name"),
        "infer_backend": get_path(cfg, "stage.infer_backend"),
        "model": get_path(cfg, "model.served_model_name"),
        "model_path": get_path(cfg, "model.model_path"),
        "status": "started",
    }
    write_json(manifest_path, manifest)

    try:
        repo_dir = _prepare_repo(row, cfg, inst_dir)
        base_prompt = _render_prompt(row, cfg)
        (inst_dir / "prompt.md").write_text(base_prompt, encoding="utf-8")
        codex_home = inst_dir / "codex_home"
        _render_codex_config(cfg, codex_home)
        final_path = inst_dir / "final_message.txt"
        stdout_path = inst_dir / "codex_events.ndjson"
        stderr_path = inst_dir / "codex_stderr.log"
        env_key = str(get_path(cfg, "codex.env_key", "VLLM_API_KEY"))
        env = {
            "CODEX_HOME": str(codex_home),
            env_key: str(get_path(cfg, "codex.api_key_value", "dummy")),
            "NO_COLOR": "1",
        }
        manifest.update({"prompt_sha256": sha256_text(base_prompt)})
        write_json(manifest_path, manifest)
        if dry_run:
            cmd = _codex_cmd(cfg, repo_dir, base_prompt, final_path)
            manifest["codex_command"] = quote_cmd([*cmd[:-1], "<PROMPT>"])
            (inst_dir / "DRY_RUN.txt").write_text(quote_cmd([*cmd[:-1], "<PROMPT>"]) + "\n", encoding="utf-8")
            rc = 0
        else:
            rc = None
            patch = ""
            patch_validation: dict[str, Any] = {"enabled": bool(get_path(cfg, "agent.validate_generated_patch", True)), "ok": True}
            codex_attempts: list[dict[str, Any]] = []
            retry_reason = ""
            max_attempts = max(1, int(get_path(cfg, "agent.codex_generation_attempts", 1) or 1))
            for attempt in range(1, max_attempts + 1):
                prompt = base_prompt if attempt == 1 else _build_codex_retry_prompt(base_prompt, retry_reason, attempt)
                if attempt > 1:
                    (inst_dir / f"prompt_attempt{attempt}.md").write_text(prompt, encoding="utf-8")
                attempt_final = _path_for_attempt(final_path, attempt)
                attempt_stdout = _path_for_attempt(stdout_path, attempt)
                attempt_stderr = _path_for_attempt(stderr_path, attempt)
                rc = _run_codex_command(cfg, repo_dir, prompt, attempt_final, attempt_stdout, attempt_stderr, env, manifest, attempt)
                write_json(manifest_path, manifest)
                patch = _get_patch(repo_dir)
                patch_validation = _validate_worktree_patch(repo_dir, cfg) if patch else {"enabled": bool(get_path(cfg, "agent.validate_generated_patch", True)), "ok": True}
                (inst_dir / f"patch_attempt{attempt}.diff").write_text(patch, encoding="utf-8")
                write_json(inst_dir / f"patch_validation_attempt{attempt}.json", patch_validation)
                attempt_record = {
                    "attempt": attempt,
                    "codex_returncode": rc,
                    "patch_bytes": len(patch.encode("utf-8")),
                    "patch_sha256": sha256_text(patch),
                    "patch_validation": patch_validation,
                    "final_message": str(attempt_final),
                    "events": str(attempt_stdout),
                    "stderr": str(attempt_stderr),
                }
                codex_attempts.append(attempt_record)
                if attempt > 1 and attempt_final.exists():
                    shutil.copyfile(attempt_final, final_path)
                if attempt > 1 and attempt_stdout.exists():
                    with stdout_path.open("a", encoding="utf-8") as out, attempt_stdout.open(encoding="utf-8", errors="replace") as src:
                        out.write(src.read())
                if patch and patch_validation.get("ok", True):
                    break
                retry_reason = _format_retry_reason(rc, patch, patch_validation, attempt_final)
                if attempt < max_attempts:
                    _reset_generated_worktree(repo_dir)
            manifest["codex_attempts"] = codex_attempts
        if dry_run:
            patch = _get_patch(repo_dir)
            patch_validation = {"enabled": bool(get_path(cfg, "agent.validate_generated_patch", True)), "ok": True}
        patch_source = "git_worktree"
        direct_patch_fallback = {
            "enabled": bool(get_path(cfg, "agent.allow_model_output_patch_fallback", False)),
            "used": False,
            "applied": False,
        }
        if direct_patch_fallback["enabled"] and not patch and not dry_run:
            direct_patch_fallback = _run_direct_patch_fallback(row, cfg, repo_dir, inst_dir)
            if direct_patch_fallback.get("applied"):
                patch = _get_patch(repo_dir)
                patch_source = "model_output_diff_applied_to_worktree"
        if patch and not dry_run and not manifest.get("codex_attempts"):
            patch_validation = _validate_worktree_patch(repo_dir, cfg)
        write_json(inst_dir / "patch_validation.json", patch_validation)
        (inst_dir / "patch.diff").write_text(patch, encoding="utf-8")
        run_cmd(["git", "status", "--short"], cwd=repo_dir, timeout=60, check=False, stdout_path=inst_dir / "repo_status_final.txt")
        if stdout_path.exists():
            parse_events(stdout_path, inst_dir / "observability", keep_raw=bool(get_path(cfg, "observability.module_timeline_keep_raw_event", True)))
        pred = {
            "instance_id": iid,
            "model_name_or_path": str(get_path(cfg, "model.served_model_name")),
            "model_patch": patch,
        }
        append_jsonl(predictions_path, pred)
        if direct_patch_fallback.get("applied") and patch:
            status = "ok_direct_patch_fallback" if rc == 0 else "codex_nonzero_direct_patch_fallback"
        elif direct_patch_fallback.get("used") and not patch:
            status = "ok_zero_patch_fallback_failed" if rc == 0 else "codex_nonzero_zero_patch_fallback_failed"
        elif patch and not patch_validation.get("ok", True):
            status = "patch_validation_failed" if rc == 0 else "codex_nonzero_patch_validation_failed"
        else:
            status = "ok" if rc == 0 and patch else ("zero_patch_git_worktree" if rc == 0 else "codex_nonzero")
        manifest.update(
            {
                "status": status,
                "codex_returncode": rc,
                "patch_source": patch_source,
                "direct_patch_fallback": direct_patch_fallback,
                "patch_validation": patch_validation,
                "patch_bytes": len(patch.encode("utf-8")),
                "patch_sha256": sha256_text(patch),
                "finished_at": now_iso(),
                "duration_s": round(time.time() - start, 3),
                "prediction_file": str(predictions_path),
            }
        )
    except Exception as e:
        manifest.update(
            {
                "status": "error",
                "error_type": type(e).__name__,
                "error": str(e)[-4000:],
                "finished_at": now_iso(),
                "duration_s": round(time.time() - start, 3),
            }
        )
    write_json(manifest_path, manifest)
    append_jsonl(run_dir / "records" / "instance_manifests.jsonl", manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SWE-bench predictions with Codex CLI and a local OpenAI-compatible backend.")
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--stage-config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--instance-ids", default=None, help="Comma-separated IDs or a file containing one ID per line.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.stage_config)
    ensure_dirs(cfg)
    if args.dataset_name:
        cfg.setdefault("swebench", {})["dataset_name"] = args.dataset_name
    if args.split:
        cfg.setdefault("swebench", {})["split"] = args.split
    if args.max_instances is not None:
        cfg.setdefault("swebench", {})["max_instances"] = args.max_instances

    run_id = args.run_id or f"{get_path(cfg, 'stage.name')}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(get_path(cfg, "project.runs_root")) / run_id
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "records" / "resolved_config.json", cfg)
    write_runtime_snapshot(run_dir / "records" / "runtime_snapshot.json", cfg)

    selected_ids = _parse_instance_ids(args.instance_ids)
    rows = _load_dataset_rows(
        str(get_path(cfg, "swebench.dataset_name")),
        str(get_path(cfg, "swebench.split", "test")),
        get_path(cfg, "swebench.max_instances"),
        selected_ids,
    )
    with (run_dir / "records" / "dataset_subset.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    predictions_path = run_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        predictions_path.unlink()

    summaries = []
    for row in rows:
        summaries.append(_run_codex_instance(row, cfg, run_dir, predictions_path, args.resume, args.dry_run))
        print(json.dumps(summaries[-1], ensure_ascii=False, sort_keys=True), flush=True)
    write_json(run_dir / "records" / "run_summary_raw.json", {"run_id": run_id, "instances": summaries})
    print(f"RUN_DIR={run_dir}")
    print(f"PREDICTIONS={predictions_path}")


if __name__ == "__main__":
    main()
