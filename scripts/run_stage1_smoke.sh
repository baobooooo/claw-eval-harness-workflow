#!/usr/bin/env bash
set -euo pipefail
STAGE1_CONFIG="${STAGE1_CONFIG:-configs/stage1_vllm.yaml}"

cfg_value() {
  python - "$1" "$2" <<'PY'
import sys, yaml
key, path = sys.argv[1], sys.argv[2]
data = yaml.safe_load(open(path, encoding='utf-8')) or {}
cur = data
for part in key.split('.'):
    cur = cur.get(part, {}) if isinstance(cur, dict) else {}
print(cur if cur != {} else "")
PY
}

BASE_URL="${BASE_URL:-$(cfg_value codex.base_url "$STAGE1_CONFIG")}"
codex exec --base-url "$BASE_URL" "$@"
