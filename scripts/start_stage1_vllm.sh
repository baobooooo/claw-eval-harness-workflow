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

TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-$(cfg_value vllm.tool_call_parser "$STAGE1_CONFIG")}"
exec vllm serve \
  --config "$STAGE1_CONFIG" \
  --tool-call-parser "$TOOL_CALL_PARSER"
