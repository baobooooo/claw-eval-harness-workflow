# 可观测性设计

目标是把 agent workflow trajectory、模型服务请求、KV cache/prefix cache、GPU 状态和 SWE-bench 结果对齐到同一个 `run_id` 下。

## 1. Codex 事件

`codex exec --json` 的 stdout 保存到：

```text
runs/<run_id>/instances/<instance_id>/codex_events.ndjson
```

解析结果：

```text
runs/<run_id>/instances/<instance_id>/observability/module_timeline.jsonl
runs/<run_id>/instances/<instance_id>/observability/tool_events.jsonl
runs/<run_id>/instances/<instance_id>/observability/agent_messages.jsonl
runs/<run_id>/instances/<instance_id>/observability/codex_event_stats.json
```

解析器是保守启发式：Codex event schema 如果升级，原始 `codex_events.ndjson` 仍然保留，不会丢信息。

## 2. vLLM 指标

`monitor_metrics.py` 周期抓取 `/metrics`：

```text
runs/<run_id>/metrics/metrics_selected.jsonl
runs/<run_id>/metrics/raw_prometheus/metrics_*.prom
runs/<run_id>/metrics/nvidia_smi.jsonl
```

建议重点看：

- prefix cache hit rate / hit tokens
- KV cache usage / residency
- prompt tokens、generation tokens
- time to first token、prefill/decode latency
- request queue time

不同 vLLM 版本 metric 名称会变化，所以默认保存 raw Prometheus 文件。

## 3. KV events

Stage1 默认启用 ZMQ KV events：

```bash
--kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'
```

采集文件：

```text
runs/<run_id>/kv_events.jsonl
```

为避免反序列化格式升级造成损失，采集器保存：

- multipart 原始 payload 的 base64
- UTF-8 preview
- timestamp / seq / endpoint

## 4. Sparse-vLLM 观测

Stage2 会额外保存 Sparse-vLLM 启动命令和 request logs。若 server 暴露 prefix-cache debug endpoint，可运行：

```bash
python -m swecodex_harness.sparsevllm_cache_probe \
  --base-url http://127.0.0.1:8001/v1 \
  --out runs/<run_id>/metrics/sparsevllm_prefix_cache_probe.json
```

## 5. 时间对齐

每个文件至少保留 `seq` 和 UTC `timestamp`。后续分析建议以以下 join key 对齐：

- `run_id`
- `instance_id`
- `manifest.started_at / finished_at`
- Codex event seq
- metrics scrape seq
- KV event seq

## 6. 后处理原则

不要只保存聚合值。至少保留 raw event / raw metrics / raw patch。聚合 summary 可以重算，但 agent trajectory 和 KV block event 一旦丢失，无法从 SWE-bench report 恢复。
