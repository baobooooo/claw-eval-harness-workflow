# Stage2: Sparse-vLLM 注意事项

Stage2 目标是把 Stage1 的 Codex + SWE-bench workflow 切到 `CURRENTF/Sparse-vLLM` 的 `codex/import-deltakv-main` 分支。

## 关键兼容点

Codex CLI 的本地模型接入路径使用 OpenAI-compatible API。Stage1 使用 vLLM 的 `/v1/responses` 路径最直接。Sparse-vLLM 该分支提供 `sparsevllm-openai-server` 入口，但当前项目包不会默认假设它已经完整实现 Codex 所需的 Responses API + tool-call loop。

因此 Stage2 runner 先做：

```bash
python -m swecodex_harness.preflight_codex_api \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen3-30b-a3b
```

如果 `/v1/responses` 失败，`scripts/run_stage2_smoke.sh` 会停止，而不是产生不可比较的实验结果。

## 两种继续方式

### A. 推荐：在 Sparse-vLLM server 原生补齐 `/v1/responses`

这样 Stage2 和 Stage1 的 Codex workflow 最可比。需要保证：

- `/v1/responses` 非流式/流式都能工作；
- tool call 的输入输出结构与 Codex CLI 兼容；
- reasoning parser/tool parser 对 Qwen3 生效；
- request id/session id 可进入 request log；
- cache/prefill/sparse method 指标可按 request 关联。

### B. 仅用于 smoke：实验性 bridge

包里有一个简化 bridge：

```bash
python -m swecodex_harness.responses_to_chat_bridge \
  --target-base-url http://127.0.0.1:8001/v1 \
  --port 8011
```

然后把 `configs/stage2_sparse_vllm.yaml` 的 `codex.base_url` 改成：

```yaml
codex:
  base_url: http://127.0.0.1:8011/v1
```

该 bridge 只适合检查 Codex 能否启动，不应作为正式 SWE-bench 分数，因为它没有完整实现 Responses API 的工具调用语义。

## Sparse 方法记录

每次 Stage2 实验记录必须包含：

- `sparse_method`
- prefill policy
- `engine_prefill_chunk_size`
- `max_num_batched_tokens`
- prompt length
- batch size / decoding seqs
- `sink_keep_tokens`, `recent_keep_tokens`, `decode_keep_tokens`
- `full_attention_layers`
- DeltaKV checkpoint path / latent dim / quant bits
- CUDA graph 相关设置

这些字段写在 `records/resolved_config.json` 和 `EXPERIMENT_RECORD.md`，不要只写在启动命令里。
