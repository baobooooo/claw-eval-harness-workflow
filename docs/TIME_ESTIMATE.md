# 测试时间预估方法

真实耗时主要取决于：

- Codex 每个实例的循环次数；
- 是否让 Codex 在 repo 内执行测试；
- vLLM/Sparse-vLLM 的 TTFT、decode throughput、prefix cache hit；
- SWE-bench Docker image/env 是否已经缓存；
- `max_workers` 和磁盘 IO。

## 推荐流程

先跑 3-10 个实例 pilot：

```bash
MAX_INSTANCES=5 bash scripts/run_stage1_smoke.sh
python -m swecodex_harness.estimate_time --run-dir runs/<run_id> --target-instances 300
```

脚本会用 pilot 的实际平均 generation time 估算总耗时。没有 pilot 数据时，脚本默认用 20 分钟/实例作为占位启发值；这不是性能承诺。

## 手工估算公式

```text
generation_hours = mean_generation_seconds_per_instance * N / 3600
eval_wall_hours_before_parallelism = mean_eval_seconds_per_instance * N / 3600
eval_wall_hours_after_parallelism ≈ eval_wall_hours_before_parallelism / max_workers
```

SWE-bench eval 的 Docker env/image 首次构建可能显著慢于后续运行。建议把首次冷启动和二次热启动分开记录。

## 1 张 H100 的经验性预期区间

在 agentic workflow 中，耗时往往不是纯 decode throughput 决定，而是多轮工具调用、repo 检索、测试执行和长 prompt prefill 共同决定。建议用以下区间做调度预留：

- smoke 3 个实例：用于检查 API、Codex event、patch、SWE-bench eval 是否打通；
- pilot 10 个实例：用于估算平均每实例 generation time 和 KV/cache 指标；
- SWE-bench Lite 300 个实例：用 pilot 均值外推，并为失败重跑和 Docker cache miss 预留额外时间。

最终报告里必须写明：pilot instance IDs、均值、中位数、最长实例、是否 cold Docker cache。
