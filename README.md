Release 页面：[https://github.com/baobooooo/claw-eval-harness-workflow/releases/tag/workflow-traces-20260715](https://github.com/baobooooo/claw-eval-harness-workflow/releases/tag/workflow-traces-20260715)

Judge 分数在仓库内这个文件：

`judge_results/external_grade_results.jsonl`

Raw 链接：[https://raw.githubusercontent.com/baobooooo/claw-eval-harness-workflow/main/judge_results/external_grade_results.jsonl](https://raw.githubusercontent.com/baobooooo/claw-eval-harness-workflow/main/judge_results/external_grade_results.jsonl)

总体结论

这 161 个任务的结果说明：不存在一个在所有任务类型上都占优的 Harness。三者形成了很明显的能力分工：

Harness	平均分	Pass rate	平均耗时	中位首次工具时间	平均工具调用
Codex	0.716	60.9%	278 秒	50 秒	17.7
NanoBot	0.735	63.4%	379 秒	87 秒	28.7
OpenClaw	0.708	58.4%	472 秒	333 秒	13.7

但这个总表容易误导。只看 status=ok 的执行：

Harness	OK 数量	OK 条件平均分	OK 条件 Pass rate
Codex	128	0.732	63.3%
NanoBot	145	0.780	70.3%
OpenClaw	134	0.784	68.7%

也就是说：

Codex 的核心优势是执行效率和稳定落地。
NanoBot 的核心优势是轻量而持续的研究/上下文循环。
OpenClaw 的核心优势是复杂、多服务、有依赖关系的工作流编排。
OpenClaw 总分不高，主要不是其复杂任务能力弱，而是冷启动、上下文装配、模型 idle timeout 和最终回复交付问题拉低了结果。

三个总体平均分的差异其实很小。按任务配对 bootstrap，NanoBot 相对 Codex 的均分差为 +0.0186，95% 区间仍跨过 0；因此不能仅凭这次 161 个任务宣布某个 Harness 全局显著更强。更可靠的结论是：它们在不同任务结构上有不同优势。

还需要说明：这是上一版运行结果，其中有 26 条 Codex 和 3 条 OpenClaw 被标记为 tool_policy_violation，主要来自旧版 hidden transport history 泄漏。因此，这批结果适合分析工作流倾向，不适合作为最终公平榜单。

一、Codex：关键提升来自“低开销执行环”

你当前 Codex adapter 的核心路径是：

task prompt
    ↓
codex exec
    ↓
Codex 自己的 agent/rollout loop
    ↓
model_tool_proxy 替换模型可见 tools
    ↓
exec_command 执行 hidden transport
    ↓
--output-last-message 提取最终回复

Codex 本身是 terminal-oriented agent，官方代码结构中包含 exec、rollout、tools、message-history、thread-store、memories、skills 等模块；CLI 也明确支持通过 codex exec 进入非交互自动化工作流。

带来提升的模块
1. exec / rollout / tool loop

这是 Codex 最重要的模块。

它倾向于：

快速把任务变成可执行步骤；
并发执行相互独立的只读调用；
对命令失败进行局部恢复；
不构建过重的长期人格、技能和 memory 上下文。

T047 中，Codex 第一批四个 web_search 几乎同时发出，说明其执行环能把多个独立研究子问题并行化。

这解释了：

中位首次工具调用仅约 50 秒；
平均耗时三个 Harness 中最低；
score / wall time 最好；
单动作任务和明确 side-effect 任务表现较强。

对于恰好有一个 expected_action 的 31 个任务：

Codex:    score 0.806，pass 74.2%
NanoBot:  score 0.789，pass 64.5%
OpenClaw: score 0.719，pass 51.6%
2. --output-last-message

这是一个被低估但非常关键的模块。

Codex 有明确的“最后 assistant message”输出通道，而不是让 adapter 从 stdout 中猜最终回复。这降低了：

CLI 日志污染；
JSON envelope 污染；
启动信息进入 judge；
截取错消息的概率。

NanoBot 和 OpenClaw 当前路径都还没有做到同等级的 native final extraction。

3. 简单的 per-task 状态边界

Codex 每个 task 使用独立 CODEX_HOME 和 driver workspace，但没有 OpenClaw 那么重的 bootstrap/runtime/plugin 装配，因此一次性任务的固定成本相对较低。

Codex 的主要损失
1. 缺少通用的“需求覆盖表”

Codex 很善于执行，但容易在已经搜到大量信息后继续扩展内容，而没有始终检查：

用户要求的每个部分是否都已经覆盖？
最终答案是否明确包含结论？
是否把关键内容真正放进 final message？

T047 的 judge 中，推荐部分占 0.28 权重；参考流程也明确要求最后综合出结构化报告和建议。 Codex 已收集大量材料，但最终被 judge 认为报告在 recommendation 前截断，说明问题不在 retrieval，而在 finalization。

2. Tool schema 错误恢复过于工程化

T047 中 Codex 多次尝试 Write，因为参数名和 sandbox workspace 映射不一致，随后又尝试 Bash、路径探测、写入测试。这些调用没有提升内容质量，反而消耗时间。

3. 并发缺少预算控制

并发检索是优势，但也可能造成：

同义 query 重复；
session 搜索额度过早耗尽；
多个返回内容高度重叠；
上下文迅速膨胀。
Codex 最值得增加的模块
RequirementLedger
FinalAnswerValidator
ToolErrorClassifier
BoundedParallelScheduler

重点不是增加长期 memory，而是：

从用户 prompt 提取 required sections；
每次工具调用后更新覆盖状态；
最多并发 3–4 个独立只读调用；
写入工具 schema 错误只修正一次；
剩余 25% 时间时强制进入 finalization；
最终回复必须包含完整交付物，而不是仅说文件已生成。
二、NanoBot：关键提升来自轻量的 Loop/Runner 分层与工作区状态

NanoBot 官方架构把运行时拆成：

MessageBus
    ↓
AgentLoop：session / workspace / context
    ↓
AgentRunner：provider / tool conversation loop
    ↓
Tools

其中：

AgentLoop 管 session、workspace、context 和 outbound；
AgentRunner 管模型请求、streaming、tool call、tool result 和停止条件；
session/manager.py 管 session 与 compaction；
agent/memory.py 管长期 memory 和 Dream。

NanoBot 还明确区分近程 session history 与长期 workspace memory，并通过 MEMORY.md、history.jsonl、SOUL.md、USER.md 等文件保存状态。

带来提升的模块
1. AgentLoop / AgentRunner 清晰分层

这是 NanoBot 最关键的架构优势。

相比重型编排器，它的控制路径短：

build context
→ model
→ tools
→ results
→ model

在 easy/medium 任务上表现尤其好：

难度	Codex	NanoBot	OpenClaw
Easy 平均分	0.729	0.772	0.694
Medium 平均分	0.755	0.800	0.787
Hard 平均分	0.652	0.603	0.646

它不需要像 OpenClaw 那样先构建庞大的插件、技能和 bootstrap 世界，因此在中等复杂度任务上取得了最高质量。

2. Workspace artifact 模块

NanoBot 很自然地把长报告、计划或结果写到 workspace。T047 中它保存了一份完整报告，并返回摘要。该样本要求的是完整迁移技术评估报告，且明确规定只做调研、不发送通知。

完整 merged judge 中，T047 的分数是：

NanoBot:  0.960
Codex:    0.764
OpenClaw: 0.600

NanoBot 在该任务上的优势主要来自：

研究覆盖充分；
写出了完整结构化产物；
最终有明确迁移建议；
没有调用 send_notification。
3. Session 和短期 memory

对于十几到几十轮的研究任务，session history 能帮助模型记住：

已搜索哪些方向；
已找到哪些事实；
最后需要生成什么文件；
哪些工具已失败。

这对 easy/medium research 和 finance 任务有明显价值。

NanoBot 的主要损失
1. 缺少全局 stopping policy

NanoBot 最大的问题不是不会做，而是不知道什么时候该停。

工具调用分布：

中位数: 10
P90:    63
P95:    125
最大值: 396

这说明大多数任务很正常，但有一批任务会进入严重的长尾循环。

在 hard 任务上：

NanoBot 平均调用: 63.5
Codex 平均调用:   23.4
OpenClaw 平均调用: 16.0

例如 T164_quarterly_customer_review 中 NanoBot 调用了 396 次工具，最终只得 0.20。它不是信息不足，而是：

不断找更多信息
→ 上下文越来越大
→ 重复搜索/重复读取
→ 时间预算被耗尽
→ 没有足够时间做最终综合

任务配对后也能观察到：NanoBot 相对 Codex 多调用的工具越多，NanoBot 相对分数通常越低，相关系数约为 -0.31。

2. 长期 memory 模块在单次 benchmark 中收益有限

Dream、heartbeat、长期 MEMORY.md 对长期个人代理有价值，但对每个 instance 都是全新、隔离的一次性 ClawEval 任务时：

很难产生跨轮长期收益；
会带来初始化成本；
可能制造无关上下文；
还可能污染 final stdout。
3. stdout 与 final message 没有分离干净

当前 adapter 使用：

nanobot agent ... | tee final_message.txt trace_file

因此 final_message.txt 里不仅有 agent 回复，还可能有：

“Using config”；
创建 HEARTBEAT/AGENTS/SOUL 等启动日志；
mascot/header；
过程性状态。

这会影响 judge 的通信质量和最终文本提取。

NanoBot 最值得增加的模块
CoverageLedger
DuplicateQueryCache
AdaptiveStopController
NativeFinalParser
OneShotMemoryPolicy

建议的停止条件：

所有用户要求均已有证据
且连续两轮没有新增有效事实
→ 立即进入 finalization

硬预算可以从下面的初始值做 ablation：

search query 重复次数: 0
同一 URL 重复 fetch: 最多 1 次
同一错误签名重试: 最多 1 次
只读工具并发: 3
总工具调用 soft cap: 30–40
hard multi-service 上限: 60
至少预留: 2 个 model turn + 25% wall time
三、OpenClaw：关键提升来自 Context Engine、Session/Queue 和复杂依赖管理

OpenClaw 的核心运行路径比另外两者更重：

session resolution
→ workspace/bootstrap
→ skills snapshot
→ context engine
→ session lock / queue
→ embedded agent loop
→ model / tools
→ reply shaping
→ persistence / compaction

官方文档显示，其 agent loop 是 per-session 串行运行，包含 context assembly、model inference、tool execution、streaming 和 persistence；运行前还要加载 skills、bootstrap/context files 并准备 SessionManager。

Context Engine 负责决定：

哪些历史消息进入模型；
如何总结旧历史；
如何管理 subagent 边界；
ingest、assemble、compact 和 after-turn 四个生命周期阶段。

其工具结果还会在输出和日志前做大小及媒体 sanitization，reply shaping 会去重消息工具产生的重复确认，并支持 compaction 后重试。

带来提升的模块
1. Context Engine

这是 OpenClaw 最关键的质量模块。

它使 OpenClaw 在复杂任务上能较好维护：

多服务之间的依赖；
哪些数据已经读取；
哪些 action 已执行；
哪些 tool result 应保留；
哪些旧内容可以压缩。
2. Session Queue 和写锁

OpenClaw 会按 session 串行化 run，同时通过 global lane 控制总并发，避免多个运行同时修改 session 和工具状态。

这种结构对单个简单任务是额外负担，但对多服务 workflow 是优势。

在 multi_service 标记的 60 个任务上：

Harness	平均分	Pass rate	平均耗时	平均工具调用
Codex	0.762	63.3%	250 秒	19.7
NanoBot	0.782	68.3%	366 秒	36.4
OpenClaw	0.811	73.3%	294 秒	13.6

在 4 个及以上 service 的任务上，OpenClaw 的 pass rate 达到约 80%，而且工具调用明显少于 NanoBot。

这说明 OpenClaw 的 Context Engine、session state 和 dependency handling 确实在复杂工作流里产生了收益。

3. Tool-result sanitation 和 compaction

OpenClaw 的 compaction 会保留 tool call 与 tool result 配对，并保留较新的未压缩 tail。

对于大量跨服务数据的任务，这比简单截断历史安全得多。

4. Retry / queue policy

其 retry policy 明确要求：

按 HTTP request 重试；
只重试当前步骤；
避免重复非幂等操作。

这正适合包含邮件、CRM、日历、金融等写操作的工作流。

OpenClaw 的主要损失
1. 冷启动和 prompt assembly 太重

这批结果中：

启动阶段平均: 约 44 秒
prep 阶段平均: 约 23 秒
中位首次工具调用: 约 333 秒

OpenClaw 每个 instance 都创建新的 profile、home、agent directory 和 model catalog，又加载 workspace bootstrap、skills 和 system prompt。

官方 runtime 本身也会注入 AGENTS.md、SOUL.md、TOOLS.md、IDENTITY.md、USER.md、HEARTBEAT.md、BOOTSTRAP.md、MEMORY.md 等文件。

对一个只需要读取几封邮件或做一次简单搜索的任务，这些模块几乎都是固定成本。

2. Model idle timeout 小于 task timeout

OpenClaw 在 T047 中已经做了 15 次工具调用，但最后模型 synthesis 阶段触发 LLM idle timeout，最终只返回 timeout JSON。

OpenClaw 当前文档区分：

cloud model idle timeout: 120 秒
self-hosted model idle timeout: 300 秒

并且 provider idle timeout 仍受更低的 agent/run timeout 限制。

你的 DeepSeek bridge 是本地/self-hosted 风格路径，上一版却实际使用了 120 秒，因此在长上下文最终生成阶段很容易被误杀。

3. Final message 是完整 JSON envelope

当前 adapter 将整个：

{
  "payloads": [...],
  "meta": {...},
  "systemPromptReport": {...}
}

通过 tee 写进 final message。

这意味着 judge 可能看到：

timeout 说明；
大量 meta；
usage；
session 路径；
system prompt report；

而不是一个纯净的 assistant final text。

OpenClaw 最值得增加的模块
LightweightBenchmarkProfile
WarmRuntimeCache
TaskRelevantSkillLoader
AdaptiveIdleWatchdog
PayloadFinalParser

具体做法：

Gateway/runtime、插件发现、model catalog 可以 warm reuse；
每个 instance 仍使用全新的 session key、workspace 和 memory，避免跨任务污染；
简单任务不加载 browser automation、cron、tmux、node debugger 等无关 skills；
self-hosted provider idle timeout 使用 300 秒起步；

idle timeout 后只允许一次：

compact context → 保留 requirement ledger → retry final synthesis
成功时只提取 payloads[].text 作为 final message；
失败时明确标记 timeout，不把 20KB metadata 送给 judge
