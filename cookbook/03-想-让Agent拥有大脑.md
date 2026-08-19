# 第 3 章 · 想：让 Agent 拥有大脑

> "记忆不是把发生的一切都存下来，而是把值得记住的东西整理成能被回忆的形状。"

到这里，Agent 已经能看、能听。但看见和听见，离"想明白"还差得远。一台监控摄像头也能"看"，可它不会思考、不会记忆、不会主动做事。

"想"这一层，是要让 Agent 在**不同的时间尺度**上把感知变成知识和行动：

- 秒级——你现在问它一个关于画面的复杂问题，它要去逐帧研究、检索外部资料后回答；
- 分钟到小时级——视频流一直在放，它要在后台默默把发生的一切整理成分层记忆，供日后回忆；
- 事件级——你让它"盯着，等某件事发生就提醒我"，它要一直看着、命中了才行动。

这一章的核心矛盾是**实时 vs 深度**。主 Agent 必须快速理解用户意图并选对入口，但它不被动看直播画面。Argus 的答案是一以贯之的：**主 Agent 做语义路由，一次性视觉问答交给 QueryWorker，记忆整理、深度研究和持续监控交给各自的专职 Agent。**

## 3.1 分工的骨架：统一问答入口与两个持续工具

主 Agent 面对多模态时，手里有三张牌（写在系统提示 `MM_LIVE_GUIDANCE` 里）：

| 工具 | 干什么 | 时间尺度 | 产出 |
| --- | --- | --- | --- |
| `query_multimodal` | 一次性当前/历史/mixed 视觉问答 | 提问时刻及之前 | QueryWorker 直接回复；按需使用提问时刻帧、Recall 和 Search |
| `set_live_watcher` | 逐帧深度研究整段流 | 当下持续 | 后台多轮、累积报告 + 最终 summary |
| `set_monitor` | 盯着流、命中事件才触发 | 未来持续 | 每次命中一个提醒（永不出报告） |

> **命名的一个坑：** 代码经历过多次重命名。新会话里模型可见的一次性入口只有 **`query_multimodal`**；旧会话历史中保留的旧 tool call 只用于回放，不代表旧工具仍对模型暴露。旧笔记中的 `set_live_research` / `route_multimodal_query` 运行时真名是 **`set_live_watcher`**；`LiveResearchAgent` 现在叫 **`WatcherAgent`**（在 [agent/multimodal/watcher_engine.py:83](agent/multimodal/watcher_engine.py)）。

这三张牌的边界，schema 里写得很死：
- **`query_multimodal`** 是所有一次性视觉问题的统一入口。QueryWorker 先获取提问时刻近期帧，再根据完整问题决定直答、Recall、Search 或组合；
- **`set_live_watcher`** 是重型后台作业，在用户要求持续逐段研究、跟踪变化或累积报告时 RE-WATCH 视频流，不是普通一次性问答的降级路径；
- **`set_monitor`** 只盯离散事件，每次出现就 alert，**永不产出 summary/report**（那是 watcher 的活）。

主 Agent 的铁律是：**按完整语义路由，不用文字关键词硬匹配子路径**。当前画面、历史画面和视觉实体相关的外部知识问题都先交给 `query_multimodal`；“先看提问时刻帧，还是先 Recall/Search”是 QueryWorker 内部的决策。

`get_current_frame` 是一个更窄的诊断/取帧工具：只在用户明确要求取回、展示或检查最新原始帧时用，不应替代 `query_multimodal` 承担普通视觉问答。

这些能力装配成两个 toolset（[toolsets.py:286](toolsets.py)）：`live_watcher`（含 `query_multimodal` / `set_live_watcher` / `get_current_frame` 等）和 `monitor`（仅含 `set_monitor`；列表由前端 registry 展示），彼此平级、独立。这里有个曾经踩过的坑：`live_watcher` 是**非 CONFIGURABLE** 的——曾因为组内某个成员不在平台的 composite 工具列表里，导致整组工具在 dashboard 里凭空消失。所以它走的是"非 configurable 平台工具兜底恢复"的路径。

## 3.2 分层记忆：把连续的流整理成能回忆的形状

先讲"想"里最静默、也最基础的部分——记忆。

视频流是连续的、无结构的。要让 Agent 日后能回答"刚才那个人做了什么""半小时前屏幕上出现过什么"，就必须把这条流**整理成分层的、可检索的结构**。Argus 的记忆是三层的（SQLite 存储，schema 在 [agent/multimodal/_memory.py:910](agent/multimodal/_memory.py)）：

| 层 | 存什么 | 谁生成 |
| --- | --- | --- |
| **micro（微事件）** | 单个事件：起止时间、描述、主谓宾、关联帧、软删标记 | Writer 每 ~5s 抽一次 |
| **macro（宏事件）** | 一段 micro 聚合的场景：标签、摘要、关键实体、**叙事弧**（起承转合） | Writer 攒够触发 L2 聚合 |
| **super（超事件）** | macro 之上更高层的叙事 | Writer 触发 L3 聚合 |
| **entity（实体）** | 人/物：名字、类型、属性、别名、出现次数、代表帧、演化时间线、台词 | Writer upsert + Reviewer 精修 |

还有 `edges`（实体关系图）和 `revision_log`（Reviewer 的审计轨迹）。**所有表都带时间戳，查询时 attach `WHERE t <= ask_ts`**——这是一条防"脏读未来"的铁律：回答"某时刻我知道什么"时，绝不能看到那一刻之后才发生的事（时序评测正是靠它逼真还原）。

### 写入：一条后台流水线

记忆写入完全**独立于主 Agent**，跑在自己的 daemon 线程 + 私有 asyncio loop 上，但**共享主 Agent 同一个 FrameBuffer**（[agent/multimodal/memory_backend.py:56](agent/multimodal/memory_backend.py)）——又是"一份感知，多方消费"。它有两个 wake 循环：

**Writer 循环**（每 ~5s 醒一次）看最近的帧 + 最近的 ASR 字幕，抽出 micro 事件、落地代表帧、upsert 实体、插入关系边。当 pending 的 micro 攒够、或时长到顶、或检测到事件边界，就触发 L2 聚合成 macro（`_maybe_trigger_l2`）：

```python
trigger_by_boundary = (boundary in ("new_macro", "new_super"))
trigger_by_count = len(pending) >= self.cfg.mem_l2_macro_min_micro   # 攒够 5 个
trigger_by_duration = duration >= self.cfg.mem_l2_macro_max_duration # 或 3min
if not (trigger_by_boundary or trigger_by_count or trigger_by_duration):
    return
```

L2 聚合会把那段时间的帧均匀采样喂给 LLM，产出"带图的" macro 摘要 + 叙事弧。L3 同理，再往上聚合。这些聚合任务都是 fire-and-forget，但用一个 `_register_agg_task` 保住引用防被 GC、并记录异常——**别让后台任务悄悄死掉还没人知道**。

**Reviewer 循环**（每 ~2min 醒一次）是记忆的"自我修订"：它回看历史，做 merge_micros（合并被拆碎的事件）、split_micro（拆开被粘连的）、merge_entities（把"那个穿红衣的人"和"小李"认成同一个）、refine_entity、rewrite_macro_summary 等动作，全部写进 `revision_log`。Reviewer 还被拆成三个专项（Entity / Event / Edge），两波调度：Wave1 并发跑 Entity 和 Event，Wave2 串行跑 Edge。

> **为什么记忆必须能看图？** config 里有一条硬性前置：记忆模型的 `vision_ability` 必须为 true，否则 MemoryBackend 启动时**直接报错、不启动**。因为"从画面里抽事件、认实体"本质就是视觉任务。宁可让它显式崩掉，也不要静默跑一个看不见图的模型、产出一堆空记忆——**让错误早暴露，别让它伪装成正常**。

### 召回：一个多轮 ReAct 子 Agent

QueryWorker 判断问题需要历史证据时，会在内部调用 `RecallAgent`（[agent/multimodal/_workers.py:5700](agent/multimodal/_workers.py)）跑多轮 ReAct。它每轮**真并发**地调一批图查询工具（`search_micro` / `search_entity` / `search_by_time` / `get_entity_timeline` / `get_quotes_by_entity` …），SQLite 纯读走 WAL 多读并发：

```python
tasks = [asyncio.to_thread(self.mem_tools.call, n, a, ask_ts=ask_ts)
         for (n, a) in normalized]
raw_results = await asyncio.gather(*tasks, return_exceptions=True)
```

每轮蒸馏（`_distill`）当轮的发现，决定下一批查询或自我终止。最后还有一道**视觉验收**（`_verify_frames`）：用一次 vision-LLM 批量看召回到的帧，过滤掉不含目标的噪声帧。这个验收的返回语义很讲究——`None` 表示"没法验证"（模型挂了/解析失败），此时**保留原始帧**别乱删；`[]` 表示"验证过了���全是噪声，一张都别留"，调用方必须尊重这个空。**"没法判断"和"判断为空"是两回事**，混淆它们会误删有效证据。

## 3.3 深度研究：一条永不停止、随场景呼吸的走查线

`set_live_watcher` 背后的 `WatcherAgent`，我们在第 1 章已经从"看"的角度讲过它的 TTL+帧数双门。这里从"想"的角度，看它的 ReAct 大脑怎么运转。

### ReAct 单步：一次流式调用做三件事

`WatcherWorker.react_step`（[agent/multimodal/_workers.py:4625](agent/multimodal/_workers.py)）一轮就是一次带原生 function-calling 的流式 chat completion，把本批帧的真图内联进去：

```python
stream = await self.client.chat.completions.create(
    model=self.cfg.model, messages=msgs,
    tools=LIVE_RESEARCH_REACT_TOOLS, tool_choice="auto",
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},  # 开推理
    stream=True,
)
```

流式解析分三路：`reasoning_content` → 思考（实时回吐给面板）、`content` → 答案正文（**仅当本轮没派工具时才吐**）、`tool_calls` → 按 index 累积成完整 JSON。它只声明两个工具：`text_search`（外部检索）和 `recall_memory`（查记忆）——图搜工具刻意不声明，免得 LLM 乱派。

**终止判据很优雅**：本轮**有没有派工具**。派了工具 → 说明还需要更多信息 → 继续下一轮；没派工具 → 说明模型选择直接作答 → 收尾。这比让模型自报一个 `can_answer` 布尔更可靠，因为"派不派工具"是它的真实行动，不是它的自我声明。

### 无状态 ToolBox：并发安全的关键

ReAct 决定好要调哪些工具后，直接交给 `ToolBox` 执行。`ToolBox` **没有实例状态**——anchor 帧、进度回调都通过 `call()` 的参数显式传（[agent/multimodal/_workers.py:227](agent/multimodal/_workers.py)）：

```python
self.toolbox.call(tc["name"], _args,
                  anchor=a_frame,               # 显式传，不放实例上
                  crop_progress_cb=_make_search_prog(tid))
```

为什么不能放实例上？因为 ToolBox 可能被并发调用，如果 anchor 存在实例属性里，两个并发调用就会互相覆盖对方的 anchor。**无状态是并发安全的前提**——这个原则你在整个系统里会反复看到。

### 累积报告、增量推送、静默收尾、超时兜底

深研是长跑，用户不能干等到结束。所以有一整套"边跑边给"的机制（全在 `_run_delegation`，[agent/multimodal/watcher_engine.py:899](agent/multimodal/watcher_engine.py)）：

- **累积报告**：每批产出一条带时间窗的解读 `[mm:ss–mm:ss] ...`，append 进一个 running_report 列表，同时把上一段喂给下一段做**增量提示**（"这段只说新东西，别重复上段"）。
- **周期增量推送**：每 N 批（默认 3）就把累积报告推给前端，不等结束用户就能先看。
- **静默/无进展不自动收尾**：`no_progress` 只作日志计数。结束只靠源停、用户停、或硬轮数上限——因为网络卡顿造成的"暂时无帧"不能被误判成"流结束了"。
- **summary 超时兜底**：最终汇总有超时（默认 120s），超时就用累积报告兜底：

```python
_running_fallback = "\n\n".join(running_report).strip() or "深度分析已完成…"
try:
    summary = await asyncio.wait_for(summarize_watch(...), timeout=_sum_timeout)
except asyncio.TimeoutError:
    summary = _running_fallback   # 用累积报告兜底
if not summary.strip():
    summary = _running_fallback
```

**绝不返回空、绝不无限等待**——这是长跑任务的两条底线。

## 3.4 事件监控：一直看着，命中才行动

`set_monitor` 是"想"里最像"值班保安"的部分：你告诉它盯什么（"有人进画面就叫我""屏幕出现报错就通知我"），它就一直看着，命中了才行动，其余时候安静。

`MonitorEngine`（[agent/multimodal/monitor_engine.py:200](agent/multimodal/monitor_engine.py)）的结构和 WatcherAgent 一脉相承：一条后台线程托管私有 asyncio loop，**每个 monitor = 一个长活的 async job**，一个 tick 一次迭代。监控的注册表 `agent.mm_monitors` 按引用共享——`set_monitor` 改它，job 实时读，UI 增删即时��效。

每个 tick 做���事：聚合 flush（如果是"每 T 秒汇报"模式）→ 源关或忙就跳过 → 取游标后的新帧 → 拥塞时几何降采样 → vision LLM 判 `{status, reason}`（命中/未命中）。有个容易忽略的细节——**post-await liveness re-check**：

```python
# LLM 调用让出控制权期间，monitor 可能已被用户删除/禁用
m = self.monitors.get(mid)
if m is None or not m.get("enabled", True):
    return   # 已经没了，别再写命中事件
```

LLM 调用要几秒，这几秒里用户可能已经把这个监控删了。回来后必须重新确认它还活着，否则会给一个已删除的监控写命中事件。**异步世界里，await 之后的世界可能已经变了**。

### hook 主 Agent：让监控命中触发自主行动

监控最强大的地方，是命中后不只是弹个提醒，还能**触发主 Agent 自主执行一段指令**。你可以设"一旦快递员出现在门口画面，就帮我在门禁系统留言"。

命中时，`_speak_cb`（[tui_gateway/server.py:9030](tui_gateway/server.py)）二选一、不并存：

```python
if _hook_task:   # 配了 hook_main_agent
    with lock:   # 原子占用 busy gate
        if not session.get("running") and not session.get("_monitor_hook_running"):
            session["running"] = True
            session["_monitor_hook_running"] = True
            _go = True
    if _go:
        threading.Thread(target=_run_hook_turn, ...).start()  # 忙则丢弃，不排队
    return True
# 无 hook：只弹提醒气泡
```

`_run_hook_turn` 把命中判定的 `text` 拼进指令，作为一条**真实的 UserMessage** 提交给主 Agent 跑一整个 turn。这里的并发策略是"忙则丢弃，不排队"——如果主 Agent 正忙，这次命中就丢掉，绝不排队堆积（想象一下画面里一直有人走动、每帧都命中、每次都排队会怎样）。

## 3.5 外部检索：AnySearch 与"防撑爆上下文"

深研的 `text_search` 背后接的是 AnySearch（[agent/multimodal/_workers.py:378](agent/multimodal/_workers.py)），走 JSON-RPC 2.0 的 `tools/call`。这里有一个很实际的工程约束——**单条检索结果可能有几百 KB**：

```python
cap = int(getattr(self.cfg, "anysearch_result_max_chars", 4000) or 0)
if cap > 0 and len(text) > cap:
    dropped = len(text) - cap
    text = text[:cap] + f"\n…[结果过长, 已截断 {dropped} 字; 如需更多请换更精确的 query]"
```

为什么必须截断？因为一轮最多派 5 条 search，又逐轮累积进 ReAct 上下文——不截断，几轮下来上下文就被检索正文撑爆了，既费钱、又拖慢、还把宝贵的画面帧挤出去。截断保留头部（最相关的���前），尾部标注被���字数。**给 LLM 的上下文是稀缺资源，任何单一来源都不该无限占用它。**

## 3.6 系统提示：多模态四要素的编排

主 Agent 之所以知道"什么时候该看、该听、该记、该委派"，全靠系统提示里的编排。四个要素按固定顺序装配（[agent/system_prompt.py:115](agent/system_prompt.py)）：

**身份 → 记忆 → 实时 → 技能**

```python
if not _soul_loaded:
    stable_parts.append(DEFAULT_AGENT_IDENTITY)   # ① 身份
...
tool_guidance = []
if "memory" in agent.valid_tool_names:
    tool_guidance.append(MEMORY_GUIDANCE)         # ② 记忆
...
tool_guidance.append(MM_LIVE_GUIDANCE)            # ③ 实时（始终加入）
if "skill_manage" in agent.valid_tool_names:
    tool_guidance.append(SKILLS_GUIDANCE)         # ④ 技能
stable_parts.append("\n\n".join(tool_guidance))   # ★ 段落级分隔
```

两个细节值得学：

1. **`MM_LIVE_GUIDANCE` 按能力加入**——只要本会话可见任一多模态实时工具就加入；纯 coding 等不暴露这些工具的 posture 不会收到无法执行的指令。它先明确“主 Agent 永不被动收图”，再按完整语义划分 `query_multimodal`、`get_current_frame`、`set_monitor` 和 `set_live_watcher` 的边界；`check_video_stream` 只用于流状态查询或真实指代歧义。
2. **用 `\n\n` 而非空格分隔各块**——每个 guidance 块是独立优化的完整段落，单空格拼接会把块边界糊成一行，让模型分不清"这是另一个主题"。

而整个 stable tier 被缓存（`agent._cached_system_prompt`），**永不 per-turn 重渲染**——这是为了保住上游的 prompt cache 温热。易变的提问时刻像素由 QueryWorker 在任务内取得，稳定的路由规则留在 system prompt；主 Agent 的每轮上下文不需要为注入图片而改写过去消息。

## 3.7 一个真实的坑：空图 400

最后讲一个特别能体现"想"这一层脆弱性的 bug，也是踩坑后学到的教训。

离线记忆评测时，所有问题都答"未找到"。记忆库明明有 156 个事件，写入也正常，为什么召回全空？

层层排查后，根因是：召回时会给 LLM 附上"提问那一刻的帧"作为参考，但离线场景下 FrameBuffer 有时返回 `jpeg_b64` 为空的帧（帧被驱逐、或从未编码）。空帧构造出一个空的 data-URL（`data:image/jpeg;base64,` 逗号后什么都没有），而检索端点收到这种空图 part 会**整包返回 400**——于是每次 recall 的 LLM 调用都失败，ReAct 直接挂掉，整个 run 返回"未找到"。

**表象是"记忆召回失效"，真因是一个空字符串。** 修复是在发送前过滤掉这种坏 part（`_drop_empty_image_parts`，[agent/multimodal/_workers.py:43](agent/multimodal/_workers.py)）：

```python
if p.get("type") == "image_url":
    url = ((p.get("image_url") or {}).get("url") or "")
    if not url or url.rstrip().endswith(",") or (
            "base64," in url and not url.split("base64,", 1)[1].strip()):
        changed = True; continue   # 丢掉空/坏 image part，保留 text
```

接在 writer 和 recall 的三处发送点。这个 bug 的教训是：**多模态系统里，一个空的媒体载荷比一个缺失的字段更危险**——缺失往往有默认处理，而"看起来有、实际是空"的载荷会一路溜到端点才炸，且炸得毫无线索。

## 3.8 Watcher 的持续语音叙述

Watcher 不仅产出文字报告，还可以在每轮结束时通过 TTS 把本段发现"念"给用户听。这让"深度研究"不再只是后台静默跑批，而是一个有存在感的伙伴——你在看视频，它在旁边低声解说。

实现关键是**串行 TTS 队列**（`watcher_engine.py:508`）：一轮的中间段用 `send_final=False` 入队，前端把后续段的 PCM 追加到同一条时间线（而非每段重启播放）；只有最后一段才标 `is_final`，收尾整轮。如果用户中途说话（barge-in），`interrupt_tts()` drain 掉队列里所有等待的 segments 并 emit is_final 收尾。

> **设计取舍：** watcher 的语音叙述是"尽力而为"——TTS 失败只 log、不中断分析循环。deep analysis 的核心价值是文字洞察，语音只是附加的"温暖层"。

## 3.9 QueryWorker 的"prior QA"保持

一个容易忽略的细节：用户连续问两个关于画面的问题时（比如"这个 UI 的配色好吗？" → "那字体呢？"），第二问的 QueryWorker 需要**看到第一问的 Q&A** 才能理解"那"指什么。

修复是 `preserve prior QA in QueryWorker handoff`（commit `5caa1874`）：提交 query 时，把近 N 条同类型 QA 对作为 context 一起传入 QueryWorker 的上下文窗口。这让连续追问不再"失忆"——每一次 `query_multimodal` 都能看到自己前几轮的结果。

## 本章小结

"想"这一层，是把感知变成知识和行动，核心是**按时间尺度分工**：

- `query_multimodal` 是一次性当前/历史/mixed 视觉问答的统一入口，QueryWorker 再内部决定提问时刻帧 / Recall / Search；`set_live_watcher` 和 `set_monitor` 分别负责持续深研与事件监控；
- 分层记忆（micro/macro/entity）在后台把连续流整理成可检索的结构，Writer 写、Reviewer 修，全程 `ask_ts` 防脏读未来；
- 深研是永不停止的 ReAct 走查，用"派不派工具"判终止，无状态 ToolBox 保并发安全，累积报告+超时兜底保证绝不返回空；
- 事件监控每个是独立 async job，命中可 hook 主 Agent 自主行动，await 后必须重查存活；
- 系统提示按"身份→记忆→实时→技能"编排并缓存，稳定与易变分离；
- 空图 400 的教训：空媒体载荷比缺失字段更危险；
- Watcher 的串行 TTS 队列让深度分析可以"边跑边念"，赋予 Agent 解说员的存在感；
- QueryWorker 保持 prior QA，连续追问不失忆。

它会看、会听、会想了。最后一章，让它开口说话——像个朋友那样。

---

*上一章：[第 2 章 · 听](02-听-让Agent拥有耳朵.md) · 下一章：[第 4 章 · 说](04-说-让Agent开口说话.md)*
