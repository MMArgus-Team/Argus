# 第 1 章 · 看：让 Agent 拥有眼睛

> "看，不是把每一帧都记下来，而是知道哪一帧值得记下来。"

如果说 LLM 原本是一个只能读文字的大脑，那么"看"这一层，就是给它接上一条永不间断的视神经。但视神经不能只是一根管道——如果把摄像头每一帧原封不动灌进大脑，几秒钟就会把它淹死。所以"看"的核心，从一开始就不是"如何看见"，而是**"如何在看见和成本之间做取舍"**。

这一章我们从一帧画面进入系统的那一刻讲起，一路讲到它如何被去重、被缓冲、在一次性问题接管后作为 QueryWorker 的提问时刻画面，以及如何被后台 watcher 持续逐段深研。

## 1.1 一帧从哪来：2fps 的尽力而为通道

前端（浏览器或桌面端）以约 **2fps** 的频率，通过 WebSocket RPC `multimodal.frame` 把单张 JPEG（base64 编码）推给服务端。这个频率是配置里的唯一真相源：

```python
# hermes_cli/config.py:2191
"buffer_capture_fps": 2.0,
"buffer_seconds": 1800.0,   # 缓冲 30 分钟
```

服务端接收帧的那段代码（[tui_gateway/server.py:10822](tui_gateway/server.py)），处处透着一个信念：**采集通道绝不能拖垮事件循环**。它体现在三个刻意的"偷懒"上：

1. **agent 没就绪就静默丢帧**（`server.py:10840`）——这是一条高频尽力而为通道，宁可丢帧也不排队阻塞。
2. **不做完整 base64 解码就入库**（`server.py:10858`）——一次约 200KB 的解码要 5–10ms，如果在 WS 事件循环线程上做，会直接卡住同一条连接上正在下发的 SSE token（也就是主 Agent 正在"打字"的流）。
3. **客户端传的时间戳被故意忽略**（`server.py:10860`）——时间戳由服务端权威生成。

> **为什么忽略客户端时间戳？** 因为画面帧和音频来自不同的客户端时钟，如果各用各的 epoch，后面做"音视频对齐"时就会错位。统一到服务端单调时钟，是整个时间基对齐设计的第一块基石——我们会在第 2 章看到音频如何戳到同一条时钟上。

## 1.2 FrameBuffer：一份感知，多方消费

帧进来之后，落进 `FrameBuffer`——整个多模态子系统的��享内存。它的定义��素得几乎让人失望：

```python
# agent/multimodal/_memory.py:154
@dataclass
class Frame:
    ts: float
    jpeg_b64: str
```

一个时间戳，一张图。但 `FrameBuffer` 本身承载了远不止存储的职责（[agent/multimodal/_memory.py:200](agent/multimodal/_memory.py)）：

```python
def __init__(self, cfg: Config):
    maxlen = max(8, int(cfg.buffer_seconds * cfg.buffer_capture_fps) + 4)
    self._dq: Deque[Frame] = deque(maxlen=maxlen)      # 环形缓冲
    self._lock = threading.Lock()
    self._dhash_threshold = int(cfg.framebuffer_dhash_threshold_init or 6)
    self._recent_dhash: Deque[int] = deque(maxlen=2)   # 最近保留帧的 dHash
    self._current_scene: Optional[dict] = None         # 场景/节奏，多消费者共享
    # ... mono_epoch / wall_epoch：服务端权威时间锚
```

容量约 `1800 × 2 + 4 ≈ 3604` 帧，正好是 30 分钟的 2fps 流。这里有几个设计选择值得停下来看：

**（一）它是环形的（`deque(maxlen=...)`）。** 超过 30 分钟的老帧自动淘汰，内存不会无限膨胀。README 和 config 注释里算过账：1800s × 2fps × ~80KB/张 ≈ 280MB，可接受；想省内存把 `buffer_capture_fps` 调成 1.0 就减半。

**（二）它是唯一的共享抽象。** QueryWorker、Memory Writer/Reviewer、watcher 引擎和场景控制器共享同一个 `FrameBuffer`；Monitor 在同一次 `push_live` 中另外保留一条原始短队列。这就是序章那句"一份感知，多方消费"的字面实现。场景标签、攒帧节奏都存在 `_current_scene` 里，各模块零改管道即可读到。

**（三）为什么 buffer 从 60s 扩到 30min？** config 里留了一段完整的推理（`_config.py:321`）：原版只存 60s 滑动窗，导致做长程记忆聚合的 L2/L3 和 Reviewer 回看历史时，大部分帧早被淘汰了，只能拿到 Writer 自己挑过的少量关键帧（有损）。扩到 30min，长程聚合就能拿到接近"原始所有帧"的密集采样。

读接口是一组按不同"看的方式"设计的游标：`latest(n)`（尾部 n 帧，给需要最新画面的消费者）、`all_after(ts)`（严格增量游标，给 watcher 逐段推进用）、`all_le(ts)`（≤某时刻，给 QueryWorker 锁定"提问那一刻"用）、`sample_uniform(window, n)`（均匀抽样，给场景探针用）。**不同的消费者，需要不同的"看法"**——这组接口就是把这些看法固化下来。

## 1.3 dHash 入口去重：在门口就把重复挡掉

2fps 的流里，绝大多数相邻帧是几乎一样的——你盯着一份文档看十秒，就是 20 张几乎全等的图。如果全存下来、全喂给下游，就是纯粹的浪费。

Argus 的选择是：**在 `push` 入口就去重**，而不是让每个下游各自去重。

去重用的是 **dHash（差分哈希）**（[agent/multimodal/_memory.py:457](agent/multimodal/_memory.py)）：

```python
def _compute_dhash(jpeg_b64: str) -> Optional[int]:
    img = Image.open(...).convert("L").resize((9, 8), Image.LANCZOS)  # 灰度缩到 9×8
    # 每行比较相邻像素：left > right 置 1，共 8×8 = 64 位
    ...
    return bits   # 解码失败返回 None；0 是合法哈希

def _hamming(a, b):
    return bin(a ^ b).count("1")   # 汉明距离
```

新帧进来时，跟最近**最多 2 张**已保留帧比对，只要与任何一张的汉明距离 `< threshold` 就判为重复、直接丢弃（`_dedup_and_store`，`_memory.py:228`）。一个小而关键的 fail-safe：解码失败用 `None` 表示，这类帧**永不判重**、直接存；而纯色画面得到的 `dHash == 0` 是合法值，必须像其他哈希一样参与去重。

这个"去重上移到入口"的决策用于长期消费链：watcher、memory writer 和 QueryWorker 的提问时刻视图读到的是去重后的稀疏帧，**不再各自做画面级 dHash**。Monitor 是刻意的例外：它从同一次 `push_live` 旁路保存最近 60 秒、服务端实际收到的全部 2fps 帧，避免只出现半秒的手机/水杯被长期记忆去重策略吃掉。watcher_engine 里那些旧的 stride 降采样、批级 dHash 代码仍被标记为 DEPRECATED、不再读取。

> **反模式警告：** 如果让每个下游各自去重，你会得到四份不一致��去重逻辑、四份重���计算，还会在某个模块忘记去重时悄悄退化成"全量喂"。共享感知层的价值，一半就在这种"公共决策只做一次"上。

## 1.4 场景感知：让去重阈值随内容"呼吸"

固定阈值的去重是笨的。一份静止的文档和一场激烈的球赛，需要的去重力度天差地别：文档你恨不得二十张只留一张，球赛你每一帧的走位都可能有用。

于是有了 `SceneDhashController`（[agent/multimodal/scene_dhash.py](agent/multimodal/scene_dhash.py)）——**每 20 秒，用一个辅助视觉模型看几张小图，判断当前是什么场景，动态调整去重阈值和攒帧节奏。**

```python
# _config.py:341
scene_probe_interval_s: float = 20.0   # 每隔多久判一次场景
scene_probe_window_s:  float = 20.0    # 从近 20 秒里抽样
scene_probe_frames:    int   = 3       # 均匀抽 3 张
scene_probe_maxside:   int   = 256     # 抽样图压到最长边 256（省 token）
scene_probe_quality:   int   = 50      # JPEG 质量 50
```

探针图故意做得很廉价：3 张、256px、质量 50。判场景不需要高清，能看出"这是文档还是球赛"就够。用的模型是 `auxiliary.vision`（辅助视觉模型），它为 None 时整个控制器空转、阈值保持不变——又一处"绝不阻塞采集"的优雅降级。

模型只输出**受约束的场景大类**。代码再用同一张硬映射表同时决定攒帧节奏和 `dhash_threshold`，保证"场景标签"、"节奏"与"去重强度"永远一致；旧模型残留的阈值字段只在场景无法解析时作为兼容兜底：

| pace | 阈值≈ | 典型场景 |
| --- | --- | --- |
| **slow**（基本不动，激进去重） | 10–12+ | 会议 / 办公 / 编程 / 阅读 / 监控 |
| **medium**（稳定推进） | 6–9 | 影视 / 对话 / 教学 / 新闻 / 音乐 |
| **fast**（频繁变化） | 4–5 | 体育 / 游戏 / 户外 / 棋牌 / 驾驶 |
| **live**（剧烈实时） | 2–3 | 直播 / 实时竞技 / 通话 / 实操 |

阈值是“重复距离截止值”：因为代码在 `distance < threshold` 时丢帧，所以阈值越大去重越激进，阈值越小保留的动态细节越多。阈值范围被 clamp 在 `[2, 20]`（`framebuffer_dhash_threshold_min/max`）。任何一步失败或超时，都保持当前阈值不变——这条"尽力而为、永不阻塞"的原则，你会发现它贯穿"看"的每一个环节。

> **设计哲学：** 让模型判"是什么场景"（它擅长的语义判断），让代码查表定"节奏该多快"（确定性映射）。不要让模型直接猜一个魔数——那样既不稳定也不可解释。把语义判断和数值决策分开，是这套设计的一个反复出现的模式。

## 1.5 QueryWorker 的提问时刻帧：按需看，不改写主对话

到这里，帧已经被去重、被缓冲。现行合同的第一条是：**主 Agent 永不被动收到直播画面**。它先根据用户完整语义做路由；当一次性问题依赖当前画面、历史画面，或需要把画面实体与外部事实绑定时，调用 `query_multimodal` 将原问题交给 QueryWorker。

QueryWorker 在接管时获得三样东西：原始用户问题、原消息的回答位，以及服务端记录的 `ask_ts`。它优先从 `FrameBuffer.all_le(ask_ts)` 取提问时刻及之前的最近帧（当前上限 3 张）；若没有可用时间锚，才兜底取 buffer 尾部。这些图只进入该次 QueryWorker 任务，不会被动塞入主 Agent 的当前 turn，也不会在主对话历史里逐轮累积。

QueryWorker 是 VLM-first 的一次性回答者：

- **提问时刻帧已足够**：直接读图回答，不调工具；
- **需要更早内容**：在内部调用 Recall，并查看召回的历史关键帧；
- **需要画面外事实**：先从图或记忆绑定目标，再调 Search；
- **证据不足**：明确说明缺口，不拿无关当前帧猜历史答案。

答案由 QueryWorker 直接回写原用户消息的 answer slot，不再回到主 Agent 进行第二次改写。这既避免了多一次模型延时，也避免图像与中间证据撑大主对话上下文。

`get_current_frame` 是更窄的显式原始帧入口：只在用户明确要求“把最新帧取回/展示给我”，或需要做原始帧诊断时使用。它**不是**“现在画面里是什么”这类普通视觉问答的默认入口。

> **旧架构说明：** 早期版本曾在每轮对话前用 `vision_inject_to_main` / `_maybe_apply_multimodal_vision()` 向主 Agent 被动附加尾部帧。这会让纯文本问题也支付图像 prefill，并且容易把“当前画面”与“历史证据”混为一谈。旧文档、旧注释或历史记录中若仍出现“主 Agent 当下一瞥”，它描述的是这个已退出的设计，不是现行合同。

## 1.6 分工的边界：一次性 QueryWorker vs 持续 Watcher

现在我们能清楚地划出"看"这一层最重要的分工线。

**QueryWorker 处理一次性问题**——它以提问时刻近期帧为起点，再根据问题需要补 Recall 或 Search。“现在画面里是什么”、“刚才出现的银行是哪家”、“查一下画面里这家银行某日收盘价”都是同一入口：`query_multimodal`。

**Watcher 处理持续研究**——它沿 buffer 游标逐批向 live 边缘推进，由场景节奏决定每轮帧数和 TTL。它适合“持续分析这段视频”、“跟踪这个人后续做了什么”、“边看边累积一份报告”这类任务。Watcher 不是普通一次性问答失败后的默认降级路径。

这条边界在两个层面同时实现：

- **数据层**：QueryWorker 取固定 `ask_ts` 的近期快照，必要时补历史关键帧；Watcher 持有持续游标，逐批消费后续流。
- **决策层**：稳定系统提示 `MM_LIVE_GUIDANCE`（[agent/prompt_builder.py:163](agent/prompt_builder.py)）要求主 Agent 根据用户完整语义路由，不依赖“当前/刚才/监控”等单个关键词硬分支。

另外还有两个边界清楚的入口：

- **“把最新原始帧给我 / 检查采集到的图”** → `get_current_frame`；
- **“等某个事件出现就提醒我”** → `set_monitor`，由 Monitor 持续盯事件，而不是 Watcher 累积报告。

## 1.7 watcher 逐段深研：TTL + 帧数双门

最后，我们深入"看"这一层最精巧的机制——watcher 是怎么"逐段"看完一整条永不停止的视频流的。

难题在于：视频流是无限的、节奏是变化的。攒多少帧跑一轮分析？攒太少，几帧就仓促下结论；攒太多，一轮要等半天、用户没反馈。固定帧数或固定时间都不对。

Argus 的解法是 **TTL + 帧数双门**（[agent/multimodal/watcher_engine.py:899](agent/multimodal/watcher_engine.py)），由场景节奏（1.4 节那张表）驱动，分 4 档：

| 档 | TTL（秒） | 目标帧数 | 场景 |
| --- | --- | --- | --- |
| slow | 200 | 100 | 会议 / 办公 / 文档 |
| medium | 60 | 60 | 影视 / 交谈 / 教学 |
| fast | 30 | 40 | 比赛 / 游戏 / 户外 |
| live | 10 | 15 | 通话 / 连麦 / 实操 |

一轮的结束条件是**双门取先到**（`watcher_engine.py:1344`）：

> **(a)** 攒够 `target_frames` 帧 —— 快场景下画面变化多，很快攒够，立即跑；
> **(b)** 或距本轮开始已过 `ttl_sec` —— 慢场景下画面不怎么变，攒不够就靠 TTL 兜底，到点了拿手头全部帧跑；
> **(c)** 或视频源停 / 用户停。

节奏值的取值是三级 fallback（`_round_pacing`，`watcher_engine.py:938`）：先看 `set_live_watcher` 调用时显式写的 TTL/帧数 → 兜底用 FrameBuffer 自动探测的场景 → 再兜底用 config override → 最终兜底 medium（120s / 64 帧）。每个值都过 `_clamp` 设正数下限，防止坏配置把门变成 0（TTL=0 会空转，target=0 会跑空批）。

几个细节体现了它对真实场景的打磨：

- **首轮特殊**（`watcher_engine.py:1327`）：只取最近 `target_frames` 帧、不从头吞整段积压；而且要攒到 ≥60% 才立即分析，否则继续等，避免"刚开始盯就用寥寥三帧仓促下结论"。
- **暂停不倒计时**（`watcher_engine.py:1388`）：TTL 到了但一帧没攒到（比如画面完全静止），不硬跑空批，而是挂起等新帧，前端显示"等待新画面…"而**不倒计时**；新帧一到就重开一轮、重置 TTL。
- **拥塞降采样**（`_even_downsample`，`watcher_engine.py:1152`）：如果上一轮跑太久、积压超过了 target，就在首尾之间等间隔降采样到 target，既保时序覆盖又不超预算。
- **无进展只计数、不自动收尾**（`watcher_engine.py:1186`）：这是应用户要求的选择——网络卡顿造成的"暂时无帧"不能被误判为"流结束"。结束只靠源显式停、用户停、或硬轮数上限（默认不限）。

而 watcher 里的 LLM 到底怎么"看图"？关键是**图像优先**——它自己就在看本段画面，真图已内联进 user message（`_workers.py:4718`），prompt 立了一条铁律（`_workers.py:4713`）：

> **★ 图像优先**：你自己就在看本段画面（真图已附），先直接读图理解，画面读不到才调工具。本轮二选一：需要外部背景/历史 → 调 text_search / recall_memory（本轮不写正文）；画面已足够 → 不调工具，直接输出本段解读。

每轮报告都累积进 watcher 自己的 running report，并每 N 批增量推送给前端（不等结束就先给用户看）；它不会把每批帧或中间观察追加进主 Agent history。最后再做一次带超时兜底的汇总——超时就用累积报告，**绝不返回空**（`watcher_engine.py:1660`）。深研的"想"的部分，我们留到第 3 章细讲；这里你只需记住：watcher 的"看"，是一条**永不停止、随场景呼吸、逐段推进**的走查线。

## 1.7 从像素到文字：OCR reflow 与窗口文字桥

纯视觉模型对小字 / 长文档的 OCR 准确率有限。当用户在桌面端共享一个具体窗口（`Frame.source_id` 形如 `window:12345:0`）时，Argus 有一条更快更准的路径：

1. **window_text_bridge**（[agent/multimodal/window_text_bridge.py](agent/multimodal/window_text_bridge.py)）：利用操作系统的辅助功能 API（macOS AX / Windows UIA）直接抓取窗口的结构化正文——零 OCR 延迟、零识别错误。还能顺带拿到浏览器 URL / 文件路径作为"锚点"，给记忆和 watcher 的上下文更具体的定位。
2. **失败回退**：如果 AX/UIA 拿不到（权限未授权、或非文本型应用），则回落到 RapidOCR，走 **ocr_reflow** 增强管线。

`ocr_reflow`（[agent/multimodal/ocr_reflow.py](agent/multimodal/ocr_reflow.py)）把"给定一张图就能跑"的 OCR 后处理逻辑抽成共享模块，两个消费者（window_text 和视频流 OCR worker）复用同一份算法，消除重复防漂移：

- **尺寸规整**（`resize_for_ocr`）：小图放大到 1280px 救小字，大图缩到 2048px 省算力——保证 OCR 始终在"字够大、图不爆"的甜区运行。
- **乱码过滤**（`is_readable_text`）：控制字符 / 私用区占比 >15% 直接丢弃。
- **版面重排**（`reflow_ocr_lines`）：先检测多栏 → 按 x 坐标分列 → 每列按 y 排序合并行 → 输出自然阅读序段落。这一步把 OCR 引擎返回的碎片坐标变成"人类从左到右从上到下阅读的文本流"。
- **Top-N 过滤**：只保留面积最大的前 15 个段落块（正文主体），标题栏 / 状态栏等小碎片自动丢弃。

> **设计选择：** window_text_bridge 故意不调用 window_text.py 自身的 OCR 兜底（它会自己截屏 + rapidocr，与视频流 OCR worker 重复）。让 worker 走 ocr_reflow 的增强路径就够了，且它用的是共享帧数据而非额外截屏。**一份感知，多方消费**——这条原则从 FrameBuffer 一直贯穿到 OCR。

## 1.8 追踪的简化：从 DINO+Kalman 到 EMA+线性运动

早期 Argus 曾尝试 DINOv3 视觉特征 + Kalman 滤波的 MOT 方案。在批量评测后发现：Kalman 和两阶段关联（先外观后运动）引入了大量复杂度，但 HOTA 指标仅比简单方案高不到 1 分，且在低帧率（2fps）下频繁失配。

最终方案极度简化：**EMA 外观向量 + 线性运动预测**，删除 Kalman、CMC、两阶段匹配，HOTA 50.90 比复杂方案持平甚至更稳。DINOv3 权重（~88MB）和 YOLO 检测器也从仓库删除（转为按需外部提供），代码体积大幅缩减。

> **教训：在 2fps 的稀疏帧率下，复杂运动模型的边际收益趋近于零。** 简单方案更容易调参、更稳定、更省算力。这与"看"层一以贯之的信念一致：不追求理论最优，追求在实时约束下的稳定最优。

## 本章小结

"看"这一层的每一个设计，都在回答同一个问题：**在无限的视频流和有限的成本之间，怎么只留下值得看的？**

- 2fps 尽力而为采集，绝不阻塞事件循环；
- FrameBuffer 是"一份感知，多方消费"的共享地基；
- dHash 在入口去重，公共决策只做一次；
- 场景感知让去重阈值和攒帧节奏随内容呼吸；
- 主 Agent 不被动收帧；一次性当前/历史/mixed 视觉问题由 `query_multimodal` 交给 QueryWorker；
- `get_current_frame` 只负责显式原始最新帧，持续逐段研究则交给 watcher；
- watcher 用 TTL + 帧数双门，随场景节奏逐段深研整条流；
- 窗口文字桥优先用 AX/UIA 抓结构化正文，回退到 OCR reflow 增强管线；
- 追踪从重型 DINO+Kalman 简化到 EMA+线性运动，在 2fps 约束下找到性价比甜区。

下一章，我们给这个会看的 Agent 装上耳朵。

---

*上一章：[序章](00-序章-一个会看会听会想会说的Agent.md) · 下一章：[第 2 章 · 听](02-听-让Agent拥有耳朵.md)*
