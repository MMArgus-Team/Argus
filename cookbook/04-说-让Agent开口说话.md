# 第 4 章 · 说：让 Agent 开口说话

> "一个只会念答案的助手是工具，一个会被你打断、会说'嗯我在'、会关心你的助手，才像个朋友。"

到这里，Agent 已经能看、能听、能想。最后一步——开口。

"说"看起来最简单：把文字丢给 TTS，播出来就行。但真正做一个**对话式**的语音助手，你会撞上一堆微妙的问题：它该不该说这句话？说的时候该用什么语气？你一开口它能不能立刻闭嘴？喇叭关着的时候它会不会偷偷出声？后台一堆监控、深研的结果同时想说话，该先说哪个？

这一章的核心矛盾是**流畅 vs 可中断，以及自然 vs 克制**。让它说得流畅、有温度，同时随时能被打断、该闭嘴时闭嘴。这一切，都收敛在一个叫 `VoiceAgentV2` 的常驻 Agent 里。

## 4.0 三条并存的"说"通道

先分清全局。代码里其实有**三套 TTS 路径**：

1. **VoiceAgent v2 实时流式路径（本章主角）**——多模态页 `/multimodal` 用的。经 DashScope WS 流式吐 PCM24k，前端 WebAudio 播放。
2. **`voice_rewrite.py`（温暖化改写层）**——**已是死代码**。它早期的职责（把生硬文本改成"说人话"）已被 v2 内部的拟词逻辑取代，如今只剩一个离线重跑脚本还引用它。
3. **`tts_registry.py` / `tts_provider.py`（文件式可插拔 TTS）**——经典 CLI/TUI 的 `text_to_speech` 工具用的，与实时多模态路径**完全无关**（4.5 节简述）。

本章绝大部分讲第一条。

## 4.1 VoiceAgentV2：一个常驻的对话大脑

`VoiceAgentV2`（[agent/multimodal/voice_agent_v2.py:72](agent/multimodal/voice_agent_v2.py)）的线程模型，你现在应该很熟悉了——**1 个后台 daemon 线程 + 私有 asyncio loop**，和 MonitorEngine、WatcherAgent 同一个模子。里面跑两个 worker 和三个队列：

- **理解队列 + 理解 worker**：收"需要委派给主 Agent 的用户作业"，提交主 Agent、等结果、结果最高优先入回播。
- **回播队列（优先级队列）+ 交互 worker**：所有要"说出口"的内容都在这里排队，交互 worker 逐个取出、拟词、送 TTS。

回播队列的优先级分三级、严格隔离（`voice_agent_v2.py:40`）：

```python
PRI_FAST_REPLY  = 100    # L1: 快速回复（用户直问的秒答，"好的"）
PRI_TASK_RESULT = 50     # L2: 主 Agent 作业结果
PRI_AMBIENT     = 10     # L3: monitor/watcher 的环境告知
```

为什么要分级？想象你刚问了句话，同时后台一个深研作业正好完成、一个监控正好命中——三件事都想说。分级保证：**你直接问的，永远最先答**；后台的环境告知，排在最后，绝不插队打断你的对话。

## 4.2 输入的三层过滤：什么话才值得回应

用户说的每一句 ASR final 进来（`submit_user`，`voice_agent_v2.py:286`），要过三道关，一道比一道贵：

**层 2——本地规则过滤（零成本）**（`_layer2_filter`）：太短的、纯语气词的（"嗯""啊""哦"）、单字重复的、短窗口内和上一句高度重复的（ASR 打字机幻觉会复述前一句），直接丢：

```python
_fillers = {"嗯", "啊", "哦", "呃", "唉", "喂", "咦", "唔", "哈", "哎"}
_stripped = text.strip("。！？，、,.?!… ")
if _stripped and all(c in _fillers for c in _stripped):
    return "pure_filler"          # 纯语气词，丢
if self._last_user_text and (time.time() - self._last_user_ts) < dedup_win:
    if _text_similarity(text, self._last_user_text) >= dedup_ratio:
        return "dedup_recent"     # 和刚才那句太像，丢
```

**层 3——LLM 意图分类**（`_admit_user_with_intent_check`）：调 `judge_addressed_to_me` 判"这话是不是在跟我说话"（就是第 2 章那个意图过滤器）。判决很讲究：

```python
if addressed is False:            # 明确判为环境语音 → 丢弃（不打断、不处理）
    return
# addressed is None（超时/失败）→ 保守放行（宁误接不误拒）
self._interrupt_current_playback()   # 确认是真用户话 → 立即打断当前 TTS
await self._route_user(text)
```

注意 `None`（判断失败）的处理是**保守放行**——宁可误接一句环境音，也不要因为分类器抽风而漏掉用户真正的话。**在"过度响应"和"漏掉用户"之间，选择偏向用户。**

## 4.3 快答 vs 委派：route=self 的独立 QA 队列

放行的用户话，会分诊成两条路（`_route_user`）：

- **route = self**：像"你好""几点了""再说一遍"这种，VoiceAgent 用一个轻量 LLM 自己产一句口语回答，**秒回**，根本不惊动主 Agent。
- **route = main_agent**：真正需要主 Agent 能力的（"帮我查一下""分析这段视频"），先说一句承接语（"好嘞，这个我让 Argus 去办，稍等哈"），同时把作业丢进理解队列。

route=self 的秒答，走一条特殊的**快路径**（`skip_phrase=True`）——不进队列、不拟词、fire-and-forget 直送 TTS，保证话音一落立刻响应。

这里有一个精巧的设计：**fast-reply 的 QA 有一个独立的对话队列 `_qa_dialogue`，不进主 Agent 的 history**（`_record_qa`，`voice_agent_v2.py:410`）：

```python
def _record_qa(self, user_text, self_answer):
    """把一轮 fast-reply QA (route=self) 成对记入独立 QA 对话队列。
    该队列不进主 Agent，仅作 VoiceAgent 后续对话的 context（与 history 独立）。"""
    if u: self._qa_dialogue.append({"role": "user", "content": u})
    if a: self._qa_dialogue.append({"role": "assistant", "content": a})
    if len(self._qa_dialogue) > 12:
        self._qa_dialogue = self._qa_dialogue[-12:]   # 只留最近 ~6 轮
```

为什么要独立？因为这些秒答是 "VoiceAgent 自己和你的闲聊"，不是"主 Agent 和你的正式对话"。如果把"几点了—三点半"这种闲聊混进主 Agent 的 history，会污染主 Agent 的上下文。但 VoiceAgent 自己后续要接得上话（你问完"几点了"再问"那还有多久下班"），又需要记得刚才聊过什么。所以给它一条独立的、有上限的短记忆——**闲聊归闲聊，正事归正事，两本账分开记。**

在对话模式下，这些快问快答**完全不发用户气泡**（前端发 `text=""` 只清"正在聆听"信号），既不污染主���天区，也不进 history。

## 4.4 唯一强制门 `_flush_to_tts`：喇叭没开就绝不出声

这是整个"说"层最重要的一个设计。

一个语音 Agent 有太多可能"出声"的地方：用户秒答、主 Agent 作业结果、监控命中、深研推送……如果每个地方各自判断"现在该不该出声"，迟早有一个地方判漏，出现"用户明明关了喇叭，它却还在说话"的尴尬。

Argus 的解法：**所有 TTS 出口，无一例外，全部收敛到一个函数** `_flush_to_tts`（`voice_agent_v2.py:862`），把"喇叭是否开"的最终判定收在这一行：

```python
async def _flush_to_tts(self, spoken, source):
    """★ 统一强制门（唯一出口）：全 v2 的 TTS 都从这里出去 —— 秒回快路径、
      交互 worker、决策层放行，无一例外。所以把"喇叭是否开"的最终判定收在
      这一行，杜绝任何"喇叭没开但仍能出声"的旁路。"""
    if not self.is_speaker_on():
        return                            # ★ 喇叭关 → 直接静默
    rid = "va2_" + source + "_" + secrets.token_hex(3)
    self._engine.enqueue_tts(spoken, rid)
    self._engine.finish_tts(rid)
```

`is_speaker_on` 的语义，是"对话模式统一接管喇叭"的关键：

```python
return bool(s.get("_mm_tts_on") or s.get("_mm_voice_dialog_on"))
# 喇叭单独开 → 照播；对话模式开 → 后台强制 TTS，哪怕前端喇叭按钮显示"关"。
```

一个容易踩错的细节：交互 worker 的消费门故意收在 `is_speaker_on`，**而不是** `is_interactive`（是否对话模式）。因为如果收在 `is_interactive`，那"喇叭开着、但对话模式关着"时，深研/监控结果会入队却永远没人消费——积压。**门要开在"喇叭"这个最终物理开关上，不是开在某个模式状态上。**

> **设计哲学：** 当一个副作用（出声、写文件、发网络请求）有很多触发路径时，与其在每个路径上重复判断"该不该做"，不如让所有路径汇流到一个"唯一出口"，把判断收在那里。多一个旁路，就多一个漏判的可能。

## 4.5 per-segment 串行队列：一句话分段说，时间线不断

主 Agent 的回答是流式吐出来的，一段一段来。TTS 也应该一段说完接着说下一段，而不是等整段答案生成完再一次性合成（那样延迟太大）。

引擎侧（[agent/multimodal/watcher_engine.py:476](agent/multimodal/watcher_engine.py)）用**单队列 + 单消费者**做 per-segment 串行播放。关键在于：**同一轮回答的所有 segment 共用同一个 `response_id`**，前端就把它们的 PCM 接到同一条播放时间线上：

```python
async def _tts_consumer(self):
    while True:
        kind, text, rid = await q.get()
        if kind == "seg":
            # segment 之间不发 is_final —— 保持时间线开着，下一段接续而不是重启
            await self._run_tts(text, rid, send_final=False)
        elif kind == "end":
            emit("multimodal.tts", {"response_id": rid, "pcm_b64": "",
                                    "is_final": True})   # 只有结束才发终止
```

段与段之间不发 `is_final`（否则前端会以为这轮播完了、下一段变成新的一轮从头播），只有全部说完才发一个 `"end"` 哨兵触发终止。

底层 TTS 优先走 DashScope 实时（`QwenRealtimeTTS`，流式 PCM），没配 key 才回落到文件式 `TTSClient`。

### TTS 1009：一句太长会撑爆 WS 帧

一个真实的坑：DashScope 的 WS 单帧上限 256KB，一次 `input_text_buffer.append` 文本太长，服务端直接回 **1009（message too big）** 关连接。修复是按 UTF-8 字节安全上限把长文本切片 append（[agent/multimodal/qwen_realtime.py:41](agent/multimodal/qwen_realtime.py)），**不切坏多字节字符**：

```python
_TTS_APPEND_MAX_BYTES = 60000    # 留足 JSON 包裹余量（≈2万汉字）
def _chunk_text_by_bytes(text, max_bytes):
    ...
    for ch in text:
        b = len(ch.encode("utf-8"))
        if size + b > max_bytes and buf:
            yield "".join(buf); buf, size = [], 0
        buf.append(ch); size += b
```

短文本仍是一次 append（行为不变），只有超长才分片。**协议层的硬限制，要在应用层优雅地绕过，而不是让它偶发地炸。**

## 4.6 barge-in：两端约定的"立刻闭嘴"

第 2 章我们从"听"的角度讲过 barge-in 的触发。这里从"说"的角度，看被打断的一方怎么干净利落地停下来。

后端打断（`_interrupt_current_playback`，`voice_agent_v2.py:373`）做两件事：cancel 底层 TTS 消费任务 + drain 引擎队列；清 v2 侧的回播队列和已调度记录。但光这样不够——**前端可能还有一队 PCM 在 WebAudio 里排着播**。后端不知道当前正在播的 rid 是什么，于是发一个通用哨兵（`watcher_engine.py:570`）：

```python
emit("multimodal.tts", {"response_id": "__interrupt__", "pcm_b64": "",
                        "sample_rate": 24000, "is_final": True})
```

前端 `onTtsChunk` 第一件事就是认这个哨兵（`MultimodalChatPage.tsx:2072`）：

```typescript
if (rid === "__interrupt__") { stopAllTts(true); return; }
```

`stopAllTts` 停掉所有活跃的 WebAudio 源、把 rid 加进"已取消"集合、并**立即解除麦静音**（打断成功了，该马上听用户说）。

这个哨兵机制，是踩坑踩出来的：早期前端只按 rid 匹配当前正播的响应，这个哨兵匹配不上任何 rid、被忽略了，已��到的 PCM 继续播完——"打断没效果"。**打断这件事，必须两端约定一个都认识的信号。** 一端喊停，另一端得听得懂。

## 4.7 温暖化：让它像朋友，而不是客服

一个语音助手最容易掉进的坑，是说话像客服："好的""收到""正在为您服务""根据您的问题……"。这种腔调，听三句就烦。

Argus 的温暖化，藏在 v2 的**三处 prompt** 里（[agent/multimodal/voice_agent_v2_context.py](agent/multimodal/voice_agent_v2_context.py)）：

**分诊 prompt**（`_DECIDE_ROUTE_SYSTEM`）：

> 你说话像用户熟悉的一位朋友——自然、有温度、会共情。可以用"嗯我在""这个我知道呀""哈哈"这样的口语和轻松语气词，让人觉得是在和真人聊天，而不是查资料。但别肉麻、别客套、别啰嗦，也不要用"好的""收到""为您服务"这类客服腔。

**口播编辑 prompt**（`_PHRASE_UTTERANCE_SYSTEM`）：

> 你是语音助手的"口播编辑"。上游已经决定"现在要说这条内容"，你的唯一任务是把它改造成一句自然口语的话交给 TTS 播出去——就像一位熟悉的朋友温暖、自然地把这件事讲给用户听：有温度、抓重点、不啰嗦。

**决策 prompt**（`_DECIDE_SPEAK_SYSTEM`）：定调"你和用户正在语音面对面交流"。

三处分别管"怎么分诊""怎么措辞""要不要说"，但共享同一种人格。**温度不是一个开关，是渗透在每个决策环节里的一致风格。**

（注：温暖化不用本地 0.5B——那个小模型只做意图分类，而且目前全停用、走远端 deepseek。旧的 `voice_rewrite.py` 也承载过温暖化，但已被这三处取代、成了死代码。）

## 4.8 对话模式统一门控：一个开关，接管麦和喇叭

对话模式（voice dialog）是个总开关：打开它，就进入"面对面聊天"状态——自动开麦、强制 TTS、锁住单独的麦/喇叭按钮；关闭它，一切复位。

后端极简（`server.py:11447`）：`voice_dialog_toggle` 只写一个标志位 `_mm_voice_dialog_on` + 懒建 VoiceAgent。这个标志被两处读：
- **输入侧** `is_interactive()` 读它——对话开时，ASR final 走 v2 分诊（快问快答不进主聊天区）；
- **输出侧** `is_speaker_on()` 读它——`_mm_tts_on OR _mm_voice_dialog_on`，对话开时强制 TTS，无需前端单独开喇叭。

这个 OR 有个隐藏的连带修复：主 Agent turn 完成的播报也用同一个 OR（`server.py:10334`）。否则对话模式下 `notify_main_reply` 永不触发，委派给主 Agent 的作业 Future 永不 resolve，300 秒后必然"任务处理超时了"。**一个状态判断写漏一处，就是一个 5 分钟后才暴露的诡异超时。**

前端的联动是用户明确要的方案——**UI 按钮态保持不变，仅后台联动**（`MultimodalChatPage.tsx:3675`）：对话开就后台 `startMic`，关就 `stopMic`；对话开着时单点麦按钮被拦截 + toast 提示"麦克风已由对话接管"；反向地，如果在对话态下停了麦，就强制关掉对话（对话必须有活麦，否则哑火）。一个开关，把麦和喇叭都收编了。

## 4.9 快照双字段：为什么"我说的"和"主 Agent 说的"不能合并

VoiceAgent 做每个决策（该不该说、怎么措辞、分诊到哪）时，需要一份"世界快照"（`WorldSnapshot`，`voice_agent_v2_context.py:29`）。快照里有两个看起来可以合并、却刻意分开的字段：

- `recent_dialogue`——主 Agent history 里最近几轮（后台主 Agent 和用户的正式对话）；
- `voice_qa_dialogue`——VoiceAgent 自己的 fast-reply QA 队列（4.3 节那条独立闲聊）。

为什么不合并成一份"对话历史"？三条根据：

1. **语义不同**：一个是"主 Agent 说的"，一个是"你自己（VoiceAgent）说的"。合并会让判断 LLM 混淆"这话到底是谁说的"。
2. **判重复的准确性**：`recent_dialogue` 里可能有 watcher 分析的原文，但那是后台推送记录，**不代表"你已经用嘴播过"**。prompt 特意点明：判重复只看内容是否真一样，别因为"话题相似"就压制该说的话。
3. **各 LLM 按需取子集**：分诊只需要 `voice_qa_dialogue + trigger_event`（主 Agent 的正式对话对"self 还是 main_agent"的判断反而是干扰）；拟词去掉 `recent_dialogue` 省 token；决策两者都留。合并成一份，就没法做这种细粒度裁剪了。

**同一段"对话历史"，在不同决策里扮演不同角色——把它拆成语义清晰的字段，比揉成一坨更好用。**

## 本章小结

"说"这一层，是把知识变回自然、有温度、可打断的人类对话：

- VoiceAgentV2 是常驻对话大脑，三级优先级队列保证"用户直问永远最先答"；
- 输入三层过滤（本地规则→意图分类），`None` 时保守偏向用户；
- route=self 秒答走独立 QA 队列，闲聊不污染主 Agent history；
- `_flush_to_tts` 是唯一强制门，喇叭没开绝不出声——副作用汇流到一个出口；
- per-segment 串行队列用同一 rid 保持播放时间线；1009 靠字节安全分片绕过；
- barge-in 靠 `__interrupt__` 哨兵两端约定，一端喊停另一端听得懂；
- 温暖化渗透在三处 prompt 里，让它像朋友而非客服；
- 对话模式一个开关接管麦和喇叭，OR 逻辑连带修复播报超时；
- 快照双字段隔离"我说的"与"主 Agent 说的"，是判重复和裁剪的前提。

至此，一个会看、会听、会想、会说的 Agent 完整了。最后一章，我们跳出四层，看那些贯穿全局、决定系统能不能真正跑起来的工程经验。

---

*上一章：[第 3 章 · 想](03-想-让Agent拥有大脑.md) · 下一章：[第 5 章 · 贯穿全局的工程经验](05-贯穿全局的工程经验.md)*
