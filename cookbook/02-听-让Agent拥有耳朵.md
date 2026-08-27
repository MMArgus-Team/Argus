# 第 2 章 · 听：让 Agent 拥有耳朵

> "听见声音很容易，难的是分清哪句话是在跟你说。"

给 Agent 装耳朵，比装眼睛更微妙。画面是被动的——你共享什么它看什么。但声音是混杂的：你在跟它说话，电视在放新闻，旁边同事在打电话，视频里的人在讲解。这些声音一股脑涌进麦克风，Agent 得先分清"哪句话是给我的"，才谈得上回应。

而且声音有画面没有的紧迫性：你一开口，它就该停下正在说的话来听你——这是"打断"，是人类对话最基本的礼貌，也是这一层最难做对的地方。

这一章我们讲 Argus 怎么听：音频怎么采、怎么送进 ASR、上游连接死了怎么自愈、麦克风的生命周期怎么管、怎么用一个本地小模型判"这话是不是在跟我说"、以及怎么在你一开口的瞬间就闭嘴。

## 2.0 两条并行的"听"链

先建立地图。Argus 的"听"其实是**两条并行的链**，共用同一个云端实时端点，但服务完全不同的目的：

**（一）用户语音链——低延迟、单说话人。** 你对着助手说话。前端麦 → 降采样成 16k PCM → `multimodal.asr_audio` → 流式 ASR → 识别出的 partial/final 文本 → 送去分诊、判断是不是真在跟助手说话 → 提交给主 Agent 或 VoiceAgent。这条链要**快**，快到能支持打断。

**（二）环境音频链——多说话人、和画面对齐。** 视频/会议里的人在说话。前端每 5 秒切一片 → `multimodal.env_audio` → 批式 STT → 写成 `audio_observation` 存进记忆，时间戳**对齐到帧时钟**。这条链不追求实时，追求的是"这段画面里，谁在什么时候说了什么"。

两条链的分工，正对应"听当下"和"听历史"——和"看"那一层一次性 QueryWorker vs 持续 Watcher 的分工，是同一种哲学。

## 2.1 音频怎么采：专用线程上的降采样

采集从麦克风开始（[apps/desktop/src/store/multimodal-voice.ts:78](apps/desktop/src/store/multimodal-voice.ts)）。`getUserMedia` 拿麦流时，强制单声道 + 全套软件 3A：

```ts
audio: {
  echoCancellation: true,   // 回声消除
  noiseSuppression: true,   // 降噪
  autoGainControl: true,    // 自动增益
  channelCount: 1           // 单声道（ASR 要 16k mono）
}
```

但注释里有一句很清醒的自知之明：这是浏览器/Electron 的天花板——能压稳态噪音，**做不到波束成形或指向性拾音**，所以附近的人声照样漏进来。硬件层做不到的事，只能在软件层兜。

降采样的活儿，交给一个**专用音频线程**（AudioWorklet），而不是主 JS 线程——这样 UI 不会被卡（[apps/desktop/public/pcm-worklet.js](apps/desktop/public/pcm-worklet.js)）：

```js
this.inRate = Number(opts.inRate) || sampleRate;  // AudioContext 原生率，通常 48k
this.outRate = 16000;                              // ASR 要的 16k
this.ratio = this.inRate / this.outRate;
// 攒够 ~200ms 一批才 postMessage
this.batchTargetSamples = Math.max(1600, Math.floor(200*16000/1000));
```

- **输出格式**：PCM16（int16 小端）、16000 Hz、单声道、raw bytes。
- **降采样算法**：线性取点 + **跨帧的分数相位 carry-over**（`pcm-worklet.js:44`）。128 样本的处理帧对不齐任何整数抽取比，不带相位就会在帧边界产生咔哒声——这是音频处理里典型的坑。
- **批处理**：攒够 ~200ms 才发一批，比旧的 ScriptProcessor（85ms）消息率降 2–3 倍。服务端 VAD 的静音判断不在乎包节奏，所以批大点无害、还省开销。

主线程只做最轻的事：base64 编码 + 一次 RPC。中间还有一道**回采门** `micGatedForTts()`（2.6 节会讲），TTS 正在播的时候直接把麦音丢掉，不发给 ASR。

## 2.2 流式 ASR：DashScope 实时 WebSocket 协议

用户语音链的核心是 `QwenRealtimeASR`（[agent/multimodal/qwen_realtime.py](agent/multimodal/qwen_realtime.py)），基于 async `websockets` 库，接的是 DashScope（通义千问）的实时端点：

```python
_BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
url = f"{_BASE_URL}?model={self.model}"     # qwen3-asr-flash-realtime
headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}
```

key 为空则整个 ASR 是 no-op——又一处优雅降级，没配 key 不报错，只是实时语音功能静默不可用。

### 配置 VAD：多耐心地判"说完了"

连上后发一个 `session.update` 配置服务端 VAD（`qwen_realtime.py:172`）：

```python
"turn_detection": {
    "type": "server_vad",
    "threshold": self.vad_threshold,            # 默认 0.5
    "silence_duration_ms": self.vad_silence_ms, # 默认 1200
}
```

这两个数字是"听觉的性格"：`threshold` 越高越不敏感（不容易被噪音触发，但轻声说话可能被吞）；`silence_ms` 越大越"耐心"（容忍说话中间的停顿，不会你一喘气就切句）。config 注释解释了为什么把默认值调宽：demo 常用的 `0.7 / 500ms` 会在用户停顿思考时切断句子，`0.5 / 1200ms`（约 2 倍的等待静音）更像"一个耐心听你说完的人"。

### session.updated ack：防第一句被吞

这是整个协议里最容易被忽略、却最影响体验的一个细节。`connect()` 的步骤顺序是精心设计的（`qwen_realtime.py:154`）：

```python
self._reader_task = asyncio.create_task(self._reader())  # ① 先起 reader
await self._send_session_update()                        # ② 再发 session.update
try:
    await asyncio.wait_for(self._session_ready.wait(), timeout=3.0)  # ③ 阻塞等 ack
except asyncio.TimeoutError:
    log.warning("session.updated not received within 3s — first utterance may clip")
```

为什么必须等这个 ack？因为**如果不等就开始喂音频，服务端的 VAD 还没 armed，第一句话的开头会被直接丢掉**——用户会觉得"我说的第一句它没听见，从第二句才开始识别"。3s 超时是兜底：服务端万一不回 ack，也不能把麦克风永久挂死，降级为"当作 ready"继续。

> **工程细节：** `_session_ready` 是个 `asyncio.Event`，reader 收到 `session.updated` ��� `session.created` 就 set。这也是为什么 `connect()` 开头要先 cancel 旧 reader、close 旧 ws、`_session_ready.clear()`——因为死 socket 自愈会用**同一个对象**重连，不清理干净就会卡在旧的 ready 事件上。

### 事件处理：partial、final、speech

reader 循环分发服务端事件（`qwen_realtime.py:201`）：

```python
et = data.get("type")
if et in ("session.updated", "session.created"):     # ack
    self._session_ready.set()
elif et == "conversation.item.input_audio_transcription.text":       # partial
    await self.on_partial(data.get("text"))          # 字段名 text
elif et == "conversation.item.input_audio_transcription.completed":  # final
    await self.on_final(data.get("transcript"))      # 字段名 transcript
elif et == "input_audio_buffer.speech_started":      # VAD 开口 → barge-in
    await self.on_speech_started()
elif et == "input_audio_buffer.speech_stopped":
    await self.on_speech_stopped()
```

- **partial**（`text`）：渲染"正在聆听…"的实时预览。
- **final**（`transcript`）：一整句说完，提交处理。
- **speech_started**：VAD 检测到开口，这是 barge-in 打断的触发点（2.6 节）。

reader 退出时（无论是 WS 自己断还是报错），`finally` 里把 `self._connected = False`——**这就是死 socket 检测的锚点**。

## 2.3 死 socket 自愈：长会话不"聋"

长会话里最阴险的 bug 是：上游 DashScope 的 WS **静默死掉了**（网络抖动、服务端 idle 超时……），死了之后 `append_audio` 只是默默把音频丢进虚空，前端毫无察觉——表现就是"聊着聊着，语音突然没反应了"。

自愈分两半：**怎么知道它死了**，和**死了怎么复活**。

### 怎么知道死了：`is_connected`

```python
# qwen_realtime.py:112
@property
def is_connected(self) -> bool:
    return bool(self._connected and self._ws is not None)
```

`_connected` 只在 `connect()` 成功时为 True，在 reader 的 `finally`（WS 断掉）里变 False。所以这个属性就是一句问话："上游还活着吗？"

### 死了怎么复活：喂音频前探活 + 去抖重连

关键在喂音频的热路径上做探活（[agent/multimodal/watcher_engine.py:687](agent/multimodal/watcher_engine.py)）：

```python
def asr_audio(self, key: str, pcm: bytes) -> None:
    asr = self._asr.get(key)
    if asr is None or not pcm: return
    if not getattr(asr, "is_connected", True):        # ★ 上游死了
        if key not in self._asr_reconnecting:         # 去抖：同 key 只重连一次
            self._asr_reconnecting.add(key)
            asyncio.run_coroutine_threadsafe(self._asr_reconnect(key), loop)
        return                                        # 当前这片音频丢弃
    asyncio.run_coroutine_threadsafe(asr.append_audio(pcm), loop)
```

**去抖是这里的灵魂。** 音频每 ~200ms 来一片，如果不去抖，一旦上游死了，就会疯狂并发地发起几十个 `connect()`。用一个 `_asr_reconnecting` 集合做"in-flight 门"，同一个 key 只允许一个重连在跑：

```python
async def _asr_reconnect(self, key: str) -> None:
    try:
        asr = self._asr.get(key)
        if asr is None: return       # 已被 asr_stop 移除 → 放弃
        await asr.connect()          # 复用同对象 + 同回调重建 WS
    finally:
        self._asr_reconnecting.discard(key)   # 无论成败都清门
```

重连**复用同一个 ASR 对象和同一组回调**——这也是前面 `connect()` 要小心清理旧状态的原因。当前这片音频丢了没关系，下一片到时如果已经重连好，就正常识别了。用户几乎无感，最多丢半句话的开头。

前端还有一层对称的自愈：网关重连后服务端 session 被回收，前端会拆掉本地音频图、用新 sid 重开麦（`rearmMicAfterReconnect`，`multimodal-voice.ts:171`）。两端各自守住自己那一半。

## 2.4 麦克风生命周期：3 态按钮与"第一句不丢"

麦克风按钮有三个状态（`multimodal-voice.ts:21`）：`idle`��灰）→ `connecting`（转圈）→ `recording`（红点）。

- 点击开麦：先 `connecting` → 调 `asr_start` 等 WS 真连上 → `getUserMedia` + 建 worklet → `recording`。任何一步失败都回 `idle` 并释放资源。
- 点击停麦：**立即** `idle`——停录瞬间红点就消失，不残留。

"第一句不丢"靠前后端两处协作：

1. **`asr_start` 是阻塞的**（`watcher_engine.py:609`）：`fut.result(timeout=12.0)` 一直等到 WS 真连上才返回。前端拿到返回才开始采集——**采集晚于连接**，不会往还没连好的 socket 里灌音频。
2. **`connect()` 内等 `session.updated` ack**（2.2 节）：即便前端立刻开喂，服务端 VAD 也已 armed。

两道保险叠加，第一句话的开头才稳稳落进 ASR。

还有一个藏得很深的坑，值得单独点出。当 ASR 识别出 final、要提交处理时，这个回调是跑在 WatcherAgent 的 asyncio loop 上的。如果在这里**内联**跑完整个同步的主 Agent turn（里面还有串行 TTS 队列、还在喂 ASR 音频），就会把那个 loop 饿死——表现是"打字不卡，但一用语音就卡"。解法是把 turn 扔到**专用线程**去跑（`server.py:11235`）。这个"别在事件循环上干重活"的教训，你在"看""听""说"每一层都会再遇到。

## 2.6 barge-in：你一开口，它就闭嘴

打断，是对话式 AI 最能体现"活人感"的一个细节。你话说到一半觉得它答偏了，一开口，它就该立刻停下来听你。做不到这一点，再聪明的 Agent 也像个只会自顾自念稿的机器。

Argus 的 barge-in 有**两条触发，快���互补**。

### 快路：VAD 一开口就打断（不等识别）

这是 `speech_started` 事件的核心用途。链路是：ASR reader 收到 `speech_started`（VAD 检测到有人开口）→ `on_speech_started` 回调 → 网关（`server.py:11255`）：

```python
def _on_speech_started() -> None:
    va2_ = session.get("_mm_voice_agent")
    if va2_ is None or not va2_.is_interactive(): return  # 仅对话模式
    va2_._interrupt_current_playback()                    # 立即停 TTS + 清回播队列
```

为什么要用 VAD 而不是等识别结果？因为等 ASR 出 final + 意图分类要 2–3 秒，用户会明显觉得"我都说话了它还在念"。**用户一开口（VAD 层）就先停**，哪怕后面发现是环境噪音误触，也比"打断有 3 秒延迟"体验好。

### 确认路：意图过滤放行后再打断

快路可能被环境音误触（电视里有人说话，VAD 也会响）。所以还有一条确认路：`VoiceAgentV2` 里，本地规则（太短/纯语气词/短窗重复）+ 意图分类都放行，确认是"真用户在说话"，再打断一次（`voice_agent_v2.py:369`）。两条路一快一稳，配合兜底。

### `__interrupt__` sentinel：让前端立即停播

后端打断时，光 cancel 自己的 TTS 任务不够——前端可能还有一队 PCM 在排着播。所以引擎会发一个**哨兵**告��前端"立刻全停"（`watcher_engine.py:570`）：

```python
emit("multimodal.tts", {"response_id": "__interrupt__", "pcm_b64": "",
                        "sample_rate": 24000, "is_final": True})
```

前端识别这个特殊 rid 就全停（`multimodal-voice.ts:286`）：

```ts
if (rid === '__interrupt__') { stopAllTts(); return }
```

这里藏着一个真实修过的 bug：早期前端只按 rid 匹配当前正在播的响应，这个哨兵匹配不上任何 rid、被忽略了，已经收到的 PCM 继续播完——表现就是"打断没效果"。加上对 `__interrupt__` 的特判才修好。**打断这件事，必须两端约定一个都认识的信号**。

### 回采门：防它听见自己

除了主动打断，"听"侧还有一道防自己听自己的门 `micGatedForTts()`（`multimodal-voice.ts:252`）：TTS 播放期间（加 300ms 尾巴），**直接把麦音丢掉、不发给 ASR**。否则喇叭放出来的 TTS 会被麦重新采进 ASR，形成"它听见自己说话，以为用户在说话"的回环。浏览器的 echoCancellation 压不住大音量外放，所以要这道软门补上。

## 2.7 音视频时间基对齐：让声音和画面对得上

最后一个问题，回到序章埋的伏笔：画面和音频来自不同的客户端时钟，怎么让它们对齐？

如果音频用客户端自己的 `performance.now`（recorder 启动时锚定的 epoch），画面用服务端单调时钟，这就是两个不同的 epoch。记忆写入时按"提问那一刻往前 N 秒"去取音频窗，就会取错甚至取空，omni 模型引用的话和当时的画面对不上。

解法是**让音频也戳到帧时钟上**。帧时间线的锚点在 FrameBuffer（`_memory.py:257`）：第一帧到时锚定 epoch，之后每帧 ts = `monotonic - epoch`。它暴露一个 `now_ts()`：

```python
def now_ts(self) -> Optional[float]:
    with self._lock:
        if self._mono_epoch is None:
            return None           # 还没帧 → 让调用方兜底
        return monotonic() - self._mono_epoch
```

环境音频进来时，就用这个 `now_ts()` 打戳，而不是客户端的时钟（`server.py:11128`）：

```python
window_ts = buf.now_ts()          # ★ 用帧时钟戳环境音频
if window_ts is None:             # 还没帧锚定 epoch → 才退回客户端 window_ts
    window_ts = params.get("window_ts")
backend.submit_env_audio(audio_bytes, window_ts=window_ts)
```

这样，画面帧和音频观察就落在**同一条相对时间轴**上。下游记忆写手按帧 ts 开一个时间窗，就能同时拿到"这一刻的画面"和"这一刻的话"——声音和画面终于对得上了。这条对齐，是第 3 章"想"这一层能把"谁在什么时候说了什么"写进记忆的前提。

## 本章小结

"听"这一层要解决的，是"混杂"和"紧迫"两个词：

- 两条并行链：用户语音（快、单人、可打断）与环境音频（对齐画面、多人、进记忆）；
- 音频在专用线程降采样成 16k PCM，主线程只做最轻的转发；
- 流式 ASR 的 WebSocket 协议里，等 `session.updated` ack 是"第一句不丢"的关键；
- 死 socket 用 `is_connected` 探活 + 去抖重连自愈，长会话不"聋"；
- 麦克风 3 态生命周期 + 阻塞式 asr_start 保证首句稳落；
- barge-in 快慢两路 + `__interrupt__` 哨兵 + 回采门，做出"一开口就闭嘴"；
- 音视频统一戳到帧时钟，声音和画面才对得上。

它听见了，也听懂了是在跟自己说话。下一章，我们进入它的大脑——看它如何"想"。

---

*上一章：[第 1 章 · 看](01-看-让Agent拥有眼睛.md) · 下一章：[第 3 章 · 想](03-想-让Agent拥有大脑.md)*
