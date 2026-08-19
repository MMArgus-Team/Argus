# Argus — 多模态实时视频 Agent

[English](README.md) · [简体中文](README.zh-CN.md)

Argus 是一个能"边看边聊"的多模态 AI 助手：主 Agent 处理用户文本与语义路由，**不会被动收到直播画面**；一次性视觉问题和持续视频任务分别交给专用 worker。三条多模态能力：

- **一次性视觉问答** —— 当前、历史或“画面实体 + 外部事实”的 mixed 问题统一进入 `query_multimodal`。QueryWorker 读取提问时刻的近期帧，再按需直接作答、召回历史记忆或检索外部资料。
- **显式原始帧读取** —— `get_current_frame` 只用于用户明确要求取回/展示/诊断最新原始帧，不是普通视觉问答的默认入口。
- **持续监控 / 深度研究** —— `set_monitor`（等某事件出现→提醒）与 `set_live_watcher`（后台持续逐段深研，产出过程与最终报告）两个后台 Agent。watcher 只有一种模式：从最近一段起、按 TTL + 帧数双门逐轮分析，直到视频流停止或用户停止（无 qa/analysis/research 分类）。

> 项目品牌为 **Argus**。命令行命令、环境变量和所有用户可见文案均已是 `argus` / `ARGUS_*`。仍保留上游 `hermes_*` 命名的只剩*内部实现*：Python 模块路径（`hermes_cli/`、`hermes_constants.py`）、部分函数名，以及少量后端契约值（如 `speech_provider: hermes`、`hermes-index` 技能来源）。这些用户看不到，刻意不改。

## Demo 演示

两段"边看边聊"的实况演示 —— 屏幕共享一段视频，向 Argus 提问，多模态 Agent 结合真实画面给出回答。

### 🇬🇧 英文演示

https://code.devops.xiaohongshu.com/liguankai/video_stream_demo/-/raw/dev/demos/demo_en.mp4

### 🇨🇳 中文演示

https://code.devops.xiaohongshu.com/liguankai/video_stream_demo/-/raw/dev/demos/demo_cn.mp4

<sub>720p H.264 压缩版（约 11-30 MB）—— 原始 4K 高清版本请见 <a href="https://github.com/MMArgus-Team/Argus/releases/tag/v0.1.0-demos">GitHub Release 页面</a>。</sub>

## 三种使用入口

同一套 Agent 运行时 + 记忆 + 技能，三种前端可选，按需取用：

| 入口 | 启动方式 | 适合场景 |
| --- | --- | --- |
| **Web 仪表盘** | `argus dashboard`（或 `python -m hermes_cli.main dashboard`，见 §3） | 最快上手；`/multimodal` 页边看边聊，本文档的评测流程也基于它。 |
| **桌面 App** | `argus desktop`，或见 [`apps/desktop`](apps/desktop/README.zh-CN.md) | 跨平台原生窗口，融合多模态视频 + 主聊天 + 文件预览 + 语音 + Computer Use，无需终端。 |
| **命令行 / 评测** | `argus ...`（等价于 `python -m hermes_cli.main ...`） | 离线记忆评测（§6）、批量跑数、脚本化调用。 |

## 系统架构

主 Agent **不被动持有画面**，只做文本理解与语义路由；视觉工作分派给专用 worker，全部角色共享同一个 `FrameBuffer`：

```
                    ┌─────────────────────────────────────────────┐
   用户文本 ───────▶ │  主 Agent（同步 ReAct 循环，语义路由）           │
                    └───────┬───────────────┬───────────────┬──────┘
       一次性视觉问题        │      set_monitor│  set_live_watcher│
       query_multimodal     ▼               ▼               ▼
                    ┌────────────┐   ┌────────────┐   ┌──────────────┐
                    │ QueryWorker│   │MonitorEngine│  │ WatcherAgent │
                    │ 当前/历史/  │   │ 事件出现→   │  │ 后台逐段深研  │
                    │ 混合问答    │   │ 提醒        │  │ 过程+最终报告 │
                    └─────┬──────┘   └─────┬──────┘   └──────┬───────┘
                          │                │                 │
      记忆召回 ◀──────────┤                │                 │
                    ┌─────▼────────────────▼─────────────────▼──────┐
                    │  FrameBuffer（前端固定 fps 推帧）                │
                    │   · 长期记忆 / watcher 读 dHash 去重后的稀疏流   │
                    │   · Monitor 读最近 60s 原始 2fps 短队列         │
                    │   · QueryWorker 只取提问时刻近期帧             │
                    └───────────────────────┬───────────────────────┘
                    ┌───────────────────────▼───────────────────────┐
                    │  MemoryBackend（分层视觉记忆 writer/reviewer）   │
                    │   + SceneDhashController（~20s 判场景，动态调    │
                    │     去重强度与 watcher 节奏）                   │
                    └────────────────────────────────────────────────┘
```

各角色职责详见文末 [架构速览](#架构速览)。

## 目录导览

| 路径 | 内容 |
| --- | --- |
| `hermes_cli/` | CLI 与 dashboard 入口（`python -m hermes_cli.main ...`）。dashboard 从 `web_dist/` 伺服静态文件，该目录是**构建产物** —— 不入 git，由 `npm run build` 生成（见 §1）。 |
| `web/` | Web 仪表盘前端（React + Vite），含多模态页 `/multimodal`。 |
| `apps/desktop/` | 跨平台桌面 App（Electron + React），见其 [README](apps/desktop/README.zh-CN.md)。 |
| `apps/shared/`、`apps/bootstrap-installer/` | 桌面/web 共享代码与引导安装器。 |
| `gateway/` | `tui_gateway` / dashboard 后端 API，前端经它与 Agent 运行时通信。 |
| `tools/`、`toolsets.py` | 工具实现与工具集分发。 |
| `run_agent.py`、`cli.py` | Agent 主循环与命令行核心。 |
| `config.yaml` | 所有 Key / Endpoint / 模型选择的唯一真相源（§2）。 |
| `convert_annotation_to_json.py`、`download_0618_videos.py` | 评测数据预处理（§6.2）。 |

## 环境要求

- Python 3.11（用 uv 创建隔离环境，避免系统 Python 版本问题）
- Node.js + npm（仅前端构建时需要；已构建过则不需要）
- ffmpeg（语音用 `speech_provider: hermes` 时需要；批量评测下载**最佳画质**视频时也需要——`bestvideo+bestaudio` 是分离流，靠 ffmpeg 合并成 mp4。只下单一 progressive 流则不需要）
- Node.js（仅批量评测下载 YouTube 时需要，用于解 YouTube 的 n-challenge）
- **cua-driver >= 0.20.0**（仅 Computer Use / 桌面控制需要）。低版本不支持：0.12.3 上 `launch_app` 会以 `exit_code: 1` 失败，连续失败后驱动会把整个 session 判死，之后所有工具调用都被 `session has ended` 拒绝。用 `cua-driver --version` 查看版本；升级用 `cua-driver update --apply`（先关掉 Argus 主进程，让旧的 stdio 子进程退出），或 `argus computer-use install --upgrade`。

## 1. 初始化（用 uv 创建 .venv 并安装）

```bash
cd /path/to/video_stream_demo

# 创建 Python 3.11 虚拟环境
uv venv --python 3.11 .venv

# 激活环境
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   Windows (cmd):
.venv\Scripts\activate.bat
#   Linux / macOS:
source .venv/bin/activate

# 安装 Argus 本体(editable)+ web 依赖
uv pip install -e ".[web]"

# 安装 JS 依赖 —— 必须在仓库根目录执行，不要在 web/ 或 apps/desktop/ 里装
# 本仓库是 npm workspaces monorepo，原因见下方警告
npm ci

# 构建前端 —— 新 clone 必做（web_dist/ 不入 git，初始为空）
cd web
npm run build
cd ..

# 可选:多模态记忆用 gemini 后端时需要 aiohttp
uv pip install aiohttp

# 可选:批量评测要下 YouTube 视频时需要 yt-dlp + openpyxl(读 Excel 标注)
uv pip install yt-dlp openpyxl
```

> editable 安装(`-e`)后,改后端 `.py` 源码无需重装,重启进程即可生效。
> `hermes_cli/web_dist/` **不在 git 里**（它是构建产物，和 `hermes_cli/tui_dist/` 一样）。clone 后需先跑一次 `npm run build`，之后每次改了 `web/**/*.tsx` 再重新构建 —— 否则 dashboard 会返回 404。发布流程会自动重建（见 `scripts/release.py`）。

### ⚠ npm：只能在仓库根目录安装依赖

本仓库是 **npm workspaces monorepo**（`apps/*`、`ui-tui`、`ui-tui/packages/*`、`web`）。所有 workspace 的依赖都会**提升(hoist)到唯一的根 `node_modules/`** —— lockfile 期望顶层有约 1178 个包，而根 `package.json` 自己只声明了 3 个。`vitest`、`typescript`、`jsdom` 这些其实属于 `web` / `ui-tui` / `apps/desktop`，出现在根目录纯粹是因为 hoist。所以 `web/node_modules` 和 `ui-tui/node_modules` **是空的，这是设计如此**，不是装坏了。

```bash
npm ci          # ✅ 在仓库根目录执行 —— 严格按 lockfile 安装，装前先清空 node_modules
npm install     # ⚠ 同样只能在根目录；可能与 lockfile 产生偏移
```

**永远不要按 workspace 单独安装：**

```bash
npm install --workspace web        # ❌ 不要
cd web && npm install              # ❌ 不要
cd apps/desktop && npm install     # ❌ 不要
```

按 workspace 安装会**以那一个 workspace 的视角重算整棵依赖树**，并把它认为多余的包**从共享的根 `node_modules` 里删掉**。于是"只装 web"会静默删掉 `apps/desktop` 的 `vitest` / `typescript` / `jsdom`。这个症状很有迷惑性，因为看起来谁都没动过东西：*"我从来没动过 `node_modules`，怎么包总是莫名少掉。"*（`install:web` / `install:tui` / `install:desktop` / `install:root` 这几个 npm script 就是因此被移除的。`npm audit --workspace <name>` 是只读的、安全；但 `npm audit fix --workspace <name>` 和 install 一样会重写依赖树，别用。）

**修复已经坏掉的依赖树** —— 把根目录和各 workspace 下的 `node_modules` 都删掉（残留的局部目录会继续遮蔽根目录），再重装：

```bash
rm -rf node_modules web/node_modules ui-tui/node_modules apps/desktop/node_modules && npm ci
```

**遇到 `tarball ... seems to be corrupted` 和满屏 `TAR_ENTRY_ERROR ENOENT`：** 那些 `ENOENT` 是*连带现象*、不是病因 —— npm 删掉解包失败的半成品目录去重试，而此时另一个并发任务还在往那个目录写。真正的故障是 tarball 校验和不匹配，也就是 `~/.npm/_cacache` 或私服上存了坏副本：

```bash
npm cache clean --force && npm ci
```

如果清完缓存还报，说明坏包在**私服**而不是本机（用 `npm config get registry` 看当前源）。可以先用公共源验证 `npm ci --registry=https://registry.npmjs.org/`，确认后找镜像维护方处理 —— 这种情况本地清缓存解决不了。

**检查依赖树是否可用。** 不要拿目录数去和 lockfile 比 —— npm 会对相同版本做去重(dedupe)，所以 `ls node_modules | wc -l` 本来就远小于 lockfile 的条目数，这个差距**不能说明任何问题**。真正有意义的是下面两项：

```bash
# 1. 可执行文件是否存在。安装中途失败会留下"包解开了但 .bin 是空的"状态
#    （npm 建软链接是最后一步）—— 这就是 "sh: concurrently: command not found" 的成因。
ls node_modules/.bin | wc -l && ls node_modules/.bin | grep -E "concurrently|vite|tsc|vitest"

# 2. 各 workspace 的直接依赖是否都能解析到（期望输出 missing 0）。
python3 -c "
import json, os
miss=[]; tot=0
for w in ('apps/desktop','web','ui-tui'):
    p=json.load(open(f'{w}/package.json'))
    need=set(p.get('dependencies',{}))|set(p.get('devDependencies',{})); tot+=len(need)
    miss += [f'{w}: {n}' for n in sorted(need)
             if not os.path.exists(f'node_modules/{n}') and not os.path.exists(f'{w}/node_modules/{n}')]
print(f'direct deps {tot}, missing {len(miss)}'); [print('  x', m) for m in miss[:25]]
"
```

!!注意：需要设置 ARGUS_HOME 环境变量为 config.yaml 所在位置（当前路径）

## 2. 配置模型

**`config.yaml` 位于项目根目录，是所有 Key / Endpoint / 模型选择的唯一真相源**（git 跟踪）。`argus dashboard` 启动时会把它单向复制到 `<ARGUS_HOME>/config.yaml`（项目版覆盖）。改配置就改项目根的这份，然后**重启 dashboard**。

每个子模块的模型都可独立配置。主 Agent 在 `model` 顶层；多模态各角色作为 `model` 的子段并列，用一致的 4 元组 `{provider, base_url, api_key, model}`（`base_url` 留空 = 跟随主 Agent）。行为旋钮在 `settings:`，语音接口(ASR/TTS)在 `audio:`：

```yaml
model:
  # 主 Agent（负责文本理解与语义路由；一次性视觉问答由 QueryWorker 接手）
  default: "kimi-k2.6"
  provider: "custom"
  base_url: "https://<your-openai-compatible-endpoint>/v1"
  api_key: "<your-key>"
  supports_vision: true

  # ① 监控 Agent（set_monitor：always-on 视频 SPEAK/SILENT + 事件合并）
  monitor:
    provider: "custom"
    base_url: "https://<endpoint>/v1"
    api_key: "<key>"
    model: "<vision-model>"
  # ② watcher = 深度研究 worker（set_live_watcher 的后台引擎）
  watcher:
    provider: "custom"
    base_url: "https://<endpoint>/v1"
    api_key: "<key>"
    model: "<vision-model>"
  # ③ 分层记忆 writer/reviewer（QueryWorker 可按需召回这个库）
  memory:
    provider: "gemini"               # 或 openai / 留空跟随主 Agent
    base_url: "https://<endpoint>"
    api_key: "<key>"
    model: "<vision-model>"
    vision_ability: true             # ★ 必须 true，否则记忆后端拒绝启动（见下）
    audio_ability: false             # true=原始音频直送 omni 模型；false=音频走外部 ASR
    recall:                          # 记忆召回子 Agent；留空跟随主 Agent
      provider: ""
      base_url: ""
      api_key: ""
      model: ""

settings:                            # 多模态行为旋钮（enabled / 提问时刻帧 / 监控 / anysearch …）
  enabled: true
  memory_enabled: true

audio:                               # 多模态语音接口（ASR / TTS / 环境音）
  asr_url: "http://<asr-endpoint>/asr"
  dashscope_api_key: "<key>"         # 流式麦克风 ASR + 流式 TTS
```

> 「边看边问」以及监控 / 深研 / 记忆写入都需要 **支持视觉(vision)的模型**。`argus doctor` 会校验：某视觉必需角色配了已知的纯文本模型（或 `supports_vision: true` 与实际能力矛盾）会打印警告——只提示、不强改。
>
> **记忆是硬性前置**：`model.memory` 段的 `vision_ability` 必须为 `true`（记忆抽取本质是视觉任务）。若配成 `false`，MemoryBackend 启动会**直接报错、不启动**（视频流也无法正常带记忆开启），而不是静默跑一个看不了图的模型。

## 3. 启动 Web 仪表盘(含多模态页面)

```bash
# 主 Agent + 多模态后端都在这个 dashboard 进程里
启动：argus dashboard --skip-build --port 9119
停止：argus dashboard --stop
```

| 参数 | 说明 |
| --- | --- |
| `--skip-build` | 跳过前端构建,直接用已构建的 `web_dist`(省时间) |
| `--no-open` | 不自动打开浏览器 |
| `--port <n>` | 指定端口(默认 9119) |
| `--stop` | 停止所有 dashboard 进程 |
| `--status` | 列出运行中的 dashboard 进程 |

启动后浏览器访问:

```
http://127.0.0.1:9119/multimodal
```

> 改了配置或后端代码后,需 **重启 dashboard** 才生效(进程不热加载)。浏览器记得 **Ctrl+Shift+R** 硬刷新。

## 4. 前端开发(改 .tsx 时,二选一)

```bash
# 方式 A:改完构建一次(dashboard 用 --skip-build serve 产物)
cd web && npm run build

# 方式 B:热更新 dev server(改 tsx 即时刷新,免 build)
#   终端1:后端
argus dashboard --skip-build --no-open
#   终端2:前端
cd web && npm run dev
```

## 5. 界面语言

| 界面 | 支持语言 | 如何决定 |
| --- | --- | --- |
| **Web 仪表盘** | English、简体中文 | 首次访问跟随浏览器/系统语言（任何 `zh-*` 标签 → 中文；不支持的语言 → 英文）。可在侧边栏切换，选择按浏览器存在 `localStorage`（`argus-locale`），因此不会影响共用同一仪表盘的其他人。 |
| **桌面 App** | English、简体中文 | 同样的首次探测逻辑，但持久化到 `config.yaml` 的 `display.language`。 |
| **命令行 / gateway** | 16 种语言（`en zh zh-hant ja de es fr tr uk af ko it ga pt ru hu`） | 依次读取 `ARGUS_LANGUAGE` 环境变量 → `config.yaml` 的 `display.language` → 英文。 |

> 命令行词表（`locales/*.yaml`）比 UI 覆盖更广是刻意的：它只覆盖 gateway 消息和审批提示（约 290 个 key），其余命令行输出仍是英文。两个 UI 界面只发布英文和简体中文，两者都完整且受类型检查——若英文加了 key 而中文漏了，构建会直接失败。

## 6. 记忆系统离线评测

离线读一个视频、走**与在线完全一致**的多模态记忆流水线建记忆(不涉及 monitor/watcher),再逐条 QA 经记忆召回作答,把预测答案写回 JSON。官方入口是主项目的 `mm-memory-eval` 子命令。

### 6.1 评测命令(单视频 / 整个文件夹)

```bash
argus mm-memory-eval <视频文件 或 视频文件夹> <QA的JSON>
```

第一个参数支持两种形态:
- **单个视频文件** → 评这一个视频(从 JSON 命中它对应的 qa_list)。
- **视频文件夹** → 评 JSON 里的**所有**视频。**开跑前先遍历检查**:JSON 里每个视频(按 `?v=<id>`)都必须在该文件夹里找到对应 `<id>.mp4`(等常见容器),**缺任一就报错、不开跑**。随后逐视频**全新建栈**评测(记忆互不串场),**单个失败跳过继续、末尾汇总**(有失败则退出码非 0)。文件夹模式必须用 §6.2 的全量数组 JSON。

- **输入 JSON**:`{"title": "<必须==去扩展名的视频文件名>", "qa_list": [{"query": "...", "answer": "...", "time": "HH:MM:SS"(可选)}, ...]}`。
  `title` 会**严格校验** = 视频文件名(去扩展名),不一致直接报错。
- **输出**:给每个 qa 元素追加 `"answer_predict"`(系统预测答案),默认原地写回输入 JSON(用 `--out` 可另存)。
  **注意:它只产出预测答案,不判对错、不算准确率** —— 判分是单独一步(见 §6.3)。

**命令行参数**(`argus mm-memory-eval <video> <json> [选项]`):

| 参数 | 位置/类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `video` | 位置参数 | (必填) | 视频文件路径,**或视频文件夹**。文件夹 → 评 JSON 里所有视频(先遍历检查全部存在,缺则不开跑)。单文件时文件名去扩展名须能在 JSON 里命中。 |
| `json` | 位置参数 | (必填) | QA JSON 路径,格式 `{title, qa_list:[{query, answer, time?}]}`。 |
| `--mode` | `tool` \| `agent` | `tool` | `tool`=直接记忆召回作答；`agent`=在离线评测器内先显式生成 Recall evidence，再与最近 3 帧一起交给主 Agent 综合作答。这是无交互 answer slot 的评测路径，不模拟在线 `query_multimodal` / QueryWorker handoff。 |
| `--source` | `camera` \| `screen` | `camera` | 离线视频模拟的线上输入源。`screen` 会开启屏幕共享 OCR/table 路径,并默认使用更高分辨率。 |
| `--answer-timing` | `before` \| `after` | `after` | 时序题到达 `time` 时的答题时机。`after` 表示先推入该帧并跑 OCR/writer wake,再答题,更接近线上用户看见画面后提问。 |
| `--query-types` | 字符串 | 不筛选 | 只评指定 `query_type`,逗号分隔,如 `b` 或 `a,b,d,e`。未命中的题保留原样,不会写 `answer_predict`。 |
| `--asr-vtt` | `auto` \| `none` \| 路径 | `auto` | 给静音视频补充已有 ASR 字幕,按 cue 结束时间流式注入 `audio_observation`。`auto` 会在视频同目录自动找 `<视频名>.auto.asr.vtt` / `.asr.vtt` / `.vtt`。 |
| `--scene-probe` / `--no-scene-probe` | bool | 开启 | 是否离线驱动 scene/dHash controller,用于对齐线上动态去重阈值。 |
| `--out` | 路径 | (原地覆盖输入 JSON) | 预测结果另存路径。不指定则把 `answer_predict` 写回输入 JSON。 |
| `--trace-out` | 路径 | 自动 sidecar | 每题 recall trace JSON 路径。不指定则写到结果 JSON 旁边,记录工具、frame_ids、OCR/table 命中摘要。 |
| `--capture-fps` | float | config 的 `buffer_capture_fps`(=2.0) | 视频解码采样帧率(与在线屏幕共享同帧率)。 |
| `--max-side` | int | `camera=720`, `screen=1536/ocr_max_side` | 帧最长边缩放上限(像素)。 |
| `--jpeg-quality` | int | `80` | 帧 JPEG 编码质量。 |
| `--timeout` | float | `120` | 单条 recall 召回的超时秒数。 |

**★ 时序评测(默认路径,按提问时间点作答)**:标注数据里每题都带 `time`(提问时间点,`HH:MM:SS` / `MM:SS` / 纯秒),所以**正常评测都走时序模式**:视频喂帧推进到某题的 `time` 那一刻,就用**截至此刻的记忆现状**回答该题(而不是整片喂完再统一答),然后继续往后喂——逼真还原"看到该时间点时系统知道什么"。tool / agent 两种模式都适用(答的是那一刻的记忆 / 主 Agent 现状)。
  - 触发条件:**qa_list 里只要有任一题带合法 `time` 就进入时序模式**,且此时**每一题都必须有合法 `time`**(缺失/无法解析 → 报错退出)。
  - 时间点晚于视频总长的题,在喂帧耗尽后用整片记忆补答(不丢题)。
  - (边角情形)若你手写一份**所有题都不带 `time`** 的 QA,则退化为非时序模式:整片喂完再逐题答。标注数据集不会走到这里。

例:

```bash
# 单视频 (第一个参数是文件)
argus mm-memory-eval test_0618/QyNunAw0sx4.mp4 test_0618/0618.json
argus mm-memory-eval test_0618/QyNunAw0sx4.mp4 test_0618/0618.json \
    --mode agent --out QyNunAw0sx4.pred.json

# 整个文件夹 (第一个参数是目录 → 评 0618.json 里所有视频, 先检查全在)
argus mm-memory-eval test_0618 test_0618/0618.json
```

### 6.2 生成评测数据 `xx.json`

`mm-memory-eval` 的输入 JSON 有两种:**单视频对象** `{title, qa_list}`(手写测一个视频),或 **全量数组** `[{video_url, qa_list}, ...]`(数据集,可配合文件夹模式一次评所有)。

**输入 JSON 的完整结构**:

```jsonc
{
  "title": "QyNunAw0sx4",           // 必须 == 视频文件名去扩展名 (会严格校验)
  "qa_list": [
    {
      "query":  "视频里那个人先拿起了什么?",   // 必填: 问题
      "answer": "一把螺丝刀",                  // 必填: 标准答案 (判分用, 系统不会看)
      "time":   "00:01:30"                     // 提问时间点 → 走时序评测 (§6.1); 标注数据每题都有
    },
    { "query": "他最后把工具放回哪了?", "answer": "工具箱第二层", "time": "00:03:12" }
  ]
}
```

字段规则:
- `title` **严格校验** = 视频文件名去扩展名(如 `QyNunAw0sx4.mp4` → `"QyNunAw0sx4"`),不一致直接报错。
- 每个 qa **必须**有 `query`(非空) + `answer`。
- `time`(提问时间点,`HH:MM:SS` / `MM:SS` / 纯秒)决定评测模式:**标注数据每题都带 `time`,所以正常都走时序评测**(§6.1)。**要么每题都有 `time`(时序,常规用法),要么整份都不带 `time`(非时序,边角用法)**;二者混用(一部分有一部分没有)→ 校验报错。

只测一个视频时,照上面手写一个 `xx.json`(`xx` == 视频名)即可,跳过下面的 Excel 流程。

---

**从标注 Excel 批量生成**(数据集场景),只需两步预处理,**无需拆分**:整个流程都围绕 `test_0618` 文件夹(标注 JSON 与视频 mp4 同目录)。

**① Excel 标注 → `test_0618/0618.json`**(转换脚本在项目根目录):

```bash
python convert_annotation_to_json.py                       # 默认: xlsx → test_0618/0618.json
python convert_annotation_to_json.py <标注.xlsx> --out-dir test_0618   # 显式指定
```

- 位置参数 `input`:标注 xlsx 路径(默认 `C:\Users\luyuan2\Downloads\0618.xlsx`,自行改)。
- `--out-dir`:输出目录(默认 `./test_0618`),产物是 `<out-dir>/0618.json`;或用 `--out <完整路径>` 直接指定文件。

产物是一个**数组**,每个元素 `{video_url, accuracy(标注), qa_list:[{query, query_type, answer, time, analyse}]}`。
列映射:`video_url←youtube URL`、`accuracy←该视频整体准确率`、`query←问题`、`query_type←题目类型`、`answer←回答（标准答案）`、`time←提问时间点`、`analyse←题目解析`(url + accuracy 每个视频块首行给值、块内向下填充)。

**② 下载视频 → `test_0618/<id>.mp4`**(id = url `?v=<id>` 的 name):

```bash
python download_0618_videos.py                             # 默认: 读 test_0618/0618.json → 下到 test_0618/
python download_0618_videos.py <目标文件夹> --json <清单.json>   # 显式指定
```

- 位置参数 `out_dir`:下载目标文件夹(默认 `./test_0618`)。
- `--json`:视频清单 JSON(默认 `./test_0618/0618.json`,即上一步产物)。断点续传(`.download-archive.txt` 记进度)。

> YouTube 会弹「确认非机器人」—— 需要 cookies。用浏览器扩展「Get cookies.txt LOCALLY」导出
> `youtube` 的 cookies.txt,放到 `./.yt_cookies.txt`(**勿提交**)。YouTube 的 n-challenge 需要 JS
> 运行时:脚本用 `--js-runtimes node:<node路径>` 指向本机 Node 解决;合并最佳画质(bv*+ba)需 ffmpeg。
> 下载完记得删掉 `.yt_cookies.txt`(含登录凭证)。

**③ 直接用全量 `0618.json` 评测(不拆分)**:`mm-memory-eval` 现在**直接吃全量数组 JSON** —— 命令内部按传入的**视频文件名**(去扩展名)匹配数组里 `video_url` 的 `?v=<id>`,命中就取那一段 qa_list 评测,**匹配不到直接报错**(不静默跳过)。`answer_predict` 写回该视频对应的数组块(其它块原样保留),整份 `0618.json` 就是结果文件。

```bash
# <video> 的文件名(去扩展名)必须能在 0618.json 里按 ?v=<id> 命中
argus mm-memory-eval test_0618/QyNunAw0sx4.mp4 test_0618/0618.json
```

**批量评所有视频 —— 直接传文件夹**(推荐):第一个参数给视频目录,命令会先遍历检查 `0618.json` 里每个视频都在目录内(缺则报错不开跑),再逐个全新建栈评测、单个失败跳过继续、末尾汇总:

```bash
argus mm-memory-eval test_0618 test_0618/0618.json
```

也可以外层自己对 `test_0618/*.mp4` 循环(等价,但少了"开跑前全在"的整体检查):

```bash
for f in test_0618/*.mp4; do
  python -m hermes_cli.main mm-memory-eval "$f" test_0618/0618.json
done
```

> 标注数据每题都带 `time`,所以命中的 qa_list **自动走时序评测**(§6.1)。命令仍兼容旧的单视频对象 JSON `{title, qa_list}`(见 §6.2 开头的手写格式)。
>
> 想每个视频结果分开存、不覆盖全量 `0618.json`,给 `--out <路径>`(仍写整份数组,只是另存)。

### 6.3 判分与准确率(单独一步)

`mm-memory-eval` 只写 `answer_predict`,**准确率要单独判**。对每题拿 `[query / answer(标准) / answer_predict]` 交给一个 LLM 裁判判**语义**对/错(同义等价即算对、不看字面;关键事实错/缺 → 错;预测为"没找到/记忆为空" → 错),再按视频/整体汇总成准确率 —— 与标注 Excel 的"是否答对(Gemini 评测)"列同思路。这一步与官方评测解耦,可复用 config 里已配的任一模型当裁判。

> **成本提示**:每个视频要先离线建记忆(逐拍跑 writer 模型),再逐条 recall 作答;全量 49 视频 /
> 651 题的 LLM 消耗与耗时都不小。建议先跑 1-2 个视频验证整条链,再全量。

## 架构速览

- **主 Agent**（同步 ReAct 循环）——处理用户文本与语义路由，不被动持有当前或历史画面；一次性视觉问题调 `query_multimodal`。
- **QueryWorker**（一次性回答所有者）——接管原用户问题和提问时刻的近期帧，根据问题本身选择当前画面 VQA、多模态 Recall、Search 或组合链路，并直接回复原消息。
- **WatcherAgent**（独立线程 + async loop）——`set_live_watcher` 的后台深研引擎（标准多模态 ReAct worker）。每轮过程/流式解读进右侧面板，每轮报告以折叠气泡进主 Agent 页，最终报告在完成时汇总；与主 Agent 聊天历史解耦（不写 history）。
- **MonitorEngine**（独立线程 + async loop，作业容器）——每个 `set_monitor` 监控 = 一个长期 async 作业。
- **MemoryBackend**——分层视觉记忆的 writer / reviewer，写入 QueryWorker 可按需召回的记忆库；同时挂一个 **SceneDhashController**，每 ~20s 用 `auxiliary.vision` 判场景、动态调 FrameBuffer 的去重强度与 watcher 节奏。

这些角色共享同一个 **`FrameBuffer`**：前端固定 fps 推帧；长期记忆和 watcher 读取 dHash 去重后的稀疏流，QueryWorker 只在一次性问题接管后获取提问时刻近期帧，Monitor 则读取同一 buffer 内最近 60 秒的原始 2fps 短队列，避免短暂目标被去重漏掉。

## 注意事项

- **不要提交含密钥的文件**：项目根 `config.yaml` 含明文 Key（按项目约定），`check_qwen.py` 等临时脚本可能硬编码了测试 key，批量评测会生成 `.yt_cookies.txt`（含你的 YouTube 登录凭证）——推送到公开仓库前务必清理，评测跑完建议 `rm .yt_cookies.txt`。
- **批量评测的大文件别入库**：`test_0618/`（下载的视频，动辄上百 GB）、`0618.json` / `results_0618.json` 视需要决定是否跟踪，视频目录建议加进 `.gitignore`。
- Windows 下若直接敲 `argus` 打开了明文文件,是因为 PATH 没找到 `argus.exe`——请先激活 `.venv`,或直接用 `.venv\Scripts\python.exe -m hermes_cli.main ...`。

## 常见问题（FAQ）

**Q：`ARGUS_HOME` 到底该设成什么？**
A：设为 `config.yaml` 所在目录（即项目根，当前路径）。`argus dashboard` 启动时会把根 `config.yaml` 单向复制到 `<ARGUS_HOME>/config.yaml`；改配置永远改项目根这份，再重启 dashboard。

**Q：改了配置 / 后端代码，为什么没生效？**
A：dashboard 进程不热加载——需**重启 dashboard**。前端记得 **Ctrl+Shift+R** 硬刷新。editable 安装（`uv pip install -e`）下改 `.py` 不用重装，但仍要重启进程。

**Q：改了 `web/**/*.tsx` 怎么办？**
A：要么 `cd web && npm run build` 重建产物（dashboard 用 `--skip-build` serve 它），要么开 `npm run dev` 热更新 dev server（见 §4）。注意 `hermes_cli/web_dist/` 不入 git，所以即使你没改过前端，新 clone 也必须先构建一次。

**Q：报「记忆后端拒绝启动」？**
A：`model.memory` 段的 `vision_ability` 必须为 `true`（记忆抽取本质是视觉任务）。配成 `false` 会直接报错不启动（见 §2）。

**Q：想要不用终端的图形界面？**
A：用桌面 App —— `argus desktop`，或见 [`apps/desktop`](apps/desktop/README.zh-CN.md)。它融合了多模态视频、主聊天、文件预览、语音与 Computer Use。

**Q：批量评测怎么从零跑通？**
A：§6.2 两步预处理（Excel→JSON、下载视频），再 §6.1 传视频文件夹一次评所有，最后 §6.3 单独判分。**先跑 1-2 个视频验证整条链，再全量**（49 视频 / 651 题成本不小）。
