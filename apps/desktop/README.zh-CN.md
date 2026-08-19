# Argus Desktop

[English](README.md) · [简体中文](README.zh-CN.md)

**Argus 的原生桌面客户端** —— 把[根项目 Argus](../../README.zh-CN.md)（多模态实时视频 Agent）的全部能力装进一个跨平台窗口：边看边聊的多模态对话、摄像头/屏幕捕获、流式工具输出、右侧预览与文件浏览、语音、Computer Use，无需终端。基于 **Electron + React 19 + Vite**，可打包为 **macOS / Windows / Linux** 安装包。

> 应用内品牌为 **Argus**（`productName: "Argus"`、`appId: com.argus.desktop`），桌面端环境变量也已全部是 `ARGUS_DESKTOP_*`。仍保留上游 `hermes` 命名的只剩*内部实现*：Python 后端包（`hermes_cli/`）和 Electron 在 `PATH` 上查找的后端可执行文件名（`hermes`）。路径与环境变量均已是 `argus` / `ARGUS_*`，与 `install.sh` / `install.ps1` 实际创建的一致，因此桌面安装与 CLI 安装可以互换。

---

## 它能做什么

桌面端把 web 多模态页面 + 主 Agent 聊天融合进一个原生外壳，核心界面（见 `src/app/`）：

| 模块 | 目录 | 说明 |
| --- | --- | --- |
| **主聊天** | `app/chat` | 与主 Agent 对话：流式响应、实时工具活动行、结构化工具卡、会话历史，与 CLI / gateway 共享同一份记忆与技能。 |
| **多模态视频页** | `app/multimodal` | web 多模态页的桌面移植：摄像头 / 屏幕捕获推帧、边看边问、深度研究面板、观测瀑布流、记忆调试面板。 |
| **右侧栏** | `app/right-sidebar` | 文件浏览与预览、内置终端、代码 review，聊天时并排查看工具产物。 |
| **Agents / Skills / Cron** | `app/agents`、`app/skills`、`app/cron` | 管理子 Agent、技能库、定时任务。 |
| **设置与引导** | `app/settings` | Provider / 模型 / 工具集 / 凭证 / MCP / 语音 / Computer Use 的可视化管理；首次启动引导你配好第一个模型。 |
| **命令中心 / 命令面板** | `app/command-center`、`app/command-palette` | 全局搜索、快捷跳转、主题与桌宠面板。 |
| **桌宠 (Pet)** | `app/pet-overlay`、`app/pet-generate` | 可选的桌面浮层伙伴。 |

其它亮点：

- **Computer Use** —— 让 Agent 驱动你的桌面（点击、输入、截屏）。macOS 走 TCC 权限行，Windows/Linux 各有平台说明（见 `src/app/settings/computer-use-panel.tsx`）。
- **语音** —— 麦克风语音输入 + 语音朗读回复（流式 ASR / TTS）。
- **多模态就绪提示** —— 若多模态子系统缺必需能力，顶部弹出可关闭的建议卡（数据来自后端 `mm.readiness` RPC，与 `argus mm doctor` 同源），不阻塞应用。
- **内置更新** —— 后台检查更新，一键拉取最新 Agent 并原地重建。

渲染层（React，`src/`）通过 `tui_gateway` / dashboard API 与一个 `argus dashboard` 后端进程通信，复用 Agent 运行时，而不是内嵌 `argus --tui`。

---

## 界面语言

支持英文与简体中文。首次运行跟随系统语言（任何 `zh-*` 标签解析为简体中文，其余回退英文），之后把你的选择持久化到 `config.yaml` 的 `display.language`。可随时在设置里切换。

两份词表都在 `src/i18n/`，且都是完整的：各自以 `: Translations` 声明，因此英文加了 key 而中文漏了会直接导致 `npm run typecheck` 失败。

> 注意命令行支持的语言比桌面端更多（16 种，但只覆盖 gateway 与审批消息）。详见根 README 的 *界面语言* 一节。

---

## 安装（终端用户）

### 用 Argus CLI 安装（推荐）

已经装好根项目的 CLI？直接：

```bash
argus desktop
```

它会用你现有的安装（同一份 config、密钥、会话、技能）构建并启动 GUI。首次启动会引导你选 Provider 和模型。

### 预构建安装包

从发布渠道下载对应平台的安装包（DMG / NSIS / AppImage 等）。

### 更新

应用后台检查更新并提供一键升级；也可随时用 CLI：

```bash
argus update
```

---

## 本地开发

从**仓库根目录**装一次 workspace 依赖（会 link `apps/desktop`、`web`、`apps/shared`），再从本目录起 dev server：

```bash
npm install          # 在仓库根目录执行
cd apps/desktop
npm run dev          # Vite 渲染进程 (127.0.0.1:5174) + Electron，Electron 负责拉起 Python 后端
```

dev server 会先跑 `scripts/assert-root-install.cjs` 校验你确实是从根装的依赖。

指向特定源码 checkout，或把它与你真实配置隔离：

```bash
ARGUS_DESKTOP_HERMES_ROOT=/path/to/clone npm run dev   # 用指定的后端源码
ARGUS_HOME=/tmp/throwaway npm run dev                  # 用一次性配置目录
npm run dev:fake-boot                                  # 用确定性延迟演练启动遮罩
```

### 常用脚本

| 命令 | 作用 |
| --- | --- |
| `npm run dev` | 渲染进程 + Electron 并行开发（`concurrently`）。 |
| `npm run build` | 全量构建：stage 后端源码 + 原生依赖 → `tsc -b` → `vite build` → 打包 Electron 主进程。 |
| `npm run typecheck` | `tsc --noEmit` 类型检查。 |
| `npm run lint` / `npm run lint:fix` | ESLint 检查 / 自动修。 |
| `npm run fix` | `lint:fix` + Prettier 格式化。 |
| `npm run test:ui` | 渲染层组件测试（Vitest，jsdom 环境）。 |
| `npm run test:desktop:platforms` | Electron 主进程 (`.cjs`) 的 node:test 套件（引导、后端探测、更新、窗口等）。 |
| `npm run test:desktop:all` | 端到端打包/安装冒烟测试。 |

> **测试从本目录跑**：`vitest.config.ts` 里的 `setupFiles: ['./src/test-setup.ts']` 按 CWD 解析，所以在 `apps/desktop/` 下执行 `npx vitest run ...`，不要从仓库根带 `--config` 跑（会解析错 setup 路径）。jsdom 环境已在 config 中固定，无需再传 `--environment`。

### 构建安装包

```bash
npm run dist:mac     # DMG + zip
npm run dist:win     # NSIS + MSI
npm run dist:linux   # AppImage + deb + rpm
npm run pack         # 只出 release/ 下未打包的 app（不生成安装包）
```

macOS/Windows 签名与公证在环境里有对应凭证时自动进行（macOS：`CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*`；Windows：`WIN_CSC_*`）。

---

## 工作原理

- 打包后的 app = Electron 外壳 + 原生 React 聊天界面。首次启动可把 Argus 运行时装进 `ARGUS_HOME`（macOS/Linux 默认 `~/.argus`，Windows 默认 `%LOCALAPPDATA%\argus`），代码本身放在 `<ARGUS_HOME>/argus` —— 与 `install.sh` / `install.ps1` 产出的布局相同，因此桌面安装与 CLI 安装可互换。
- **后端定位顺序**：`ARGUS_DESKTOP_HERMES_ROOT` → 已完成的托管安装 → `PATH` 上探测到的 `hermes` 可执行文件（除非设了 `ARGUS_DESKTOP_IGNORE_EXISTING=1`）→ 最后是给打包/排障用的显式覆盖 `ARGUS_DESKTOP_HERMES`。
- 安装、后端定位、自更新逻辑都在 `electron/main.cjs`；渲染进程与主进程的桥接在 `electron/preload.cjs`。

### 目录导览

```
apps/desktop/
├─ electron/          # 主进程 (main.cjs)、preload、引导/后端探测/更新，及其 node:test
├─ src/
│  ├─ app/            # 各功能页面（chat / multimodal / settings / right-sidebar / …）
│  ├─ components/     # 复用 UI（assistant-ui 聊天原语、多模态部件等）
│  ├─ store/          # nanostores 状态（gateway / multimodal / session / …）
│  ├─ lib/            # 纯逻辑与工具（chat-messages、工具视图模型等）
│  ├─ i18n/           # 英文 + 简体中文词表
│  └─ styles.css      # Tailwind v4 + 设计 token 主题
└─ scripts/           # 构建/打包/测试辅助脚本 (.cjs / .mjs)
```

---

## 排障

启动日志在 `ARGUS_HOME/logs/desktop.log`（含后端输出与最近的 Python traceback）——报启动失败先看它。

**macOS / Linux：**

```bash
# 强制重跑首次启动引导
rm "$HOME/.argus/argus/.hermes-bootstrap-complete"
# 重建损坏的 Python venv
rm -rf "$HOME/.argus/argus/venv"
# 重置卡住的 macOS 麦克风授权（仅 macOS）
tccutil reset Microphone com.argus.desktop
```

**Windows (PowerShell)：**

```powershell
# 强制重跑首次启动引导
Remove-Item "$env:LOCALAPPDATA\argus\argus\.hermes-bootstrap-complete"
# 重建损坏的 Python venv
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\argus\argus\venv"
```

> Windows 默认 home 目录是 `%LOCALAPPDATA%\argus`；若你迁移过，设 `ARGUS_HOME` 环境变量指向新位置。

---

## License

MIT —— 见 [LICENSE](../../LICENSE)。
