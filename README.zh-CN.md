<p align="center">
  <img src="assets/mmargus_logo.jpeg" alt="MM-Argus 标志" width="420">
</p>

# Argus

Argus 是一个能"边看边聊"的多模态 AI 助手：主 Agent 处理用户文本与语义路由，**不会被动收到直播画面**；一次性视觉问题和持续视频任务分别交给专用 worker。两种对外多模态模式：

- **一次性视觉问答** —— 当前、历史或“画面实体 + 外部事实”的 mixed 问题统一进入 `query_multimodal`。QueryWorker 读取提问时刻的近期帧，再按需直接作答、召回历史记忆或检索外部资料。
- **持续监控 / 深度研究** —— `set_monitor`（等某事件出现→提醒）与 `set_live_watcher`（后台持续逐段深研，产出过程与最终报告）两个后台 Agent。watcher 只有一种模式：从最近一段起、按 TTL + 帧数双门逐轮分析，直到视频流停止或用户停止（无 qa/analysis/research 分类）。

[English](README.md) · [Español](README.es.md) · [اردو](README.ur-pk.md)

> Argus 基于 Nous Research 的
> [Hermes Agent](https://github.com/NousResearch/hermes-agent) 修改而来。
> 项目保留原始版权声明并继续使用 [MIT License](LICENSE)。

## Demo 演示

两段"边看边聊"的实况演示 —— 屏幕共享一段视频，向 Argus 提问，多模态 Agent 结合真实画面给出回答。**点击缩略图即可在 YouTube 上观看。**

<table>
<tr>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=suX31-o6lLM">
  <img src="assets/demo_en.png" alt="英文演示预览" width="480"><br/>
  <b>🇬🇧 英文演示</b>（点击播放）
</a>
</td>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=iCijSbVFRu8">
  <img src="assets/demo_cn.png" alt="中文演示预览" width="480"><br/>
  <b>🇨🇳 中文演示</b>（点击播放）
</a>
</td>
</tr>
</table>

<sub>视频托管在 YouTube。原始 4K 高清版本仍可在 <a href="https://github.com/MMArgus-Team/Argus/releases/tag/v0.1.0-demos">v0.1.0-demos Release 页面</a> 下载。</sub>

## 主要能力

- 使用 `query_multimodal` 回答当前画面和历史画面问题。
- 使用 `set_monitor` 持续等待并提醒指定视频事件。
- 使用 `set_live_watcher` 在后台持续分析视频并生成研究报告。
- Web 和桌面端支持屏幕、摄像头、麦克风及共享系统音频采集。
- 对画面、语音、事件和实体建立分层多模态记忆。
- 保留 Hermes 兼容的 Agent 核心、工具、技能、消息网关、TUI 和桌面应用。

## 环境要求

- Python 3.11–3.13
- Node.js 20.19+ 或 22.12+（构建 Web/桌面端时需要）
- `ffmpeg`（音频处理）
- macOS 桌面共享需要“屏幕与系统音频录制”权限

## 从 PyPI 安装

```bash
python -m pip install "mm-argus[web]"
argus setup
argus
```

PyPI 包名是 `mm-argus`，推荐命令是 `argus`。为兼容继承自 Hermes 的已有集成，
暂时保留 `hermes`、`hermes-agent` 和 `hermes-acp` 命令。

## 从源码安装

```bash
git clone https://github.com/MMArgus-Team/Argus.git
cd argus

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[web]"
```

Windows PowerShell 激活方式：

```powershell
.venv\Scripts\Activate.ps1
```

## 配置

仓库不保存真实凭证。复制公开模板到本机 Argus 目录，再填写自己使用的服务：

```bash
mkdir -p ~/.argus
cp config.example.yaml ~/.argus/config.yaml
cp .env.example ~/.argus/.env
```

- `~/.argus/config.yaml`：行为、模型、endpoint 和多模态配置。
- `~/.argus/.env`：只存放 API key、token 和密码。

也可以运行交互式配置：

```bash
argus setup
```

## 启动

```bash
argus                         # 交互式 CLI
argus dashboard               # Web 仪表盘
argus gateway                 # 消息网关
argus mm doctor               # 多模态诊断
```

## 桌面端开发

```bash
npm install
npm --workspace apps/desktop run dev
```

桌面版捕获共享屏幕声音时，需要授予 macOS“屏幕与系统音频录制”权限。修改权限后，
必须彻底退出并重新启动桌面应用，再重新共享屏幕。

## Web 开发

```bash
npm install
npm --workspace web run dev
```

## 测试

请使用仓库测试脚本，避免读取本机凭证和 Argus 状态：

```bash
scripts/run_tests.sh
npm --workspace apps/desktop run test:desktop
```

## 文档与反馈

- [文档源码](website/docs)
- [问题反馈](https://github.com/MMArgus-Team/Argus/issues)
- [安全策略](SECURITY.md)
- [贡献指南](CONTRIBUTING.md)

## 许可证与署名

Argus 使用 MIT License 发布，是 Nous Research 的 Hermes Agent 的修改衍生项目；
所有 upstream 版权与许可证声明均予以保留。详见 [LICENSE](LICENSE)。

因为这层渊源，仓库里仍有少量内部模块沿用 `hermes_*` 命名，最显眼的是存放 CLI
实现的 `hermes_cli` 包。规范入口是 `argus_cli`（`argus` 命令解析到
`argus_cli.main:main`），它转发到该实现；`hermes`、`hermes-agent`、`hermes-acp`
这几个命令作为别名保留，以便已安装的环境继续可用。这些名字只是导入层的兼容面，
并不代表你运行的是 upstream Hermes。
