<p align="center">
  <img src="assets/mmargus_logo.jpeg" alt="MM-Argus logo" width="420">
</p>

# Argus

Argus is a multimodal AI assistant that can "watch and chat": the main Agent handles user text and semantic routing and **does not passively receive the live video feed**; one-shot visual questions and ongoing video tasks are delegated to dedicated workers. Two user-facing multimodal modes:

- **One-shot visual Q&A** — Present, historical, or mixed "on-screen entity + external facts" questions all go through `query_multimodal`. QueryWorker reads recent frames at question time, then answers directly, recalls historical memory, or searches external sources as needed.
- **Continuous monitoring / deep research** — Two background Agents: `set_monitor` (wait for an event → alert) and `set_live_watcher` (background segment-by-segment deep research, producing progress and a final report). Watcher has one mode only: start from the most recent segment, analyze round-by-round under TTL + frame-count dual gates until the stream stops or the user stops it (no qa/analysis/research taxonomy).

[简体中文](README.zh-CN.md) · [Español](README.es.md) · [اردو](README.ur-pk.md)

> Argus is derived from [Hermes Agent](https://github.com/NousResearch/hermes-agent)
> by Nous Research. The original copyright and MIT license are preserved in
> [LICENSE](LICENSE).

## Related projects

[MM-DSH](https://github.com/MMArgus-Team/MM-DSH) brings Argus to **DeepSeek Harness** with live screen and camera understanding, voice interaction, event monitoring, and offline video Q&A. See its [installation guide](https://github.com/MMArgus-Team/MM-DSH/blob/main/INSTALL.md) to get started.

[MM-CC](https://github.com/MMArgus-Team/MM-CC) brings Argus to **Claude Code** with headless screen and camera capture, offline video Q&A, event Monitors, continuous Watchers, and visual memory. See its [installation guide](https://github.com/MMArgus-Team/MM-CC#install) to get started.

## Demo

Live "watch-and-chat" walkthroughs of Argus — screen share a video, ask a question, get a grounded answer from the multimodal Agent — plus an office workflow demo powered by MM-DSH. **Click a thumbnail to watch on YouTube.**

<table>
<tr>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=suX31-o6lLM">
  <img src="assets/demo_en.png" alt="English demo preview" width="480"><br/>
  <b>🇬🇧 English demo</b> (click to play)
</a>
</td>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=iCijSbVFRu8">
  <img src="assets/demo_cn.png" alt="中文 demo 预览" width="480"><br/>
  <b>🇨🇳 中文 demo</b> (click to play)
</a>
</td>
</tr>
<tr>
<td align="center" colspan="2">
<a href="https://www.youtube.com/watch?v=35TsiaIgPRo">
  <img src="assets/demo_mm_dsh_office.jpg" alt="MM-DSH office workflow demo preview" width="720"><br/>
  <b>💼 MM-DSH office demo</b> (click to play)
</a>
</td>
</tr>
</table>

<sub>Hosted on YouTube. Full 4K originals of the English and Chinese watch-and-chat demos are also available on the <a href="https://github.com/MMArgus-Team/Argus/releases/tag/v0.1.0-demos">v0.1.0-demos Release</a>.</sub>

## StreamArena Benchmark

Results on the StreamArena streaming benchmark, comparing offline models and streaming harnesses across Tool-use, memory recall, Real Time perception, and proactive interaction tasks.

<table>
<thead>
<tr>
  <th align="left">setting</th>
  <th align="left">model / harness</th>
  <th align="left">Tool-use</th>
  <th align="left">memory recall</th>
  <th align="left">Real Time<br>perception</th>
  <th align="left">proactive<br>interaction</th>
  <th align="left">overall</th>
</tr>
</thead>
<tbody>
<tr>
  <td rowspan="3" valign="top">Offline</td>
  <td>Qwen3.5-397B</td>
  <td align="left">0.622</td>
  <td align="left">0.415</td>
  <td align="left">0.441</td>
  <td align="left">-</td>
  <td align="left"></td>
</tr>
<tr>
  <td>Kimi K2.6</td>
  <td align="left">0.609</td>
  <td align="left">0.438</td>
  <td align="left">0.479</td>
  <td align="left">-</td>
  <td align="left"></td>
</tr>
<tr>
  <td>Gemini 3.5 Flash</td>
  <td align="left">0.708</td>
  <td align="left">0.514</td>
  <td align="left">0.513</td>
  <td align="left">-</td>
  <td align="left"></td>
</tr>
<tr>
  <td rowspan="3" valign="top">Streaming<br>Harness</td>
  <td>StreamMind (Qwen3.5-397B)</td>
  <td align="left">0.561</td>
  <td align="left">0.349</td>
  <td align="left">0.445</td>
  <td align="left">0.116</td>
  <td align="left"><b>0.407</b></td>
</tr>
<tr>
  <td>MMArgus (Qwen3.5-397B)</td>
  <td align="left">0.571</td>
  <td align="left">0.408</td>
  <td align="left">0.449</td>
  <td align="left">0.194</td>
  <td align="left"><b>0.443</b></td>
</tr>
<tr>
  <td>MMArgus (Kimi K2.6 + Gemini 3.5 Flash)</td>
  <td align="left">0.702</td>
  <td align="left">0.491</td>
  <td align="left">0.559</td>
  <td align="left">0.233</td>
  <td align="left"><b>0.543</b></td>
</tr>
</tbody>
</table>

Benchmark dataset: [StreamArena on Hugging Face](https://huggingface.co/datasets/hkuzxc/StreamArena).

Higher is better. Overall scores are highlighted in bold. A dash or an empty cell indicates a score that was not reported.

## Highlights

- Current and historical visual question answering through `query_multimodal`.
- Continuous event monitoring with `set_monitor`.
- Long-running video research with `set_live_watcher`.
- Screen, camera, microphone, and shared-system-audio capture in the web and
  desktop clients.
- Layered multimodal memory for scenes, speech, events, and entities.
- The Hermes-compatible agent core, tools, skills, gateway, TUI, and desktop
  application.

## Requirements

- Python 3.11–3.13
- Node.js 20.19+ or 22.12+ for web and desktop builds
- `ffmpeg` for audio processing
- macOS screen and system-audio permissions for desktop screen sharing

## Install from PyPI

```bash
python -m pip install "mm-argus[web]"
argus setup
argus
```

The PyPI distribution is `mm-argus`; the primary command is `argus`. Legacy
`hermes`, `hermes-agent`, and `hermes-acp` entry points remain available for
compatibility with inherited integrations.

## Install from source

```bash
git clone https://github.com/MMArgus-Team/Argus.git
cd argus

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[web]"
```

On Windows PowerShell, run the native installer, then activate the environment:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/install.ps1
.venv\Scripts\Activate.ps1
```

## Configuration

Secrets are never committed to the repository. Copy the public templates into
your local Argus home and fill in only the providers you use:

```bash
mkdir -p ~/.argus
cp config.example.yaml ~/.argus/config.yaml
cp .env.example ~/.argus/.env
```

- `~/.argus/config.yaml` contains behavior, model, endpoint, and multimodal
  settings.
- `~/.argus/.env` contains API keys, tokens, and passwords only.

Run the guided setup at any time:

```bash
argus setup
```

## Run

```bash
argus                         # Interactive CLI
argus dashboard               # Web dashboard
argus gateway                 # Messaging gateway
argus mm doctor               # Multimodal diagnostics
```

## Desktop development

```bash
npm install
npm --workspace apps/desktop run dev
```

The desktop app needs macOS Screen & System Audio Recording permission to
capture audio from a shared screen. After changing that permission, fully quit
and restart the desktop app before sharing again.

## Web development

```bash
npm install
npm --workspace web run dev
```

## Tests

Use the repository wrapper so credentials and local Argus state are isolated:

```bash
scripts/run_tests.sh
npm --workspace apps/desktop run test:desktop
```

## Documentation and support

- [Documentation source](website/docs)
- [Issues](https://github.com/MMArgus-Team/Argus/issues)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)

## License and attribution

Argus is distributed under the MIT License. It is a modified derivative of
Hermes Agent by Nous Research; upstream notices and copyright statements are
retained. See [LICENSE](LICENSE) for details.

Because of that lineage a few internal module names still read `hermes_*` —
most visibly the `hermes_cli` package, which holds the CLI implementation.
`argus_cli` is the canonical entry point (`argus` resolves to
`argus_cli.main:main`) and forwards into it, and the `hermes`, `hermes-agent`
and `hermes-acp` console scripts are kept as aliases so existing installs
keep working. The names are import-compatibility surface, not a sign that
you are running upstream Hermes.
