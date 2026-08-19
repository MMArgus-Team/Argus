# Argus — Real-Time Multimodal Video Agent

[English](README.md) · [简体中文](README.zh-CN.md)

Argus is a multimodal AI assistant that can "watch and chat": the main Agent handles user text and semantic routing and **does not passively receive the live video feed**; one-shot visual questions and ongoing video tasks are delegated to dedicated workers. Three multimodal capabilities:

- **One-shot visual Q&A** — Present, historical, or mixed "on-screen entity + external facts" questions all go through `query_multimodal`. QueryWorker reads recent frames at question time, then answers directly, recalls historical memory, or searches external sources as needed.
- **Explicit raw-frame fetch** — `get_current_frame` is only for when the user explicitly asks to retrieve/show/diagnose the latest raw frame; it is not the default entry for ordinary visual Q&A.
- **Continuous monitoring / deep research** — Two background Agents: `set_monitor` (wait for an event → alert) and `set_live_watcher` (background segment-by-segment deep research, producing progress and a final report). Watcher has one mode only: start from the most recent segment, analyze round-by-round under TTL + frame-count dual gates until the stream stops or the user stops it (no qa/analysis/research taxonomy).

> The product brand is **Argus**. The CLI command, environment variables and all user-facing strings are already `argus` / `ARGUS_*`. What still carries upstream `hermes_*` names is *internal only*: Python module paths (`hermes_cli/`, `hermes_constants.py`), some function names, and a few backend contract values (e.g. `speech_provider: hermes`, the `hermes-index` skill source). Those are invisible to users and are deliberately left alone.

## Demo

Live "watch-and-chat" walkthroughs of Argus — screen share a video, ask a question, get a grounded answer from the multimodal Agent.

### 🇬🇧 English demo

https://code.devops.xiaohongshu.com/liguankai/video_stream_demo/-/raw/dev/demos/demo_en.mp4

### 🇨🇳 中文 demo

https://code.devops.xiaohongshu.com/liguankai/video_stream_demo/-/raw/dev/demos/demo_cn.mp4

<sub>Compressed 720p H.264 previews (~11-30 MB) — full 4K originals available on the <a href="https://github.com/MMArgus-Team/Argus/releases/tag/v0.1.0-demos">GitHub Release page</a>.</sub>

## Three Entry Points

Same Agent runtime + memory + skills, three frontends — pick what fits:

| Entry | How to start | Best for |
| --- | --- | --- |
| **Web dashboard** | `argus dashboard` (or `python -m hermes_cli.main dashboard`, see §3) | Fastest onboarding; `/multimodal` page for watch-and-chat; eval workflow in this doc is based on it. |
| **Desktop app** | `argus desktop`, or see [`apps/desktop`](apps/desktop/README.md) | Cross-platform native window combining multimodal video + main chat + file preview + voice + Computer Use — no terminal required. |
| **CLI / eval** | `argus ...` (equivalently `python -m hermes_cli.main ...`) | Offline memory eval (§6), batch runs, scripted calls. |

## System Architecture

The main Agent **does not passively hold video** — it only understands text and routes semantically; visual work goes to dedicated workers. All roles share one `FrameBuffer`:

```
                    ┌─────────────────────────────────────────────┐
   User text ──────▶│  Main Agent (sync ReAct loop, semantic routing) │
                    └───────┬───────────────┬───────────────┬──────┘
       One-shot visual     │      set_monitor│  set_live_watcher│
       query_multimodal    ▼               ▼               ▼
                    ┌────────────┐   ┌────────────┐   ┌──────────────┐
                    │ QueryWorker│   │MonitorEngine│  │ WatcherAgent │
                    │ present/   │   │ event →     │  │ background   │
                    │ history/   │   │ alert       │  │ segment deep │
                    │ mixed Q&A  │   │             │  │ research     │
                    └─────┬──────┘   └─────┬──────┘   └──────┬───────┘
                          │                │                 │
      Memory recall ◀─────┤                │                 │
                    ┌─────▼────────────────▼─────────────────▼──────┐
                    │  FrameBuffer (frontend pushes at fixed fps)      │
                    │   · long-term memory / watcher: dHash-deduped    │
                    │     sparse stream                                │
                    │   · Monitor: raw 2fps queue for last 60s         │
                    │   · QueryWorker: recent frames at question time  │
                    └───────────────────────┬───────────────────────┘
                    ┌───────────────────────▼───────────────────────┐
                    │  MemoryBackend (layered visual memory writer/   │
                    │    reviewer) + SceneDhashController (~20s scene │
                    │    probe, dynamic dedup strength & watcher pace) │
                    └────────────────────────────────────────────────┘
```

Role responsibilities are summarized at the end in [Architecture Overview](#architecture-overview).

## Directory Guide

| Path | Contents |
| --- | --- |
| `hermes_cli/` | CLI and dashboard entry (`python -m hermes_cli.main ...`). Serves the dashboard from `web_dist/`, which is a **build artifact** — git-ignored, produced by `npm run build` (see §1). |
| `web/` | Web dashboard frontend (React + Vite), includes multimodal page `/multimodal`. |
| `apps/desktop/` | Cross-platform desktop app (Electron + React); see its [README](apps/desktop/README.md). |
| `apps/shared/`、`apps/bootstrap-installer/` | Shared desktop/web code and bootstrap installer. |
| `gateway/` | `tui_gateway` / dashboard backend API; frontends talk to the Agent runtime through it. |
| `tools/`、`toolsets.py` | Tool implementations and toolset dispatch. |
| `run_agent.py`、`cli.py` | Agent main loop and CLI core. |
| `config.yaml` | Single source of truth for all keys / endpoints / model selection (§2). |
| `convert_annotation_to_json.py`、`download_0618_videos.py` | Eval data preprocessing (§6.2). |

## Requirements

- Python 3.11 (use uv for an isolated env to avoid system Python version issues)
- Node.js + npm (only needed for frontend builds; skip if already built)
- ffmpeg (needed when `speech_provider: hermes` for voice; also needed when batch eval downloads **best-quality** video — `bestvideo+bestaudio` are separate streams merged by ffmpeg into mp4. Not needed if downloading a single progressive stream only)
- Node.js (only needed for batch eval YouTube downloads, to solve YouTube's n-challenge)
- **cua-driver >= 0.20.0** (only needed for Computer Use / desktop control). Older builds are not supported: on 0.12.3, `launch_app` fails with `exit_code: 1`, and after repeated failures the driver marks the whole session dead — every subsequent tool call is then rejected with `session has ended`. Check with `cua-driver --version`; upgrade with `cua-driver update --apply` (stop the Argus process first, so the old stdio child exits), or `argus computer-use install --upgrade`.

## 1. Setup (create `.venv` with uv and install)

```bash
cd /path/to/video_stream_demo

# Create Python 3.11 virtual environment
uv venv --python 3.11 .venv

# Activate
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   Windows (cmd):
.venv\Scripts\activate.bat
#   Linux / macOS:
source .venv/bin/activate

# Install Argus (editable) + web deps
uv pip install -e ".[web]"

# Install JS deps — ALWAYS from the repo root, never from web/ or apps/desktop/.
# This is an npm workspaces monorepo (see the warning below).
npm ci

# Build frontend — REQUIRED on a fresh clone (web_dist/ is git-ignored, so it starts empty)
cd web
npm run build
cd ..

# Optional: aiohttp for gemini multimodal memory backend
uv pip install aiohttp

# Optional: yt-dlp + openpyxl for batch eval (YouTube download + Excel annotations)
uv pip install yt-dlp openpyxl
```

> Editable install (`-e`): backend `.py` changes take effect after restart — no reinstall.
> `hermes_cli/web_dist/` is **not** in git (it's a build artifact, like `hermes_cli/tui_dist/`). Run `npm run build` once after cloning, then again whenever you change `web/**/*.tsx` — otherwise the dashboard serves 404s. Release builds regenerate it automatically (`scripts/release.py`).

### ⚠ npm: always install from the repo root

This is an **npm workspaces monorepo** (`apps/*`, `ui-tui`, `ui-tui/packages/*`, `web`). Every workspace's dependencies **hoist into the single root `node_modules/`** — the lockfile expects ~1178 top-level packages, while the root `package.json` declares only 3 of them. `vitest`, `typescript`, `jsdom` etc. belong to `web` / `ui-tui` / `apps/desktop` and live at the root only because of hoisting. `web/node_modules` and `ui-tui/node_modules` being empty is **by design**, not a broken install.

```bash
npm ci          # ✅ from the repo root — strict lockfile install, wipes node_modules first
npm install     # ⚠ also root-only; may drift from the lockfile
```

**Never run a per-workspace install:**

```bash
npm install --workspace web        # ❌ don't
cd web && npm install              # ❌ don't
cd apps/desktop && npm install     # ❌ don't
```

A per-workspace install recomputes the whole tree from that one workspace's point of view and **deletes packages it considers extraneous** from the shared root `node_modules`. So a "web only" install silently removes `apps/desktop`'s `vitest` / `typescript` / `jsdom`. The symptom is confusing because nothing looks like it touched anything: *"I never touched `node_modules`, but packages keep disappearing."* (The `install:web` / `install:tui` / `install:desktop` / `install:root` npm scripts were removed for exactly this reason. `npm audit --workspace <name>` is read-only and safe; `npm audit fix --workspace <name>` has the same tree-rewrite hazard as install — avoid it.)

**Recovering a broken tree** — delete every `node_modules` (root *and* per-workspace, or leftovers keep shadowing the root) and reinstall:

```bash
rm -rf node_modules web/node_modules ui-tui/node_modules apps/desktop/node_modules && npm ci
```

**`tarball ... seems to be corrupted` / a flood of `TAR_ENTRY_ERROR ENOENT`:** the `ENOENT` lines are a *side effect*, not the cause — npm deletes a half-extracted package and retries while another task is still writing to that directory. The real failure is a checksum mismatch on the tarball, i.e. a bad copy in `~/.npm/_cacache` or on the registry:

```bash
npm cache clean --force && npm ci
```

If it persists, the bad package is on the registry rather than your machine (check `npm config get registry`); verify against the public one with `npm ci --registry=https://registry.npmjs.org/` and report it to whoever runs the mirror.

**Verify the tree is usable.** Don't compare the directory count against the lockfile — npm dedupes shared versions, so `ls node_modules | wc -l` is legitimately far lower than the lockfile's entry count and says nothing about health. Two checks that do:

```bash
# 1. Executables exist. An interrupted install leaves packages unpacked but .bin EMPTY
#    (npm links binaries last) — that is the "sh: concurrently: command not found" case.
ls node_modules/.bin | wc -l && ls node_modules/.bin | grep -E "concurrently|vite|tsc|vitest"

# 2. Every workspace's direct dependencies actually resolve (expects "missing 0").
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

**Important:** Set `ARGUS_HOME` to the directory containing `config.yaml` (the project root).

## 2. Model Configuration

**`config.yaml` at the project root is the single source of truth** for all keys / endpoints / model selection (tracked in git). On `argus dashboard` startup it is copied one-way to `<ARGUS_HOME>/config.yaml` (project version wins). Edit the root copy, then **restart dashboard**.

Each submodule can have its own model. The main Agent lives under top-level `model`; multimodal roles are sibling sub-sections under `model`, each using the same 4-tuple `{provider, base_url, api_key, model}` (`base_url` empty = follow main Agent). Behavior knobs are under `settings:`; speech (ASR/TTS) under `audio:`:

```yaml
model:
  # Main Agent (text understanding + semantic routing; one-shot visual Q&A goes to QueryWorker)
  default: "kimi-k2.6"
  provider: "custom"
  base_url: "https://<your-openai-compatible-endpoint>/v1"
  api_key: "<your-key>"
  supports_vision: true

  # ① Monitor Agent (set_monitor: always-on video SPEAK/SILENT + event merge)
  monitor:
    provider: "custom"
    base_url: "https://<endpoint>/v1"
    api_key: "<key>"
    model: "<vision-model>"
  # ② Watcher = deep-research worker (backend for set_live_watcher)
  watcher:
    provider: "custom"
    base_url: "https://<endpoint>/v1"
    api_key: "<key>"
    model: "<vision-model>"
  # ③ Layered memory writer/reviewer (QueryWorker can recall this store)
  memory:
    provider: "gemini"               # or openai / empty to follow main Agent
    base_url: "https://<endpoint>"
    api_key: "<key>"
    model: "<vision-model>"
    vision_ability: true             # ★ must be true or memory backend refuses to start (see below)
    audio_ability: false             # true = raw audio to omni model; false = external ASR for audio
    recall:                          # memory recall sub-agent; empty = follow main Agent
      provider: ""
      base_url: ""
      api_key: ""
      model: ""

settings:                            # multimodal behavior knobs (enabled / question-time frames / monitor / anysearch …)
  enabled: true
  memory_enabled: true

audio:                               # multimodal speech (ASR / TTS / ambient audio)
  asr_url: "http://<asr-endpoint>/asr"
  dashscope_api_key: "<key>"         # streaming mic ASR + streaming TTS
```

> Watch-and-chat plus monitor / deep research / memory writes all require **vision-capable models**. `argus doctor` warns when a vision-required role is assigned a known text-only model (or `supports_vision: true` contradicts actual capability) — advisory only, no forced override.
>
> **Memory is a hard prerequisite**: `model.memory.vision_ability` must be `true` (memory extraction is inherently visual). If set to `false`, MemoryBackend **fails at startup** (video stream cannot run with memory enabled) rather than silently using a blind model.

## 3. Start Web Dashboard (includes multimodal page)

```bash
# Main Agent + multimodal backend both run in this dashboard process
Start: argus dashboard --skip-build --port 9119
Stop:  argus dashboard --stop
```

| Flag | Description |
| --- | --- |
| `--skip-build` | Skip frontend build; serve existing `web_dist` (faster) |
| `--no-open` | Do not auto-open browser |
| `--port <n>` | Port (default 9119) |
| `--stop` | Stop all dashboard processes |
| `--status` | List running dashboard processes |

Then open:

```
http://127.0.0.1:9119/multimodal
```

> After config or backend code changes, **restart dashboard** (no hot reload). Hard-refresh the browser with **Ctrl+Shift+R**.

## 4. Frontend Development (when editing `.tsx`, pick one)

```bash
# Option A: build once (dashboard serves with --skip-build)
cd web && npm run build

# Option B: Vite dev server (hot reload, no build)
#   Terminal 1: backend
argus dashboard --skip-build --no-open
#   Terminal 2: frontend
cd web && npm run dev
```

## 5. Interface Language

| Surface | Languages | How it is chosen |
| --- | --- | --- |
| **Web dashboard** | English, 简体中文 | Follows the browser/OS language on first visit (any `zh-*` tag → Chinese; anything we don't ship → English). Switch in the sidebar; the choice is stored per-browser in `localStorage` (`argus-locale`), so it never leaks to other users of a shared dashboard. |
| **Desktop app** | English, 简体中文 | Same first-run detection, but persisted to `display.language` in `config.yaml`. |
| **CLI / gateway** | 16 languages (`en zh zh-hant ja de es fr tr uk af ko it ga pt ru hu`) | `ARGUS_LANGUAGE` env var, else `display.language` in `config.yaml`, else English. |

> The CLI catalog (`locales/*.yaml`) is intentionally wider than the UI: it covers
> gateway messages and approval prompts only (~290 keys). The rest of the CLI
> output is English-only. The two UI surfaces ship English and Simplified
> Chinese, both complete and type-checked — a key added to English but missing in
> Chinese fails the build.

## 6. Offline Memory System Evaluation

Reads a video offline, runs the **same** multimodal memory pipeline as online (no monitor/watcher), then answers each QA item via memory recall and writes predicted answers back to JSON. Official entry is the main project's `mm-memory-eval` subcommand.

### 6.1 Eval Command (single video / whole folder)

```bash
argus mm-memory-eval <video file or folder> <QA JSON>
```

First argument supports two shapes:
- **Single video file** → evaluate that one video (match its `qa_list` from JSON).
- **Video folder** → evaluate **all** videos in JSON. **Pre-flight check**: every video in JSON (by `?v=<id>`) must exist as `<id>.mp4` (or common container) in the folder — **missing any one aborts before run**. Then each video gets a **fresh stack** (memories don't cross-contaminate); **single failures are skipped, summary at end** (non-zero exit if any failed). Folder mode requires the full-array JSON from §6.2.

- **Input JSON**: `{"title": "<must match video filename without extension>", "qa_list": [{"query": "...", "answer": "...", "time": "HH:MM:SS"(optional)}, ...]}`.
  `title` is **strictly validated** = video filename without extension; mismatch errors out.
- **Output**: appends `"answer_predict"` to each QA item; by default overwrites input JSON (use `--out` to save elsewhere).
  **Note: it only produces predictions — no automatic scoring or accuracy** — scoring is a separate step (§6.3).

**CLI flags** (`argus mm-memory-eval <video> <json> [options]`):

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `video` | positional | (required) | Video file path, **or video folder**. Folder → eval all videos in JSON (pre-check all exist). Single file: basename without extension must match JSON. |
| `json` | positional | (required) | QA JSON path, format `{title, qa_list:[{query, answer, time?}]}`. |
| `--mode` | `tool` \| `agent` | `tool` | `tool` = direct memory recall answer; `agent` = offline eval first generates Recall evidence explicitly, then main Agent synthesizes with last 3 frames. Non-interactive eval path — does not simulate online `query_multimodal` / QueryWorker handoff. |
| `--source` | `camera` \| `screen` | `camera` | Simulated online input source. `screen` enables screen-share OCR/table path and higher default resolution. |
| `--answer-timing` | `before` \| `after` | `after` | For temporal questions at `time`, when to answer. `after` = push frame at that timestamp, run OCR/writer wake, then answer — closer to online "user sees frame then asks". |
| `--query-types` | string | no filter | Only eval specified `query_type`, comma-separated, e.g. `b` or `a,b,d,e`. Non-matching items kept unchanged, no `answer_predict`. |
| `--asr-vtt` | `auto` \| `none` \| path | `auto` | Inject existing ASR subtitles for silent video, streamed by cue end time into `audio_observation`. `auto` looks for `<video>.auto.asr.vtt` / `.asr.vtt` / `.vtt` beside video. |
| `--scene-probe` / `--no-scene-probe` | bool | on | Whether to drive scene/dHash controller offline to align dynamic dedup thresholds with online. |
| `--out` | path | (overwrite input) | Save predictions elsewhere. Default writes `answer_predict` back to input JSON. |
| `--trace-out` | path | auto sidecar | Per-question recall trace JSON. Default beside result JSON — tools, frame_ids, OCR/table hit summaries. |
| `--capture-fps` | float | config `buffer_capture_fps` (=2.0) | Video decode sample rate (same as online screen share). |
| `--max-side` | int | `camera=720`, `screen=1536/ocr_max_side` | Max frame long-edge scale (pixels). |
| `--jpeg-quality` | int | `80` | Frame JPEG quality. |
| `--timeout` | float | `120` | Per-recall timeout (seconds). |

**★ Temporal eval (default when annotations have `time`)**: Each item has `time` (question timestamp, `HH:MM:SS` / `MM:SS` / seconds), so **normal eval is temporal**: feed frames until that item's `time`, answer using **memory state up to that moment** (not after full video), then continue — mimics "what the system knew at that point". Applies to both `tool` and `agent` modes.
  - Trigger: **any valid `time` in qa_list → temporal mode**, and then **every item must have valid `time`** (missing/unparseable → error exit).
  - Items with `time` past video end are answered with full-video memory after frames exhaust (no dropped items).
  - (Edge case) A hand-written QA with **no `time` on any item** falls back to non-temporal: feed full video then answer all. Annotation datasets don't use this path.

Examples:

```bash
# Single video (first arg is a file)
python -m hermes_cli.main mm-memory-eval test_0618/QyNunAw0sx4.mp4 test_0618/0618.json
python -m hermes_cli.main mm-memory-eval test_0618/QyNunAw0sx4.mp4 test_0618/0618.json \
    --mode agent --out QyNunAw0sx4.pred.json

# Whole folder (first arg is directory → eval all videos in 0618.json, pre-check all present)
python -m hermes_cli.main mm-memory-eval test_0618 test_0618/0618.json
```

### 6.2 Generating Eval Data `xx.json`

`mm-memory-eval` input JSON has two shapes: **single-video object** `{title, qa_list}` (hand-write for one video), or **full array** `[{video_url, qa_list}, ...]` (dataset; use with folder mode to eval all at once).

**Full input JSON structure**:

```jsonc
{
  "title": "QyNunAw0sx4",           // must == video filename without extension (strictly validated)
  "qa_list": [
    {
      "query":  "What did the person pick up first?",   // required: question
      "answer": "A screwdriver",                        // required: gold answer (for scoring; system never sees it)
      "time":   "00:01:30"                              // question time → temporal eval (§6.1); annotations always have this
    },
    { "query": "Where did he put the tool back?", "answer": "Second layer of the toolbox", "time": "00:03:12" }
  ]
}
```

Field rules:
- `title` **strictly validated** = video filename without extension (e.g. `QyNunAw0sx4.mp4` → `"QyNunAw0sx4"`); mismatch errors out.
- Each QA **must** have non-empty `query` + `answer`.
- `time` (question timestamp, `HH:MM:SS` / `MM:SS` / seconds) sets eval mode: **annotation data always has `time`, so normal runs are temporal** (§6.1). **Either every item has `time` (temporal, usual) or none do (non-temporal, edge case)**; mixing → validation error.

For a single video, hand-write `xx.json` (`xx` == video name) and skip the Excel flow below.

---

**Batch generation from annotation Excel** (dataset scenario), two preprocessing steps only, **no splitting** — everything lives under `test_0618` (annotation JSON and video mp4 in the same folder).

**① Excel annotations → `test_0618/0618.json`** (conversion script at project root):

```bash
python convert_annotation_to_json.py                       # default: xlsx → test_0618/0618.json
python convert_annotation_to_json.py <annotations.xlsx> --out-dir test_0618   # explicit
```

- Positional `input`: annotation xlsx path (default `C:\Users\luyuan2\Downloads\0618.xlsx`, change as needed).
- `--out-dir`: output directory (default `./test_0618`), produces `<out-dir>/0618.json`; or `--out <full path>` for direct file path.

Output is an **array**, each element `{video_url, accuracy(annotation), qa_list:[{query, query_type, answer, time, analyse}]}`.
Column mapping: `video_url←youtube URL`, `accuracy←per-video overall accuracy`, `query←question`, `query_type←item type`, `answer←gold answer`, `time←question timestamp`, `analyse←explanation` (url + accuracy filled on first row per video block, forward-filled within block).

**② Download videos → `test_0618/<id>.mp4`** (id = `?v=<id>` from url):

```bash
python download_0618_videos.py                             # default: read test_0618/0618.json → download to test_0618/
python download_0618_videos.py <target folder> --json <manifest.json>   # explicit
```

- Positional `out_dir`: download target (default `./test_0618`).
- `--json`: video manifest (default `./test_0618/0618.json`, output of step ①). Resumable (`.download-archive.txt` tracks progress).

> YouTube may show "confirm you're not a bot" — cookies required. Export YouTube cookies with browser extension "Get cookies.txt LOCALLY" to `./.yt_cookies.txt` (**do not commit**). YouTube n-challenge needs JS runtime: script uses `--js-runtimes node:<node path>`; merging best quality (bv*+ba) needs ffmpeg.
> Delete `.yt_cookies.txt` after download (contains login credentials).

**③ Eval directly with full `0618.json` (no split)**: `mm-memory-eval` now **accepts the full array JSON** — internally matches by **video filename** (no extension) to `?v=<id>` in `video_url`, takes that block's `qa_list`, **errors if no match** (no silent skip). `answer_predict` writes back to that video's array block (others unchanged); whole `0618.json` is the result file.

```bash
# <video> basename (no extension) must match ?v=<id> in 0618.json
python -m hermes_cli.main mm-memory-eval test_0618/QyNunAw0sx4.mp4 test_0618/0618.json
```

**Batch all videos — pass folder directly** (recommended): first arg is video directory; command pre-checks every JSON video exists in folder (abort if missing), then fresh stack per video, skip failures, summary at end:

```bash
python -m hermes_cli.main mm-memory-eval test_0618 test_0618/0618.json
```

Or loop externally (equivalent, but no upfront "all present" check):

```bash
for f in test_0618/*.mp4; do
  python -m hermes_cli.main mm-memory-eval "$f" test_0618/0618.json
done
```

> Annotation data always has `time`, so matched `qa_list` **automatically uses temporal eval** (§6.1). Command still supports legacy single-object JSON `{title, qa_list}` (see §6.2 hand-write format).
>
> To save per-video results without overwriting full `0618.json`, use `--out <path>` (still writes full array, just to another file).

### 6.3 Scoring and Accuracy (separate step)

`mm-memory-eval` only writes `answer_predict`; **accuracy must be scored separately**. For each item, send `[query / answer(gold) / answer_predict]` to an LLM judge for **semantic** right/wrong (paraphrase counts; wrong/missing key facts → wrong; prediction "not found / empty memory" → wrong), then aggregate per-video and overall — same idea as the annotation Excel "correct (Gemini eval)" column. Decoupled from official eval; reuse any model from config as judge.

> **Cost note**: Each video runs offline memory build (writer model per frame) then per-item recall; full 49 videos / 651 items is substantial LLM cost and time. **Run 1–2 videos to validate the pipeline before full batch**.

## Architecture Overview

- **Main Agent** (sync ReAct loop) — handles user text and semantic routing; does not passively hold present or historical video; one-shot visual questions call `query_multimodal`.
- **QueryWorker** (one-shot answer owner) — takes the original user question and recent frames at question time; chooses present-frame VQA, multimodal Recall, Search, or a combination based on the question; replies directly to the original message.
- **WatcherAgent** (dedicated thread + async loop) — background deep-research engine for `set_live_watcher` (standard multimodal ReAct worker). Per-round progress/streaming interpretation goes to the right panel; per-round reports as collapsible bubbles on the main Agent page; final report summarized on completion; decoupled from main Agent chat history (not written to history).
- **MonitorEngine** (dedicated thread + async loop, job container) — each `set_monitor` monitor = one long-lived async job.
- **MemoryBackend** — layered visual memory writer / reviewer, building the store QueryWorker recalls from; also hosts **SceneDhashController**, using `auxiliary.vision` every ~20s to probe scene and dynamically adjust FrameBuffer dedup strength and watcher pace.

These roles share one **`FrameBuffer`**: frontend pushes at fixed fps; long-term memory and watcher read dHash-deduped sparse stream; QueryWorker only fetches recent frames at question time after handoff; Monitor reads the raw 2fps queue for the last 60 seconds in the same buffer so brief targets aren't lost to dedup.

## Notes

- **Do not commit secret-bearing files**: root `config.yaml` has plaintext keys (project convention), scratch scripts like `check_qwen.py` may carry hardcoded test keys, and batch eval writes `.yt_cookies.txt` (YouTube login) — scrub before pushing to public repos; `rm .yt_cookies.txt` after eval.
- **Don't commit large batch eval artifacts**: `test_0618/` (downloaded videos, can be 100+ GB), `0618.json` / `results_0618.json` as needed — video dir should be in `.gitignore`.
- On Windows, if typing `argus` opens a plain file, PATH didn't find `argus.exe` — activate `.venv` first, or use `.venv\Scripts\python.exe -m hermes_cli.main ...`.

## FAQ

**Q: What should `ARGUS_HOME` be set to?**
A: The directory containing `config.yaml` (project root). `argus dashboard` copies root `config.yaml` one-way to `<ARGUS_HOME>/config.yaml`; always edit the project root copy, then restart dashboard.

**Q: Changed config / backend code — why no effect?**
A: Dashboard doesn't hot-reload — **restart dashboard**. Hard-refresh frontend **Ctrl+Shift+R**. Editable install (`uv pip install -e`): `.py` changes need no reinstall, but process restart is required.

**Q: Changed `web/**/*.tsx`?**
A: Either `cd web && npm run build` (dashboard serves with `--skip-build`), or run `npm run dev` hot-reload dev server (§4). Note `hermes_cli/web_dist/` is git-ignored, so a fresh clone needs that first build even if you never touched the frontend.

**Q: "Memory backend refused to start"?**
A: `model.memory.vision_ability` must be `true` (memory extraction is visual). `false` fails at startup (see §2).

**Q: Want a GUI without terminal?**
A: Use the desktop app — `argus desktop`, or see [`apps/desktop`](apps/desktop/README.md). Combines multimodal video, main chat, file preview, voice, and Computer Use.

**Q: How to run batch eval from scratch?**
A: §6.2 two preprocessing steps (Excel→JSON, download videos), then §6.1 pass video folder to eval all, finally §6.3 score separately. **Validate with 1–2 videos first, then full batch** (49 videos / 651 items is expensive).
