# Argus Desktop

[English](README.md) · [简体中文](README.zh-CN.md)

**The native desktop client for Argus** — packs everything from [the root Argus project](../../README.md) (real-time multimodal video Agent) into one cross-platform window: watch-and-chat multimodal conversation, camera/screen capture, streaming tool output, right-hand preview and file browsing, voice, and Computer Use — no terminal required. Built on **Electron + React 19 + Vite**, packageable as **macOS / Windows / Linux** installers.

> In-app branding is **Argus** (`productName: "Argus"`, `appId: com.argus.desktop`), and every desktop environment variable is `ARGUS_DESKTOP_*`. What still carries upstream `hermes` names is *internal only*: the Python backend package (`hermes_cli/`) and the backend executable Electron looks for on `PATH` (`hermes`). Paths and env keys are all `argus` / `ARGUS_*` and match what `install.sh` / `install.ps1` create, so a desktop install and a CLI install stay interchangeable.

---

## What it can do

The desktop app fuses the web multimodal page with the main Agent chat inside a native shell. Core surfaces (see `src/app/`):

| Module | Directory | Description |
| --- | --- | --- |
| **Main chat** | `app/chat` | Talk to the main Agent: streaming responses, live tool-activity rows, structured tool cards, session history. Shares one memory and skill store with the CLI / gateway. |
| **Multimodal video page** | `app/multimodal` | Desktop port of the web multimodal page: camera / screen capture frame push, watch-and-ask, deep-research panel, observation waterfall, memory debug panel. |
| **Right sidebar** | `app/right-sidebar` | File browsing and preview, built-in terminal, code review — inspect tool output side-by-side with the chat. |
| **Agents / Skills / Cron** | `app/agents`, `app/skills`, `app/cron` | Manage sub-Agents, the skill library, and scheduled tasks. |
| **Settings & onboarding** | `app/settings` | Visual management of providers / models / toolsets / credentials / MCP / voice / Computer Use. First launch walks you through configuring your first model. |
| **Command center / palette** | `app/command-center`, `app/command-palette` | Global search, quick navigation, theme and pet panels. |
| **Pet** | `app/pet-overlay`, `app/pet-generate` | Optional desktop overlay companion. |

Other highlights:

- **Computer Use** — let the Agent drive your desktop (click, type, screenshot). macOS goes through TCC permission rows; Windows/Linux have their own platform notes (see `src/app/settings/computer-use-panel.tsx`).
- **Voice** — microphone voice input plus spoken replies (streaming ASR / TTS).
- **Multimodal readiness hints** — if the multimodal subsystem is missing a required capability, a dismissible suggestion card appears at the top (data comes from the backend `mm.readiness` RPC, same source as `argus mm doctor`). Never blocks the app.
- **Built-in updates** — checks for updates in the background, pulls the latest Agent and rebuilds in place with one click.

The renderer (React, `src/`) talks to an `argus dashboard` backend process through the `tui_gateway` / dashboard API, reusing the Agent runtime rather than embedding `argus --tui`.

---

## Interface language

English and Simplified Chinese. On first run the app follows your system language (any `zh-*` tag resolves to Simplified Chinese; anything else falls back to English), then persists your choice to `display.language` in `config.yaml`. Switch it any time from Settings.

Both catalogs live in `src/i18n/` and are complete: each is declared `: Translations`, so adding a key to English while forgetting Chinese fails `npm run typecheck`.

> Note the CLI supports a wider set of languages than the app (16, for gateway and approval messages only). See the root README's *Interface Language* section.

---

## Install (end users)

### Via the Argus CLI (recommended)

Already have the root project's CLI installed? Just run:

```bash
argus desktop
```

It builds and launches the GUI against your existing install — same config, keys, sessions, and skills. First launch walks you through picking a provider and model.

### Prebuilt installers

Download the installer for your platform (DMG / NSIS / AppImage, etc.) from the release channel.

### Updating

The app checks for updates in the background and offers a one-click upgrade. You can also use the CLI at any time:

```bash
argus update
```

---

## Local development

Install workspace dependencies once from the **repository root** (this links `apps/desktop`, `web`, and `apps/shared`), then start the dev server from this directory:

```bash
npm install          # run in the repository root
cd apps/desktop
npm run dev          # Vite renderer (127.0.0.1:5174) + Electron; Electron starts the Python backend
```

The dev server first runs `scripts/assert-root-install.cjs` to verify you really installed from the root.

To point at a specific source checkout, or isolate it from your real config:

```bash
ARGUS_DESKTOP_HERMES_ROOT=/path/to/clone npm run dev   # use a specific backend checkout
ARGUS_HOME=/tmp/throwaway npm run dev                  # use a throwaway config directory
npm run dev:fake-boot                                  # rehearse the boot overlay with deterministic delays
```

### Common scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Renderer + Electron in parallel (`concurrently`). |
| `npm run build` | Full build: stage backend source + native deps → `tsc -b` → `vite build` → bundle the Electron main process. |
| `npm run typecheck` | `tsc --noEmit` type check. |
| `npm run lint` / `npm run lint:fix` | ESLint check / autofix. |
| `npm run fix` | `lint:fix` plus Prettier formatting. |
| `npm run test:ui` | Renderer component tests (Vitest, jsdom environment). |
| `npm run test:desktop:platforms` | node:test suites for the Electron main process (`.cjs`): bootstrap, backend discovery, updates, windows. |
| `npm run test:desktop:all` | End-to-end packaging / install smoke tests. |

> **Run tests from this directory**: `setupFiles: ['./src/test-setup.ts']` in `vitest.config.ts` resolves against the CWD, so run `npx vitest run ...` inside `apps/desktop/` — don't run it from the repository root with `--config` (the setup path would resolve wrong). The jsdom environment is pinned in the config, so there's no need to pass `--environment`.

### Building installers

```bash
npm run dist:mac     # DMG + zip
npm run dist:win     # NSIS + MSI
npm run dist:linux   # AppImage + deb + rpm
npm run pack         # unpacked app under release/ only (no installer)
```

macOS/Windows signing and notarization happen automatically when the matching credentials are in the environment (macOS: `CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*`; Windows: `WIN_CSC_*`).

---

## How it works

- A packaged app = Electron shell + native React chat UI. On first launch it can install the Argus runtime into `ARGUS_HOME` (default `~/.argus` on macOS/Linux, `%LOCALAPPDATA%\argus` on Windows), with the code itself under `<ARGUS_HOME>/argus` — the same layout `install.sh` / `install.ps1` produce, so a desktop install and a CLI install are interchangeable.
- **Backend discovery order**: `ARGUS_DESKTOP_HERMES_ROOT` → a completed managed install → a `hermes` executable found on `PATH` (unless `ARGUS_DESKTOP_IGNORE_EXISTING=1` is set) → finally `ARGUS_DESKTOP_HERMES`, an explicit override for packaging and troubleshooting.
- Installation, backend discovery, and self-update logic all live in `electron/main.cjs`; the renderer ↔ main bridge is in `electron/preload.cjs`.

### Directory guide

```
apps/desktop/
├─ electron/          # main process (main.cjs), preload, bootstrap/backend-probe/update, plus their node:test suites
├─ src/
│  ├─ app/            # feature pages (chat / multimodal / settings / right-sidebar / …)
│  ├─ components/     # reusable UI (assistant-ui chat primitives, multimodal widgets, …)
│  ├─ store/          # nanostores state (gateway / multimodal / session / …)
│  ├─ lib/            # pure logic and helpers (chat-messages, tool view models, …)
│  ├─ i18n/           # English + Simplified Chinese catalogs
│  └─ styles.css      # Tailwind v4 + design-token theme
└─ scripts/           # build / packaging / test helper scripts (.cjs / .mjs)
```

---

## Troubleshooting

Startup logs are at `ARGUS_HOME/logs/desktop.log` (includes backend output and the most recent Python traceback) — check it first when launch fails.

**macOS / Linux:**

```bash
# Force the first-run bootstrap to run again
rm "$HOME/.argus/argus/.hermes-bootstrap-complete"
# Rebuild a broken Python venv
rm -rf "$HOME/.argus/argus/venv"
# Reset a stuck macOS microphone grant (macOS only)
tccutil reset Microphone com.argus.desktop
```

**Windows (PowerShell):**

```powershell
# Force the first-run bootstrap to run again
Remove-Item "$env:LOCALAPPDATA\argus\argus\.hermes-bootstrap-complete"
# Rebuild a broken Python venv
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\argus\argus\venv"
```

> The default Windows home is `%LOCALAPPDATA%\argus`. If you moved it, point the `ARGUS_HOME` environment variable at the new location.

---

## License

MIT — see [LICENSE](../../LICENSE).
