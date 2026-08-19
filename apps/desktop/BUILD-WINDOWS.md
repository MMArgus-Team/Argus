# 构建 Windows 安装程序

本文档说明如何把 Argus 桌面版打成 Windows 安装程序，以及如何安装。

> 所有命令基于本仓库当前的 `apps/desktop/package.json`（electron-builder 26.x，
> Electron 40，appId `com.argus.desktop`）。

---

## ⚠️ 必读：在 Windows 机器上构建

桌面版依赖原生模块 **`node-pty`**（终端功能），它需要平台专属的 `.node` 二进制。
electron-builder **无法可靠地从 macOS / Linux 交叉编译出 Windows 的原生二进制**。

**结论：Windows 安装程序必须在一台 Windows 电脑上构建。** 在 mac/linux 上跑
`dist:win` 即使产出了安装包，里面的 `node-pty` 也是错误平台的二进制，装到 Windows
会在启动时报 “PTY support is unavailable”。

---

## 一、前置环境（Windows）

1. **Node.js** — 版本需满足 `^20.19.0 || >=22.12.0`（见 `package.json` 的 `engines`）。
   建议装 **Node 20 LTS** 或 **Node 22**。
2. **Git** — 用于拉取代码。
3. **原生模块构建工具**（编译 `node-pty` 需要 C++ 工具链，二选一）：
   - 安装 Node.js 时勾选 **“Tools for Native Modules”**；或
   - 单独安装 **Visual Studio Build Tools**，勾选 **“Desktop development with C++”** 工作负载。

---

## 二、构建步骤（PowerShell 或 CMD）

```powershell
# 1) 到仓库根目录，安装全部 workspace 依赖。
#    这一步会编译 node-pty 的 Windows 二进制，并链接 apps/desktop、web、apps/shared。
#    ★ 必须从仓库根运行，不是从 apps/desktop（构建脚本 assert-root-install.cjs 会校验）。
cd <仓库根>
npm ci                # 首次或干净构建；或 npm install

# 2) 到桌面应用目录，打 Windows 安装包。
cd apps\desktop
npm run dist:win      # 同时产出 NSIS(.exe 安装器) + MSI
```

`npm run dist:win` 会依次：`build`（tsc + vite build + 打包 Electron 主进程 +
staging native-deps）→ `builder --win`。

### 只要某一种格式

```powershell
npm run dist:win:nsis   # 只产出 NSIS .exe 安装器（推荐给终端用户）
npm run dist:win:msi    # 只产出 .msi（适合企业/批量部署，如 GPO/Intune）
npm run pack            # 免安装解压版（release\ 下，本地快速试跑，不含安装器）
```

---

## 三、产物

构建产物位于：

```
apps\desktop\release\
```

文件名格式为 `Argus-${version}-win-${arch}.${ext}`（见 `build.artifactName`），
以当前 `version: 0.17.0`、x64 为例：

- **`Argus-0.17.0-win-x64.exe`** — NSIS 安装器（**推荐**）
- **`Argus-0.17.0-win-x64.msi`** — MSI 安装包
- `win-unpacked\`（若跑了 `pack`）— 免安装目录，`Argus.exe` 可直接双击

---

## 四、安装

双击 `Argus-0.17.0-win-x64.exe`：

- **非一键安装**，可**自选安装目录**（`nsis.allowToChangeInstallationDirectory: true`）。
- **按当前用户安装**（`nsis.perMachine: false`），**无需管理员权限**。
- 快捷方式名为 **Argus**（开始菜单 / 桌面）。
- 卸载项显示为 **Argus**（控制面板 → 程序和功能）。

### 首次运行

应用会把 Argus 运行时安装到 `%LOCALAPPDATA%\hermes`（即 `HERMES_HOME`，
与 CLI 安装同一布局，二者可互换），随后引导你选择 provider / model。

---

## 五、SmartScreen 与代码签名

未签名的 `.exe` 首次运行时，Windows SmartScreen 会弹警告。个人测试可点
**“更多信息” → “仍要运行”**。

正式分发建议做**代码签名**：在构建环境设置以下变量，electron-builder 会自动签名，
消除 SmartScreen 警告：

```powershell
$env:WIN_CSC_LINK = "C:\path\to\cert.pfx"   # 证书文件路径或 base64
$env:WIN_CSC_KEY_PASSWORD = "证书密码"
npm run dist:win
```

---

## 六、多模态功能与权限（摄像头 / 屏幕共享 / 麦克风 / 托盘）

打包版与开发版行为一致（全部走标准 Electron/Chromium API，跨平台）：

- **摄像头 / 麦克风**：首次使用时 Windows 会弹系统权限请求，授权即可。
- **屏幕共享**：由主进程的 `setDisplayMediaRequestHandler`（`electron/main.cjs`）驱动，
  Windows 上可用系统源选择器；无需额外配置。
- **系统托盘 + 关窗隐藏 + 后台采集**：关闭主窗口会隐藏到托盘而非退出，视频/音频采集
  在窗口隐藏时继续运行；从托盘菜单“退出”才真正结束进程。
- **流式语音 ASR / TTS**：需要网关侧配置语音后端（如 `dashscope_api_key`）才生效，
  否则麦克风按钮会提示“流式语音未启用”。

---

## 七、常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 构建报 `Run from repo root: ... npm ci` | 你在 `apps/desktop` 里跑了 `npm ci`。请**先在仓库根**执行 `npm ci`，再回到 `apps/desktop` 跑 `dist:win`。 |
| 启动报 “PTY support is unavailable” | 安装包是在非 Windows 平台构建的（node-pty 二进制平台不匹配）。请在 Windows 上重新构建。 |
| `node-pty` 编译失败 | 缺 C++ 构建工具。安装 VS Build Tools 的 “Desktop development with C++”，或重装 Node 时勾选 “Tools for Native Modules”。 |
| SmartScreen 拦截 | 未签名安装器。测试点“仍要运行”；分发请配置 `WIN_CSC_*` 代码签名。 |
| Node 版本报错 | 需 `^20.19.0 || >=22.12.0`；换成 Node 20 LTS 或 22。 |

---

## 八、其它平台（参考）

```powershell
npm run dist:mac     # macOS：DMG + zip（须在 macOS 上构建）
npm run dist:linux   # Linux：AppImage + deb + rpm（须在 Linux 上构建）
```

macOS / Windows 的签名与公证会在相应凭据存在时自动进行
（macOS：`CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*`；Windows：`WIN_CSC_*`）。
