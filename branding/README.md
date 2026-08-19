# Argus Branding Assets

**Design**: 观察光圈 (Observation Aperture) — three concentric layers convey
"always-on observation": dashed outer arc (radar sweep) + solid middle ring
(lens) + amber center dot (pupil / focus).

## Colours

| Role | Hex |
|---|---|
| Background | `#0B1220` (midnight blue) |
| Ring / arc | `#34D399` (fluorescent green — "alive") |
| Pupil | `#F59E0B` (amber — "attention") |
| Pupil highlight | `#FEF3C7` (55% opacity) |

## Source files

- **`argus-logo.svg`** — full-colour master (1024×1024 viewBox, exports to any size)
- **`argus-logo-mono.svg`** — single-colour version for tray / menu-bar (uses `currentColor` so the OS auto-tints it based on theme)

## How to generate the raster/binary variants

Each platform needs a specific binary:

### macOS (.icns)

```bash
# 1. Export 10 PNG sizes into an iconset folder:
mkdir argus.iconset
for size in 16 32 64 128 256 512; do
  rsvg-convert -w $size  -h $size  argus-logo.svg > argus.iconset/icon_${size}x${size}.png
  rsvg-convert -w $((size*2)) -h $((size*2)) argus-logo.svg > argus.iconset/icon_${size}x${size}@2x.png
done
# 2. Build .icns:
iconutil -c icns argus.iconset -o icon.icns
# 3. Replace apps/desktop/assets/icon.icns
mv icon.icns ../apps/desktop/assets/icon.icns
```

### Windows (.ico)

```bash
# ImageMagick required
magick -background none argus-logo.svg -define icon:auto-resize=16,32,48,64,128,256 ../apps/desktop/assets/icon.ico
```

### PNG (favicon, banner, tray)

```bash
# 512×512 for apps/desktop/assets/icon.png (Linux / fallback)
rsvg-convert -w 512 -h 512 argus-logo.svg > ../apps/desktop/assets/icon.png

# 16×16 favicon (web)
rsvg-convert -w 16 -h 16 argus-logo.svg > ../web/public/favicon-16.png
# ...combine into .ico for web/public/favicon.ico

# Tray icon: single-colour, template-mode (macOS) requires PNG named *Template@2x.png
rsvg-convert -w 44 -h 44 argus-logo-mono.svg > trayTemplate@2x.png
```

### Wordmark (splash / about page)

Font suggestion: **Inter Display** for UI, **Fraunces** for the splash wordmark
(现代衬线, 呼应希腊神话来源).

## Files to replace with the new icon

| Path | Purpose |
|---|---|
| `apps/desktop/assets/icon.icns` | macOS app icon |
| `apps/desktop/assets/icon.ico` | Windows app icon |
| `apps/desktop/assets/icon.png` | Linux app icon / fallback |
| `apps/desktop/public/apple-touch-icon.png` | Web/PWA touch icon |
| `apps/desktop/public/hermes.png` | ⚠️ Rename to `argus.png` after regenerating |
| `apps/desktop/public/hermes-sprite.png` | ⚠️ Animation frames — need custom re-render |
| `apps/desktop/public/hermes-frames/` | ⚠️ Animation frames dir — need custom re-render |
| `apps/desktop/public/nous-girl.jpg` | ⚠️ Third-party mascot image — remove or replace |
| `assets/banner.png` | Repo README banner |
| `acp_registry/icon.svg` | ACP registry entry icon |
| `web/public/favicon.ico` | Web favicon |
