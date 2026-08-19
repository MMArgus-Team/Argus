/**
 * Screen-share source picker (Part 2 / desktop 补齐).
 *
 * Users pick which screen or app window they want to share, instead of the old
 * "auto-pick screen[0]" behaviour. The picked source id/name is stored in the
 * Electron main process and later stamped onto every frame we push, so Part 3
 * (AX/UIA window text capture) can look up the exact target window.
 *
 * Usage:
 *   1. Mount <ScreenSourcePickerHost/> once at the app root (see app/index.tsx).
 *   2. From anywhere in the app: `const src = await openSourcePicker()`.
 *      Resolves to `{id, name, shareAudio} | null` (null = user cancelled).
 *
 * Audio decision (aligns with web + platform reality):
 *   - Web/Chrome: getDisplayMedia({audio:true}) only yields audio when the user
 *     picks a TAB with "share tab audio" checked. Screens/windows → video-only.
 *   - Electron Windows and macOS 13+: main.cjs sets audio="loopback" and the
 *     host capture stack returns system audio with the selected screen/window.
 *   - Electron Linux: no supported path is wired yet, so the toggle is disabled.
 */
import { atom } from 'nanostores'
import { useStore } from '@nanostores/react'
import * as React from 'react'
import { useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

// ── Module-scoped picker state ────────────────────────────────────────────
export interface PickedSource {
  id: string
  name: string
  shareAudio: boolean
}

interface PickerState {
  open: boolean
  resolve: ((v: PickedSource | null) => void) | null
}

const $picker = atom<PickerState>({ open: false, resolve: null })

/** Show the picker modal; resolves to the chosen source or null on cancel. */
export function openSourcePicker(): Promise<PickedSource | null> {
  const cur = $picker.get()
  if (cur.open && cur.resolve) {
    return new Promise<PickedSource | null>(resolve => {
      const prev = cur.resolve!
      $picker.set({
        open: true,
        resolve: v => {
          prev(v)
          resolve(v)
        }
      })
    })
  }
  return new Promise<PickedSource | null>(resolve => {
    $picker.set({ open: true, resolve })
  })
}

function closePicker(v: PickedSource | null): void {
  const cur = $picker.get()
  if (cur.resolve) cur.resolve(v)
  $picker.set({ open: false, resolve: null })
}

// ── The host component (mount once) ───────────────────────────────────────
interface SourceItem {
  id: string
  name: string
  kind: 'screen' | 'window'
  displayId: string
  thumbnailDataUrl: string
  appIconDataUrl: string
}

export function ScreenSourcePickerHost(): React.JSX.Element | null {
  const { t } = useI18n()
  const { open } = useStore($picker)
  const [sources, setSources] = useState<SourceItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [selectedId, setSelectedId] = useState<string>('')
  // Default OFF to match web behaviour (Chrome's picker also defaults tab-audio
  // off; screens/windows never yield audio on the web either).
  const [shareAudio, setShareAudio] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState<'screen' | 'window'>('screen')
  const systemAudioSupported = window.hermesDesktop?.screenShareSystemAudio === true
  const effectiveShareAudio = systemAudioSupported && shareAudio

  useEffect(() => {
    if (!open) {
      setSources([])
      setSelectedId('')
      setError('')
      setActiveTab('screen')
      setShareAudio(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setError('')
    void (async () => {
      try {
        const bridge = window.hermesDesktop?.multimodalSourcePicker
        if (!bridge) throw new Error('multimodalSourcePicker unavailable (renderer only)')
        const res = await bridge.listSources()
        if (cancelled) return
        if (!res?.ok) {
          setError(res?.error || 'listSources failed')
          setSources([])
        } else {
          setSources(res.sources || [])
        }
      } catch (e) {
        if (!cancelled) setError(String((e as Error)?.message || e))
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [open])

  const handleConfirm = useCallback(() => {
    const picked = sources.find(s => s.id === selectedId)
    if (!picked) return
    closePicker({ id: picked.id, name: picked.name, shareAudio: effectiveShareAudio })
  }, [effectiveShareAudio, sources, selectedId])

  const handleCancel = useCallback(() => {
    closePicker(null)
  }, [])

  const screens = sources.filter(s => s.kind === 'screen')
  const windows = sources.filter(s => s.kind === 'window')

  return (
    <Dialog
      onOpenChange={o => {
        if (!o) handleCancel()
      }}
      open={open}
    >
      <DialogContent className="max-w-4xl sm:max-w-4xl">
        <DialogHeader>
          <DialogTitle>{t.multimodal.screenPicker.title}</DialogTitle>
        </DialogHeader>
        {loading ? (
          <div className="p-6 text-center text-sm text-(--ui-text-tertiary)">
            {t.multimodal.screenPicker.loading}
          </div>
        ) : error ? (
          <div className="p-4 text-sm text-red-500 whitespace-pre-wrap">
            {error}
            {error.toLowerCase().includes('failed to get sources') && (
              <div className="mt-2 text-(--ui-text-tertiary)">
                {t.multimodal.screenPicker.macPermission}
              </div>
            )}
          </div>
        ) : sources.length === 0 ? (
          <div className="p-6 text-center text-sm text-(--ui-text-tertiary)">
            {t.multimodal.screenPicker.noSources}
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {/* Tab bar: 整个屏幕 / 窗口 — 只渲染有内容的 tab, 避免空 section。
                切换 tab 时清掉当前选中项 (跨 tab 选择无意义)。 */}
            <div className="flex items-center gap-1 border-b border-(--ui-stroke-tertiary)">
              {(
                [
                  { key: 'screen', label: t.multimodal.screenPicker.tabScreen, count: screens.length },
                  { key: 'window', label: t.multimodal.screenPicker.tabWindow, count: windows.length }
                ] as const
              )
                .filter(t => t.count > 0)
                .map(t => {
                  const on = activeTab === t.key
                  return (
                    <button
                      className={cn(
                        'relative -mb-px px-3 py-2 text-sm transition-colors',
                        on
                          ? 'font-semibold text-(--ui-text-primary)'
                          : 'text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)'
                      )}
                      key={t.key}
                      onClick={() => {
                        setActiveTab(t.key)
                        setSelectedId('')
                      }}
                      style={{
                        borderBottom: '2px solid',
                        borderBottomColor: on ? 'var(--ui-accent)' : 'transparent'
                      }}
                      type="button"
                    >
                      {t.label}
                      <span className="ml-1.5 text-(--ui-text-tertiary)">{t.count}</span>
                    </button>
                  )
                })}
            </div>
            <div className="max-h-[60vh] overflow-y-auto pr-1">
              <SourceGrid
                items={activeTab === 'screen' ? screens : windows}
                onConfirm={(s: SourceItem) =>
                  closePicker({ id: s.id, name: s.name, shareAudio: effectiveShareAudio })
                }
                onPick={setSelectedId}
                selectedId={selectedId}
              />
            </div>
          </div>
        )}
        <div className="mt-2 flex items-center justify-between gap-3 pt-2">
          <label
            className={cn(
              'flex items-center gap-2 text-sm select-none',
              systemAudioSupported
                ? 'cursor-pointer'
                : 'cursor-not-allowed text-(--ui-text-tertiary)'
            )}
            htmlFor="share-audio-switch"
            title={
              systemAudioSupported
                ? t.multimodal.screenPicker.shareAudioEnabled
                : t.multimodal.screenPicker.shareAudioDisabled
            }
          >
            <Switch
              checked={shareAudio}
              disabled={!systemAudioSupported}
              id="share-audio-switch"
              onCheckedChange={setShareAudio}
              size="xs"
            />
            <span>{t.multimodal.screenPicker.shareAudioLabel}</span>
            {!systemAudioSupported && (
              <span className="text-[0.6875rem]">{t.multimodal.screenPicker.shareAudioNote}</span>
            )}
          </label>
          <div className="flex gap-2">
            <Button onClick={handleCancel} variant="ghost">
              {t.multimodal.screenPicker.cancel}
            </Button>
            <Button disabled={!selectedId} onClick={handleConfirm}>
              {t.multimodal.screenPicker.startShare}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

interface SourceGridProps {
  items: SourceItem[]
  selectedId: string
  onPick: (id: string) => void
  onConfirm: (s: SourceItem) => void
}

function SourceGrid({
  items,
  onConfirm,
  onPick,
  selectedId
}: SourceGridProps): React.JSX.Element {
  const { t } = useI18n()
  return (
    // Tailwind 编译产物里没有 plain `grid-cols-3`, 所以关键布局(display: grid +
    // template-columns)走 inline style 100% 兜底; 其余仍走 Tailwind class。
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
        gap: '1rem'
      }}
    >
      {items.map(s => {
        const label = s.name || t.multimodal.screenPicker.noTitle
        const active = selectedId === s.id
        return (
          <button
            aria-pressed={active}
            className={cn(
              // ★ min-w-0 + w-full: 打破 grid item 默认 min-width:auto,
              //   否则 img natural 1024px 会撑爆 1fr track。
              'group flex w-full min-w-0 flex-col gap-2 rounded-xl p-2 text-left transition-all duration-150',
              active
                ? 'bg-(--ui-row-active-background)'
                : 'hover:bg-(--ui-row-hover-background)'
            )}
            key={s.id}
            onClick={() => onPick(s.id)}
            onDoubleClick={() => {
              onPick(s.id)
              queueMicrotask(() => onConfirm(s))
            }}
            // inline style: active 时 2px 主题色描边; idle 时保留 2px 透明 border
            // 保证尺寸稳定, 不会有 layout shift。--ui-accent 是项目主色 token。
            style={{
              border: '2px solid',
              borderColor: active ? 'var(--ui-accent)' : 'transparent'
            }}
            title={label}
            type="button"
          >
            <div
              className="aspect-video w-full overflow-hidden rounded-lg"
              // 无预览时的 fallback bg; 有 img 时 object-cover 铺满, 不会露出。
              style={{ background: 'var(--ui-bg-elevated, rgba(255,255,255,0.04))' }}
            >
              {s.thumbnailDataUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  alt={label}
                  className="h-full w-full object-cover"
                  draggable={false}
                  src={s.thumbnailDataUrl}
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-sm text-(--ui-text-tertiary)">
                  {t.multimodal.screenPicker.noPreview}
                </div>
              )}
            </div>
            <div className="flex items-center gap-2 px-0.5">
              {s.appIconDataUrl && (
                // eslint-disable-next-line @next/next/no-img-element
                <img alt="" className="h-4 w-4 flex-none" src={s.appIconDataUrl} />
              )}
              <div className="min-w-0 truncate text-sm">{label}</div>
            </div>
          </button>
        )
      })}
    </div>
  )
}
