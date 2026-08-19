import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef } from 'react'

import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'
import {
  $mmDeepRailWidth,
  $mmMediaRailWidth,
  bindMultimodalToMainSession,
  setDeepRailWidth,
  setMediaRailWidth
} from '@/store/multimodal'
import { $mmBgItems, $mmMonitors, $mmToasts, $mmWatchers, type MmToast } from '@/store/multimodal-deep'

import { DeepPanel } from './deep-panel'
import { ObservationPanels } from './observation-panels'
import { VideoStage } from './video-stage'

/**
 * 多模态在主聊天 shell 里的两列 (对齐 web 布局):
 *   - MediaRail: 视频流 + 注入帧/画面观察/音频观察/搜索事实。放在【会话列表和主 Agent
 *     之间】(左第2列), 宽度可拖拽调节 (带 min/max)。
 *   - DeepRail: 深度研究进度 + monitor 注册表。放在【主 Agent 右侧】(第4列), 宽度可拖拽,
 *     仅当有深度研究/监控任务时才出现 (无任务自隐藏, 不占横向空间)。
 *
 * side-channel 绑定 (bindMultimodalToMainSession: $activeSessionId → $mmSessionId +
 * 事件订阅) 挂在 MediaRail (它始终 mount)。幂等, gateway/profile 切换时自重绑。
 * 不在 unmount 解绑: capture/mic/observations 在 module scope (后台常驻), 普通路由切换
 * 不能拆绑定。
 */

/** 竖栏拖拽把手。edge='right' → 装在栏右边缘 (向右拖变宽, MediaRail 用);
 *  edge='left' → 装在栏左边缘 (向左拖变宽, DeepRail 用)。指针捕获, 带 min/max 夹取。 */
function RailResizeHandle({
  edge,
  getWidth,
  setWidth
}: {
  edge: 'left' | 'right'
  getWidth: () => number
  setWidth: (px: number) => void
}) {
  const dragRef = useRef<{ startX: number; startW: number } | null>(null)
  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault()
      dragRef.current = { startX: e.clientX, startW: getWidth() }
      e.currentTarget.setPointerCapture(e.pointerId)
    },
    [getWidth]
  )
  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const d = dragRef.current
      if (!d) return
      const dx = e.clientX - d.startX
      // right 边缘: 向右 (dx>0) 变宽; left 边缘: 向左 (dx<0) 变宽。
      setWidth(edge === 'right' ? d.startW + dx : d.startW - dx)
    },
    [edge, setWidth]
  )
  const onPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    dragRef.current = null
    try {
      e.currentTarget.releasePointerCapture(e.pointerId)
    } catch {
      /* noop */
    }
  }, [])
  return (
    <div
      aria-hidden="true"
      className={`absolute inset-y-0 z-20 w-1.5 cursor-col-resize hover:bg-(--ui-accent)/30 [-webkit-app-region:no-drag] ${
        edge === 'right' ? '-right-[3px]' : '-left-[3px]'
      }`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    />
  )
}

export function MediaRail() {
  const gateway = useStore($gateway)
  const width = useStore($mmMediaRailWidth)

  useEffect(() => {
    if (!gateway) return
    bindMultimodalToMainSession()
  }, [gateway])

  return (
    <aside
      data-selectable-text="true"
      // 顶部 OS 标题栏空间由共享标题栏那一行占据; 宽度可拖拽 (右边缘把手)。
      className="relative flex min-h-0 shrink-0 flex-col gap-3 overflow-y-auto border-r border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) px-3 pb-3 pt-3"
      style={{ width: `${width}px` }}
    >
      <VideoStage />
      <ObservationPanels />
      <RailResizeHandle edge="right" getWidth={() => $mmMediaRailWidth.get()} setWidth={setMediaRailWidth} />
    </aside>
  )
}

/**
 * DeepRail — 深度研究列 (主 Agent 右侧, 第4列)。无任务时整列不渲染 (对齐 web:
 * 有深度研究/监控才出现, 否则不占横向空间)。深度分析事件多 → 独立可滚动。
 */
export function DeepRail() {
  const bgItems = useStore($mmBgItems)
  const monitors = useStore($mmMonitors)
  const watchers = useStore($mmWatchers)
  const toasts = useStore($mmToasts)
  const width = useStore($mmDeepRailWidth)
  // 只按【未完成】任务决定是否自动打开: watcher status ∈ {running,interrupted,disabled}
  // 都算未完成; monitor 在册即算 (无完成概念); bgItems 含实时进度 + 用户手动重开的只读窗口。
  const hasIncompleteWatcher = watchers.some(w =>
    ['running', 'interrupted', 'disabled'].includes(String(w.status || ''))
  )
  // 有 toast 时也保持面板可见 (监控停用后可能已无活跃任务, 否则 toast 无处可显)。
  const hasAny = bgItems.length > 0 || monitors.length > 0 || hasIncompleteWatcher || toasts.length > 0
  if (!hasAny) return null
  return (
    <aside
      data-selectable-text="true"
      // 宽度可拖拽 (左边缘把手)。无任务时整列 return null 自隐藏 (见上)。
      className="relative flex min-h-0 shrink-0 flex-col overflow-hidden border-l border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) px-3 pb-3 pt-3"
      style={{ width: `${width}px` }}
    >
      <RailResizeHandle edge="left" getWidth={() => $mmDeepRailWidth.get()} setWidth={setDeepRailWidth} />
      <DeepPanel />
      <MmToastStack toasts={toasts} />
    </aside>
  )
}

/** 右侧面板底部的 toast 小框栈: 监控/深度研究过程失败/停用通知, 3s 后淡出 (数据侧删除,
 *  这里 fade-out 动画)。绝对定位在面板底部, 不挤占内容。 */
function MmToastStack({ toasts }: { toasts: MmToast[] }) {
  if (toasts.length === 0) return null
  return (
    <div className="pointer-events-none absolute inset-x-2 bottom-2 z-30 flex flex-col gap-1.5">
      {toasts.map(t => (
        <div
          key={t.id}
          className={cn(
            'mm-toast-in pointer-events-auto rounded-md border px-2.5 py-1.5 text-[0.7rem] leading-snug shadow-md backdrop-blur-sm',
            t.level === 'error'
              ? 'border-(--ui-red)/40 bg-(--ui-red)/12 text-(--ui-red)'
              : t.level === 'warning'
                ? 'border-(--ui-yellow)/40 bg-(--ui-yellow)/12 text-(--ui-yellow)'
                : 'border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) text-(--ui-text-secondary)'
          )}
        >
          {t.text}
        </div>
      ))}
    </div>
  )
}
