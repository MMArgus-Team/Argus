import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef } from 'react'

import { useI18n } from '@/i18n'
import { $mmAnchor, $mmCtx } from '@/store/multimodal'

/**
 * ObservationPanels — the memory backend's live observation surfaces for the
 * multimodal right rail: 注入帧 / 画面观察 / 音频观察 / 搜索事实.
 *
 * Matches the existing MonitorList / DeepWindow card idiom (rounded border,
 * chat-surface bg, small text) so it reads as part of the same rail. Reads the
 * existing store atoms only — $mmAnchor (injected frames) and $mmCtx (obs /
 * audioObs / facts) — no new data plumbing.
 */
export function ObservationPanels() {
  return (
    <>
      <AnchorFrames />
      <SceneObs />
      <AudioObs />
      <Facts />
    </>
  )
}

function Card({ title, trailing, children }: {
  title: string
  trailing?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section className="rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) p-2.5 text-xs">
      <div className="mb-1.5 flex items-center gap-1.5 font-medium text-(--ui-text-secondary)">
        <span className="uppercase tracking-wide text-[0.6875rem] text-(--ui-text-tertiary)">{title}</span>
        {trailing != null && <span className="ml-auto">{trailing}</span>}
      </div>
      {children}
    </section>
  )
}

// 🎯 注入帧
function AnchorFrames() {
  const { t } = useI18n()
  const frames = useStore($mmAnchor)
  return (
    <Card
      title={t.multimodal.observations.injectedFrames}
      trailing={frames.length > 0 ? <span className="text-(--ui-accent)">{frames.length}</span> : undefined}
    >
      {frames.length === 0 ? (
        <div className="italic text-(--ui-text-quaternary)">{t.multimodal.observations.injectedFramesHint}</div>
      ) : (
        <div className="flex gap-1.5 overflow-x-auto">
          {frames.map((f, i) => (
            <button
              key={i}
              type="button"
              className="flex-shrink-0"
              title={t.multimodal.observations.clickToEnlarge}
              onClick={() => window.open(`data:image/jpeg;base64,${f.jpeg_b64}`, '_blank')}
            >
              <img
                src={`data:image/jpeg;base64,${f.jpeg_b64}`}
                alt={`frame ${i}`}
                className="h-16 w-auto rounded border border-(--ui-stroke-secondary)"
              />
              {f.ts != null && (
                <div className="mt-0.5 text-center text-[0.5625rem] text-(--ui-text-tertiary)">{f.ts.toFixed(1)}s</div>
              )}
            </button>
          ))}
        </div>
      )}
    </Card>
  )
}

// 画面观察
function SceneObs() {
  const { t } = useI18n()
  const ctx = useStore($mmCtx)
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [ctx.obs, ctx.version])
  return (
    <Card title={t.multimodal.observations.videoObs} trailing={<span className="text-(--ui-accent)">v{ctx.version}</span>}>
      <div ref={ref} className="max-h-52 space-y-1.5 overflow-y-auto">
        {ctx.obs.length === 0 ? (
          <span className="italic text-(--ui-text-quaternary)">{t.multimodal.observations.empty}</span>
        ) : (
          ctx.obs.map((o, i) => (
            <ObsRow key={i} ts={o.ts} speaker={o.speaker} text={o.text} tone="violet" />
          ))
        )}
      </div>
    </Card>
  )
}

// 音频观察
function AudioObs() {
  const { t } = useI18n()
  const ctx = useStore($mmCtx)
  const ref = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [ctx.audioObs])
  return (
    <Card title={t.multimodal.observations.audioObs}>
      <div ref={ref} className="max-h-44 space-y-1.5 overflow-y-auto">
        {ctx.audioObs.length === 0 ? (
          <span className="italic text-(--ui-text-quaternary)">{t.multimodal.observations.audioObsHint}</span>
        ) : (
          ctx.audioObs.map((o, i) => (
            <ObsRow key={i} ts={o.ts} speaker={o.speaker && `🗣 ${o.speaker}`} text={o.text} tone="sky" />
          ))
        )}
      </div>
    </Card>
  )
}

// SearchFactStore 的外部检索证据投影
function Facts() {
  const { t } = useI18n()
  const ctx = useStore($mmCtx)
  const list = useMemo(() => Object.entries(ctx.facts || {}), [ctx.facts])
  return (
    <Card title={t.multimodal.observations.searchFacts}>
      <div className="max-h-48 overflow-y-auto">
        {list.length === 0 ? (
          <span className="italic text-(--ui-text-quaternary)">{t.multimodal.observations.noneYet}</span>
        ) : (
          <ul className="space-y-1">
            {list.map(([k, v]) => (
              <li key={k} className="leading-snug">
                <span className="text-(--ui-purple)">{k}</span>
                <span className="text-(--ui-text-secondary)">：{String(v)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  )
}

function ObsRow({ ts, speaker, text, tone }: {
  ts?: string
  speaker?: string | null
  text?: string
  tone: 'violet' | 'sky'
}) {
  const tsClass = tone === 'violet' ? 'bg-violet-500/15 text-violet-400' : 'bg-sky-500/15 text-sky-400'
  return (
    <div className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-1.5">
      <div className="mb-0.5 flex items-center gap-1">
        {ts ? <span className={`rounded px-1 py-px font-mono text-[0.5625rem] ${tsClass}`}>{ts}</span> : null}
        {speaker ? <span className="text-[0.5625rem] text-(--ui-text-tertiary)">{speaker}</span> : null}
      </div>
      <div className="whitespace-pre-wrap leading-snug text-(--ui-text-secondary)">{text || ''}</div>
    </div>
  )
}
