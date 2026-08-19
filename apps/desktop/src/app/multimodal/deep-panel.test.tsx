import { cleanup, render, screen } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import {
  $mmBgItems,
  $mmMonitorAlerts,
  $mmMonitors,
  $mmWatchers,
  resetDeepUi
} from '@/store/multimodal-deep'

import { DeepPanel } from './deep-panel'

function renderZh(ui: ReactElement) {
  return render(<I18nProvider configClient={null} initialLocale="zh">{ui}</I18nProvider>)
}

describe('desktop watcher registry presentation', () => {
  beforeEach(() => {
    resetDeepUi()
    $mmBgItems.set([])
    $mmMonitors.set([])
    $mmMonitorAlerts.set({})
  })

  afterEach(() => {
    cleanup()
    resetDeepUi()
  })

  it('hides deleted rows and presents terminal and transitional states accurately', () => {
    $mmWatchers.set([
      { watcher_id: 'done', label: '完成任务', status: 'done' },
      { watcher_id: 'stopping', label: '停止任务', status: 'stopping' },
      { watcher_id: 'interrupted', label: '中断任务', status: 'interrupted' },
      { watcher_id: 'deleted', label: '已删任务', status: 'deleted' }
    ])

    renderZh(<DeepPanel />)

    expect(screen.queryByText('已删任务')).toBeNull()
    expect(screen.getByText('· 已完成')).toBeTruthy()
    expect(screen.getByText('· 正在停止')).toBeTruthy()
    expect(screen.getByText('· 已中断')).toBeTruthy()
    expect(screen.getByRole('switch', { name: '完成任务：已完成' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('switch', { name: '停止任务：正在停止' }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('switch', { name: '中断任务：已中断' }).hasAttribute('disabled')).toBe(false)
  })

  it('renders nothing when deleted watchers are the only deep-panel state', () => {
    $mmWatchers.set([{ watcher_id: 'deleted', label: '已删任务', status: 'deleted' }])

    const { container } = renderZh(<DeepPanel />)

    expect(container.innerHTML).toBe('')
  })

  it('renders bounded Monitor evidence with the true model-input count', () => {
    $mmMonitors.set([{
      monitor_id: 'mon-evidence',
      brief: '看到手机时提醒',
      status: 'running'
    }])
    $mmMonitorAlerts.set({
      'mon-evidence': [{
        id: 'alert-evidence',
        text: '手机出现了',
        ts: Date.now(),
        evidence: {
          input_count: 18,
          shown_count: 2,
          frames: [
            { ts: 2, source_type: 'screen', thumb_b64: 'dGh1bWIx' },
            { ts: 8, source_type: 'screen', thumb_b64: 'dGh1bWIy' }
          ]
        }
      }]
    })

    renderZh(<DeepPanel />)

    expect(screen.getByText('本轮模型输入 18 帧 · 展示 2 张证据缩略图')).toBeTruthy()
    expect(screen.getByAltText('Monitor 证据帧 1').getAttribute('src')).toBe(
      'data:image/jpeg;base64,dGh1bWIx'
    )
    expect(screen.getByAltText('Monitor 证据帧 2').getAttribute('src')).toBe(
      'data:image/jpeg;base64,dGh1bWIy'
    )
  })
})
