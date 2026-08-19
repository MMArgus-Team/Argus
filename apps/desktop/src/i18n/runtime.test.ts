import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { fieldCopyForSchemaKey } from '@/app/settings/field-copy'

import { TRANSLATIONS } from './catalog'
import { setRuntimeI18nLocale, translateNow } from './runtime'
import { zh } from './zh'

describe('desktop i18n runtime translator', () => {
  beforeEach(() => {
    setRuntimeI18nLocale('en')
  })

  afterEach(() => {
    setRuntimeI18nLocale('en')
  })

  it('translates string paths for the active runtime locale', () => {
    setRuntimeI18nLocale('zh')

    expect(translateNow('boot.ready')).toBe('Argus 桌面版已就绪')
    expect(translateNow('notifications.voice.noSpeechDetected')).toBe('没有检测到语音')
    expect(translateNow('composer.lookupNoMatches')).toBe('没有匹配项。')
    expect(translateNow('assistant.tool.statusRecovered')).toBe('已恢复')
  })

  it('passes arguments to function translations', () => {
    expect(translateNow('notifications.updateReadyMessage', 2)).toBe('2 new changes available.')
  })

  it('translates migrated overlap keys', () => {
    setRuntimeI18nLocale('zh')
    expect(translateNow('common.save')).toBe('保存')
    expect(translateNow('cron.promptPlaceholder')).toBe('总结我未读的 Slack 话题，并把前 5 条邮件发给我…')
  })

  it('translates settings copy', () => {
    setRuntimeI18nLocale('zh')
    expect(translateNow('settings.appearance.title')).toBe('外观')
    expect(translateNow('settings.nav.providers')).toBe('提供方')
    expect(translateNow('settings.nav.providerApiKeys')).toBe('API 密钥')
  })

  it('keeps translated settings field copy addressable from schema keys', () => {
    const field = ['display', 'show_reasoning'].join('.')

    expect(fieldCopyForSchemaKey(zh.settings.fieldLabels, field)).toBe('推理过程块')
    expect(fieldCopyForSchemaKey(zh.settings.fieldDescriptions, field)).toBe('当后端提供推理内容时予以显示。')
  })

  it('falls back to English when the active locale cannot resolve a key', () => {
    const boot = TRANSLATIONS.zh.boot as { ready?: string }
    const originalReady = boot.ready

    try {
      boot.ready = undefined
      setRuntimeI18nLocale('zh')

      expect(translateNow('boot.ready')).toBe('Argus Desktop is ready')
    } finally {
      boot.ready = originalReady
    }
  })

  it('returns the key when no locale can resolve a path', () => {
    setRuntimeI18nLocale('zh')

    expect(translateNow('missing.path')).toBe('missing.path')
  })
})
