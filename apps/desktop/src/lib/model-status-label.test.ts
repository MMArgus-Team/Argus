import { describe, expect, it } from 'vitest'

import {
  currentPickerSelection,
  displayModelName,
  formatModelStatusLabel,
  isCurrentProviderRow,
  reasoningEffortLabel
} from './model-status-label'

describe('model-status-label', () => {
  it('formats display names consistently', () => {
    expect(displayModelName('anthropic/claude-opus-4.8-fast')).toBe('Opus 4.8')
    expect(displayModelName('openai/gpt-5.5-fast')).toBe('GPT-5.5')
    expect(displayModelName('deepseek/deepseek-v4-pro-thinking')).toBe('Deepseek V4 Pro')
    expect(displayModelName('openai/gpt-5.5')).toBe('GPT-5.5')
  })

  it('strips trailing date-pin snapshots from the display name', () => {
    expect(displayModelName('claude-opus-4-5-20251101')).toBe('Opus 4 5')
    expect(displayModelName('anthropic/claude-haiku-4-5-20251001')).toBe('Haiku 4 5')
  })

  it('maps reasoning effort to compact labels', () => {
    expect(reasoningEffortLabel('high')).toBe('High')
    expect(reasoningEffortLabel('xhigh')).toBe('Max')
    expect(reasoningEffortLabel('')).toBe('')
  })

  it('appends fast + effort session state to the status label', () => {
    expect(formatModelStatusLabel('openai/gpt-5.5', { fastMode: true, reasoningEffort: 'high' })).toBe(
      'GPT-5.5 · Fast High'
    )
  })

  it('always surfaces the effort (default medium) so the level is visible', () => {
    expect(formatModelStatusLabel('openai/gpt-5.5', { reasoningEffort: 'medium' })).toBe('GPT-5.5 · Med')
    expect(formatModelStatusLabel('openai/gpt-5.5')).toBe('GPT-5.5 · Med')
  })

  it('returns just the placeholder name when there is no model', () => {
    expect(formatModelStatusLabel('')).toBe('No model')
  })

  describe('currentPickerSelection', () => {
    const store = { model: 'opus', provider: 'anthropic' }
    const options = { model: 'hermes-4', provider: 'nous' }

    it('prefers the sticky composer pick over the profile default pre-session', () => {
      expect(currentPickerSelection(false, store, options)).toEqual(store)
    })

    it('lets the live session model.options win when a session exists', () => {
      expect(currentPickerSelection(true, store, options)).toEqual(options)
    })

    it('falls back to options when the store is empty', () => {
      expect(currentPickerSelection(false, { model: '', provider: '' }, options)).toEqual(options)
    })

    it('falls back to the store while options are still loading', () => {
      expect(currentPickerSelection(true, store, undefined)).toEqual(store)
    })
  })

  // A `custom:` provider row's slug is derived from the endpoint HOSTNAME, while
  // model.options reports the bare configured provider name. Comparing the two by
  // string never matched, so the active model was never "current": the composer
  // pill showed live state ("Off") while the picker row fell back to a preset
  // ("Low"). These are the real values observed from the gateway.
  describe('isCurrentProviderRow', () => {
    it('trusts is_current when the slug cannot match the bare provider name', () => {
      expect(isCurrentProviderRow({ is_current: true, slug: 'custom:open.bigmodel.cn' }, 'custom')).toBe(true)
    })

    it('does not mark a row current just because it shares the custom: prefix', () => {
      expect(isCurrentProviderRow({ is_current: false, slug: 'custom:api.other.example' }, 'custom')).toBe(false)
    })

    it('regression: a bare string compare would have failed this case', () => {
      const row = { is_current: true, slug: 'custom:open.bigmodel.cn' }

      // The old logic, kept here as the tripwire.
      expect(row.slug === 'custom').toBe(false)
      expect(isCurrentProviderRow(row, 'custom')).toBe(true)
    })

    it('falls back to the slug compare when is_current is absent', () => {
      expect(isCurrentProviderRow({ slug: 'anthropic' }, 'anthropic')).toBe(true)
      expect(isCurrentProviderRow({ slug: 'nous' }, 'anthropic')).toBe(false)
    })

    it('lets an explicit is_current:false win over a matching slug', () => {
      expect(isCurrentProviderRow({ is_current: false, slug: 'anthropic' }, 'anthropic')).toBe(false)
    })
  })
})
