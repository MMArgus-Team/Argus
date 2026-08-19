import { describe, expect, it } from 'vitest'

import {
  DEFAULT_LOCALE,
  isLocale,
  isSupportedLocaleValue,
  localeConfigValue,
  matchSystemLocale,
  normalizeLocale
} from './languages'

describe('desktop i18n languages', () => {
  it('normalizes supported locale aliases', () => {
    expect(normalizeLocale('en')).toBe('en')
    expect(normalizeLocale('EN-US')).toBe('en')
    expect(normalizeLocale('zh')).toBe('zh')
    expect(normalizeLocale('zh-CN')).toBe('zh')
    expect(normalizeLocale('zh-Hans')).toBe('zh')
    expect(normalizeLocale(' zh_hans_cn ')).toBe('zh')
  })

  // Traditional tags stay in the alias table on purpose: an existing
  // config.yaml carrying `language: zh-TW` should keep a Chinese UI rather than
  // silently revert to English now that the zh-hant catalog is gone.
  it('folds Traditional Chinese tags into Simplified', () => {
    expect(normalizeLocale('zh-Hant')).toBe('zh')
    expect(normalizeLocale('zh-TW')).toBe('zh')
    expect(normalizeLocale('zh_HK')).toBe('zh')
    expect(normalizeLocale('zh-MO')).toBe('zh')
  })

  it('falls back to English for empty or unsupported values', () => {
    expect(normalizeLocale(null)).toBe(DEFAULT_LOCALE)
    expect(normalizeLocale('')).toBe(DEFAULT_LOCALE)
    expect(normalizeLocale('de')).toBe(DEFAULT_LOCALE)
    // Removed locales must not resolve any more.
    expect(normalizeLocale('ja')).toBe(DEFAULT_LOCALE)
    expect(normalizeLocale('ja-JP')).toBe(DEFAULT_LOCALE)
  })

  it('distinguishes exact locale ids from supported config aliases', () => {
    expect(isSupportedLocaleValue('zh-CN')).toBe(true)
    expect(isSupportedLocaleValue('zh-TW')).toBe(true)
    expect(isSupportedLocaleValue('ja-JP')).toBe(false)
    expect(isSupportedLocaleValue('de')).toBe(false)
    expect(isLocale('zh-CN')).toBe(false)
    expect(isLocale('zh')).toBe(true)
    expect(isLocale('zh-hant')).toBe(false)
    expect(isLocale('ja')).toBe(false)
  })

  it('returns the persisted config value for supported locales', () => {
    expect(localeConfigValue('en')).toBe('en')
    expect(localeConfigValue('zh')).toBe('zh')
  })

  // matchSystemLocale backs "follow the system language on first run". It must
  // return null rather than English for languages we don't ship, so the caller
  // can tell "no match" from "the user picked English".
  describe('matchSystemLocale', () => {
    it('matches English and Chinese system tags', () => {
      expect(matchSystemLocale('en')).toBe('en')
      expect(matchSystemLocale('en-GB')).toBe('en')
      expect(matchSystemLocale('zh')).toBe('zh')
      expect(matchSystemLocale('zh-CN')).toBe('zh')
      expect(matchSystemLocale('zh-Hans-CN')).toBe('zh')
      expect(matchSystemLocale('zh-SG')).toBe('zh')
    })

    it('routes Traditional Chinese regions to Simplified too', () => {
      // No Traditional catalog ships now; Simplified beats English for them.
      expect(matchSystemLocale('zh-TW')).toBe('zh')
      expect(matchSystemLocale('zh-HK')).toBe('zh')
      expect(matchSystemLocale('zh-MO')).toBe('zh')
      expect(matchSystemLocale('zh-Hant')).toBe('zh')
      expect(matchSystemLocale('zh-Hant-TW')).toBe('zh')
    })

    it('returns null for languages this app does not ship', () => {
      expect(matchSystemLocale('de')).toBeNull()
      expect(matchSystemLocale('fr-FR')).toBeNull()
      expect(matchSystemLocale('ja')).toBeNull()
      expect(matchSystemLocale('ja-JP')).toBeNull()
      expect(matchSystemLocale('')).toBeNull()
      expect(matchSystemLocale(null)).toBeNull()
      expect(matchSystemLocale(undefined)).toBeNull()
    })
  })
})
