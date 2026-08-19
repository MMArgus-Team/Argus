import type { Locale } from './types'

export const DEFAULT_LOCALE: Locale = 'en'

export const LOCALE_OPTIONS = [
  {
    id: 'en',
    name: 'English',
    englishName: 'English',
    configValue: 'en'
  },
  {
    id: 'zh',
    name: '简体中文',
    englishName: 'Simplified Chinese',
    configValue: 'zh'
  },
] as const satisfies readonly { configValue: string; englishName: string; id: Locale; name: string }[]

// `name` is the endonym (native name) shown in the picker so users recognize
// their language regardless of the current UI language. No country flags:
// languages are not countries. `englishName` is search-only (not shown) so an
// English speaker can type "chinese"/"simplified" to filter the list.
export const LOCALE_META: Record<Locale, { name: string; englishName: string }> = Object.fromEntries(
  LOCALE_OPTIONS.map(locale => [locale.id, { name: locale.name, englishName: locale.englishName }])
) as Record<Locale, { name: string; englishName: string }>

// Every zh tag maps to Simplified, including the Traditional ones (zh-TW /
// zh-HK / zh-MO / zh-Hant). We no longer ship a Traditional catalog, and
// Simplified is far closer for those readers than falling back to English.
// Kept as aliases (rather than dropped) so an existing config.yaml carrying
// `language: zh-TW` keeps a Chinese UI instead of silently reverting to en.
const LOCALE_ALIASES: Record<string, Locale> = {
  en: 'en',
  'en-us': 'en',
  en_us: 'en',
  zh: 'zh',
  'zh-cn': 'zh',
  zh_cn: 'zh',
  'zh-hans': 'zh',
  zh_hans: 'zh',
  'zh-hans-cn': 'zh',
  zh_hans_cn: 'zh',
  'zh-tw': 'zh',
  zh_tw: 'zh',
  'zh-hk': 'zh',
  zh_hk: 'zh',
  'zh-mo': 'zh',
  zh_mo: 'zh',
  'zh-hant': 'zh',
  zh_hant: 'zh',
  'zh-hant-tw': 'zh',
  zh_hant_tw: 'zh',
  'zh-hant-hk': 'zh',
  zh_hant_hk: 'zh'
}

export function isLocale(value: unknown): value is Locale {
  return typeof value === 'string' && LOCALE_OPTIONS.some(locale => locale.id === value)
}

export function normalizeLocale(value: unknown): Locale {
  if (typeof value !== 'string') {
    return DEFAULT_LOCALE
  }

  return LOCALE_ALIASES[value.trim().toLowerCase()] ?? DEFAULT_LOCALE
}

export function isSupportedLocaleValue(value: unknown): boolean {
  return typeof value === 'string' && LOCALE_ALIASES[value.trim().toLowerCase()] != null
}

// Resolve an OS/browser BCP 47 tag onto a locale we ship. The alias table
// already covers the common exact tags; beyond that, match Traditional Chinese
// by script/region and otherwise fall back to the bare language subtag. Returns
// null (rather than English) when we ship nothing for the tag, so callers can
// tell "no match" apart from "the user really wants English".
export function matchSystemLocale(value: unknown): Locale | null {
  if (typeof value !== 'string') {
    return null
  }

  const lower = value.trim().toLowerCase()

  if (!lower) {
    return null
  }

  const aliased = LOCALE_ALIASES[lower]

  if (aliased) {
    return aliased
  }

  const [language] = lower.split('-')

  // All zh variants, Simplified and Traditional alike, route to zh.
  if (language === 'zh') {
    return 'zh'
  }

  return LOCALE_ALIASES[language] ?? null
}

// The system language, used only when the user has not chosen one yet.
// Electron's renderer exposes the OS languages through navigator.
export function resolveSystemLocale(): Locale | null {
  if (typeof navigator === 'undefined') {
    return null
  }

  const tags = navigator.languages?.length ? navigator.languages : [navigator.language]

  for (const tag of tags) {
    const matched = matchSystemLocale(tag)

    if (matched) {
      return matched
    }
  }

  return null
}

export function localeConfigValue(locale: Locale): string {
  return LOCALE_OPTIONS.find(item => item.id === locale)?.configValue ?? DEFAULT_LOCALE
}
