import { en } from './en'
import type { Locale, Translations } from './types'
import { zh } from './zh'

// English and Simplified Chinese only. The ja / zh-hant catalogs upstream
// shipped were removed: both were missing the entire `multimodal` (244 keys)
// and `keybinds` sections, and they were built with a deep-merge helper over
// `en`, so those gaps fell back to English silently — no type error, no failing
// test. Both remaining locales are declared `: Translations` directly, which
// makes tsc the guard instead.
export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  zh
}
