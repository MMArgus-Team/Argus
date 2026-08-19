import type { Locale, Translations } from "./types";
import { en } from "./en";
import { zh } from "./zh";

// Split out of context.tsx so non-React callers (runtime.ts / translateNow) can
// reach the catalog without importing the React provider.
//
// This app ships English and Simplified Chinese only. Upstream carried 14 more
// locales, but they were half-finished — each aliased 8 whole sections
// (multimodal, system, mcp, webhooks, files, channels, pairing, profileBuilder)
// straight to `en`, so ~49% of their UI rendered in English anyway. Worse, they
// forced every newly added key to be declared optional in Translations, which
// removed tsc's ability to catch a missing zh translation. They were removed so
// the two locales we actually ship are fully type-checked.
export const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  zh,
};
