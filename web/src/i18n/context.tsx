import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from "react";
import type { Locale, Translations } from "./types";
import { en } from "./en";
import { TRANSLATIONS } from "./catalog";
import { setRuntimeI18nLocale } from "./runtime";

// Display metadata for the language picker — endonym (native name) so users
// recognize their language even if they don't speak the current UI language.
// Exposed as a constant so the LanguageSwitcher and any future settings page
// can share the same list.
//
// We intentionally do NOT pair locales with country flags. Languages are not
// countries (English ≠ GB, Chinese variants ≠ any single jurisdiction).
// Endonyms are unambiguous and avoid the political mismapping that flag
// pairings inevitably create.
export const LOCALE_META: Record<Locale, { name: string }> = {
  en: { name: "English" },
  zh: { name: "简体中文" },
};

const SUPPORTED_LOCALES = Object.keys(TRANSLATIONS) as Locale[];
const STORAGE_KEY = "argus-locale";

function isLocale(value: string): value is Locale {
  return (SUPPORTED_LOCALES as string[]).includes(value);
}

// Map a BCP 47 tag from the browser/OS onto a locale we ship: exact tag first,
// then the bare language subtag ("zh-CN" -> "zh"). Anything we don't ship falls
// through to English rather than guessing a near neighbour.
function matchSystemLocale(tag: string): Locale | null {
  const lower = tag.trim().toLowerCase();
  if (!lower) return null;
  if (isLocale(lower)) return lower;

  const [language] = lower.split("-");

  // Every zh tag routes to Simplified, including the Traditional ones
  // (zh-Hant / zh-TW / zh-HK / zh-MO). We no longer ship a Traditional
  // catalog, and Simplified is far closer for those readers than English.
  if (language === "zh") return "zh";

  return isLocale(language) ? language : null;
}

// Exported for tests: the pure part of first-run locale selection, independent
// of navigator and React.
export function resolveSystemLocale(tags: readonly string[]): Locale | null {
  for (const tag of tags) {
    const matched = tag ? matchSystemLocale(tag) : null;
    if (matched) return matched;
  }

  return null;
}

function getSystemLocale(): Locale | null {
  if (typeof navigator === "undefined") return null;

  // navigator.languages is ordered by user preference; fall back to the
  // single-value form for older browsers.
  const tags = navigator.languages?.length
    ? navigator.languages
    : [navigator.language];

  return resolveSystemLocale(tags);
}

function getInitialLocale(): Locale {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && isLocale(stored)) return stored;
  } catch {
    // SSR or privacy mode
  }

  // No explicit choice yet: follow the system language, and fall back to
  // English when the OS/browser language is one we don't ship.
  return getSystemLocale() ?? "en";
}

// Our internal locale ids are not all valid BCP 47 tags, and `<html lang>` is
// read by screen readers, the CJK line-breaker, and font fallback — so publish
// a proper tag rather than the raw id.
export const HTML_LANG: Record<Locale, string> = {
  en: "en",
  zh: "zh-Hans",
};

function syncHtmlLang(locale: Locale) {
  if (typeof document === "undefined") return;
  document.documentElement.lang = HTML_LANG[locale] ?? locale;
}

interface I18nContextValue {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: Translations;
}

const I18nContext = createContext<I18nContextValue>({
  locale: "en",
  setLocale: () => {},
  t: en,
});

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(getInitialLocale);

  // Publish during render, not in an effect: pure helpers reached via
  // translateNow() are called while the tree renders, so an effect would leave
  // them one paint behind (English on first render after a locale switch).
  setRuntimeI18nLocale(locale);

  // `<html lang>` is a DOM mutation, so it belongs in an effect rather than in
  // the render path. Nothing reads it back synchronously during render.
  useEffect(() => {
    syncHtmlLang(locale);
  }, [locale]);

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    setRuntimeI18nLocale(l);
    try {
      localStorage.setItem(STORAGE_KEY, l);
    } catch {
      // ignore
    }
  }, []);

  const value: I18nContextValue = {
    locale,
    setLocale,
    t: TRANSLATIONS[locale],
  };

  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  return useContext(I18nContext);
}

/**
 * Subscribe a component to locale changes without reading `t`.
 *
 * `translateNow()` resolves against a module-level `runtimeLocale`, not React
 * state, so it is invisible to the reconciler: a `memo()` component whose props
 * did not change will NOT re-render on a language switch and keeps painting the
 * previous language until something else happens to dirty it. Calling this hook
 * gives such a component a real context dependency, so switching languages
 * re-renders it and its `translateNow()` calls resolve against the new locale.
 *
 * Use this in `memo()` components that call `translateNow()` but have no other
 * reason to read the i18n context. Components that already destructure `t` from
 * `useI18n()` are subscribed by that call and do not need this.
 */
export function useLocaleRevision(): Locale {
  return useContext(I18nContext).locale;
}
