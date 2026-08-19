import { TRANSLATIONS } from "./catalog";
import type { Locale, Translations } from "./types";

// Non-React translator, mirroring apps/desktop/src/i18n/runtime.ts. Pure helper
// functions (e.g. the QueryWorker trajectory mappers) run outside the React
// tree and cannot call useI18n(), so they resolve copy through translateNow()
// against the locale the provider last published.

const DEFAULT_LOCALE: Locale = "en";

let runtimeLocale: Locale = DEFAULT_LOCALE;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function resolvePath(catalog: Translations, key: string): unknown {
  return key.split(".").reduce<unknown>((current, part) => {
    if (!isRecord(current)) {
      return undefined;
    }

    return current[part];
  }, catalog);
}

function renderTranslation(value: unknown, args: unknown[]): string | null {
  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "function") {
    return (value as (...args: unknown[]) => string)(...args);
  }

  return null;
}

export function setRuntimeI18nLocale(locale: Locale) {
  runtimeLocale = locale;
}

export function translateNow(key: string, ...args: unknown[]): string {
  const active = renderTranslation(resolvePath(TRANSLATIONS[runtimeLocale], key), args);

  if (active !== null) {
    return active;
  }

  if (runtimeLocale !== DEFAULT_LOCALE) {
    const fallback = renderTranslation(resolvePath(TRANSLATIONS[DEFAULT_LOCALE], key), args);

    if (fallback !== null) {
      return fallback;
    }
  }

  return key;
}
