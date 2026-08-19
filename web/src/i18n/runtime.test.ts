import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { TRANSLATIONS } from "./catalog";
import { setRuntimeI18nLocale, translateNow } from "./runtime";

describe("web i18n runtime translator", () => {
  beforeEach(() => {
    setRuntimeI18nLocale("en");
  });

  afterEach(() => {
    setRuntimeI18nLocale("en");
  });

  it("translates string paths for the active runtime locale", () => {
    expect(translateNow("multimodal.recall.completeNotFound")).toBe(
      TRANSLATIONS.en.multimodal.recall.completeNotFound,
    );

    setRuntimeI18nLocale("zh");
    expect(translateNow("multimodal.recall.completeNotFound")).toBe(
      TRANSLATIONS.zh.multimodal.recall.completeNotFound,
    );
  });

  it("passes arguments to function translations", () => {
    expect(translateNow("multimodal.recall.started", 3)).toBe(
      TRANSLATIONS.en.multimodal.recall.started(3),
    );
  });

  // Uses zh, not one of the aliasing locales: most locale files declare
  // `multimodal: en.multimodal`, so mutating theirs would mutate en's own
  // object and there would be nothing left to fall back to.
  it("falls back to English when the active locale cannot resolve a key", () => {
    const recall = TRANSLATIONS.zh.multimodal.recall as {
      completeNotFound?: string;
    };
    const original = recall.completeNotFound;

    try {
      recall.completeNotFound = undefined;
      setRuntimeI18nLocale("zh");

      expect(translateNow("multimodal.recall.completeNotFound")).toBe(
        TRANSLATIONS.en.multimodal.recall.completeNotFound,
      );
    } finally {
      recall.completeNotFound = original;
    }
  });

  it("returns the key when no locale can resolve a path", () => {
    setRuntimeI18nLocale("zh");

    expect(translateNow("missing.path")).toBe("missing.path");
  });
});
