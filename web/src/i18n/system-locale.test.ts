import { describe, expect, it } from "vitest";

import { HTML_LANG, LOCALE_META, resolveSystemLocale } from "./context";
import { TRANSLATIONS } from "./catalog";
import type { Locale } from "./types";

// First-run behaviour: with no stored choice the provider follows the system
// language (see getInitialLocale), and falls back to English for languages we
// don't ship. resolveSystemLocale is the pure core of that decision.

describe("resolveSystemLocale", () => {
  it("matches Chinese system tags to Simplified Chinese", () => {
    expect(resolveSystemLocale(["zh"])).toBe("zh");
    expect(resolveSystemLocale(["zh-CN"])).toBe("zh");
    expect(resolveSystemLocale(["zh-Hans"])).toBe("zh");
    expect(resolveSystemLocale(["zh-Hans-CN"])).toBe("zh");
    expect(resolveSystemLocale(["zh-SG"])).toBe("zh");
  });

  it("routes Traditional Chinese regions to Simplified too", () => {
    // We no longer ship a Traditional catalog; Simplified is much closer for
    // those readers than falling through to English.
    expect(resolveSystemLocale(["zh-TW"])).toBe("zh");
    expect(resolveSystemLocale(["zh-HK"])).toBe("zh");
    expect(resolveSystemLocale(["zh-MO"])).toBe("zh");
    expect(resolveSystemLocale(["zh-Hant"])).toBe("zh");
    expect(resolveSystemLocale(["zh-Hant-TW"])).toBe("zh");
  });

  it("matches English system tags", () => {
    expect(resolveSystemLocale(["en"])).toBe("en");
    expect(resolveSystemLocale(["en-US"])).toBe("en");
    expect(resolveSystemLocale(["en-GB"])).toBe("en");
  });

  it("returns null for a language we do not ship", () => {
    // Caller turns null into English; returning "en" here would hide the
    // difference between "no match" and "the user asked for English".
    expect(resolveSystemLocale(["qps-ploc"])).toBeNull();
    expect(resolveSystemLocale([""])).toBeNull();
    expect(resolveSystemLocale([])).toBeNull();
    // Locales upstream shipped but we removed must NOT resolve — a Japanese
    // system should get English, not a half-translated Japanese UI.
    expect(resolveSystemLocale(["ja-JP"])).toBeNull();
    expect(resolveSystemLocale(["de-DE"])).toBeNull();
    expect(resolveSystemLocale(["pt-BR"])).toBeNull();
  });

  it("prefers the first system language we actually ship", () => {
    expect(resolveSystemLocale(["qps-ploc", "zh-CN", "en-US"])).toBe("zh");
    expect(resolveSystemLocale(["qps-ploc", "en-US"])).toBe("en");
    // A removed locale must not shadow a shipped one later in the list.
    expect(resolveSystemLocale(["ja-JP", "zh-CN"])).toBe("zh");
  });
});

describe("HTML_LANG", () => {
  it("covers every shipped locale", () => {
    const shipped = Object.keys(TRANSLATIONS) as Locale[];
    const missing = shipped.filter((locale) => !HTML_LANG[locale]);

    expect(missing).toEqual([]);
    // Keep it in step with the picker as well.
    expect(Object.keys(HTML_LANG).sort()).toEqual(
      Object.keys(LOCALE_META).sort(),
    );
  });

  it("maps Chinese to an explicit script subtag", () => {
    // Bare "zh" is ambiguous to a screen reader and the CJK line-breaker.
    expect(HTML_LANG.zh).toBe("zh-Hans");
    expect(HTML_LANG.en).toBe("en");
  });
});
