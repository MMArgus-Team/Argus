import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { TRANSLATIONS } from "./catalog";
import type { Locale } from "./types";

// `translateNow("a.b.c")` takes a STRING path, so TypeScript cannot check it.
// A typo, or a key that only exists in en, renders the raw key text ("multimodal
// .errors.camraFailed") straight into the UI — silently, and only in the locale
// that is missing it. tsc stays green the whole time. This test closes that gap
// by resolving every key literal in src/ against every shipped locale.
//
// Note this only covers the string-path API. `t.multimodal.x.y` member access is
// already checked by tsc against the Translations interface.

const SRC = new URL("..", import.meta.url).pathname;
const CODE = /\.tsx?$/;
const KEY_CALL = /translateNow\(\s*["'`]([^"'`]+)["'`]/g;

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) {
      walk(p, out);
    } else if (CODE.test(entry) && !entry.includes(".test.")) {
      out.push(p);
    }
  }
  return out;
}

function resolve(catalog: unknown, key: string): unknown {
  return key.split(".").reduce<unknown>(
    (cur, part) => (cur && typeof cur === "object"
      ? (cur as Record<string, unknown>)[part]
      : undefined),
    catalog,
  );
}

const referenced = new Map<string, string>(); // key -> first file that uses it
for (const file of walk(SRC)) {
  for (const [, key] of readFileSync(file, "utf8").matchAll(KEY_CALL)) {
    if (!referenced.has(key)) referenced.set(key, file.slice(SRC.length));
  }
}

describe("i18n key integrity", () => {
  it("finds translateNow call sites to check", () => {
    // Guards against the regex silently breaking and the suite passing vacuously.
    expect(referenced.size).toBeGreaterThan(50);
  });

  for (const locale of Object.keys(TRANSLATIONS) as Locale[]) {
    it(`every translateNow key resolves in ${locale}`, () => {
      const broken = [...referenced].filter(([key]) => {
        const value = resolve(TRANSLATIONS[locale], key);
        return typeof value !== "string" && typeof value !== "function";
      }).map(([key, file]) => `${key} (${file})`);

      expect(broken).toEqual([]);
    });
  }
});
