import { describe, expect, it } from "vitest";

import { en } from "./en";
import { zh } from "./zh";

// This project ships exactly two locales, en and zh, and both are complete —
// so every key in Translations is required and tsc already rejects a key added
// to en but forgotten in zh. (That was not true while 14 half-finished upstream
// locales were in the tree: each new key had to be `optional` to keep them
// compiling, and an optional key is one tsc lets zh silently omit. That is how
// `profiles.manageSkills` / `profiles.activeSetHint` went missing and rendered
// English in the Chinese UI.)
//
// These assertions cover what the type system still cannot see:
//   - a key present in zh but not en (dead weight nothing can render)
//   - a value whose *kind* drifted (string where en has a function → `t.x.y(3)`
//     throws; function where en has a string → renders nothing)
//   - `{placeholder}` tokens renamed or dropped in translation, which either
//     prints a literal "{count}" or silently loses the interpolated value
// The plain "zh defines every en key" check is kept as a belt-and-braces guard
// in case someone reintroduces optional keys.

type Node = Record<string, unknown>;

function isRecord(value: unknown): value is Node {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Every leaf path in `node`, dot-joined. */
function leafPaths(node: Node, prefix = ""): string[] {
  return Object.entries(node).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return isRecord(value) ? leafPaths(value, path) : [path];
  });
}

function at(root: Node, path: string): unknown {
  return path
    .split(".")
    .reduce<unknown>(
      (cur, part) => (isRecord(cur) ? cur[part] : undefined),
      root,
    );
}

const enPaths = leafPaths(en as unknown as Node);

describe("en/zh catalog parity", () => {
  it("finds a catalog to check", () => {
    // Guards against the walker silently returning [] and the suite passing
    // vacuously.
    expect(enPaths.length).toBeGreaterThan(1000);
  });

  it("zh defines every key that en defines", () => {
    const missing = enPaths.filter(
      (path) => at(zh as unknown as Node, path) === undefined,
    );

    expect(missing).toEqual([]);
  });

  it("zh does not define keys that en lacks", () => {
    // A key only in zh is dead weight: nothing can render it, because the
    // Translations type is written against en's shape.
    const zhPaths = leafPaths(zh as unknown as Node);
    const extra = zhPaths.filter(
      (path) => at(en as unknown as Node, path) === undefined,
    );

    expect(extra).toEqual([]);
  });

  it("zh matches en's value kind for every key", () => {
    // A string where en has a function (or vice versa) breaks the call site:
    // `t.x.y(3)` on a plain string throws, and rendering a function as a React
    // child renders nothing.
    const mismatched = enPaths
      .map((path) => ({
        path,
        en: typeof at(en as unknown as Node, path),
        zh: typeof at(zh as unknown as Node, path),
      }))
      .filter(({ en: a, zh: b }) => b !== undefined && a !== b)
      .map(({ path, en: a, zh: b }) => `${path}: en=${a} zh=${b}`);

    expect(mismatched).toEqual([]);
  });

  it("zh keeps the same {placeholder} tokens as en", () => {
    // Call sites interpolate with `.replace("{count}", …)`. A translated string
    // that renamed or dropped the token renders the literal "{count}" — or, for
    // a dropped token, silently loses the number.
    const tokens = (value: unknown) =>
      typeof value === "string"
        ? [...value.matchAll(/\{[a-zA-Z_][a-zA-Z0-9_]*\}/g)]
            .map((m) => m[0])
            .sort()
        : [];

    const drifted = enPaths
      .map((path) => ({
        path,
        want: tokens(at(en as unknown as Node, path)),
        got: tokens(at(zh as unknown as Node, path)),
      }))
      // `{s}` is an English-only plural hook; zh has no plural inflection and
      // correctly omits it.
      .map(({ path, want, got }) => ({
        path,
        want: want.filter((token) => token !== "{s}"),
        got: got.filter((token) => token !== "{s}"),
      }))
      .filter(
        ({ want, got }) => want.join(",") !== got.join(","),
      )
      .map(({ path, want, got }) => `${path}: en=[${want}] zh=[${got}]`);

    expect(drifted).toEqual([]);
  });
});
