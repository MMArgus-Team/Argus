// @vitest-environment jsdom
/**
 * DeepWindow interaction tests — regression guard for the report: after a
 * deep-research run completes ("深度调研完成后"), the segment cards inside
 * the right-side window (第1段 / 第2段 …) must still expand on click.
 *
 * The window renders older segments folded (defaultOpen=false) and the last
 * segment open (defaultOpen=true). A completed run must not fight a user
 * click: toggling a card header must stick, and the window header must still
 * collapse/expand the whole panel.
 */
import { createElement, type ReactNode } from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { I18nProvider } from "../i18n/context";
import { setRuntimeI18nLocale } from "../i18n/runtime";
import {
  DeepWindow,
  type BgItem,
  type BgSegment,
} from "./MultimodalChatPage";

// jsdom: allow act() (the default vitest node env does not set this).
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  setRuntimeI18nLocale("zh");
  // I18nProvider reads localStorage first, then navigator; pin zh so the
  // segment labels render as "第N段" regardless of jsdom defaults.
  localStorage.setItem("argus-locale", "zh");
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function segment(seg: number, answer: string): BgSegment {
  return { seg, lookups: [], ready: true, answer };
}

function completedItem(segs: BgSegment[]): BgItem {
  return {
    id: "rid1",
    requestId: "rid1",
    segments: segs,
    done: true,
    finalReport: "# 最终报告\n完成内容",
    waiting: null,
  };
}

function renderWindow(item: BgItem, expanded: boolean, onToggle: (rid: string) => void) {
  act(() => {
    root.render(
      createElement(I18nProvider, null,
        createElement(DeepWindow, {
          rid: "rid1", item, msgs: [], model: "test", expanded, onToggle,
        }) as ReactNode,
      ),
    );
  });
}

function cardHeader(seg: number): HTMLButtonElement {
  const btns = Array.from(container.querySelectorAll("button"));
  const b = btns.find((x) => new RegExp(`第\\s*${seg}\\s*段`).test(x.textContent || ""));
  if (!b) throw new Error(`segment card header for 第${seg}段 not found`);
  return b;
}

function windowHeader(): HTMLButtonElement {
  const btns = Array.from(container.querySelectorAll("button"));
  const b = btns.find((x) => x.textContent?.includes("深度分析"));
  if (!b) throw new Error("window header not found");
  return b;
}

function click(el: HTMLElement) {
  act(() => el.dispatchEvent(new MouseEvent("click", { bubbles: true })));
}

describe("DeepWindow after completion", () => {
  it("opens a folded older segment on click", () => {
    const item = completedItem([segment(1, "第一段内容"), segment(2, "第二段内容")]);
    renderWindow(item, true, () => {});
    // Last segment (2) open by default; older (1) folded → its answer hidden.
    expect(container.textContent).toContain("第二段内容");
    expect(container.textContent).not.toContain("第一段内容");

    click(cardHeader(1));

    expect(container.textContent).toContain("第一段内容");
    expect(cardHeader(1).textContent).toContain("▾");
  });

  it("keeps the segment open across a parent re-render (completion flush)", () => {
    const item = completedItem([segment(1, "第一段内容"), segment(2, "第二段内容")]);
    renderWindow(item, true, () => {});
    click(cardHeader(1));
    expect(container.textContent).toContain("第一段内容");

    // Simulate a trailing bg flush: same done item, new object identity.
    act(() => {
      root.render(
        createElement(I18nProvider, null,
          createElement(DeepWindow, {
            rid: "rid1", item: { ...item }, msgs: [], model: "test",
            expanded: true, onToggle: () => {},
          }) as ReactNode,
        ),
      );
    });

    expect(container.textContent).toContain("第一段内容");
    click(cardHeader(1));
    expect(container.textContent).not.toContain("第一段内容");
  });

  it("window header still collapses and re-expands the panel", () => {
    const item = completedItem([segment(1, "第一段内容")]);
    // Stateful parent harness: onToggle flips `expanded` like DeepColumn does.
    let expanded = true;
    const onToggle = () => { expanded = !expanded; };
    renderWindow(item, expanded, onToggle);

    click(windowHeader());
    expect(expanded).toBe(false);
    // Parent collapsed → re-render with expanded=false hides the body.
    act(() => {
      root.render(
        createElement(I18nProvider, null,
          createElement(DeepWindow, {
            rid: "rid1", item, msgs: [], model: "test",
            expanded, onToggle: () => { expanded = !expanded; },
          }) as ReactNode,
        ),
      );
    });
    expect(container.textContent).not.toContain("第一段内容");
    expect(windowHeader().textContent).toContain("▸");
  });

  it("long completed runs: every folded segment opens independently", () => {
    const segs = Array.from({ length: 6 }, (_, i) => segment(i + 1, `内容${i + 1}`));
    const item = completedItem(segs);
    renderWindow(item, true, () => {});
    // Older folded segments: answers hidden.
    for (let i = 1; i <= 5; i++) expect(container.textContent).not.toContain(`内容${i}`);
    // Last open.
    expect(container.textContent).toContain("内容6");

    click(cardHeader(3));
    expect(container.textContent).toContain("内容3");
    click(cardHeader(5));
    expect(container.textContent).toContain("内容5");
  });
});
