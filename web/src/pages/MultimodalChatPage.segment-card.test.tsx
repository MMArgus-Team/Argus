// @vitest-environment jsdom
/**
 * Segment-card expand/collapse interaction tests.
 *
 * Regression guard: after a deep-research run completes, the user must still
 * be able to click a segment header (第N段) to expand/collapse it. The card
 * owns its `open` state locally; the `defaultOpen` effect must only seed the
 * initial state and re-apply when the DEFAULT flips (e.g. a segment stops
 * being the "last/current" one) — it must never fight a user click.
 */
import { createElement } from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { SegmentCard, type BgSegment } from "./MultimodalChatPage";

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function segment(overrides: Partial<BgSegment> = {}): BgSegment {
  return { seg: 1, lookups: [], ...overrides };
}

function renderCard(s: BgSegment, defaultOpen = false, terminal = false) {
  act(() => {
    root.render(
      createElement(SegmentCard, { s, defaultOpen, terminal }),
    );
  });
}

function header() {
  // The header is the only <button> in the card.
  return container.querySelector("button") as HTMLButtonElement;
}

function click(el: HTMLElement) {
  act(() => el.dispatchEvent(new MouseEvent("click", { bubbles: true })));
}

describe("SegmentCard expand/collapse", () => {
  it("opens a collapsed (older, completed) segment on click", () => {
    renderCard(segment({ ready: true, answer: "findings text" }), false, true);
    // Collapsed: the answer body is NOT rendered.
    expect(container.textContent).not.toContain("findings text");

    click(header());

    // Expanded: answer visible, header flips ▸ → ▾.
    expect(container.textContent).toContain("findings text");
    expect(header().textContent).toContain("▾");
  });

  it("re-collapses on a second click", () => {
    renderCard(segment({ ready: true, answer: "findings text" }), true, true);
    expect(container.textContent).toContain("findings text");

    click(header());

    expect(container.textContent).not.toContain("findings text");
    expect(header().textContent).toContain("▸");
  });

  it("a user click survives a parent re-render with an unchanged defaultOpen", () => {
    const s = segment({ ready: true, answer: "findings text" });
    renderCard(s, false, true);
    click(header());
    expect(container.textContent).toContain("findings text");

    // Simulate the completion transition re-render (terminal prop flips,
    // defaultOpen stays false): the open state must NOT be reset.
    act(() => {
      root.render(createElement(SegmentCard, { s, defaultOpen: false, terminal: true }));
    });
    expect(container.textContent).toContain("findings text");

    // And it still closes on click afterwards.
    click(header());
    expect(container.textContent).not.toContain("findings text");
  });

  it("still opens when the card is the terminal (done) segment", () => {
    renderCard(segment({ ready: true, answer: "final seg text" }), false, true);
    click(header());
    expect(container.textContent).toContain("final seg text");
  });
});
