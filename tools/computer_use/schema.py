"""Schema for the generic `computer_use` tool.

Model-agnostic. Any tool-calling model can drive this. Vision-capable models
should prefer `capture(mode='som')` then `click(element=N)` — much more
reliable than pixel coordinates. Pixel coordinates remain supported for
models that were trained on them (e.g. Claude's computer-use RL).
"""

from __future__ import annotations

from typing import Any, Dict


# One consolidated tool with an `action` discriminator. Keeps the schema
# compact and the per-turn token cost low.
COMPUTER_USE_SCHEMA: Dict[str, Any] = {
    "name": "computer_use",
    "description": (
        "Drive the desktop in the background via cua-driver — screenshots, "
        "mouse, keyboard, scroll, drag — without stealing the user's cursor "
        "or keyboard focus. Supported on macOS, Windows, and Linux. "
        "Preferred workflow: call with "
        "action='capture' (mode='som' gives numbered element overlays), "
        "then click by `element` index for reliability. Pixel coordinates "
        "are supported for models trained on them. Works on any window — "
        "hidden, minimized, or behind another app. Requires cua-driver to "
        "be installed.\n"
        "★ In a multimodal video session, do not use computer_use merely to "
        "answer an ordinary visual question about the current or past live screen; "
        "use query_multimodal. If the user explicitly asks only for a raw/latest "
        "capture or to show/inspect the current live frame (no interaction), use "
        "get_current_frame. Reserve computer_use for desktop INTERACTION (click, "
        "type, scroll, drag, launch_app), including captures needed to carry out "
        "that interaction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "capture",
                    "click",
                    "double_click",
                    "right_click",
                    "middle_click",
                    "drag",
                    "scroll",
                    "type",
                    "key",
                    "set_value",
                    "wait",
                    "list_apps",
                    "focus_app",
                    "launch_app",
                ],
                "description": (
                    "Which action to perform. `capture` is free (no side "
                    "effects). All other actions require approval unless "
                    "auto-approved. Use `set_value` for select/popup elements "
                    "and sliders — it selects the matching option directly "
                    "without opening the native menu (no focus steal). Use "
                    "`launch_app` to OPEN / START an application that is not "
                    "yet running (e.g. 'open Tencent Meeting'): it launches in the "
                    "background without stealing focus and returns the new "
                    "pid + window. This is the RIGHT way to open an app — do "
                    "NOT try to press Win/Win+R or click the taskbar, which "
                    "need a focused window and fail on a bare desktop capture."
                ),
            },
            # ── capture ────────────────────────────────────────────
            "mode": {
                "type": "string",
                "enum": ["som", "vision", "ax"],
                "description": (
                    "Capture mode. `som` (default) is a screenshot with "
                    "numbered overlays on every interactable element — best "
                    "for vision models, lets you click by element index (only "
                    "when an `app` is targeted; a whole-screen capture has no "
                    "AX overlay). `vision` is a plain screenshot. `ax` is the "
                    "accessibility tree only (no image; text-only models). "
                    "NOTE: element indices only exist for app-targeted "
                    "captures; a full-screen capture (no `app`) is vision-only "
                    "— act on it with x/y screen coordinates."
                ),
            },
            "app": {
                "type": "string",
                "description": (
                    "Optional. Target a SPECIFIC app window (by name, e.g. "
                    "'Safari', or bundle ID 'com.apple.Safari') to get its "
                    "accessibility tree + numbered element overlay, then click "
                    "by element index. Works even on hidden/minimized/"
                    "background windows. IF OMITTED (or app='screen'/"
                    "'desktop'), captures the ENTIRE display — a true "
                    "full-screen screenshot — and subsequent click(x,y)/"
                    "scroll(x,y) use TRUE SCREEN pixel coordinates (no element "
                    "indices). Use the full-screen capture to see the whole "
                    "desktop / find where things are, then either act via x/y "
                    "screen coords or re-capture with app='<AppName>' for "
                    "precise element-index clicks. (A single image can't span "
                    "multiple monitors — capture one display at a time.)"
                ),
            },
            "max_elements": {
                "type": "integer",
                "description": (
                    "Optional cap on the AX `elements` array returned by "
                    "`action='capture'`. Default 100, hard maximum 1000. "
                    "Dense UIs (Electron apps such as Obsidian or VS Code, "
                    "JetBrains IDEs) can publish 500+ AX nodes — capping "
                    "prevents a single capture from blowing session "
                    "context. When the cap trims the response, "
                    "`total_elements` and `truncated_elements` are "
                    "surfaced in the result so you can re-call with "
                    "`app=` to narrow scope or raise `max_elements` when "
                    "the full tree is required. Has no effect on "
                    "`mode='som'` / `mode='vision'` when a screenshot is "
                    "included in the response; only the rare image-"
                    "missing fallback returns an `elements` array and is "
                    "subject to the cap."
                ),
                "default": 100,
                "minimum": 1,
                "maximum": 1000,
            },
            # ── click / drag / scroll targeting ────────────────────
            "element": {
                "type": "integer",
                "description": (
                    "The 1-based SOM index returned by the last "
                    "`capture(mode='som')` call. Strongly preferred over "
                    "raw coordinates."
                ),
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Pixel coordinates [x, y] in logical screen space (as "
                    "returned by capture width/height). Only use this if "
                    "no element index is available."
                ),
            },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button. Defaults to left.",
            },
            "modifiers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "cmd", "shift", "option", "alt", "ctrl", "fn",
                        "win", "windows", "super", "meta",
                    ],
                },
                "description": "Modifier keys held during the action.",
            },
            # ── drag ───────────────────────────────────────────────
            "from_element": {"type": "integer",
                              "description": "Source element index (drag)."},
            "to_element": {"type": "integer",
                            "description": "Target element index (drag)."},
            "from_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
                "description": "Source [x,y] (drag; use when no element available).",
            },
            "to_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
                "description": "Target [x,y] (drag; use when no element available).",
            },
            # ── scroll ─────────────────────────────────────────────
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction.",
            },
            "amount": {
                "type": "integer",
                "description": "Scroll wheel ticks. Default 3.",
            },
            # ── set_value ──────────────────────────────────────────
            "value": {
                "type": "string",
                "description": (
                    "For action='set_value': the value to set on the element. "
                    "For AXPopUpButton / select dropdowns, pass the option's "
                    "display label (e.g. 'Blue'). For sliders and other "
                    "AXValue-settable elements, pass the numeric or string value."
                ),
            },
            # ── type / key / wait ──────────────────────────────────
            "text": {
                "type": "string",
                "description": "Text to type (respects the current layout).",
            },
            "keys": {
                "type": "string",
                "description": (
                    "Key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return', "
                    "'escape', 'tab'. Use '+' to combine."
                ),
            },
            "seconds": {
                "type": "number",
                "description": "Seconds to wait. Max 30.",
            },
            # ── focus_app ──────────────────────────────────────────
            "raise_window": {
                "type": "boolean",
                "description": (
                    "Only for action='focus_app'. Default TRUE — brings the "
                    "window to the front so a subsequent capture/click can "
                    "actually see it. Set FALSE only for background input "
                    "routing where you already know the window is visible "
                    "and you deliberately don't want to disturb the user's "
                    "z-order. Note: if the window is on another Space / "
                    "minimized, the raise may fail on macOS — the focus_app "
                    "result says so; do NOT loop-retry, ask the user to move "
                    "the window to the current Space instead."
                ),
            },
            # ── launch_app ─────────────────────────────────────────
            "path": {
                "type": "string",
                "description": (
                    "For action='launch_app': full path to the executable "
                    "(Windows, e.g. 'C:\\\\Program Files\\\\Tencent\\\\WeMeet\\\\"
                    "wemeetapp.exe'). Most reliable launch method on Windows. "
                    "If you don't know the path, first try `name` (display "
                    "name); the driver resolves it via the Start-menu / "
                    "AppsFolder index. Launches in the background without "
                    "stealing focus."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "For action='launch_app': app display name to launch "
                    "(e.g. 'Tencent Meeting', 'wemeetapp', 'Notepad'). Resolved via "
                    "the Start-menu / shell:AppsFolder index, falling back to "
                    "a PATH search. Prefer `path` when known. (Distinct from "
                    "`app`, which TARGETS an already-running window for "
                    "capture/focus_app.)"
                ),
            },
            # ── return shape ───────────────────────────────────────
            "capture_after": {
                "type": "boolean",
                "description": (
                    "If true, take a follow-up capture after the action "
                    "and include it in the response. Saves a round-trip "
                    "when you need to verify an action's effect."
                ),
            },
        },
        "required": ["action"],
    },
}


def get_computer_use_schema() -> Dict[str, Any]:
    """Return the generic OpenAI function-calling schema."""
    return COMPUTER_USE_SCHEMA
