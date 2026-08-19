"""Entry point for the `computer_use` tool.

Universal (any-model) desktop control across macOS, Windows, and Linux via
cua-driver's background computer-use primitive. Replaces #4562's
Anthropic-native `computer_20251124` approach — the schema here is standard
OpenAI function-calling so every tool-capable model can drive it.

Linux is the most recent runtime (X11 + Wayland, via cua-driver-rs's
AT-SPI tree path); it is enabled here alongside macOS and Windows. When a
host's display server or accessibility stack isn't reachable, cua-driver's
`health_report` (surfaced by `hermes computer-use doctor`) reports the
exact blocked check rather than the toolset silently failing.

Return contract
---------------
For text-only results (wait, key, list_apps, focus_app, failures, etc.):
  JSON string.

For captures / actions with `capture_after=True`:
  A dict wrapped as the OpenAI-style multi-part tool-message content:

      {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "<human-readable summary + SOM index>"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,<b64>"}},
        ],
        "text_summary": "<text used for fallback string content>",
      }

  run_agent.py's tool-message builder inspects `_multimodal` and emits a
  list-shaped `content` for OpenAI-compatible providers. The Anthropic
  adapter splices the base64 image into a `tool_result` block (see
  `agent/anthropic_adapter.py`). Every provider that supports multi-part
  tool content gets the image; text-only providers see the summary only.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import struct
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approval & safety
# ---------------------------------------------------------------------------

_approval_callback = None


def set_approval_callback(cb) -> None:
    """Register a callback for computer_use approval prompts (used by CLI).

    Matches the terminal_tool._approval_callback pattern. The callback
    receives (action, args, summary) and returns one of:
      "approve_once" | "approve_session" | "always_approve" | "deny".
    """
    global _approval_callback
    _approval_callback = cb


# Actions that read, not mutate. Always allowed.
_SAFE_ACTIONS = frozenset({"capture", "wait", "list_apps"})

# Actions that mutate user-visible state. Go through approval.
_DESTRUCTIVE_ACTIONS = frozenset({
    "click", "double_click", "right_click", "middle_click",
    "drag", "scroll", "type", "key", "set_value", "focus_app",
    "launch_app",
})

# Hard-blocked key combinations. Mirrored from #4562 — these are destructive
# regardless of approval level (e.g. logout kills the session Hermes runs in).
_BLOCKED_KEY_COMBOS = {
    frozenset({"cmd", "shift", "backspace"}),   # empty trash
    frozenset({"cmd", "option", "backspace"}),   # force delete
    frozenset({"cmd", "ctrl", "q"}),             # lock screen
    frozenset({"cmd", "shift", "q"}),            # log out
    frozenset({"cmd", "option", "shift", "q"}),  # force log out
    # Windows secure/session shortcuts. The Windows driver accepts Win-key
    # combos, and Alt is canonicalized to option below, so block the
    # destructive variants before any backend sees them.
    frozenset({"win", "l"}),
    frozenset({"ctrl", "option", "delete"}),
    frozenset({"ctrl", "option", "del"}),
    frozenset({"option", "f4"}),
}

_KEY_ALIASES = {
    "command": "cmd", "control": "ctrl", "alt": "option", "⌘": "cmd", "⌥": "option",
    "windows": "win", "super": "win", "meta": "win",
}


def _canon_key_combo(keys: str) -> frozenset:
    parts = [p.strip().lower() for p in re.split(r"\s*\+\s*", keys) if p.strip()]
    parts = [_KEY_ALIASES.get(p, p) for p in parts]
    return frozenset(parts)


# Dangerous text patterns for the `type` action. Same list as #4562.
_BLOCKED_TYPE_PATTERNS = [
    re.compile(r"curl\s+[^|]*\|\s*bash", re.IGNORECASE),
    re.compile(r"curl\s+[^|]*\|\s*sh", re.IGNORECASE),
    re.compile(r"wget\s+[^|]*\|\s*bash", re.IGNORECASE),
    re.compile(r"\bsudo\s+rm\s+-[rf]", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}", re.IGNORECASE),  # fork bomb
]


def _is_blocked_type(text: str) -> Optional[str]:
    for pat in _BLOCKED_TYPE_PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None


# ---------------------------------------------------------------------------
# Backend selection — env-swappable for tests
# ---------------------------------------------------------------------------

# Per-process cached backend; lazily instantiated on first call.
_backend_lock = threading.Lock()
_backend: Optional[ComputerUseBackend] = None
# Session-scoped approval state.
_session_auto_approve = False
_always_allow: set = set()  # action names the user unlocked for the session

# --------------------------------------------------------------------------- #
# No-progress loop guard
# --------------------------------------------------------------------------- #
# computer_use can deadlock into "click something -> re-screenshot -> click the
# same something -> ..." when a click lands nowhere (wrong coords, dead pixel,
# frontmost-window mismatch). The model sees a byte-identical screenshot every
# round, so it keeps issuing the same action forever. Guard: fingerprint each
# screenshot; when an ACTION that should change the screen (click/type/scroll/
# ...) is followed by an unchanged screenshot _NO_PROGRESS_LIMIT times in a row,
# surface an explicit no-progress warning so the model stops and re-strategises
# (or tells the user) instead of spinning.
_NO_PROGRESS_LIMIT = 3
# Actions that are EXPECTED to mutate the screen. A repeated capture/wait with
# no change is legitimate (nothing was asked to change), so those don't count.
_SCREEN_MUTATING_ACTIONS = {
    "click", "double_click", "right_click", "middle_click",
    "drag", "scroll", "type", "key", "set_value", "focus_app",
}
_last_shot_fp: Optional[str] = None
_no_progress_count = 0
# Fingerprint of the screenshot produced by the most recent _capture_response
# call, read back by the main entrypoint after _dispatch returns.
_pending_shot_fp: Optional[str] = None

# --------------------------------------------------------------------------- #
# Coordinate rescale
# --------------------------------------------------------------------------- #
# The screenshot handed to the main model is downscaled (longest side capped at
# _MAX_MAIN_MODEL_DIM) to save tokens. The model therefore reports click
# coordinates measured on the SMALLER image it sees. cua-driver expects REAL
# screen pixels. Without correction a click on a 1920px screen shown at 1024px
# lands at ~0.53x the intended position — systematically missing every target
# outside the top-left, which is the root of the open-app loop. We record the
# most recent downscale factor here and multiply raw model coordinates by it
# before dispatch. 1.0 means the image was sent at full size (no correction).
_last_coord_scale: float = 1.0


def _screenshot_fingerprint(png_b64: Optional[str]) -> Optional[str]:
    """Cheap stable fingerprint of a screenshot for no-progress detection.

    Hash of the encoded base64 (byte-identical image => identical fingerprint).
    Returns None when there is no image (AX-only captures), so those turns
    neither set nor break the streak.
    """
    if not png_b64:
        return None
    import hashlib
    return hashlib.sha1(png_b64.encode("ascii", "ignore")).hexdigest()


def reset_progress_guard() -> None:
    """Clear the no-progress streak (call at the start of a fresh user turn)."""
    global _last_shot_fp, _no_progress_count
    _last_shot_fp = None
    _no_progress_count = 0


def _get_backend() -> ComputerUseBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            backend_name = os.environ.get("ARGUS_COMPUTER_USE_BACKEND", "cua").lower()
            if backend_name in {"cua", "cua-driver", ""}:
                from tools.computer_use.cua_backend import CuaDriverBackend
                _backend = CuaDriverBackend()
            elif backend_name == "noop":  # pragma: no cover
                _backend = _NoopBackend()
            else:
                raise RuntimeError(f"Unknown ARGUS_COMPUTER_USE_BACKEND={backend_name!r}")
            try:
                _backend.start()
            except Exception:
                # Don't cache a backend whose start() failed (e.g. a lazy
                # dependency install was declined / failed). The next call
                # retries cleanly instead of returning a half-initialised
                # backend.
                _backend = None
                raise
        return _backend


def reset_backend_for_tests() -> None:  # pragma: no cover
    """Test helper — tear down the cached backend."""
    global _backend, _session_auto_approve, _always_allow
    with _backend_lock:
        if _backend is not None:
            try:
                _backend.stop()
            except Exception:
                pass
        _backend = None
    _session_auto_approve = False
    _always_allow = set()


class _NoopBackend(ComputerUseBackend):  # pragma: no cover
    """Test/CI stub. Records calls; returns trivial results."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self._started = False

    def start(self) -> None: self._started = True
    def stop(self) -> None: self._started = False
    def is_available(self) -> bool: return True

    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        self.calls.append(("capture", {"mode": mode, "app": app}))
        return CaptureResult(mode=mode, width=1024, height=768, png_b64=None,
                             elements=[], app=app or "", window_title="")

    def click(self, **kw) -> ActionResult:
        self.calls.append(("click", kw))
        return ActionResult(ok=True, action="click")

    def drag(self, **kw) -> ActionResult:
        self.calls.append(("drag", kw))
        return ActionResult(ok=True, action="drag")

    def scroll(self, **kw) -> ActionResult:
        self.calls.append(("scroll", kw))
        return ActionResult(ok=True, action="scroll")

    def type_text(self, text: str) -> ActionResult:
        self.calls.append(("type", {"text": text}))
        return ActionResult(ok=True, action="type")

    def key(self, keys: str) -> ActionResult:
        self.calls.append(("key", {"keys": keys}))
        return ActionResult(ok=True, action="key")

    def list_apps(self) -> List[Dict[str, Any]]:
        self.calls.append(("list_apps", {}))
        return []

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        self.calls.append(("focus_app", {"app": app, "raise": raise_window}))
        return ActionResult(ok=True, action="focus_app")

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        self.calls.append(("set_value", {"value": value, "element": element}))
        return ActionResult(ok=True, action="set_value")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle_computer_use(args: Dict[str, Any], **kwargs) -> Any:
    """Main entry point — dispatched by tools.registry.

    Returns either a JSON string (text-only) or a dict marked `_multimodal`
    (image + summary) which run_agent.py wraps into the tool message.
    """
    action = (args.get("action") or "").strip().lower()
    if not action:
        return json.dumps({"error": "missing `action`"})

    # Safety: validate actions before approval prompt.
    if action == "type":
        text = args.get("text", "")
        pat = _is_blocked_type(text)
        if pat:
            return json.dumps({
                "error": f"blocked pattern in type text: {pat!r}",
                "hint": "Dangerous shell patterns cannot be typed via computer_use.",
            })

    if action == "key":
        keys = args.get("keys", "")
        combo = _canon_key_combo(keys)
        for blocked in _BLOCKED_KEY_COMBOS:
            if blocked.issubset(combo) and len(blocked) <= len(combo):
                return json.dumps({
                    "error": f"blocked key combo: {sorted(blocked)}",
                    "hint": "Destructive system shortcuts are hard-blocked.",
                })

    # Approval gate (destructive actions only).
    if action in _DESTRUCTIVE_ACTIONS:
        err = _request_approval(action, args)
        if err is not None:
            return err

    # Map model-space coordinates back to real screen pixels. The model reports
    # coordinates on the downscaled screenshot it sees; cua-driver wants real
    # pixels. Element-index actions are unaffected (the backend resolves element
    # geometry itself) — only raw coordinate fields are rescaled.
    _rescale_coordinates(args)

    # Dispatch to backend.
    try:
        backend = _get_backend()
    except Exception as e:
        return json.dumps({
            "error": f"computer_use backend unavailable: {e}",
            "hint": "If the cua-driver binary is missing, run `argus computer-use install`. "
                    "If a Python dependency is missing, the error above shows the exact install command.",
        })

    global _last_shot_fp, _no_progress_count, _pending_shot_fp
    _pending_shot_fp = None
    try:
        result = _dispatch(backend, action, args)
    except Exception as e:
        logger.exception("computer_use %s failed", action)
        return json.dumps({"error": f"{action} failed: {e}"})

    # No-progress loop guard: a screen-mutating action that leaves the
    # screenshot byte-identical to the previous one is a wasted move. Count
    # consecutive such moves; once the streak hits the limit, tell the model to
    # stop repeating and re-strategise.
    fp = _pending_shot_fp
    if fp is not None:
        if action in _SCREEN_MUTATING_ACTIONS and fp == _last_shot_fp:
            _no_progress_count += 1
        else:
            _no_progress_count = 0
        _last_shot_fp = fp
        if _no_progress_count >= _NO_PROGRESS_LIMIT:
            warn = (
                f"[no-progress guard] The screen has not changed after "
                f"{_no_progress_count} consecutive '{action}'-type actions — the "
                f"action is having NO effect (wrong target/coordinates, an "
                f"unfocused window, or a click landing on nothing). STOP "
                f"repeating the same action. Try a different approach: re-capture "
                f"with mode='som' or 'ax' and act via an element index instead of "
                f"raw coordinates, focus_app the target window first, or tell the "
                f"user this cannot be completed automatically. Do NOT issue the "
                f"same action again."
            )
            _no_progress_count = 0  # reset so one warning per streak, not spam
            result = _inject_progress_warning(result, warn)
    return result


def _inject_progress_warning(result: Any, warn: str) -> Any:
    """Attach the no-progress warning to a dispatch result of either shape.

    ``result`` is either the ``_multimodal`` dict envelope or a JSON string.
    """
    try:
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                content.insert(0, {"type": "text", "text": warn})
            if "text_summary" in result:
                result["text_summary"] = warn + "\n" + str(result["text_summary"])
            result["no_progress"] = True
            return result
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            parsed["no_progress_warning"] = warn
            parsed["no_progress"] = True
            return json.dumps(parsed)
    except Exception:
        pass
    return json.dumps({"no_progress_warning": warn, "no_progress": True,
                       "original": str(result)[:2000]})


def _request_approval(action: str, args: Dict[str, Any]) -> Optional[str]:
    """Return None if approved, or a JSON error string if denied."""
    global _session_auto_approve, _always_allow
    if _session_auto_approve:
        return None
    if action in _always_allow:
        return None
    cb = _approval_callback
    if cb is None:
        # No CLI approval wired — default allow. Gateway approval is handled
        # one layer out via the normal tool-approval infra.
        return None
    summary = _summarize_action(action, args)
    try:
        verdict = cb(action, args, summary)
    except Exception as e:
        logger.warning("approval callback failed: %s", e)
        verdict = "deny"
    if verdict == "approve_once":
        return None
    if verdict == "approve_session" or verdict == "always_approve":
        _always_allow.add(action)
        if verdict == "always_approve":
            _session_auto_approve = True
        return None
    return json.dumps({"error": "denied by user", "action": action})


def _rescale_one(coord: Any, scale: float) -> Any:
    """Multiply a ``[x, y]`` coordinate by ``scale`` (real-pixel mapping).

    Leaves the value untouched if it isn't a 2-element numeric sequence, so a
    malformed coordinate reaches the backend as-is (which returns its own error)
    rather than crashing here.
    """
    try:
        if (isinstance(coord, (list, tuple)) and len(coord) == 2
                and coord[0] is not None and coord[1] is not None):
            return [int(round(float(coord[0]) * scale)),
                    int(round(float(coord[1]) * scale))]
    except (TypeError, ValueError):
        pass
    return coord


def _rescale_coordinates(args: Dict[str, Any]) -> None:
    """In-place: map raw model coordinates in ``args`` back to real pixels.

    No-op when the last screenshot was sent at full size (scale 1.0). Only
    touches raw coordinate fields — element-index actions carry no coordinates
    and are resolved by the backend, so they need no correction.
    """
    scale = _last_coord_scale
    if not scale or scale == 1.0:
        return
    for key in ("coordinate", "from_coordinate", "to_coordinate"):
        if args.get(key) is not None:
            args[key] = _rescale_one(args[key], scale)


def _summarize_action(action: str, args: Dict[str, Any]) -> str:
    if action in {"click", "double_click", "right_click", "middle_click"}:
        if args.get("element") is not None:
            return f"{action} element #{args['element']}"
        coord = args.get("coordinate")
        if coord:
            return f"{action} at {tuple(coord)}"
        return action
    if action == "drag":
        src = args.get("from_element") or args.get("from_coordinate")
        dst = args.get("to_element") or args.get("to_coordinate")
        return f"drag {src} → {dst}"
    if action == "scroll":
        return f"scroll {args.get('direction', '?')} x{args.get('amount', 3)}"
    if action == "type":
        text = args.get("text", "")
        return f"type {text[:60]!r}" + ("..." if len(text) > 60 else "")
    if action == "key":
        return f"key {args.get('keys', '')!r}"
    if action == "focus_app":
        raise_arg = args.get("raise_window")
        # Default is now True; only annotate the non-default case explicitly.
        if raise_arg is False:
            suffix = " (no raise)"
        else:
            suffix = ""  # default (raise) — keep display quiet for the common case
        return f"focus {args.get('app', '')!r}" + suffix
    return action


def _dispatch(backend: ComputerUseBackend, action: str, args: Dict[str, Any]) -> Any:
    capture_after = bool(args.get("capture_after"))

    if action == "capture":
        mode = str(args.get("mode", "som"))
        if mode not in {"som", "vision", "ax"}:
            return json.dumps({"error": f"bad mode {mode!r}; use som|vision|ax"})
        cap = backend.capture(mode=mode, app=args.get("app"))
        return _capture_response(cap, max_elements=_coerce_max_elements(args.get("max_elements")))

    if action == "wait":
        seconds = float(args.get("seconds", 1.0))
        res = backend.wait(seconds)
        return _text_response(res)

    if action == "list_apps":
        apps = backend.list_apps()
        return json.dumps({"apps": apps, "count": len(apps)})

    if action == "focus_app":
        app = args.get("app")
        if not app:
            return json.dumps({"error": "focus_app requires `app`"})
        # Default raise_window=True: after a launch/switch, capture and click
        # need the window visible or they see a blank 0x0 frame. Only skip
        # the raise when the caller explicitly asks for background input.
        raise_arg = args.get("raise_window")
        raise_window = True if raise_arg is None else bool(raise_arg)
        res = backend.focus_app(app, raise_window=raise_window)
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "launch_app":
        # Open / start an application that is not yet running. The RIGHT way to
        # "打开腾讯会议" — launches in the background (no focus steal) and returns
        # the new pid. Prefer `path` (full exe path, most reliable on Windows);
        # fall back to `name` (resolved via Start-menu / AppsFolder index).
        path = args.get("path")
        name = args.get("name") or args.get("app")
        bundle_id = args.get("bundle_id")
        if not (path or name or bundle_id):
            return json.dumps({
                "error": "launch_app requires `path` (full exe path) or `name` "
                         "(app display name).",
            })
        try:
            launch_kwargs: Dict[str, Any] = {}
            if bundle_id:
                launch_kwargs["bundle_id"] = bundle_id
            if name:
                launch_kwargs["name"] = name
            # `path` is a Windows-only field on the backend's call; pass it
            # through the generic escape hatch when provided so ShellExecuteEx
            # gets the exact executable.
            if path:
                info = backend.call_tool("launch_app", {"path": path})
                if info.get("isError"):
                    return json.dumps({
                        "error": f"launch_app failed: {info.get('data')}",
                        "hint": "Check the exe path is correct and current "
                                "(version-numbered install dirs change on "
                                "update); or retry with `name` to resolve via "
                                "the Start-menu index.",
                    })
                sc = info.get("structuredContent") or {}
                return json.dumps({
                    "ok": True, "action": "launch_app",
                    "pid": sc.get("pid"), "name": sc.get("name"),
                    "message": info.get("data") if isinstance(info.get("data"), str) else "launched",
                })
            # name / bundle_id path via the typed backend method.
            try:
                info = backend.launch_app(**launch_kwargs)
                return json.dumps({
                    "ok": True, "action": "launch_app",
                    "pid": info.get("pid"), "name": info.get("name"),
                    "windows": len(info.get("windows", []) or []),
                })
            except Exception as name_err:
                # cua-driver's name resolution (shell:AppsFolder) misses many
                # third-party desktop apps (e.g. 腾讯会议). Fall back to the
                # Start-menu shortcut → real exe path, then launch by path.
                resolved = None
                if name:
                    try:
                        from tools.computer_use.cua_backend import (
                            resolve_exe_from_start_menu,
                        )
                        resolved = resolve_exe_from_start_menu(name)
                    except Exception:
                        resolved = None
                if resolved:
                    info = backend.call_tool("launch_app", {"path": resolved})
                    if not info.get("isError"):
                        sc = info.get("structuredContent") or {}
                        return json.dumps({
                            "ok": True, "action": "launch_app",
                            "pid": sc.get("pid"), "name": sc.get("name"),
                            "resolved_path": resolved,
                            "message": info.get("data") if isinstance(info.get("data"), str) else "launched",
                        })
                raise name_err
        except Exception as e:
            return json.dumps({
                "error": f"launch_app failed: {e}",
                "hint": "Retry with `path` (full exe path) if `name` didn't "
                        "resolve, or verify the app is installed.",
            })

    if action in {"click", "double_click", "right_click", "middle_click"}:
        button = args.get("button")
        click_count = 1
        if action == "double_click":
            click_count = 2
        elif action == "right_click":
            button = "right"
        elif action == "middle_click":
            button = "middle"
        else:
            button = button or "left"
        element = args.get("element")
        coord = args.get("coordinate") or (None, None)
        x, y = (coord[0], coord[1]) if coord and coord[0] is not None else (None, None)
        res = backend.click(
            element=element if element is not None else None,
            x=x, y=y, button=button or "left", click_count=click_count,
            modifiers=args.get("modifiers"),
        )
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "drag":
        has_elements = args.get("from_element") is not None and args.get("to_element") is not None
        has_coords = args.get("from_coordinate") and args.get("to_coordinate")
        if not has_elements and not has_coords:
            return json.dumps({
                "error": "drag requires from_coordinate/to_coordinate or from_element/to_element",
            })
        res = backend.drag(
            from_element=args.get("from_element"),
            to_element=args.get("to_element"),
            from_xy=tuple(args["from_coordinate"]) if args.get("from_coordinate") else None,
            to_xy=tuple(args["to_coordinate"]) if args.get("to_coordinate") else None,
            button=args.get("button", "left"),
            modifiers=args.get("modifiers"),
        )
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "scroll":
        coord = args.get("coordinate") or (None, None)
        res = backend.scroll(
            direction=args.get("direction", "down"),
            amount=int(args.get("amount", 3)),
            element=args.get("element"),
            x=coord[0] if coord and coord[0] is not None else None,
            y=coord[1] if coord and coord[1] is not None else None,
            modifiers=args.get("modifiers"),
        )
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "type":
        res = backend.type_text(args.get("text", ""))
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "key":
        res = backend.key(args.get("keys", ""))
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "set_value":
        value = args.get("value")
        if value is None:
            return json.dumps({"error": "set_value requires `value`"})
        res = backend.set_value(value=str(value), element=args.get("element"))
        return _maybe_follow_capture(backend, res, capture_after)

    return json.dumps({"error": f"unknown action {action!r}"})


# ---------------------------------------------------------------------------
# Response shaping
# ---------------------------------------------------------------------------

def _text_response(res: ActionResult) -> str:
    payload: Dict[str, Any] = {"ok": res.ok, "action": res.action}
    if res.message:
        payload["message"] = res.message
    if res.meta:
        payload["meta"] = res.meta
    return json.dumps(payload)


# Default cap for the AX `elements` array returned by capture. Dense UIs
# (Electron apps, Obsidian, JetBrains IDEs) can publish 500+ AX nodes, which
# can exhaust session context after a single capture. The model-facing
# `max_elements` argument lets callers raise this when they need the full tree.
_DEFAULT_MAX_ELEMENTS = 100
# Hard upper bound on caller-supplied `max_elements`. Without this, a tool
# call passing a very large integer would silently disable the safeguard and
# reintroduce the original unbounded behavior.
_MAX_ALLOWED_MAX_ELEMENTS = 1000
_MIN_PROVIDER_IMAGE_DIMENSION = 8


def _image_dimensions_from_b64(image_b64: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) for common inline screenshot formats.

    Some providers reject images below 8x8 before the model sees the tool
    result. Inspecting the encoded bytes here lets computer_use fall back to
    its AX/SOM text payload instead of sending an unusable placeholder.
    """
    if not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except Exception:
        return None

    # PNG: signature + IHDR width/height.
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        try:
            width, height = struct.unpack(">II", raw[16:24])
            return int(width), int(height)
        except Exception:
            return None

    # JPEG: scan for SOF markers that carry dimensions.
    if raw.startswith(b"\xff\xd8") and len(raw) > 4:
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            i += 2
            while marker == 0xFF and i < len(raw):
                marker = raw[i]
                i += 1
            if marker in {0xD8, 0xD9}:
                continue
            if marker == 0xDA:
                break
            if i + 2 > len(raw):
                break
            segment_len = int.from_bytes(raw[i:i + 2], "big")
            if segment_len < 2 or i + segment_len > len(raw):
                break
            if marker in {
                0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
            } and segment_len >= 7:
                height = int.from_bytes(raw[i + 3:i + 5], "big")
                width = int.from_bytes(raw[i + 5:i + 7], "big")
                return int(width), int(height)
            i += segment_len
    return None


def _coerce_max_elements(value: Any) -> int:
    """Validate the caller-supplied ``max_elements``.

    Falls back to :data:`_DEFAULT_MAX_ELEMENTS` for missing / non-integer /
    sub-1 inputs so the cap can never be silently disabled by a malformed
    tool-call argument. Clamps oversized values to
    :data:`_MAX_ALLOWED_MAX_ELEMENTS` so a caller cannot bypass the
    safeguard by passing a very large integer.
    """
    if value is None:
        return _DEFAULT_MAX_ELEMENTS
    try:
        n = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_ELEMENTS
    if n < 1:
        return _DEFAULT_MAX_ELEMENTS
    if n > _MAX_ALLOWED_MAX_ELEMENTS:
        return _MAX_ALLOWED_MAX_ELEMENTS
    return n


def _capture_response(cap: CaptureResult, max_elements: int = _DEFAULT_MAX_ELEMENTS) -> Any:
    # Record this screenshot's fingerprint for the no-progress loop guard in
    # the main entrypoint (None on AX-only captures, which don't carry an image).
    global _pending_shot_fp
    _pending_shot_fp = _screenshot_fingerprint(cap.png_b64)

    total_elements = len(cap.elements)
    visible_elements = cap.elements[:max_elements]
    truncated_elements = max(0, total_elements - len(visible_elements))
    image_dimensions = _image_dimensions_from_b64(cap.png_b64 or "") if cap.png_b64 else None
    response_width = image_dimensions[0] if image_dimensions else cap.width
    response_height = image_dimensions[1] if image_dimensions else cap.height
    image_too_small = bool(
        image_dimensions
        and (
            image_dimensions[0] < _MIN_PROVIDER_IMAGE_DIMENSION
            or image_dimensions[1] < _MIN_PROVIDER_IMAGE_DIMENSION
        )
    )

    # Index only what's actually surfaced in the response — otherwise the
    # human-readable summary references element indices the model cannot
    # find in the JSON `elements` array (e.g. max_elements=10 vs the default
    # 40-line index window).
    element_index = _format_elements(visible_elements)
    summary_lines = [
        f"capture mode={cap.mode} {response_width}x{response_height}"
        + (f" app={cap.app}" if cap.app else "")
        + (f" window={cap.window_title!r}" if cap.window_title else ""),
        f"{total_elements} interactable element(s):",
    ]
    if element_index:
        summary_lines.extend(element_index)
    # Multimodal and AX paths both reference `summary`; build it once up-front
    # so the aux-vision routing branch (which fires before either path is
    # selected) has a valid value to hand to _route_capture_through_aux_vision.
    # The AX path appends the "truncated to N of M" note to summary_lines
    # below and rebuilds; the multimodal path keeps this version untouched.
    if image_too_small:
        summary_lines.append(
            f"  (screenshot omitted: {image_dimensions[0]}x{image_dimensions[1]} "
            f"is below the {_MIN_PROVIDER_IMAGE_DIMENSION}x{_MIN_PROVIDER_IMAGE_DIMENSION} "
            "provider minimum)"
        )
    summary = "\n".join(summary_lines)

    if cap.png_b64 and cap.mode != "ax" and not image_too_small:
        # Decide whether to hand the screenshot to the auxiliary.vision
        # pipeline (text-only result) or keep the multimodal envelope (main
        # model handles vision natively). Issue #24015: previously the
        # multimodal envelope was returned unconditionally, so non-vision
        # main models tripped HTTP 404 / 400 at the provider boundary even
        # when auxiliary.vision was explicitly configured to handle this.
        if _should_route_through_aux_vision():
            routed = _route_capture_through_aux_vision(cap, summary)
            if routed is not None:
                return routed
            # Aux routing was requested but failed (vision node down, aux call
            # raised, empty analysis, etc.). Routing being requested means the
            # main model may not be able to consume images; falling through to
            # the multimodal envelope can break the capture with a provider
            # error. Degrade to the AX/SOM text payload instead so element
            # indices remain usable while vision is unavailable.
            summary_lines.append(
                "  (vision unavailable: the auxiliary vision model could not "
                "be reached; screenshot omitted. Element-index actions still "
                "work — drive via the element list above.)"
            )
            if truncated_elements:
                summary_lines.append(
                    f"  (response truncated to {len(visible_elements)} of "
                    f"{total_elements} elements; raise max_elements or pass "
                    "app= to narrow)"
                )
            payload = {
                "mode": cap.mode,
                "width": response_width,
                "height": response_height,
                "app": cap.app,
                "window_title": cap.window_title,
                "elements": [_element_to_dict(e) for e in visible_elements],
                "total_elements": total_elements,
                "summary": "\n".join(summary_lines),
                "vision_unavailable": True,
            }
            if truncated_elements:
                payload["truncated_elements"] = truncated_elements
            return json.dumps(payload)

        # Prefer the explicit MIME type cua-driver attaches to its image
        # parts (Surface 7 of NousResearch/hermes-agent#47072 — trycua/cua#1961
        # made `mimeType` part of every MCP image-part response). Fall back
        # to base64-prefix sniffing for older cua-driver builds that didn't
        # carry the field. JPEG base64 starts with /9j/; PNG with iVBOR.
        _mime = cap.image_mime_type
        if not _mime:
            _b64_prefix = cap.png_b64[:8]
            _mime = "image/jpeg" if _b64_prefix.startswith("/9j/") else "image/png"
        # Downscale the screenshot before handing it to the main model. A raw
        # 4K desktop capture is ~12MB of base64 — big enough to hit provider
        # image-size limits, balloon tokens, and make the turn slow/flaky.
        # Cap the longest side at _MAX_MAIN_MODEL_DIM (no-op if already small).
        _img_b64, _mime, _scale, _out_w, _out_h = _shrink_b64_for_main_model(
            cap.png_b64, _mime)
        # Record the downscale factor so the entrypoint can map the model's
        # coordinates (measured on the image it sees) back to real pixels.
        global _last_coord_scale
        _last_coord_scale = _scale if _scale and _scale > 0 else 1.0
        # Report the dimensions of the image the model ACTUALLY sees, so its
        # coordinate frame matches the picture (we rescale on the way back).
        _mm_w = _out_w or response_width
        _mm_h = _out_h or response_height
        # The multimodal response carries the screenshot, not the AX
        # elements array, so a "response truncated to N of M elements"
        # note would be inaccurate — skip it on this branch.
        return {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": summary},
                {"type": "image_url",
                 "image_url": {"url": f"data:{_mime};base64,{_img_b64}"}},
            ],
            "text_summary": summary,
            "meta": {"mode": cap.mode, "width": _mm_w, "height": _mm_h,
                     "elements": total_elements, "png_bytes": cap.png_bytes_len},
        }
    # AX-only (or image-missing fallback): text path actually carries the
    # `elements` array, so the truncation note applies here.
    if truncated_elements:
        summary_lines.append(
            f"  (response truncated to {len(visible_elements)} of {total_elements} elements; "
            f"raise max_elements or pass app= to narrow)"
        )
    summary = "\n".join(summary_lines)
    payload: Dict[str, Any] = {
        "mode": cap.mode,
        "width": response_width,
        "height": response_height,
        "app": cap.app,
        "window_title": cap.window_title,
        "elements": [_element_to_dict(e) for e in visible_elements],
        "total_elements": total_elements,
        "summary": summary,
    }
    if truncated_elements:
        payload["truncated_elements"] = truncated_elements
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# auxiliary.vision routing for captured screenshots (#24015)
# ---------------------------------------------------------------------------

# Longest image side handed to the aux vision model. Full-resolution desktop
# captures tokenize heavily and can overflow small local-model context windows;
# ~1456px keeps SOM badges legible while cutting per-capture vision latency.
_MAX_VISION_DIM = 1456

# Longest image side for the screenshot handed DIRECTLY to a vision-capable
# main model (the `_multimodal` envelope path). A raw 4K desktop capture is
# ~12MB of base64 — big enough to blow past provider image-size limits (400/
# 413), balloon token counts, and make the turn slow/flaky. 1024px keeps the
# screenshot legible for coarse navigation while cutting the payload ~10x.
_MAX_MAIN_MODEL_DIM = 1024


def _shrink_capture_for_vision(raw: bytes, ext: str,
                               max_dim: int = _MAX_VISION_DIM) -> Tuple[bytes, str]:
    """Re-encode a capture as JPEG (quality 85) for the aux-vision model.

    Desktop screenshots re-encode to JPEG at a fraction of PNG's byte size —
    smaller payload = faster vision round-trip — while staying legible for
    description. Always re-encodes to JPEG (even when the image already fits
    ``max_dim``, so a raw PNG capture never goes out as lossless PNG), and
    downscales the longest side to ``max_dim`` when it exceeds it.

    Returns ``(bytes, ext)`` where ``ext`` is the extension that MATCHES the
    encoded bytes (always ``".jpg"`` on success) so the caller names the temp
    file consistently. On any failure (Pillow missing / decode error) returns
    ``(raw, ext)`` unchanged — never worse than the pre-shrink behavior.
    """
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(raw))
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        img = img.convert("RGB")
        out = BytesIO()
        img.save(out, format="JPEG", quality=85)
        return out.getvalue(), ".jpg"
    except Exception as exc:
        logger.debug("computer_use: vision downscale skipped: %s", exc)
        return raw, ext


def _shrink_b64_for_main_model(
    image_b64: str, mime: str, max_dim: int = _MAX_MAIN_MODEL_DIM,
) -> Tuple[str, str, float, int, int]:
    """Downscale a base64 screenshot so its longest side is <= max_dim.

    Operates on base64 (the shape the `_multimodal` envelope carries) and
    returns ``(b64, mime, scale, out_w, out_h)`` where:
      * ``scale`` = original_longest_side / downscaled_longest_side, i.e. the
        multiplier that maps a coordinate the MODEL gives (measured on the
        downscaled image it actually sees) back to a REAL screen pixel. It is
        ``1.0`` when no downscale happened.
      * ``out_w / out_h`` = dimensions of the image actually sent to the model.

    Re-encodes as JPEG (quality 85) to further shrink the payload. Returns the
    input unchanged (scale 1.0, out dims 0 = unknown) when it already fits, or
    on any failure (Pillow missing / decode error), so the path is never worse
    than sending the original.
    """
    if not image_b64:
        return image_b64, mime, 1.0, 0, 0
    try:
        from io import BytesIO
        from PIL import Image
        raw = base64.b64decode(image_b64, validate=False)
        img = Image.open(BytesIO(raw))
        orig_w, orig_h = img.size
        if max(img.size) <= max_dim:
            return image_b64, mime, 1.0, orig_w, orig_h
        img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim))
        new_w, new_h = img.size
        # Longest-side ratio; thumbnail preserves aspect so x/y share one scale.
        scale = float(max(orig_w, orig_h)) / float(max(new_w, new_h) or 1)
        out = BytesIO()
        img.save(out, format="JPEG", quality=85)
        return (base64.b64encode(out.getvalue()).decode("ascii"),
                "image/jpeg", scale, new_w, new_h)
    except Exception as exc:
        logger.debug("computer_use: main-model downscale skipped: %s", exc)
        return image_b64, mime, 1.0, 0, 0

def _should_route_through_aux_vision() -> bool:
    """Return True when ``_capture_response`` should hand the PNG to aux vision.

    Reads the active main provider/model and the loaded config and asks the
    routing helper. Any failure (config import, runtime override missing,
    etc.) returns False so the existing multimodal envelope continues to be
    returned — fail open on the routing decision so a broken config can
    never silently drop the screenshot for vision-capable main models.
    """
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider
        from hermes_cli.config import load_config
        from tools.computer_use.vision_routing import (
            should_route_capture_to_aux_vision,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing import failed: %s", exc)
        return False
    try:
        provider = _read_main_provider()
        model = _read_main_model()
        cfg = load_config()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing config read failed: %s", exc)
        return False
    try:
        return bool(should_route_capture_to_aux_vision(provider, model, cfg))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing decision failed: %s", exc)
        return False


def _route_capture_through_aux_vision(
    cap: CaptureResult,
    summary: str,
) -> Optional[str]:
    """Pre-analyse the captured PNG via ``vision_analyze`` and return a text result.

    The captured base64 PNG is materialised to ``$HERMES_HOME/cache/vision/``
    and handed to ``vision_analyze_tool`` with a generic describe prompt.
    The resulting text description is merged into the existing AX/SOM
    summary so the main model receives a single text payload that mentions
    every interactable element AND a description of what the screenshot
    looked like.

    Returns:
      A JSON-encoded text response on success.
      ``None`` on failure (caller falls back to the multimodal envelope).
    """
    if not cap.png_b64:
        return None
    try:
        import base64 as _base64
        import os as _os
        import uuid as _uuid

        from hermes_constants import get_hermes_dir
        from model_tools import _run_async
        from tools.vision_tools import vision_analyze_tool
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision import failed: %s", exc)
        return None

    temp_image_path = None
    try:
        try:
            raw = _base64.b64decode(cap.png_b64, validate=False)
        except Exception as exc:
            logger.debug("computer_use: failed to decode capture base64: %s", exc)
            return None

        # Pick an extension that matches the on-disk bytes so vision_analyze's
        # MIME sniffing returns the right content-type.
        # Surface 7: prefer the explicit MIME type cua-driver supplied.
        _mime_for_ext = cap.image_mime_type or ""
        if _mime_for_ext == "image/jpeg" or (not _mime_for_ext and cap.png_b64[:8].startswith("/9j/")):
            ext = ".jpg"
        else:
            ext = ".png"
        cache_dir = get_hermes_dir("cache/vision", "temp_vision_images")
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Re-encode to JPEG first, then name the temp file with the extension
        # that actually matches the bytes (vision_analyze_tool sniffs by suffix).
        raw, ext = _shrink_capture_for_vision(raw, ext)
        temp_image_path = cache_dir / f"computer_use_{_uuid.uuid4().hex}{ext}"
        temp_image_path.write_bytes(raw)

        prompt = (
            "Describe what is visible in this desktop application screenshot in "
            "concise but specific terms. Mention the app name and window "
            "title if visible, the overall layout, any labelled buttons, "
            "menus or text fields, and any prominent text content the user "
            "would need to know about. Do not invent details that are not "
            "actually visible.\n\n"
            f"AX/SOM index for cross-reference:\n{summary}"
        )

        result_json = _run_async(
            vision_analyze_tool(str(temp_image_path), prompt)
        )
    except Exception as exc:
        logger.warning(
            "computer_use: auxiliary.vision pre-analysis failed (%s); "
            "returning to caller without aux analysis",
            exc,
        )
        return None
    finally:
        if temp_image_path is not None:
            try:
                _os.unlink(str(temp_image_path))
            except Exception:
                pass

    analysis_text = ""
    if isinstance(result_json, str):
        try:
            parsed = json.loads(result_json)
            if isinstance(parsed, dict):
                analysis_text = str(parsed.get("analysis") or "").strip()
        except (TypeError, json.JSONDecodeError):
            analysis_text = result_json.strip()

    if not analysis_text:
        return None

    return json.dumps({
        "mode": cap.mode,
        "width": cap.width,
        "height": cap.height,
        "app": cap.app,
        "window_title": cap.window_title,
        "elements": [_element_to_dict(e) for e in cap.elements],
        "summary": summary,
        "vision_analysis": analysis_text,
        "vision_analysis_routed_via": "auxiliary.vision",
    })


def _maybe_follow_capture(
    backend: ComputerUseBackend, res: ActionResult, do_capture: bool,
) -> Any:
    if not do_capture:
        return _text_response(res)
    # Skip the follow-up capture when the action itself failed: showing a
    # normal-looking screenshot after a failure misleads the model into thinking
    # the action succeeded. Return the error text instead.
    if not res.ok:
        return _text_response(res)
    try:
        # Preserve the app context established by the preceding capture/focus_app so
        # that capture_after=True re-captures the same app rather than the frontmost
        # window (which may have changed if the action caused a focus shift).
        last_app = getattr(backend, "_last_app", None)
        cap = backend.capture(mode="som", app=last_app)
    except Exception as e:
        logger.warning("follow-up capture failed: %s", e)
        return _text_response(res)
    # Combine action summary with the capture.
    resp = _capture_response(cap)
    if isinstance(resp, dict) and resp.get("_multimodal"):
        prefix = f"[{res.action}] ok={res.ok}" + (f" — {res.message}" if res.message else "")
        resp["content"][0]["text"] = prefix + "\n\n" + resp["content"][0]["text"]
        resp["text_summary"] = prefix + "\n\n" + resp["text_summary"]
        return resp
    # Fallback: action + text capture merged.
    try:
        data = json.loads(resp)
    except (TypeError, json.JSONDecodeError):
        data = {"capture": resp}
    data["action"] = res.action
    data["ok"] = res.ok
    if res.message:
        data["message"] = res.message
    return json.dumps(data)


def _format_elements(elements: List[UIElement], max_lines: int = 40) -> List[str]:
    out: List[str] = []
    for e in elements[:max_lines]:
        label = e.label.replace("\n", " ")[:60]
        out.append(f"  #{e.index} {e.role} {label!r} @ {e.bounds}"
                   + (f" [{e.app}]" if e.app else ""))
    if len(elements) > max_lines:
        out.append(f"  ... +{len(elements) - max_lines} more (call capture with app= to narrow)")
    return out


def _element_to_dict(e: UIElement) -> Dict[str, Any]:
    return {
        "index": e.index,
        "role": e.role,
        "label": e.label,
        "bounds": list(e.bounds),
        "app": e.app,
    }


# ---------------------------------------------------------------------------
# Availability check (used by the tool registry check_fn)
# ---------------------------------------------------------------------------

def check_computer_use_requirements() -> bool:
    """Return True iff computer_use can run on this host.

    Conditions: macOS, Windows, or Linux + cua-driver binary installed (or
    override via env). cua-driver runs on all three; the Linux path is
    headed/X11 today (Wayland via XWayland), pure-Wayland progress tracked
    upstream. Linux users see specific blocked checks via
    `hermes computer-use doctor` if their session is incomplete (e.g. no
    DISPLAY set).
    """
    if sys.platform not in ("darwin", "win32", "linux"):
        return False
    from tools.computer_use.cua_backend import cua_driver_binary_available
    return cua_driver_binary_available()


def get_computer_use_schema() -> Dict[str, Any]:
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA
    return COMPUTER_USE_SCHEMA
