"""Focused verification of the WatcherAgent deep-research delegation loop.

These tests exercise the *incremental / self-terminating* behaviour added in the
deep-research overhaul (the "十项优化"), which the older test_watch.py does
NOT cover. Specifically they prove, on the real `_run_delegation` coroutine:

  #10  termination without hanging  — silent_streak auto-finish (stream never
       stops, batches keep producing nothing new) still ends the run, and the
       final summary always threads back to the main agent (never blank / never
       an infinite wait, thanks to the summary timeout + running_report fallback).

  #9   progress_report / incremental report — running_report accumulates one
       entry per productive batch and a `progress_report` bg event is pushed
       mid-run (not only at the end).

  #8   画面级 entry dedup — near-identical frames are removed by FrameBuffer
       before watcher batches are assembled, with cutoff strength following the
       conventional dHash distance contract.

The engine's thread/`_build()` machinery is bypassed: we drive `_run_delegation`
directly on a fresh event loop with fake `responder`/`cfg`/callbacks, so the test
is deterministic and offline (no model, no gateway).
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest

from agent.multimodal._memory import Frame, FrameBuffer, _compute_dhash, _hamming


# --------------------------------------------------------------------------- #
# Helpers: real JPEGs so _compute_dhash returns a meaningful (non-zero) hash.
# --------------------------------------------------------------------------- #
def _jpeg_b64(color) -> str:
    """A small solid-color JPEG as base64 (PIL is a hard dep of the module)."""
    from PIL import Image

    img = Image.new("RGB", (32, 32), color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gradient_b64(seed: int) -> str:
    """A per-seed distinct textured JPEG (distinct dHash from other seeds)."""
    from PIL import Image

    img = Image.new("L", (32, 32))
    px = img.load()
    for y in range(32):
        for x in range(32):
            px[x, y] = (x * 8 + y * 4 + seed * 37) % 256
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_engine(frame_buffer, *, responder, cfg, emit_cb, on_complete,
                 registry_cb=None, on_round_report=None):
    """Build a WatcherAgent with only the attrs `_run_delegation` reads,
    bypassing start()/_build() entirely."""
    from agent.multimodal.watcher_engine import WatcherAgent

    eng = WatcherAgent.__new__(WatcherAgent)
    eng.frame_buffer = frame_buffer
    eng.cfg = cfg
    eng.client = object()
    eng.model = "fake-model"
    eng.responder = responder
    eng._emit_cb = emit_cb
    eng._on_delegation_complete = on_complete
    eng._on_round_report = on_round_report
    eng._stop_events = {}
    eng._source_stopped = False
    eng._source_epoch = 0
    eng._state_lock = threading.RLock()
    eng._sid = "test-sid"
    eng._research_registry_cb = registry_cb
    return eng


class _FakeResponder:
    """Stand-in for WatcherWorker: each _spawn_delegation call returns an
    awaitable that emits `text` (or nothing) through the sink. Records how many
    times it was actually invoked (so we can assert dedup SKIPPED a batch), and
    the brief each call ran with (so we can assert op=update changed the goal)."""

    def __init__(self, *, texts):
        # texts: list of per-call strings; "" means the batch produced nothing.
        self._texts = list(texts)
        self.calls = 0
        self.briefs = []
        self.frame_batches = []

    async def _spawn_delegation(self, *, task_instruction, prelude, sink,
                               on_event, ask_ts=None, ask_frames_override=None,
                               seen_search_briefs=None, prev_segment=None,
                               static_tail_check=False):
        idx = self.calls
        self.calls += 1
        self.briefs.append(task_instruction)
        self.frame_batches.append([
            frame.ts for frame in (ask_frames_override or [])
        ])
        text = self._texts[idx] if idx < len(self._texts) else ""

        async def _task():
            # Mimic the real ReAct emitting a router_react event then tokens.
            await on_event({"type": "router_react", "can_answer": False,
                            "search_tasks": [], "recall_tasks": []})
            if text:
                await sink(text)

        return _task()


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch):
    """Redirect the analyse/ dir to a throwaway temp home so no real files are
    written (watch_file.watch_dir reads HERMES_HOME)."""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("ARGUS_HOME", d)
        yield d


def _run(coro, timeout=15.0):
    """Run a coroutine on a fresh loop with a hard wall-clock guard so a real
    hang FAILS the test (rather than blocking the suite forever)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    finally:
        loop.close()


def _track_running_writes_after_replacement(monkeypatch, replaced):
    """Record any stale write that tries to make an old run live again."""
    from agent.multimodal import watch_file

    stale_writes = []
    original_set_status = watch_file.set_status
    original_update_state = watch_file.update_state

    def set_status(request_id, status, *args, **kwargs):
        if replaced["value"] and status == "running":
            stale_writes.append(("set_status", request_id))
        return original_set_status(request_id, status, *args, **kwargs)

    def update_state(request_id, **patch):
        if replaced["value"] and patch.get("status") == "running":
            stale_writes.append(("update_state", request_id))
        return original_update_state(request_id, **patch)

    monkeypatch.setattr(watch_file, "set_status", set_status)
    monkeypatch.setattr(watch_file, "update_state", update_state)
    return stale_writes


# --------------------------------------------------------------------------- #
# Sanity: dHash helpers behave (guards the dedup assertions below).
# --------------------------------------------------------------------------- #
def test_dhash_similar_vs_distinct():
    a = _gradient_b64(0)
    a2 = _gradient_b64(0)   # identical content
    b = _gradient_b64(50)   # clearly different
    ha, ha2, hb = _compute_dhash(a), _compute_dhash(a2), _compute_dhash(b)
    assert ha != 0 and hb != 0
    assert _hamming(ha, ha2) < 6      # near-identical → below dedup threshold
    assert _hamming(ha, hb) >= 6      # distinct → above threshold


# --------------------------------------------------------------------------- #
# Termination + #9 progress_report. v33: a silent streak no longer auto-finishes
# a live stream (no_progress is log-only) — a live stream that never stops would
# park forever waiting for new frames. Termination comes from source-stop or
# user-stop. Here we close the source after a couple of batches and assert the
# run completes, emits mid-run progress_report events, and fires delegation_done.
# --------------------------------------------------------------------------- #
def test_research_auto_finishes_and_emits_progress_report():
    cfg = SimpleNamespace(
        watch_frame_batch=2,
        watch_min_batch=2,
        watch_poll_interval=0.01,
        watch_max_rounds=200,
        watch_silent_stop_rounds=3,   # (v33: log-only, does not terminate)
        watch_report_every_rounds=1,  # push progress every productive batch
        watch_summary_timeout=2.0,
        watch_summary_max_tokens=256,
        watch_round_ttl_sec=0.1,
    )

    fb = FrameBuffer(SimpleNamespace(buffer_seconds=60, buffer_capture_fps=2))
    # Seed round 1's frames; more are fed between rounds (round 1 anchors to the
    # recent tail, so each subsequent round needs fresh frames pushed past it).
    fb.push(Frame(ts=0.0, jpeg_b64=_gradient_b64(0)))
    fb.push(Frame(ts=1.0, jpeg_b64=_gradient_b64(40)))

    # Both productive batches yield findings.
    responder = _FakeResponder(texts=["发现A", "发现B", "", "", ""])

    events = []
    completes = []
    eng = _make_engine(
        fb, responder=responder, cfg=cfg,
        emit_cb=lambda ev, payload: None,   # replaced by _drive below
        # The contract under test only needs the request id and stop reason.
        on_complete=lambda rid, _task, _summary, reason: completes.append((rid, reason)),
    )
    eng._source_stopped = False

    # Feed round 2's frames after round 1 ran; close the source once two
    # productive batches completed → the loop drains and ends (no hang).
    fed = {"done": False}

    def _drive(ev, payload):
        events.append((ev, payload))
        if responder.calls == 1 and not fed["done"]:
            fed["done"] = True
            fb.push(Frame(ts=10.0, jpeg_b64=_gradient_b64(80)))
            fb.push(Frame(ts=11.0, jpeg_b64=_gradient_b64(120)))
        if responder.calls >= 2:
            eng._source_stopped = True
    eng._emit_cb = _drive

    _run(eng._run_delegation("rid-fin", task_instruction="盯桌面"), timeout=15.0)

    # It terminated (no hang) and signalled completion to the main agent.
    assert completes, "delegation never completed → main agent would hang"
    rid, reason = completes[0]
    assert rid == "rid-fin"

    # #9: a progress_report bg event was pushed MID-run (before the final).
    progress = [p for (ev, p) in events
                if ev == "multimodal.bg" and p.get("type") == "progress_report"]
    assert progress, "no incremental progress_report emitted"
    # The accumulated report carries the productive batches' findings.
    assert any("发现A" in p.get("report", "") for p in progress)

    # A terminal delegation_done was emitted so the right-rail window closes.
    assert any(p.get("delegation_done") for (ev, p) in events
               if ev == "multimodal.bg")


# --------------------------------------------------------------------------- #
# Entry dedup (moved to FrameBuffer): near-identical frames are dropped ON PUSH,
# so watcher no longer needs any batch-level dHash skip — it just reads the
# already-sparse buffer. This replaces the old #8 "batch dedup skip" test.
# --------------------------------------------------------------------------- #
def test_framebuffer_entry_dedup_drops_near_identical():
    fb = FrameBuffer(SimpleNamespace(
        buffer_seconds=60, buffer_capture_fps=2,
        framebuffer_dhash_threshold_init=6,
        framebuffer_dhash_threshold_min=2,
        framebuffer_dhash_threshold_max=20))
    same = _gradient_b64(0)
    fb.push(Frame(ts=0.0, jpeg_b64=same))          # kept (first)
    for i in range(1, 10):
        fb.push(Frame(ts=float(i), jpeg_b64=same))  # identical → dropped on push
    # Only the first distinct frame survived; the 9 near-identical ones were
    # deduped at the FrameBuffer entry (dHash distance 0 < threshold).
    assert fb.size == 1, f"entry dedup should keep 1 frame, got {fb.size}"

    # A visibly different frame IS kept.
    fb.push(Frame(ts=10.0, jpeg_b64=_gradient_b64(200)))
    assert fb.size == 2, f"a distinct frame must be kept, got {fb.size}"


def test_framebuffer_threshold_controls_dedup_strength():
    # Conventional cutoff semantics: distance < threshold is dropped, so a
    # larger threshold deduplicates at least as aggressively as a smaller one.
    def _count(threshold):
        fb = FrameBuffer(SimpleNamespace(
            buffer_seconds=60, buffer_capture_fps=2,
            framebuffer_dhash_threshold_init=threshold,
            framebuffer_dhash_threshold_min=0,
            framebuffer_dhash_threshold_max=64))
        # A slowly-drifting gradient: consecutive frames are similar but not equal.
        for i in range(12):
            fb.push(Frame(ts=float(i), jpeg_b64=_gradient_b64(i)))
        return fb.size
    low_cutoff = _count(2)
    high_cutoff = _count(20)
    assert high_cutoff <= low_cutoff
    # Fixture sanity: the sequence straddles these cutoffs, so the comparison
    # exercises a real policy difference rather than passing by equality.
    assert high_cutoff < low_cutoff


# --------------------------------------------------------------------------- #
# v33: there is NO final summary LLM call at completion — per-round findings are
# the only output, streamed live via progress_report. This guards that a
# productive round's finding is captured in the accumulated progress_report (so
# it is never lost) and the run completes without hanging.
# --------------------------------------------------------------------------- #
def test_finding_captured_in_progress_report():
    cfg = SimpleNamespace(
        watch_frame_batch=2,
        watch_min_batch=2,
        watch_poll_interval=0.01,
        watch_max_rounds=200,
        watch_silent_stop_rounds=2,   # (v33: log-only, does not terminate)
        watch_report_every_rounds=1,  # push progress every productive batch
        watch_summary_timeout=0.2,
        watch_summary_max_tokens=256,
        watch_round_ttl_sec=0.1,
    )

    fb = FrameBuffer(SimpleNamespace(buffer_seconds=60, buffer_capture_fps=2))
    for i in range(6):
        fb.push(Frame(ts=float(i), jpeg_b64=_gradient_b64(i * 5)))

    responder = _FakeResponder(texts=["独一无二的发现", "", ""])

    events = []
    completes = []
    eng = _make_engine(
        fb, responder=responder, cfg=cfg,
        emit_cb=lambda ev, payload: events.append((ev, payload)),
        on_complete=lambda rid, _task, _summary, reason: completes.append((rid, reason)),
    )
    eng._source_stopped = False

    # Close the source after the productive batch → loop drains and ends.
    def _stop_after_one(ev, payload):
        events.append((ev, payload))
        if responder.calls >= 1:
            eng._source_stopped = True
    eng._emit_cb = _stop_after_one

    _run(eng._run_delegation("rid-timeout", task_instruction="b"), timeout=15.0)

    assert completes, "run must complete without hanging"
    # The productive round's finding is carried in the accumulated progress_report.
    progress = [p for (ev, p) in events
                if ev == "multimodal.bg" and p.get("type") == "progress_report"]
    assert progress, "no incremental progress_report emitted"
    assert any("独一无二的发现" in p.get("report", "") for p in progress)


# --------------------------------------------------------------------------- #
# Batch-boundary op=delete: the loop finishes the current round then ends, and
# reports `_deleted` state so the completion hook can be suppressed upstream.
# --------------------------------------------------------------------------- #
def _batch_cfg(**over):
    base = dict(
        watch_frame_batch=2, watch_min_batch=2,
        watch_poll_interval=0.01, watch_max_rounds=200,
        watch_silent_stop_rounds=99,   # don't let silent-stop end it first
        watch_report_every_rounds=99,
        watch_summary_timeout=2.0, watch_summary_max_tokens=256,
        watch_round_ttl_sec=0.1,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_delete_ends_run_at_batch_boundary():
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=60, buffer_capture_fps=2))
    for i in range(20):
        fb.push(Frame(ts=float(i), jpeg_b64=_gradient_b64(i * 4)))
    # Many productive batches available; without delete this would run for a while.
    responder = _FakeResponder(texts=["发现"] * 20)

    # Registry entry flips to _deleted after the FIRST batch runs.
    entry = {"task_instruction": "研究", "label": "L"}
    state = {"rounds": 0}

    def registry_cb(rid):
        # Mark deleted once at least one batch has executed.
        if responder.calls >= 1:
            entry["_deleted"] = True
        return entry

    completes = []
    eng = _make_engine(
        fb, responder=responder, cfg=_batch_cfg(),
        emit_cb=lambda ev, payload: None,
        on_complete=lambda rid, _task, _summary, reason: completes.append((rid, reason)),
        registry_cb=registry_cb,
    )
    eng._source_stopped = False

    _run(eng._run_delegation("rid-del", task_instruction="研究"), timeout=15.0)

    # Ran the current round then stopped promptly (NOT all 20 batches).
    assert 1 <= responder.calls <= 3, f"delete didn't end at batch boundary: {responder.calls}"
    # The delegation still completed (completion signalled) so nothing hangs.
    assert completes
    assert completes[0] == ("rid-del", "deleted")


def test_update_changes_brief_next_round():
    # target=2, ttl=0.1s. Feed 2 frames at a time so multiple rounds happen; an
    # op=update between rounds must take effect on the NEXT round.
    fb = FrameBuffer(SimpleNamespace(
        buffer_seconds=60, buffer_capture_fps=2,
        framebuffer_dhash_threshold_init=6,
        framebuffer_dhash_threshold_min=2, framebuffer_dhash_threshold_max=20))
    fb.push(Frame(ts=0.0, jpeg_b64=_gradient_b64(0)))
    fb.push(Frame(ts=1.0, jpeg_b64=_gradient_b64(40)))   # round-1 frames
    responder = _FakeResponder(texts=["a", "b", "c", "d", "e", "f"])

    entry = {"task_instruction": "旧目标", "label": "L"}
    eng_ref = {}

    def registry_cb(rid):
        # After the first round ran, push more frames + an updated goal.
        if responder.calls == 1 and not entry.get("_did_update"):
            fb.push(Frame(ts=10.0, jpeg_b64=_gradient_b64(80)))
            fb.push(Frame(ts=11.0, jpeg_b64=_gradient_b64(120)))
            entry["task_instruction"] = "新目标"
            entry["_pending_update"] = True
            entry["_did_update"] = True
        # Once the new goal has run at least once, close the source so the run ends.
        if entry.get("_did_update") and "新目标" in responder.briefs and eng_ref:
            eng_ref["e"]._source_stopped = True
        return entry

    completes = []
    eng = _make_engine(
        fb, responder=responder,
        cfg=_batch_cfg(),
        emit_cb=lambda ev, payload: None,
        on_complete=lambda rid, _task, _summary, reason: completes.append((rid, reason)),
        registry_cb=registry_cb,
    )
    eng._source_stopped = False
    eng_ref["e"] = eng

    _run(eng._run_delegation("rid-upd", task_instruction="旧目标"), timeout=15.0)

    # First batch ran with the old goal, a later batch ran with the new goal.
    assert responder.briefs[0] == "旧目标"
    assert "新目标" in responder.briefs, f"update not applied next round: {responder.briefs}"


# --------------------------------------------------------------------------- #
# TTL gate: when target_frames is NOT reached but the ttl elapses, the round
# runs on the frames it has (doesn't hang waiting for the full target).
# --------------------------------------------------------------------------- #
def test_auto_pacing_refreshes_during_first_round_wait():
    cfg = _batch_cfg(
        watch_min_batch=64,
        watch_round_ttl_sec=120.0,
        watch_static_tail_flush_sec=999.0,
    )
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=60, buffer_capture_fps=2))
    for i in range(4):
        fb.push(Frame(ts=float(i), jpeg_b64=_gradient_b64(i * 7)))

    registry = {
        "pacing_mode": "auto",
        # Create-time medium snapshot; the engine must stop treating this as an
        # explicit override once SceneDhash publishes a live classification.
        "ttl_sec": 60,
        "target_frames": 60,
    }
    responder = _FakeResponder(texts=["实时段", ""])
    events = []
    eng = _make_engine(
        fb,
        responder=responder,
        cfg=cfg,
        emit_cb=lambda ev, payload: events.append((ev, payload)),
        on_complete=lambda *_args: None,
        registry_cb=lambda _rid: registry,
    )

    switched = {"done": False}

    def _switch_scene_and_stop(ev, payload):
        events.append((ev, payload))
        if (not switched["done"] and ev == "multimodal.bg"
                and payload.get("type") == "waiting"):
            switched["done"] = True
            fb.set_current_scene({
                "pace": "live", "ttl_sec": 0.05, "target_frames": 2,
            })
        if responder.calls >= 1:
            eng._source_stopped = True

    eng._emit_cb = _switch_scene_and_stop
    _run(eng._run_delegation("rid-auto-pace", task_instruction="盯实时操作"))

    assert switched["done"] is True
    assert responder.calls >= 1
    waits = [
        payload for ev, payload in events
        if ev == "multimodal.bg" and payload.get("type") == "waiting"
    ]
    # Initial UI feedback uses the medium snapshot (not the stale 120/64 cfg),
    # then the real first round observes the newly published live scene and
    # runs immediately with at most its two-frame target.
    assert waits[0]["need"] == 60
    assert waits[0]["ttl_sec"] == 60.0
    assert all(payload.get("need") != 64 for payload in waits)
    assert len(responder.frame_batches[0]) <= 2


def test_ttl_elapse_runs_on_partial_frames():
    cfg = _batch_cfg(
        watch_min_batch=64,          # target 64 frames…
        watch_round_ttl_sec=0.1,     # …but ttl is 0.1s → fires on partial
        watch_silent_stop_rounds=1,
    )
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=60, buffer_capture_fps=2))
    # Only 4 frames — far below the 64 target. ttl elapse must run anyway.
    for i in range(4):
        fb.push(Frame(ts=float(i), jpeg_b64=_gradient_b64(i * 4)))

    responder = _FakeResponder(texts=["首批解读", ""])
    events = []
    completes = []
    eng = _make_engine(
        fb, responder=responder, cfg=cfg,
        emit_cb=lambda ev, payload: events.append((ev, payload)),
        on_complete=lambda rid, _task, _summary, reason: completes.append((rid, reason)),
    )
    eng._source_stopped = False

    def _flip_after_first(_ev, _p):
        events.append((_ev, _p))
        if responder.calls >= 1:
            eng._source_stopped = True
    eng._emit_cb = _flip_after_first

    _run(eng._run_delegation("rid-ttl", task_instruction="盯"), timeout=10.0)

    # It ran on the 4 frames it had once the ttl elapsed (didn't hang for 64).
    assert responder.calls >= 1, "ttl elapse did not run the round on partial frames"
    assert completes
    # The waiting events carry ttl fields so the panel can show a countdown.
    waits = [p for (ev, p) in events
             if ev == "multimodal.bg" and p.get("type") == "waiting"]
    assert any("ttl_remaining" in p for p in waits), \
        "waiting events must carry ttl_remaining for the UI countdown"


# --------------------------------------------------------------------------- #
# Live-watcher streaming: on_round_report fires once per PRODUCTIVE round, with
# that round's report text (so the gateway can append it to the turn-2 message).
# --------------------------------------------------------------------------- #
def test_on_round_report_fires_per_productive_round():
    cfg = _batch_cfg(watch_silent_stop_rounds=2)
    # Feed frames in 3 waves so 3 rounds run; each round yields a report.
    fb = FrameBuffer(SimpleNamespace(
        buffer_seconds=60, buffer_capture_fps=2,
        framebuffer_dhash_threshold_init=6,
        framebuffer_dhash_threshold_min=2, framebuffer_dhash_threshold_max=20))
    fb.push(Frame(ts=0.0, jpeg_b64=_gradient_b64(0)))
    fb.push(Frame(ts=1.0, jpeg_b64=_gradient_b64(40)))   # wave 1
    responder = _FakeResponder(texts=["报告A", "报告B", "报告C", "", ""])

    reports = []
    eng_ref = {}

    def registry_cb(rid):
        # Feed the next wave after each productive round, then stop the source
        # once 3 rounds have run so the loop drains + ends.
        n = responder.calls
        if n == 1 and not eng_ref.get("w2"):
            eng_ref["w2"] = True
            fb.push(Frame(ts=10.0, jpeg_b64=_gradient_b64(80)))
            fb.push(Frame(ts=11.0, jpeg_b64=_gradient_b64(120)))
        elif n == 2 and not eng_ref.get("w3"):
            eng_ref["w3"] = True
            fb.push(Frame(ts=20.0, jpeg_b64=_gradient_b64(160)))
            fb.push(Frame(ts=21.0, jpeg_b64=_gradient_b64(200)))
        elif n >= 3 and eng_ref.get("e"):
            eng_ref["e"]._source_stopped = True
        return None

    eng = _make_engine(
        fb, responder=responder, cfg=cfg,
        emit_cb=lambda ev, payload: None,
        on_complete=lambda rid, _task, _summary, reason: None,
        on_round_report=lambda rid, ridx, text: reports.append((rid, ridx, text)),
        registry_cb=registry_cb,
    )
    eng._source_stopped = False
    eng_ref["e"] = eng

    _run(eng._run_delegation("rid-rr", task_instruction="盯"), timeout=10.0)

    # One callback per PRODUCTIVE round (empties don't fire it).
    assert len(reports) == 3, reports
    assert all(r[0] == "rid-rr" for r in reports)
    # Each carries that round's report text (frame-range prefix + answer).
    joined = " ".join(r[2] for r in reports)
    assert "报告A" in joined and "报告B" in joined and "报告C" in joined
    # round_idx is monotonic across productive rounds.
    assert [r[1] for r in reports] == sorted(r[1] for r in reports)


def test_source_stop_without_replacement_drains_buffered_tail():
    """A stop is a drain signal, not an immediate generation replacement."""
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=60, buffer_capture_fps=2))
    fb.push(Frame(ts=0.0, jpeg_b64=_gradient_b64(0)))
    fb.push(Frame(ts=1.0, jpeg_b64=_gradient_b64(40)))
    responder = _FakeResponder(texts=["old segment", "buffered tail"])
    completions = []
    state = {"fed_tail": False, "epoch_at_stop": None}
    eng = _make_engine(
        fb,
        responder=responder,
        cfg=_batch_cfg(),
        emit_cb=lambda _event, _payload: None,
        on_complete=lambda rid, _task, _summary, reason: completions.append(
            (rid, reason)
        ),
    )
    original_spawn = responder._spawn_delegation

    async def spawn_and_stop(**kwargs):
        task = await original_spawn(**kwargs)
        if responder.calls == 1 and not state["fed_tail"]:
            state["fed_tail"] = True
            fb.push(Frame(ts=10.0, jpeg_b64=_gradient_b64(80)))
            fb.push(Frame(ts=11.0, jpeg_b64=_gradient_b64(120)))
            before = eng._source_epoch
            eng.mark_source_stopped()
            state["epoch_at_stop"] = (before, eng._source_epoch)
        return task

    responder._spawn_delegation = spawn_and_stop

    _run(
        eng._run_delegation("rid-stop-drain", task_instruction="drain old stream"),
        timeout=10.0,
    )

    assert state["epoch_at_stop"] == (0, 0)
    assert responder.frame_batches == [[0.0, 1.0], [10.0, 11.0]]
    assert completions == [("rid-stop-drain", "source_end")]


def test_replacement_while_waiting_never_routes_new_frames_to_old_run(monkeypatch):
    from agent.multimodal import watch_file

    fb = FrameBuffer(SimpleNamespace(buffer_seconds=60, buffer_capture_fps=2))
    # Preserve evidence that capture was live while beginning with no pending
    # frames, forcing the old run into its wait loop.
    fb.push(Frame(ts=0.0, jpeg_b64=_gradient_b64(0)))
    fb.clear()
    responder = _FakeResponder(texts=["must never run"])
    completions = []
    replaced = {"value": False}
    stale_running_writes = _track_running_writes_after_replacement(
        monkeypatch, replaced
    )

    async def run():
        waiting = asyncio.Event()

        def emit(_event, payload):
            if payload.get("type") == "waiting":
                waiting.set()

        eng = _make_engine(
            fb,
            responder=responder,
            cfg=_batch_cfg(watch_round_ttl_sec=1.0),
            emit_cb=emit,
            on_complete=lambda rid, _task, _summary, reason: completions.append(
                (rid, reason)
            ),
        )
        task = asyncio.create_task(
            eng._run_delegation(
                "rid-replaced-waiting", task_instruction="old stream watcher"
            )
        )
        await asyncio.wait_for(waiting.wait(), timeout=1.0)

        epoch_before_stop = eng._source_epoch
        eng.mark_source_stopped()
        assert eng._source_epoch == epoch_before_stop
        replaced["value"] = True
        eng.mark_source_started()
        assert eng._source_epoch > epoch_before_stop
        fb.push(Frame(ts=100.0, jpeg_b64=_gradient_b64(160)))
        fb.push(Frame(ts=101.0, jpeg_b64=_gradient_b64(200)))

        await asyncio.wait_for(task, timeout=2.0)

    _run(run(), timeout=5.0)

    assert responder.calls == 0
    assert responder.frame_batches == []
    assert completions == [("rid-replaced-waiting", "source_end")]
    assert stale_running_writes == []
    assert watch_file.read_status("rid-replaced-waiting")["status"] == "done"


def test_replacement_during_inflight_batch_discards_old_result_and_new_frames(
    monkeypatch,
):
    from agent.multimodal import watch_file

    fb = FrameBuffer(SimpleNamespace(buffer_seconds=60, buffer_capture_fps=2))
    fb.push(Frame(ts=0.0, jpeg_b64=_gradient_b64(0)))
    fb.push(Frame(ts=1.0, jpeg_b64=_gradient_b64(40)))
    completions = []
    reports = []
    events = []
    replaced = {"value": False}
    stale_running_writes = _track_running_writes_after_replacement(
        monkeypatch, replaced
    )

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        class BlockingResponder:
            def __init__(self):
                self.frame_batches = []

            async def _spawn_delegation(
                self, *, sink, on_event, ask_frames_override=None, **_kwargs
            ):
                self.frame_batches.append([
                    frame.ts for frame in (ask_frames_override or [])
                ])
                entered.set()

                async def task():
                    await release.wait()
                    await on_event({
                        "type": "router_react",
                        "search_tasks": [],
                        "recall_tasks": [],
                    })
                    await sink("late old-stream answer")

                return task()

        responder = BlockingResponder()
        eng = _make_engine(
            fb,
            responder=responder,
            cfg=_batch_cfg(),
            emit_cb=lambda event, payload: events.append((event, payload)),
            on_complete=lambda rid, _task, _summary, reason: completions.append(
                (rid, reason)
            ),
            on_round_report=lambda *row: reports.append(row),
        )
        task = asyncio.create_task(
            eng._run_delegation(
                "rid-replaced-inflight", task_instruction="old stream watcher"
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        eng.mark_source_stopped()
        replaced["value"] = True
        eng.mark_source_started()
        fb.push(Frame(ts=100.0, jpeg_b64=_gradient_b64(160)))
        fb.push(Frame(ts=101.0, jpeg_b64=_gradient_b64(200)))
        release.set()

        await asyncio.wait_for(task, timeout=2.0)
        return responder.frame_batches

    frame_batches = _run(run(), timeout=5.0)

    assert frame_batches == [[0.0, 1.0]]
    assert reports == []
    assert not any(
        payload.get("type") == "answer_delta"
        and payload.get("delta") == "late old-stream answer"
        for _event, payload in events
    )
    assert completions == [("rid-replaced-inflight", "source_end")]
    assert stale_running_writes == []
    assert watch_file.read_status("rid-replaced-inflight")["status"] == "done"
