# -*- coding: utf-8 -*-
"""Live-watcher completion hook: queue once and keep its prompt internal.

Contract (differs from the monitor hook, which drops if busy):
  • On a hooked watcher completion, an internal main-agent task is enqueued into
    a persistent per-session FIFO — never dropped.
  • It is drained (dispatched to the main agent via _run_prompt_submit) only when
    the session is idle; while busy it waits.
  • The queued message is deduped by rid and is FIFO.
  • The task prompt is hidden from the chat UI/history; only the final Assistant
    response is user-visible.

Pure unittest: we stub _run_prompt_submit to record dispatches instead of running
a real agent turn, and drive the drain helpers directly.
"""
import threading
import unittest


class TestWatcherHookQueue(unittest.TestCase):
    def setUp(self):
        import tui_gateway.server as S
        self.S = S
        self._orig_submit = S._run_prompt_submit
        self._orig_emit = S._emit
        self.dispatched = []

        # Stub: record (rid, text); mimic _run_prompt_submit's real lifecycle of
        # clearing session["running"] when the (fake) turn completes.
        def _fake_submit(rid, sid, session, text, **kwargs):
            self.dispatched.append({
                "rid": rid, "sid": sid, "text": text, **kwargs,
            })
            with session["history_lock"]:
                session["running"] = False
                if kwargs.get("internal_origin") == "watcher_hook":
                    session["_monitor_hook_running"] = False
        S._run_prompt_submit = _fake_submit
        S._emit = lambda *a, **k: None

    def tearDown(self):
        self.S._run_prompt_submit = self._orig_submit
        self.S._emit = self._orig_emit

    def _session(self, running=False):
        return {"history": [], "history_lock": threading.Lock(),
                "running": running, "_monitor_hook_running": False}

    def test_enqueue_then_drain_when_idle(self):
        s = self._session(running=False)
        self.S._enqueue_watcher_hook(s, rid="req_a", label="观影心得",
                                     task="整理成一份观影报告", report="第1段…第2段…")
        # Idle → drains and dispatches.
        fired = self.S._drain_watcher_hook("d1", "sid", s)
        self.assertTrue(fired)
        self.assertEqual(len(self.dispatched), 1)
        txt = self.dispatched[0]["text"]
        self.assertTrue(self.dispatched[0].get("internal_origin") == "watcher_hook")
        self.assertIn("第1段", self.dispatched[0]["internal_fallback_text"])
        self.assertIn("整理成一份观影报告", txt)
        self.assertIn("monitor / watcher result for reference:", txt)
        self.assertIn("第1段", txt)  # report embedded
        # Queue drained empty.
        self.assertEqual(s.get("_watcher_hook_queue"), [])

    def test_not_dropped_when_busy_then_fires_when_idle(self):
        s = self._session(running=True)  # main agent busy
        self.S._enqueue_watcher_hook(s, rid="req_b", label="周报", task="生成周报")
        # Busy → drain is a no-op, hook stays queued (NOT dropped).
        self.assertFalse(self.S._drain_watcher_hook("d1", "sid", s))
        self.assertEqual(len(self.dispatched), 0)
        self.assertEqual(len(s["_watcher_hook_queue"]), 1)
        # Agent goes idle → next drain fires it.
        s["running"] = False
        self.assertTrue(self.S._drain_watcher_hook("d2", "sid", s))
        self.assertEqual(len(self.dispatched), 1)
        self.assertIn("生成周报", self.dispatched[0]["text"])

    def test_dedup_same_rid(self):
        s = self._session()
        self.S._enqueue_watcher_hook(s, rid="req_c", label="L", task="T")
        self.S._enqueue_watcher_hook(s, rid="req_c", label="L", task="T")  # dup
        self.assertEqual(len(s["_watcher_hook_queue"]), 1)
        self.assertTrue(self.S._drain_watcher_hook("d", "sid", s))
        self.assertEqual(len(self.dispatched), 1)
        # A duplicated completion callback after the first hook already ran
        # must still be ignored for this live session.
        self.S._enqueue_watcher_hook(s, rid="req_c", label="L", task="T")
        self.assertEqual(s["_watcher_hook_queue"], [])
        self.assertFalse(self.S._drain_watcher_hook("d2", "sid", s))
        self.assertEqual(len(self.dispatched), 1)

    def test_fifo_order_multiple_hooks(self):
        s = self._session(running=True)
        self.S._enqueue_watcher_hook(s, rid="req_1", label="一", task="t1")
        self.S._enqueue_watcher_hook(s, rid="req_2", label="二", task="t2")
        self.S._enqueue_watcher_hook(s, rid="req_3", label="三", task="t3")
        self.assertEqual(len(s["_watcher_hook_queue"]), 3)
        s["running"] = False
        # Each drain fires exactly one, in order; the real turn tail re-drains the
        # next, which here we simulate by looping.
        order = []
        while self.S._drain_watcher_hook("d", "sid", s):
            order.append(self.dispatched[-1]["text"])
        self.assertEqual(len(order), 3)
        self.assertIn("t1", order[0])
        self.assertIn("t2", order[1])
        self.assertIn("t3", order[2])

    def test_tail_chaining_fires_all_hooks(self):
        # Real chaining: _run_prompt_submit spawns a thread + returns; when that
        # turn finishes its tail calls _drain_watcher_hook again. Simulate that by
        # having the submit stub, AFTER clearing running (turn done), re-drain —
        # exactly like the real tail. This proves the _monitor_hook_running guard
        # timing does NOT block the chained next hook.
        s = self._session(running=True)   # busy: all three queue up
        self.S._enqueue_watcher_hook(s, rid="req_1", label="一", task="t1")
        self.S._enqueue_watcher_hook(s, rid="req_2", label="二", task="t2")
        self.S._enqueue_watcher_hook(s, rid="req_3", label="三", task="t3")

        fired = []

        def _submit_then_tail(rid, sid, session, text, **_kwargs):
            fired.append(text)
            # Turn completes → both guards are released in the real finally
            # before its tail attempts to chain the next Watcher hook.
            with session["history_lock"]:
                session["running"] = False
                session["_monitor_hook_running"] = False
            # tail: chain the next queued hook (like run()'s tail at server.py)
            self.S._drain_watcher_hook("tail", sid, session)
        self.S._run_prompt_submit = _submit_then_tail

        s["running"] = False
        # Kick off the first drain; the tail chains the rest.
        self.assertTrue(self.S._drain_watcher_hook("d", "sid", s))
        self.assertEqual(len(fired), 3, fired)
        self.assertIn("t1", fired[0])
        self.assertIn("t2", fired[1])
        self.assertIn("t3", fired[2])
        self.assertEqual(s.get("_watcher_hook_queue"), [])

    def test_message_without_task(self):
        s = self._session()
        self.S._enqueue_watcher_hook(
            s, rid="req_d", label="L", task="", report="final report")
        self.S._drain_watcher_hook("d", "sid", s)
        txt = self.dispatched[0]["text"]
        self.assertIn("monitor / watcher result for reference:", txt)
        self.assertNotIn("None", txt)

    def test_fallback_is_empty_without_a_durable_report(self):
        s = self._session()
        self.S._enqueue_watcher_hook(
            s, rid="req_no_report", label="L", task="T", report="")
        self.S._drain_watcher_hook("d", "sid", s)
        self.assertEqual(
            self.dispatched[0].get("internal_fallback_text"), "")

    def test_busy_via_monitor_hook_flag_also_blocks(self):
        s = self._session()
        s["_monitor_hook_running"] = True  # a monitor hook is mid-dispatch
        self.S._enqueue_watcher_hook(s, rid="req_e", label="L", task="T")
        self.assertFalse(self.S._drain_watcher_hook("d", "sid", s))
        self.assertEqual(len(self.dispatched), 0)
        self.assertEqual(len(s["_watcher_hook_queue"]), 1)


def test_active_watcher_hook_queues_user_without_interrupt_then_drains(
    monkeypatch,
):
    """The hook guard spans the async turn and its tail gives the user priority."""
    from tui_gateway import server

    hook_started = threading.Event()
    release_hook = threading.Event()
    user_started = threading.Event()
    both_turns_finished = threading.Event()
    prompts = []
    info_count = 0
    info_lock = threading.Lock()

    class BlockingAgent:
        compression_enabled = True
        mm_monitors = {}
        mm_watchers = {}
        session_id = "durable-watcher-guard"

        def __init__(self):
            self.interrupts = 0

        def clear_interrupt(self):
            pass

        def interrupt(self):
            self.interrupts += 1

        def run_conversation(
            self, prompt, conversation_history=None, stream_callback=None,
        ):
            prompts.append((prompt, self._ephemeral_internal_turn))
            if prompt.startswith("summarize watcher result"):
                hook_started.set()
                if not release_hook.wait(timeout=5):
                    raise TimeoutError("test did not release watcher hook")
            else:
                user_started.set()
            return {
                "final_response": "",
                "messages": list(conversation_history or []),
            }

    agent = BlockingAgent()
    session = {
        "agent": agent,
        "attached_images": [],
        "cols": 80,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "queued_prompt": None,
        "queued_prompts": [],
        "running": False,
        "session_key": "durable-watcher-guard",
        "source": "tui",
        "transport": None,
    }

    def emit(event, _sid, _payload=None):
        nonlocal info_count
        if event != "session.info":
            return
        with info_lock:
            info_count += 1
            if info_count == 2:
                both_turns_finished.set()

    monkeypatch.setitem(server._sessions, "live-watcher-guard", session)
    monkeypatch.setattr(server, "_emit", emit)
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "make_stream_renderer", lambda *_args: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_args: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_args: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *_args: "")
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)

    server._enqueue_watcher_hook(
        session,
        rid="watcher-guard",
        label="watcher",
        task="summarize watcher result",
        report="finished",
    )
    assert server._drain_watcher_hook(
        "drain", "live-watcher-guard", session) is True
    assert hook_started.wait(timeout=2)
    assert session["running"] is True
    assert session["_monitor_hook_running"] is True

    response = server._methods["prompt.submit"]("user-rpc", {
        "session_id": "live-watcher-guard",
        "text": "real user question",
        "client_request_id": "user-after-hook",
    })
    assert response["result"]["status"] == "queued"
    assert response["result"]["queue_position"] == 1
    assert agent.interrupts == 0
    assert user_started.is_set() is False
    assert [item["text"] for item in session["queued_prompts"]] == [
        "real user question"
    ]

    release_hook.set()
    assert user_started.wait(timeout=2)
    assert both_turns_finished.wait(timeout=2)
    assert prompts == [
        (
            "summarize watcher result\n"
            "monitor / watcher result for reference: finished",
            True,
        ),
        ("real user question", False),
    ]
    assert session["queued_prompts"] == []
    assert session["running"] is False
    assert session["_monitor_hook_running"] is False
    assert agent.interrupts == 0


def test_failed_watcher_hook_delivers_durable_report_fallback(monkeypatch):
    """A provider failure in the hidden synthesis turn must still complete the
    visible Assistant slot with the consolidated watcher report."""
    from tui_gateway import server

    completed = threading.Event()
    complete_payloads = []

    class FailingAgent:
        compression_enabled = True
        mm_monitors = {}
        mm_watchers = {}
        session_id = "durable-watcher-fallback"

        def clear_interrupt(self):
            pass

        def run_conversation(
            self, _prompt, conversation_history=None, stream_callback=None,
        ):
            return {
                "final_response": "",
                "messages": list(conversation_history or []),
                "failed": True,
                "error": "provider rejected tool_use",
            }

    agent = FailingAgent()
    sid = "live-watcher-fallback"
    session = {
        "agent": agent,
        "attached_images": [],
        "cols": 80,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "queued_prompt": None,
        "queued_prompts": [],
        "running": False,
        "session_key": "durable-watcher-fallback",
        "source": "tui",
        "transport": None,
    }

    def emit(event, _sid, payload=None):
        if event == "message.complete":
            complete_payloads.append(payload or {})
            completed.set()

    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_emit", emit)
    monkeypatch.setattr(server, "render_message", lambda *_args: "")
    monkeypatch.setattr(server, "make_stream_renderer", lambda *_args: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_args: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_args: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *_args: "")
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)

    server._enqueue_watcher_hook(
        session,
        rid="watcher-fallback",
        label="视频内容总结",
        task="用中文整理视频内容",
        report="第一段内容。\n\n第二段结论。",
    )
    assert server._drain_watcher_hook("drain", sid, session) is True
    assert completed.wait(timeout=3)

    assert len(complete_payloads) == 1
    payload = complete_payloads[0]
    assert payload["status"] == "complete"
    assert payload["watcher_fallback"] is True
    assert "第一段内容" in payload["text"]
    assert "第二段结论" in payload["text"]
    assert "provider rejected tool_use" not in payload["text"]


if __name__ == "__main__":
    unittest.main()
